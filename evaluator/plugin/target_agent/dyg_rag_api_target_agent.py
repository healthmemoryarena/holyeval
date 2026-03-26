"""
DygRagApiTargetAgent — DyG-RAG dynamic graph retrieval-augmented QA

Registered name: "dyg_rag_api"

Based on DyG-RAG (Dynamic Graph RAG) paper evaluation logic:
1. Load user data -> convert to event_docs format ([PROFILE] + [DATA] temporal health records)
2. Build dynamic event graph + temporal vector index via GraphRAG engine (auto-build on first use, cached afterward)
3. Query phase: time/entity parsing -> time-weighted vector retrieval -> event graph random walk -> Time-CoT -> LLM answer

Differences from hippo_rag_api:
- hippo_rag_api: Simple vector retrieval (embed -> cosine similarity -> top-k chunks)
- dyg_rag_api: Dynamic event graph + temporal weighted retrieval + graph traversal (better suited for temporal health data)

Source: GraphRAG engine from /ThetaAI/tianyuchen/dyg-rag/
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    SessionInfo,
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.utils.llm import BasicMessage

logger = logging.getLogger(__name__)

# benchmark/data/ directory
_BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"

# GraphRAG instance cache (key: user_dir path string)
_GRAPHRAG_CACHE: dict[str, Any] = {}


# ============================================================
# Data conversion: thetagen -> DyG-RAG event_docs
# ============================================================


def _email_to_dir(email: str) -> str:
    """Email -> directory name: user110@demo -> user110_AT_demo"""
    return email.replace("@", "_AT_")


def _prepare_event_docs(user_dir: Path) -> str:
    """Convert thetagen user data to DyG-RAG event_docs format

    DyG-RAG expects a single document with [PROFILE] + [DATA] sections,
    consistent with the event_docs format in /ThetaAI/tianyuchen/dyg-rag/datasets/.
    """
    parts: list[str] = []

    # === [PROFILE] section ===
    profile_file = user_dir / "profile.json"
    if profile_file.exists():
        with open(profile_file, encoding="utf-8") as f:
            profile = json.load(f)
        demographics = profile.get("demographics", {})
        health_profile = profile.get("health_profile", {})
        personality = profile.get("personality", "")
        parts.append("[PROFILE]")
        if demographics:
            parts.append(json.dumps(demographics, ensure_ascii=False, indent=2))
        if health_profile:
            parts.append(f"\nHealth Profile:\n{json.dumps(health_profile, ensure_ascii=False, indent=2)}")
        if personality:
            parts.append(f"\nPersonality: {personality[:500]}")

    # === [DATA] section: Health Events ===
    events_file = user_dir / "events.json"
    if events_file.exists():
        with open(events_file, encoding="utf-8") as f:
            events = json.load(f)
        if isinstance(events, list):
            parts.append("\n[DATA] Health Events")
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                line = (
                    f"Event: {ev.get('event_name', '?')}, "
                    f"Start: {ev.get('start_date', '?')}, "
                    f"Duration: {ev.get('duration_days', '?')} days, "
                    f"Impact: {ev.get('impact_level', '?')}, "
                    f"Interrupted: {ev.get('interrupted', False)}"
                )
                parts.append(line)
                for ai in ev.get("affected_indicators", []):
                    if isinstance(ai, dict):
                        parts.append(
                            f"  Indicator: {ai.get('indicator_name', '?')} "
                            f"({ai.get('indicator_key', '')}), "
                            f"Expected: {ai.get('expected_change', '?')}, "
                            f"Impact: {ai.get('impact_level', '?')}, "
                            f"TimeToEffect: {ai.get('time_to_effect', '?')}, "
                            f"FadeOut: {ai.get('fade_out_days', '?')}"
                        )
                for med in ev.get("medications", []):
                    if isinstance(med, dict):
                        parts.append(f"  Medication: {med.get('name', '?')}, Dosage: {med.get('dosage', '?')}")
                    elif isinstance(med, str):
                        parts.append(f"  Medication: {med}")

    # === [DATA] section: Exam Results ===
    exam_file = user_dir / "exam_data.json"
    if exam_file.exists():
        with open(exam_file, encoding="utf-8") as f:
            exam = json.load(f)
        if isinstance(exam, list) and exam:
            parts.append("\n[DATA] Exam Results")
            for item in exam:
                parts.append(json.dumps(item, ensure_ascii=False))

    # === [DATA] section: Device Data ===
    device_file = user_dir / "device_data.json"
    if device_file.exists():
        with open(device_file, encoding="utf-8") as f:
            device = json.load(f)
        if isinstance(device, list) and device:
            parts.append("\n[DATA] Device Indicators")
            for item in device:
                parts.append(json.dumps(item, ensure_ascii=False))

    return "\n".join(parts)


# ============================================================
# Config Model
# ============================================================


class DygRagApiTargetInfo(BaseModel):
    """DyG-RAG target config — dynamic graph retrieval-augmented generation"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "Default config (uses Gemini for event extraction and QA)",
                    "type": "dyg_rag_api",
                    "model": "gemini-3-pro-preview",
                    "data_group": "eslbench",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["dyg_rag_api"] = Field(description="Target type")
    model: str = Field(
        "gemini-3-pro-preview",
        description="QA generation model (default gemini-3-pro-preview, used for final answer generation)",
    )
    data_group: str = Field(description="Data directory (benchmark name, e.g. 'eslbench')")
    user_email: Optional[str] = Field(None, description="User email (maps to .data/{user_dir}/)")
    top_k: int = Field(20, description="DyG-RAG seed event count (starting nodes for graph traversal)", ge=1, le=100)
    enable_graph_traversal: bool = Field(True, description="Enable graph random walk (disabled uses vector retrieval only)")


# ============================================================
# Target Agent
# ============================================================


class DygRagApiTargetAgent(
    AbstractTargetAgent,
    name="dyg_rag_api",
    params_model=DygRagApiTargetInfo,
):
    """DyG-RAG dynamic graph retrieval-augmented QA — temporal event graph based health data RAG

    Flow:
    1. Auto-builds index on first query (LLM event extraction -> dynamic event graph + temporal VDB)
    2. Query: time/entity parsing -> time-weighted vector retrieval -> graph random walk -> Time-CoT -> LLM answer
    """

    _display_meta = {
        "icon": (
            "M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0"
            " 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997"
            " 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0"
            " 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099"
            " 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5a17.92 17.92 0 01-8.716-2.247m0"
            " 0A9.015 9.015 0 003 12c0-1.605.42-3.113 1.157-4.418"
        ),
        "color": "#10b981",
        "features": ["DyG-RAG", "Event Graph", "Temporal Search", "Graph Walk"],
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(
        self,
        target_config: DygRagApiTargetInfo,
        history: list[BaseMessage] | None = None,
    ):
        super().__init__(target_config, history=history)
        self.config: DygRagApiTargetInfo = target_config

        # Conversation history
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # User data directory
        self._user_dir: Path | None = None
        if target_config.user_email:
            dir_name = _email_to_dir(target_config.user_email)
            self._user_dir = _BENCHMARK_DATA_DIR / target_config.data_group / ".data" / dir_name
            if not self._user_dir.is_dir():
                logger.warning("[DyG-RAG] User data directory not found: %s", self._user_dir)

        # GraphRAG instance (lazy initialization)
        self._graphrag: Any = None
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    async def _ensure_graphrag(self) -> None:
        """Ensure GraphRAG instance is initialized and index building is complete"""
        if self._graphrag is not None:
            return

        cache_key = str(self._user_dir)
        if cache_key in _GRAPHRAG_CACHE:
            self._graphrag = _GRAPHRAG_CACHE[cache_key]
            logger.info("[DyG-RAG] Using cached GraphRAG instance: %s", self._user_dir.name)
            return

        from evaluator.vendor.dyg_graphrag import GraphRAG

        # Index storage directory (aligned with server-56591 work_dirs structure)
        work_dir = _BENCHMARK_DATA_DIR / self.config.data_group / ".dyg_work_dirs" / self._user_dir.name
        work_dir.mkdir(parents=True, exist_ok=True)
        index_done_flag = work_dir / ".index_done"

        # GEMINI_API_KEY compatibility: holyeval uses GOOGLE_API_KEY, DyG-RAG uses GEMINI_API_KEY
        if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

        # Set model (override DyG-RAG defaults)
        original_model = os.environ.get("GEMINI_MODEL")
        if self.config.model:
            os.environ["GEMINI_MODEL"] = self.config.model

        try:
            # Initialize GraphRAG (auto-reads GEMINI_API_KEY and other env vars)
            # Disable cross-encoder (requires PyTorch), use BM25 reranking instead
            rag = GraphRAG(
                working_dir=str(work_dir),
                enable_ce_rerank=False,
                enable_bm25_reranking=True,
                enable_graph_traversal=self.config.enable_graph_traversal,
            )
        finally:
            # Restore original env vars
            if original_model is not None:
                os.environ["GEMINI_MODEL"] = original_model
            elif "GEMINI_MODEL" in os.environ and self.config.model:
                del os.environ["GEMINI_MODEL"]

        # Auto-build index (slow on first run, cached afterward)
        if not index_done_flag.exists():
            logger.info("[DyG-RAG] Building index for %s (first time may take several minutes)...", self._user_dir.name)
            doc_context = _prepare_event_docs(self._user_dir)
            if not doc_context:
                raise ValueError(f"User data is empty: {self._user_dir}")
            await rag.ainsert(doc_context)
            index_done_flag.touch()
            logger.info("[DyG-RAG] Index build complete: %s", self._user_dir.name)
        else:
            logger.info("[DyG-RAG] Loading existing index: %s", work_dir)

        self._graphrag = rag
        _GRAPHRAG_CACHE[cache_key] = rag

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        if test_action is None:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "Hello, how can I help you?"}],
            )

        user_text = self._extract_user_input(test_action)

        if not self._user_dir or not self._user_dir.is_dir():
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": f"Error: User data directory not found ({self._user_dir})"}],
            )

        # Initialize GraphRAG (includes auto index building)
        try:
            await self._ensure_graphrag()
        except Exception as e:
            logger.error("[DyG-RAG] GraphRAG initialization failed: %s", e, exc_info=True)
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": f"GraphRAG initialization error: {e}"}],
            )

        from evaluator.vendor.dyg_graphrag import QueryParam

        logger.info("[DyG-RAG] Query: %s...", user_text[:80])

        try:
            answer = await self._graphrag.aquery(
                user_text,
                param=QueryParam(mode="dynamic", top_k=self.config.top_k),
            )
        except Exception as e:
            logger.error("[DyG-RAG] Query failed: %s", e, exc_info=True)
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": f"DyG-RAG query error: {e}"}],
            )

        # Append conversation history
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=answer))

        logger.info("[DyG-RAG] Response: %d chars", len(answer))

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": answer}],
        )

    def get_session_info(self) -> SessionInfo:
        return SessionInfo(has_user_data=bool(self._user_dir and self._user_dir.is_dir()))

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        return str(test_action.custom_content) if test_action.custom_content else ""
