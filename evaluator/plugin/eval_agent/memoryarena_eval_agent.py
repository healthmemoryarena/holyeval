"""
MemoryArenaEvalAgent — LLM-as-Judge multi-subtask evaluator

Registered name: "memoryarena"

How it works:
1. Extract multi-turn conversation from TestAgent / TargetAgent memory_list
2. Pair each assistant response with the corresponding ground_truth by turn
3. Concurrently call LLM to judge each subtask answer
4. Calculate Progress Score = passed subtasks / total subtasks
5. Result is scored, score = Progress Score

Supported domains: bundled_shopping, progressive_search, group_travel_planner,
formal_reasoning_math, formal_reasoning_phys.

Reference: https://arxiv.org/abs/2602.16313
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    SessionInfo,
    TestAgentMemory,
)
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1"

# ============================================================
# LLM Judge Prompt
# ============================================================

JUDGE_TEMPLATE = """You are an evaluator for the MemoryArena benchmark, which tests an AI agent's ability to solve interdependent multi-step tasks.

Your task: determine if the assistant's response correctly answers the given subtask.

# Subtask Question
<<question>>

# Expected Answer
<<expected_answer>>

# Assistant's Response
<<response>>

# Domain Context
This subtask belongs to the "<<domain>>" domain.

# Evaluation Guidelines
- Focus on whether the assistant's response contains the correct answer, not on formatting or verbosity.
- For product selection (bundled_shopping): check if the correct product was identified (matching ASIN or key attributes).
- For search queries (progressive_search): check if the answer matches the expected information semantically.
- For travel planning (group_travel_planner): check if the plan satisfies the key constraints mentioned in the expected answer.
- For formal reasoning (math/physics): check if the answer is logically and numerically correct.
- Be lenient on formatting differences but strict on factual correctness.

# Output Format
Return a JSON object:
```json
{
  "explanation": "Brief explanation of why the answer is correct or incorrect",
  "correct": true or false
}
```

Return ONLY the JSON object in markdown format. Do not include any other text."""


# ============================================================
# MemoryArenaEvalInfo — config model
# ============================================================


class MemoryArenaEvalInfo(BaseModel):
    """MemoryArena multi-subtask evaluation config — LLM-as-Judge + Progress Score

    Pairs assistant responses with ground_truths by turn, concurrently calls LLM to judge each subtask.
    Calculates Progress Score = passed subtasks / total subtasks, result is scored.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "memoryarena",
                    "domain": "bundled_shopping",
                    "ground_truths": ["Product A with ASIN B00XXX", "Total price is $45.99"],
                },
                {
                    "evaluator": "memoryarena",
                    "model": "gpt-4.1",
                    "domain": "formal_reasoning_math",
                    "ground_truths": ["42", "The answer is 3.14"],
                },
            ]
        },
    )

    evaluator: Literal["memoryarena"] = Field(default="memoryarena", description="Evaluator type")
    model: Optional[str] = Field(default="gpt-4.1", description="Judge LLM model (default gpt-4.1)")
    domain: str = Field(
        description=(
            "Evaluation domain — bundled_shopping, progressive_search, group_travel_planner, "
            "formal_reasoning_math, formal_reasoning_phys"
        ),
    )
    ground_truths: List[str] = Field(description="Standard answers for each subtask, aligned with conversation turns")


class MemoryArenaEvalAgent(AbstractEvalAgent, name="memoryarena", params_model=MemoryArenaEvalInfo):
    """MemoryArena multi-subtask evaluator — LLM-as-Judge + Progress Score"""

    _display_meta = {
        "icon": (
            "M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0"
            " 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987"
            " 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
        ),
        "color": "#8b5cf6",
        "features": ["MemoryArena", "Multi-Subtask", "Progress Score"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.06,  # ~6 subtasks × gpt-4.1 judging, USD/case
    }

    def __init__(self, eval_config: MemoryArenaEvalInfo, model: str | None = None, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.eval_config: MemoryArenaEvalInfo = eval_config
        self.model = model or eval_config.model or DEFAULT_MODEL
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # Framework interface
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        try:
            # Extract (question, response) pairs from multi-turn conversation
            qa_pairs = self._extract_qa_pairs(memory_list, self.history)
            if not qa_pairs:
                return EvalResult(result="fail", score=0.0, feedback="No conversation records, cannot evaluate")

            ground_truths = self.eval_config.ground_truths
            domain = self.eval_config.domain
            num_subtasks = len(ground_truths)

            if not ground_truths:
                return EvalResult(result="fail", score=0.0, feedback="No ground_truths configured, cannot evaluate")

            logger.info(
                "[MemoryArenaEval] Starting evaluation — %d subtasks, domain=%s, model=%s",
                num_subtasks,
                domain,
                self.model,
            )

            # Align subtask count (take min to avoid out-of-bounds)
            n = min(len(qa_pairs), num_subtasks)
            if len(qa_pairs) < num_subtasks:
                logger.warning(
                    "[MemoryArenaEval] Conversation turns (%d) fewer than subtasks (%d), missing parts count as failed",
                    len(qa_pairs),
                    num_subtasks,
                )

            # Concurrently evaluate all available subtasks
            judge_results = await asyncio.gather(
                *[
                    self._judge_subtask(qa_pairs[i]["question"], qa_pairs[i]["response"], ground_truths[i], domain)
                    for i in range(n)
                ]
            )

            # Missing subtasks default to not passed
            for _ in range(num_subtasks - n):
                judge_results.append({"correct": False, "explanation": "No response (conversation ended early)"})

            # Calculate Progress Score
            correct_count = sum(1 for r in judge_results if r.get("correct"))
            progress_score = correct_count / num_subtasks if num_subtasks > 0 else 0.0
            success = correct_count == num_subtasks

            # Build details
            subtask_details = []
            for i, result in enumerate(judge_results):
                subtask_details.append(
                    {
                        "subtask_index": i,
                        "question": qa_pairs[i]["question"][:200] if i < len(qa_pairs) else "(missing)",
                        "ground_truth": ground_truths[i][:200] if i < num_subtasks else "",
                        "correct": result.get("correct", False),
                        "explanation": result.get("explanation", ""),
                    }
                )

            feedback = (
                f"MemoryArena [{domain}] evaluation: {correct_count}/{num_subtasks} subtasks passed, "
                f"PS={progress_score:.3f}, SR={'1.0' if success else '0.0'}"
            )

            logger.info(
                "[MemoryArenaEval] Evaluation complete — PS=%.3f, SR=%s (%d/%d)",
                progress_score,
                "1.0" if success else "0.0",
                correct_count,
                num_subtasks,
            )

            return EvalResult(
                result="scored",
                score=progress_score,
                feedback=feedback,
                trace=EvalTrace(
                    eval_detail={
                        "domain": domain,
                        "subtask_results": subtask_details,
                        "progress_score": progress_score,
                        "success_rate": 1.0 if success else 0.0,
                        "correct_count": correct_count,
                        "total_subtasks": num_subtasks,
                    }
                ),
            )

        except Exception as e:
            logger.error("[MemoryArenaEval] Evaluation error: %s", e, exc_info=True)
            return EvalResult(result="fail", score=0.0, feedback=f"Evaluation error: {e}")

    # ----------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------

    async def _judge_subtask(self, question: str, response: str, ground_truth: str, domain: str) -> dict:
        """Call LLM to judge whether a single subtask answer is correct"""
        prompt = (
            JUDGE_TEMPLATE.replace("<<question>>", question)
            .replace("<<expected_answer>>", ground_truth)
            .replace("<<response>>", response)
            .replace("<<domain>>", domain)
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a precise evaluator for multi-step agent tasks.",
                    input=prompt,
                    max_tokens=500,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json_response(result.content)
                if parsed is not None and "correct" in parsed:
                    if isinstance(parsed["correct"], bool):
                        return parsed

                logger.warning(
                    "[MemoryArenaEval] Judge output format error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    result.content[:200],
                )
            except Exception as e:
                logger.warning(
                    "[MemoryArenaEval] Judge call failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

        logger.error("[MemoryArenaEval] Judge retries exhausted, defaulting to correct=False")
        return {"explanation": "Judging failed after retries", "correct": False}

    # ----------------------------------------------------------
    # Conversation extraction
    # ----------------------------------------------------------

    @staticmethod
    def _extract_qa_pairs(
        memory_list: List[TestAgentMemory],
        history: List[BaseMessage] | None = None,
    ) -> List[Dict[str, str]]:
        """Extract (question, response) pairs from memory_list

        In each conversation turn, user content is question, assistant reply is response.
        Messages in history are not counted in subtask evaluation (they are preloaded context).
        """
        pairs: List[Dict[str, str]] = []

        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue

            user_text = mem.test_reaction.action.semantic_content
            target_text = mem.target_response.extract_text() if mem.target_response else ""

            if user_text and target_text:
                pairs.append({"question": user_text, "response": target_text})

        return pairs

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """Extract JSON from LLM response (compatible with markdown wrapping)"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
