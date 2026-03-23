"""
SemanticEvalAgent — LLM-based general-purpose semantic evaluator

Registered name: "semantic"

How it works:
1. Extract full conversation history from TestAgent and TargetAgent memory_list
2. Build evaluation prompt based on configured evaluation dimensions (criteria)
3. Call gpt-5.2 to score each dimension independently (0-100)
4. Calculate weighted total score, compare with threshold for pass / fail

Config example (SemanticEvalInfo):
{
    "evaluator": "semantic",
    "criteria": [
        {
            "name": "goal_completion",
            "display_name": "Goal Completion",
            "description": "Whether the user's goal was effectively resolved in the conversation",
            "weight": 40,
            "evaluation_points": ["Whether user's request was understood", "Whether an effective response was given"]
        },
        {
            "name": "response_quality",
            "display_name": "Response Quality",
            "description": "Whether the response is accurate, complete, and professional",
            "weight": 35
        },
        {
            "name": "communication_experience",
            "display_name": "Communication Experience",
            "description": "Whether the interaction is natural, friendly, and efficient",
            "weight": 25
        }
    ],
    "threshold": 0.7
}

When criteria is not configured, uses default three dimensions (Goal Completion 40% + Response Quality 35% + Communication Experience 25%).

Integration test:
    agent = SemanticEvalAgent(SemanticEvalInfo())
    result = await agent.evaluate(
        conversation_text="User: I have a headache\\n\\nAI Assistant: How long has it lasted?",
        user_goal="Ask about the cause of headache",
    )
"""

import logging
from typing import Any, Dict, List, Literal, Optional

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

DEFAULT_MODEL = "gpt-5.2"


# ============================================================
# Evaluation config model
# ============================================================


class EvalCriterion(BaseModel):
    """Single evaluation dimension config"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Dimension identifier (English)")
    display_name: str = Field(default="", description="Dimension display name")
    description: str = Field(default="", description="Dimension evaluation description")
    weight: int = Field(default=0, ge=0, le=100, description="Weight percentage (0-100)")
    evaluation_points: List[str] = Field(default_factory=list, description="Specific checkpoints")


# Semantic evaluation default three dimensions
_DEFAULT_SEMANTIC_CRITERIA = [
    EvalCriterion(
        name="goal_completion",
        display_name="Goal Completion",
        description="Whether the user's goal was effectively resolved or advanced in the conversation",
        weight=40,
    ),
    EvalCriterion(
        name="response_quality",
        display_name="Response Quality",
        description="Whether the response is accurate, complete, and professional",
        weight=35,
    ),
    EvalCriterion(
        name="communication_experience",
        display_name="Communication Experience",
        description="Whether the interaction is natural, friendly, and efficient",
        weight=25,
    ),
]


class SemanticEvalInfo(BaseModel):
    """Semantic evaluation config — LLM-based multi-dimensional semantic evaluation (general-purpose, suitable for most scenarios)

    LLM independently scores the AI assistant's performance on each evaluation dimension (0-100),
    then calculates a weighted total score compared with threshold for pass / fail.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "Uses default three dimensions (Goal Completion 40% + Response Quality 35% + Communication Experience 25%), threshold 0.7",
                    "evaluator": "semantic",
                },
                {
                    "_comment": "Custom evaluation dimensions for symptom triage scenarios",
                    "evaluator": "semantic",
                    "model": "gpt-5.2",
                    "criteria": [
                        {
                            "name": "triage_accuracy",
                            "display_name": "Triage Accuracy",
                            "description": "Whether the AI correctly identified symptom urgency and gave reasonable department suggestions",
                            "weight": 50,
                            "evaluation_points": [
                                "Whether danger signals were identified (e.g. chest pain, breathing difficulty)",
                                "Whether suggested departments are reasonable",
                                "Whether urgent and non-urgent situations are distinguished",
                            ],
                        },
                        {
                            "name": "information_gathering",
                            "display_name": "Information Gathering",
                            "description": "Whether the AI collected sufficient diagnostic information through effective follow-up questions",
                            "weight": 30,
                        },
                        {
                            "name": "safety",
                            "display_name": "Safety",
                            "description": "Whether the AI avoided inappropriate diagnoses and recommended necessary medical visits",
                            "weight": 20,
                        },
                    ],
                    "threshold": 0.75,
                },
            ]
        },
    )

    evaluator: Literal["semantic"] = Field(default="semantic", description="Evaluator type")
    model: Optional[str] = Field(None, description="LLM model for EvalAgent (default gpt-5.2)")
    criteria: List[EvalCriterion] = Field(
        default_factory=lambda: [c.model_copy() for c in _DEFAULT_SEMANTIC_CRITERIA],
        description=(
            "Evaluation dimension list, each containing:\n"
            "  name: str (identifier), display_name: str (display name),\n"
            "  description: str (evaluation description), weight: int (weight 0-100),\n"
            "  evaluation_points: List[str] (optional, specific checkpoints)\n"
            "When not provided, uses default three dimensions: Goal Completion(40%) + Response Quality(35%) + Communication Experience(25%)"
        ),
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Pass threshold (0.0~1.0, default 0.7)",
    )


# ============================================================
# LLM structured output format
# ============================================================


class _DimensionScore(BaseModel):
    """Single dimension score"""

    name: str = Field(description="Dimension identifier (English, matches name in criteria)")
    score: int = Field(ge=0, le=100, description="Score for this dimension, integer 0-100")
    reason: str = Field(description="Scoring reason (brief)")


class _EvalLLMResponse(BaseModel):
    """LLM evaluation structured output"""

    dimensions: List[_DimensionScore] = Field(description="Scores for each dimension")
    overall_feedback: str = Field(description="Overall assessment (2-3 sentences)")
    strengths: List[str] = Field(description="List of strengths")
    issues: List[str] = Field(description="List of issues and improvement suggestions")


# ============================================================
# SemanticEvalAgent
# ============================================================


class SemanticEvalAgent(AbstractEvalAgent, name="semantic", params_model=SemanticEvalInfo):
    """LLM-based general-purpose semantic evaluator"""

    _display_meta = {
        "icon": (
            "M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09"
            "L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
            "M18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0"
            " 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0"
            " 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5"
            " 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423"
            " 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
        ),
        "color": "#8b5cf6",
        "features": ["LLM-Powered", "Multi-Dimensional Scoring", "Semantic Understanding"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.010,  # 1-2 LLM calls, USD/case
    }

    def __init__(self, eval_config: SemanticEvalInfo, model: str | None = None, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.model = model or eval_config.model or DEFAULT_MODEL
        # Cost tracking (using langchain UsageMetadata)
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        """Get accumulated cost statistics"""
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        """Accumulate LLM call cost"""
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # Framework interface: extract data from agent objects, forward to evaluate
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """Framework entry — extract data from memory then call evaluate"""
        try:
            logger.info(
                "[SemanticEval] Extracting conversation from memory: %d turns (model=%s)",
                len(memory_list),
                self.model,
            )
            conversation_text = self._memory_to_text(memory_list)
            logger.debug("[SemanticEval] Conversation text: %d chars", len(conversation_text))
            return await self.evaluate(
                conversation_text=conversation_text,
                user_goal=getattr(self.user_info, "goal", "") if self.user_info else "",
                user_context=getattr(self.user_info, "context", None) if self.user_info else None,
            )
        except Exception as e:
            logger.error("[SemanticEval] Evaluation error: %s", e, exc_info=True)
            return EvalResult(
                result="fail",
                score=0.0,
                feedback=f"Evaluation error: {e}",
            )

    # ----------------------------------------------------------
    # Core evaluation method: all params are simple types, can be called directly for integration testing
    # ----------------------------------------------------------

    async def evaluate(
        self,
        conversation_text: str,
        user_goal: str,
        user_context: str | None = None,
        criteria: List[Dict[str, Any]] | None = None,
        threshold: float | None = None,
    ) -> EvalResult:
        """Core evaluation logic

        Args:
            conversation_text: Formatted conversation text ("User: ...\\n\\nAI Assistant: ...")
            user_goal: User goal
            user_context: User background (optional)
            criteria: Evaluation dimension list (optional, defaults from self.eval_config)
            threshold: Pass threshold (optional, defaults from self.eval_config)

        Returns:
            EvalResult
        """
        # ---- 1. Configuration (params take priority over eval_config) ----
        criteria = criteria or [c.model_dump() for c in self.eval_config.criteria]
        threshold = threshold if threshold is not None else self.eval_config.threshold
        criteria = self._normalize_weights(criteria)

        if not conversation_text.strip():
            return EvalResult(
                result="fail",
                score=0.0,
                feedback="No conversation records, cannot evaluate",
            )

        criteria_summary = " + ".join(f"{c.get('display_name', c['name'])}({c.get('weight', 0)}%)" for c in criteria)
        logger.info(
            "[SemanticEval] Starting evaluation — dimensions: %s, threshold=%.2f",
            criteria_summary,
            threshold,
        )

        # ---- 2. Build context ----
        context_parts: list[str] = []
        if user_goal:
            context_parts.append(f"**User Goal**: {user_goal}")
        if user_context:
            context_parts.append(f"**User Background**: {user_context}")
        context_section = "\n".join(context_parts) if context_parts else "No additional context"

        # ---- 3. Build prompt ----
        criteria_prompt = self._build_criteria_prompt(criteria)

        system_prompt = """You are a professional conversation quality evaluation expert. Please evaluate the AI assistant's performance based on the given evaluation criteria.

Evaluation requirements:
1. Score strictly according to each dimension's description and checkpoints (0-100)
2. Scoring must be objective, fair, and well-reasoned
3. Provide specific scoring reasons and improvement suggestions
4. If a dimension has evaluation_points, check each one"""

        input_prompt = f"""## Evaluation Context

{context_section}

## Conversation Between AI Assistant and User

{conversation_text}

## Evaluation Criteria

Please evaluate on the following dimensions, scoring each independently (0-100):

{criteria_prompt}

Please output the evaluation result."""

        # ---- 4. Call LLM ----
        result = await do_execute(
            model=self.model,
            system_prompt=system_prompt,
            input=input_prompt,
            response_format=_EvalLLMResponse,
            max_tokens=2000,
        )

        # Accumulate cost
        self._accumulate_cost(result.usage)

        # ---- 5. Parse results, calculate weighted total score ----
        if result.data is None:
            logger.warning("[SemanticEval] LLM structured output parsing failed, raw content: %s", result.content[:500])
            return EvalResult(
                result="fail",
                score=0.0,
                feedback=f"LLM structured output parsing failed: {result.content[:200]}",
            )

        eval_response: _EvalLLMResponse = result.data

        # Build dimension name -> LLM score mapping
        llm_scores: Dict[str, _DimensionScore] = {d.name: d for d in eval_response.dimensions}

        total_score = 0.0
        total_weight = 0.0
        dimension_results: Dict[str, Any] = {}

        for criterion in criteria:
            name = criterion["name"]
            weight = criterion.get("weight", 100 // len(criteria))
            display_name = criterion.get("display_name", name)

            dim = llm_scores.get(name)
            score = dim.score if dim else 50
            reason = dim.reason if dim else "LLM did not return score for this dimension"

            dimension_results[name] = {
                "display_name": display_name,
                "score": score,
                "weight": weight,
                "reason": reason,
            }

            total_score += score * weight
            total_weight += weight

            logger.info(
                "[SemanticEval]   %s: %d pts (weight %d%%), reason: %s",
                display_name,
                score,
                weight,
                reason,
            )

        # Final score (0-1 range)
        final_score = (total_score / total_weight / 100) if total_weight > 0 else 0.0
        passed = final_score >= threshold

        # ---- 6. Build feedback ----
        feedback_parts: list[str] = []
        if eval_response.overall_feedback:
            feedback_parts.append(eval_response.overall_feedback)
        if eval_response.strengths:
            feedback_parts.append(f"Strengths: {'; '.join(eval_response.strengths)}")
        if eval_response.issues:
            feedback_parts.append(f"Issues: {'; '.join(eval_response.issues)}")
        feedback = " | ".join(feedback_parts)

        if eval_response.strengths:
            logger.info("[SemanticEval] Strengths: %s", "; ".join(eval_response.strengths))
        if eval_response.issues:
            logger.info("[SemanticEval] Issues: %s", "; ".join(eval_response.issues))
        logger.info(
            "[SemanticEval] Evaluation complete — weighted total=%.2f, threshold=%.2f, result=%s",
            final_score,
            threshold,
            "pass" if passed else "fail",
        )

        return EvalResult(
            result="pass" if passed else "fail",
            score=final_score,
            feedback=feedback,
            trace=EvalTrace(eval_detail={"dimensions": dimension_results}),
        )

    # ============================================================
    # Helper methods
    # ============================================================

    @staticmethod
    def _normalize_weights(criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize weights to sum to 100"""
        total = sum(c.get("weight", 0) for c in criteria)
        if total == 100 or total == 0:
            if total == 0:
                avg = 100 // len(criteria) if criteria else 100
                for c in criteria:
                    c["weight"] = avg
            return criteria

        logger.warning("[SemanticEval] Weight sum is %d, auto-normalizing to 100", total)
        for c in criteria:
            c["weight"] = round(c.get("weight", 0) * 100 / total)
        return criteria

    @staticmethod
    def _memory_to_text(memory_list: List[TestAgentMemory]) -> str:
        """Extract and format conversation text from TestAgent's memory_list"""
        lines: list[str] = []

        for mem in memory_list:
            # Skip is_finished ghost turns (not sent to target)
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue

            # What the user (TestAgent) said
            user_text = mem.test_reaction.action.semantic_content
            if user_text:
                lines.append(f"User: {user_text}")

            # What the system under test responded
            target_text = mem.target_response.extract_text() if mem.target_response else ""
            if target_text:
                if len(target_text) > 2000:
                    target_text = target_text[:2000] + "...(truncated)"
                lines.append(f"AI Assistant: {target_text}")

        return "\n\n".join(lines)

    @staticmethod
    def _build_criteria_prompt(criteria: List[Dict[str, Any]]) -> str:
        """Build evaluation dimension prompt"""
        lines: list[str] = []

        for i, criterion in enumerate(criteria, 1):
            name = criterion.get("name", f"criterion_{i}")
            display_name = criterion.get("display_name", name)
            description = criterion.get("description", "")
            weight = criterion.get("weight", 100 // len(criteria))
            evaluation_points = criterion.get("evaluation_points", [])

            lines.append(f"### {i}. {display_name} ({name}) — Weight {weight}%")
            if description:
                lines.append(f"Evaluation description: {description}")
            if evaluation_points:
                lines.append("Checkpoints:")
                for point in evaluation_points:
                    lines.append(f"  - {point}")
            lines.append("")

        return "\n".join(lines)
