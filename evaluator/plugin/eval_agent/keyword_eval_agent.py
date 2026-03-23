"""
KeywordEvalAgent — Keyword matching evaluator

Rule-based keyword matching evaluation, fast, free, and highly deterministic.
Suitable for structured checking scenarios, such as verifying whether a conversation contains specific keywords.

Config example (KeywordEvalInfo):
{
    "evaluator": "keyword",
    "rules": [
        {
            "rule_id": "check_location",
            "rule_type": "keyword_check",
            "keywords": ["位置", "哪里", "部位"],
            "match_mode": "any",
            "weight": 0.3,
            "threshold": 0.5
        },
        {
            "rule_id": "check_duration",
            "rule_type": "keyword_check",
            "keywords": ["多久", "多长时间", "持续"],
            "match_mode": "any",
            "weight": 0.3,
            "threshold": 0.5
        }
    ],
    "pass_threshold": 0.7
}
"""

import logging
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    SessionInfo,
    TargetAgentReaction,
    TestAgentMemory,
)

logger = logging.getLogger(__name__)


# ============================================================
# Evaluation config model
# ============================================================


class KeywordRule(BaseModel):
    """Keyword matching rule"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(description="Rule identifier")
    rule_type: str = Field(default="keyword_check", description="Rule type (currently only keyword_check)")
    keywords: List[str] = Field(description="List of keywords to match")
    match_mode: str = Field(
        default="any",
        description=(
            "Match mode:\n"
            "  - any: pass if any keyword is hit\n"
            "  - all: must hit all keywords\n"
            "  - ratio: pass if hits >= min_matches"
        ),
    )
    weight: float = Field(default=1.0, ge=0.0, description="Weight")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Pass threshold")
    target: str = Field(
        default="output",
        description="Match target: output (AI response only) | conversation (full conversation)",
    )
    min_matches: int = Field(default=1, ge=1, description="Minimum matches (ratio mode)")


class KeywordEvalInfo(BaseModel):
    """Keyword evaluation config — rule-based keyword matching evaluation (fast, free, highly deterministic)

    Suitable for structured checking scenarios, such as verifying whether a conversation contains specific keywords.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "keyword",
                    "rules": [
                        {
                            "rule_id": "check_location",
                            "keywords": ["位置", "哪里", "部位"],
                            "match_mode": "any",
                            "weight": 0.3,
                            "threshold": 0.5,
                        },
                        {
                            "rule_id": "check_duration",
                            "keywords": ["多久", "多长时间", "持续"],
                            "match_mode": "any",
                            "weight": 0.3,
                            "threshold": 0.5,
                        },
                    ],
                    "pass_threshold": 0.7,
                },
            ]
        },
    )

    evaluator: Literal["keyword"] = Field(description="Evaluator type")
    model: Optional[str] = Field(None, description="Reserved field (keyword evaluation does not use LLM)")
    rules: List[KeywordRule] = Field(description="List of keyword matching rules")
    pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall pass threshold (0.0~1.0, default 0.7)",
    )


class KeywordEvalAgent(AbstractEvalAgent, name="keyword", params_model=KeywordEvalInfo):
    """Keyword matching evaluator"""

    _display_meta = {
        "icon": (
            "M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872"
            " 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568"
            " 3z M6 6h.008v.008H6V6z"
        ),
        "color": "#f59e0b",
        "features": ["Zero LLM", "Keyword Matching", "Rule-Driven"],
    }
    _cost_meta = {
        "est_cost_per_case": 0,  # Pure rule matching, no LLM calls
    }

    def __init__(self, eval_config: KeywordEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.rules = eval_config.rules
        self.pass_threshold = eval_config.pass_threshold

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """Execute keyword evaluation"""
        try:
            if not self.rules:
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback="Missing rules configuration",
                    trace={"error": "missing rules"},
                )

            # Extract conversation history
            conversation = self._extract_conversation(memory_list)

            if not conversation:
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback="No conversation records",
                    trace={"error": "no conversation"},
                )

            # Build text
            output_text = self._build_output_text(conversation)
            conversation_text = self._build_conversation_text(conversation)

            # Evaluate each rule
            metrics = []
            for rule in self.rules:
                metric = self._evaluate_rule(rule, output_text, conversation_text)
                metrics.append(metric)

            # Calculate weighted total score
            if metrics:
                total_weight = sum(m["weight"] for m in metrics)
                overall_score = (
                    sum(m["score"] * m["weight"] for m in metrics) / total_weight
                    if total_weight > 0
                    else 0.0
                )
            else:
                overall_score = 1.0

            passed = overall_score >= self.pass_threshold

            # Build feedback
            passed_count = sum(1 for m in metrics if m["passed"])
            failed_rules = [m["metric_name"] for m in metrics if not m["passed"]]

            feedback_parts = [f"Score: {overall_score:.2f}"]
            feedback_parts.append(f"Passed: {passed_count}/{len(metrics)} rules")
            if failed_rules:
                feedback_parts.append(f"Failed: {', '.join(failed_rules)}")

            feedback = " | ".join(feedback_parts)

            logger.info(
                f"[KeywordEval] Evaluation complete: score={overall_score:.2f}, "
                f"passed={passed}, {len(metrics)} rules"
            )

            return EvalResult(
                result="pass" if passed else "fail",
                score=overall_score,
                feedback=feedback,
                trace={
                    "metrics": metrics,
                    "rule_count": len(metrics),
                    "passed_rules": passed_count,
                    "failed_rules": len(failed_rules),
                    "threshold": self.pass_threshold,
                },
            )

        except Exception as e:
            logger.error(f"[KeywordEval] Evaluation error: {e}", exc_info=True)
            return EvalResult(
                result="fail",
                score=0.0,
                feedback=f"Evaluation error: {str(e)}",
                trace={"error": str(e)},
            )

    def _extract_conversation(
        self, memory_list: list[TestAgentMemory]
    ) -> list[dict[str, str]]:
        """Extract conversation history from memory"""
        conversation = []

        for memory in memory_list:
            # Skip is_finished ghost turns (not sent to target)
            if memory.test_reaction and memory.test_reaction.is_finished and memory.target_response is None:
                continue

            # User message
            if memory.test_reaction and memory.test_reaction.action:
                content = memory.test_reaction.action.semantic_content
                if content:
                    conversation.append({"role": "user", "content": content})

            # AI response
            target_text = self._extract_target_text(memory.target_response)
            if target_text:
                conversation.append({"role": "assistant", "content": target_text})

        return conversation

    @staticmethod
    def _extract_target_text(target_reaction: Optional[TargetAgentReaction]) -> str:
        """Extract text from TargetAgentReaction"""
        if target_reaction is None:
            return ""

        if target_reaction.type == "message" and target_reaction.message_list:
            parts = []
            for msg in target_reaction.message_list:
                if isinstance(msg, dict) and msg.get("content"):
                    parts.append(str(msg["content"]))
            return "\n".join(parts)

        return ""

    def _build_output_text(self, conversation: list[dict[str, str]]) -> str:
        """Build AI output text (assistant responses only)"""
        return "\n".join(
            msg["content"] for msg in conversation if msg["role"] == "assistant"
        )

    def _build_conversation_text(self, conversation: list[dict[str, str]]) -> str:
        """Build full conversation text"""
        return "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation
        )

    def _evaluate_rule(
        self,
        rule: KeywordRule,
        output_text: str,
        conversation_text: str,
    ) -> dict[str, Any]:
        """Evaluate a single rule"""
        if rule.rule_type == "keyword_check":
            return self._evaluate_keyword_check(rule, output_text, conversation_text)
        else:
            logger.warning(f"[KeywordEval] Unknown rule type: {rule.rule_type}")
            return {
                "metric_name": rule.rule_id,
                "score": 0.0,
                "passed": False,
                "weight": rule.weight,
                "reason": f"Unknown rule type: {rule.rule_type}",
            }

    def _evaluate_keyword_check(
        self,
        rule: KeywordRule,
        output_text: str,
        conversation_text: str,
    ) -> dict[str, Any]:
        """Evaluate keyword check rule"""
        keywords = rule.keywords
        match_mode = rule.match_mode
        target = rule.target
        threshold = rule.threshold
        weight = rule.weight

        # Select target text
        text = conversation_text if target == "conversation" else output_text

        # Match keywords
        matched = [kw for kw in keywords if kw in text]

        # Calculate score based on match mode
        if match_mode == "any":
            score = 1.0 if matched else 0.0
        elif match_mode == "all":
            score = len(matched) / len(keywords) if keywords else 1.0
        elif match_mode == "ratio":
            min_matches = rule.min_matches
            score = (
                1.0 if len(matched) >= min_matches else len(matched) / min_matches
            )
        else:
            score = len(matched) / len(keywords) if keywords else 1.0

        passed = score >= threshold

        return {
            "metric_name": rule.rule_id,
            "score": score,
            "passed": passed,
            "weight": weight,
            "reason": f"Matched {len(matched)}/{len(keywords)} keywords: {matched}",
            "details": {
                "matched_keywords": matched,
                "missing_keywords": [kw for kw in keywords if kw not in matched],
                "total_keywords": len(keywords),
                "match_mode": match_mode,
            },
        }
