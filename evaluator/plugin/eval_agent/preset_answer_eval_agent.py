"""
PresetAnswerEvalAgent — Preset answer accuracy evaluator

Registered name: "preset_answer"

Deterministic comparison of AI assistant's response against standard answers, zero LLM calls.
Supports three matching modes:
  - number: extract numbers, allow relative error tolerance (default 1%)
  - keyword: keyword matching
  - exact: exact string matching

Use cases: data extraction validation, preset Q&A, regression testing.
"""

import logging
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, SessionInfo, TestAgentMemory

logger = logging.getLogger(__name__)


# ============================================================
# Evaluation config model
# ============================================================


class PresetAnswerEvalInfo(BaseModel):
    """Preset answer accuracy evaluation — single-item deterministic comparison, zero LLM

    Compares AI assistant's response against standard answers with number tolerance, keyword, and exact matching.
    Suitable for data extraction validation, preset Q&A, and other scenarios with clear standard answers.

    Each TestCase contains one question and one standard answer; batch execution calculates accuracy via TestReport.pass_rate.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "Number matching (default), allows 1% tolerance",
                    "evaluator": "preset_answer",
                    "standard_answer": "59.06",
                },
                {
                    "_comment": "Keyword matching",
                    "evaluator": "preset_answer",
                    "standard_answer": "建议就医",
                    "match_mode": "keyword",
                },
            ]
        },
    )

    evaluator: Literal["preset_answer"] = Field(description="Evaluator type")
    model: Optional[str] = Field(None, description="Reserved field (LLM not used)")
    standard_answer: str = Field(description="Standard answer — numeric types are auto-extracted and compared, text types use keyword matching")
    match_mode: str = Field(
        default="number",
        description=(
            "Matching mode:\n"
            "  - number: extract numbers, allow relative error tolerance (default)\n"
            "  - keyword: keyword matching\n"
            "  - exact: exact string matching"
        ),
    )
    number_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Number matching relative error tolerance (default 0.01 = 1%)",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Number/keyword hit ratio threshold (default 0.5 = 50% hit means correct)",
    )

# Number extraction regex: supports 1,234.56 / -10 / 0.5 etc.
# Note: supports both Unicode minus U+2212 and ASCII hyphen -
_NUMBER_RE = re.compile(r"[\-\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _normalize_minus(text: str) -> str:
    """Replace Unicode minus (U+2212) with ASCII hyphen (-) to avoid missed number extraction"""
    return text.replace("\u2212", "-")


class PresetAnswerEvalAgent(AbstractEvalAgent, name="preset_answer", params_model=PresetAnswerEvalInfo):
    """Preset answer accuracy evaluator — deterministic comparison, zero LLM"""

    _display_meta = {
        "icon": "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
        "color": "#06b6d4",
        "features": ["Zero LLM", "Exact Matching", "Deterministic Scoring"],
    }
    _cost_meta = {
        "est_cost_per_case": 0,  # Pure rule comparison, no LLM calls
    }

    def __init__(self, eval_config: PresetAnswerEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.config: PresetAnswerEvalInfo = eval_config

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """Extract AI response from conversation memory and compare against standard answer"""
        standard_answer = self.config.standard_answer

        # Extract all AI response text from memory
        generated_parts: List[str] = []
        for mem in memory_list:
            if mem.target_response is not None:
                text = mem.target_response.extract_text()
                if text:
                    generated_parts.append(text)

        generated_answer = "\n".join(generated_parts)

        if not generated_answer.strip():
            logger.warning("[PresetAnswerEval] AI response is empty")
            return EvalResult(
                result="fail",
                score=0.0,
                feedback="AI response is empty, cannot compare",
            )

        # 502 / API infrastructure error -> mark as system_error, excluded from Agent capability scoring
        if "502 Bad Gateway" in generated_answer or "API 调用失败" in generated_answer:
            logger.warning("[PresetAnswerEval] API infrastructure error detected, marking as system_error")
            return EvalResult(
                result="error",
                score=0.0,
                feedback="system_error: API infrastructure failure (502), excluded from scoring",
            )

        logger.info(
            "[PresetAnswerEval] Standard answer: %s, match mode: %s",
            standard_answer, self.config.match_mode,
        )
        logger.info(
            "[PresetAnswerEval] AI response: %s",
            generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer,
        )

        # Compare based on match mode
        mode = self.config.match_mode
        if mode == "number":
            is_correct, detail = self._match_number(
                generated_answer, standard_answer,
                tolerance=self.config.number_tolerance,
                threshold=self.config.threshold,
            )
        elif mode == "keyword":
            is_correct, detail = self._match_keyword(
                generated_answer, standard_answer,
                threshold=self.config.threshold,
            )
        elif mode == "exact":
            is_correct, detail = self._match_exact(generated_answer, standard_answer)
        else:
            return EvalResult(
                result="fail",
                score=0.0,
                feedback=f"Unknown match mode: {mode}",
            )

        score = 1.0 if is_correct else 0.0
        result = "pass" if is_correct else "fail"

        logger.info("[PresetAnswerEval] Result: %s, detail: %s", result, detail)

        return EvalResult(
            result=result,
            score=score,
            feedback=detail,
        )

    # ============================================================
    # Matching methods
    # ============================================================

    @staticmethod
    def _match_number(
        generated: str,
        standard: str,
        tolerance: float = 0.01,
        threshold: float = 0.5,
    ) -> tuple[bool, str]:
        """Number matching — extract numbers, allow relative error tolerance

        Returns:
            (is_correct, detail_message)
        """
        std_numbers = _NUMBER_RE.findall(_normalize_minus(standard))
        gen_numbers = _NUMBER_RE.findall(_normalize_minus(generated))

        if not std_numbers:
            # Standard answer has no numbers, fallback to keyword matching
            return PresetAnswerEvalAgent._match_keyword(generated, standard, threshold)

        try:
            std_floats = [float(n.replace(",", "")) for n in std_numbers]
            gen_floats = [float(n.replace(",", "")) for n in gen_numbers]
        except ValueError:
            return PresetAnswerEvalAgent._match_keyword(generated, standard, threshold)

        matched = 0
        for std_num, std_str in zip(std_floats, std_numbers):
            for gen_num, gen_str in zip(gen_floats, gen_numbers):
                if _numbers_close_with_precision(
                    std_num, gen_num, std_str, gen_str, tolerance
                ):
                    matched += 1
                    break

        ratio = matched / len(std_floats)
        is_correct = ratio >= threshold

        detail = (
            f"Number match: {matched}/{len(std_floats)} hit "
            f"(ratio={ratio:.0%}, threshold={threshold:.0%}, tolerance={tolerance:.0%})"
        )
        return is_correct, detail

    @staticmethod
    def _match_keyword(
        generated: str,
        standard: str,
        threshold: float = 0.3,
    ) -> tuple[bool, str]:
        """Keyword matching — extract standard answer keywords, check hit rate

        Chinese text without space delimiters automatically switches to 2-gram matching.
        """

        def strip_markdown(text: str) -> str:
            """Remove common markdown formatting"""
            # Remove bold/italic: **text** / *text* / __text__ / _text_
            text = re.sub(r"\*{1,3}|_{1,3}", "", text)
            # Remove headings # ## ###
            text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
            # Remove inline code `code`
            text = re.sub(r"`[^`]*`", lambda m: m.group().strip("`"), text)
            # Remove links [text](url)
            text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
            return text

        def split_camel(text: str) -> str:
            """Split CamelCase into space-separated lowercase words: DailyStepCount -> daily step count"""
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
            text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
            return text

        def normalize(text: str) -> str:
            text = strip_markdown(text)
            text = split_camel(text)
            # Replace hyphens/semicolons and other separators with spaces to avoid abbreviation concatenation
            text = re.sub(r"[-;/]", " ", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.lower().strip()

        norm_gen = normalize(generated)
        norm_std = normalize(standard)

        # Split by spaces
        space_tokens = [w for w in norm_std.split() if len(w) >= 2]

        # If multiple words extracted (English/mixed text), use word-level matching
        # If single token (Chinese without spaces), use 2-gram matching
        if len(space_tokens) > 1:
            keywords = space_tokens
        elif len(norm_std) >= 2:
            # Chinese: sliding window 2-gram
            keywords = [norm_std[i : i + 2] for i in range(len(norm_std) - 1)]
        else:
            is_correct = norm_std in norm_gen if norm_std else False
            detail = f"Empty keyword, substring match: {'hit' if is_correct else 'miss'}"
            return is_correct, detail

        matched = sum(1 for w in keywords if w in norm_gen)
        ratio = matched / len(keywords)
        is_correct = ratio >= threshold

        detail = (
            f"Keyword match: {matched}/{len(keywords)} hit "
            f"(ratio={ratio:.0%}, threshold={threshold:.0%})"
        )
        return is_correct, detail

    @staticmethod
    def _match_exact(generated: str, standard: str) -> tuple[bool, str]:
        """Exact matching — whether standard answer string appears in AI response"""
        is_correct = standard.strip() in generated
        detail = f"Exact match: {'hit' if is_correct else 'miss'}"
        return is_correct, detail


# ============================================================
# Utility functions
# ============================================================


def _decimal_places(s: str) -> int:
    """Get decimal places from raw number string: '463.74' -> 2, '25' -> 0"""
    s = s.replace(",", "")
    if "." in s:
        return len(s.split(".")[-1])
    return 0


def _numbers_close(a: float, b: float, tolerance: float) -> bool:
    """Check if two numbers are within relative error tolerance"""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < 1e-6
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance


def _numbers_close_with_precision(
    a: float, b: float,
    a_str: str, b_str: str,
    tolerance: float,
) -> bool:
    """First compare with original tolerance; if failed, try precision-aligned comparison.

    Example: standard answer 463.7415 (4 dp) vs AI answer 463.74 (2 dp)
    -> round 463.7415 to 2 dp = 463.74 -> equal -> pass
    """
    # Step 1: original tolerance comparison
    if _numbers_close(a, b, tolerance):
        return True

    # Step 2: precision alignment — round higher-precision number to lower precision then compare
    dp_a = _decimal_places(a_str)
    dp_b = _decimal_places(b_str)
    min_dp = min(dp_a, dp_b)

    rounded_a = round(a, min_dp)
    rounded_b = round(b, min_dp)
    return _numbers_close(rounded_a, rounded_b, tolerance)
