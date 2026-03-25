"""
KgQaEvalAgent — KG knowledge Q&A hybrid evaluator

Registered name: "kg_qa"

Based on thetagendata KG evaluation queries, routes to different evaluation strategies by answer_type:
  - numeric_value: rule-based number extraction + tolerance comparison (zero LLM)
  - boolean:       rule-based yes/no signal word extraction (zero LLM)
  - list:          rule-based set comparison + recall scoring (zero LLM)
  - text:          LLM-as-Judge, using key_points as scoring criteria

Scoring: 0.0~1.0 continuous score, result="scored".
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.messages import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, EvalTrace, SessionInfo, TestAgentMemory
from evaluator.utils.llm import accumulate_usage, do_execute

logger = logging.getLogger(__name__)

# ============================================================
# ANSWER: line extraction (prefer extracting answer from structured output)
# ============================================================

_ANSWER_LINE_RE = re.compile(
    r"(?:^|\n)\s*ANSWER\s*:\s*(.+?)(?:\n|$)"
)


def _extract_answer_line(text: str) -> Optional[str]:
    """Extract ANSWER: line content from response, returns None if not found"""
    m = _ANSWER_LINE_RE.search(text)
    return m.group(1).strip() if m else None


def _word_boundary_match(pattern: str, text: str) -> bool:
    """Signal word matching: short English patterns (<=5 chars, pure ASCII) use \\b word boundary regex
    to avoid false matches on substrings like 'know', 'note', 'cannot';
    Chinese and long English patterns use plain substring matching."""
    if len(pattern) <= 5 and pattern.isascii():
        return bool(re.search(rf"\b{re.escape(pattern)}\b", text))
    return pattern in text


# ============================================================
# Number extraction and comparison utilities
# ============================================================

# Number extraction regex: supports 1,234.56 / -10 / 0.5 / Unicode minus
_NUMBER_RE = re.compile(r"[\-\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _normalize_minus(text: str) -> str:
    """Replace Unicode minus (U+2212) with ASCII hyphen"""
    return text.replace("\u2212", "-")


def _decimal_places(s: str) -> int:
    """Get decimal places of a number string"""
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
    a: float, b: float, a_str: str, b_str: str, tolerance: float
) -> bool:
    """Tolerance comparison + precision alignment: round higher-precision number to lower precision then compare"""
    if _numbers_close(a, b, tolerance):
        return True
    dp_a = _decimal_places(a_str)
    dp_b = _decimal_places(b_str)
    min_dp = min(dp_a, dp_b)
    return _numbers_close(round(a, min_dp), round(b, min_dp), tolerance)


# ============================================================
# Keyword matching utilities
# ============================================================

# Strip markdown formatting
_MD_BOLD_RE = re.compile(r"\*{1,3}|_{1,3}")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_CODE_RE = re.compile(r"`[^`]*`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^\)]*\)")


_REFUSAL_PATTERNS = [
    "i cannot find", "i don't have", "i couldn't find",
    "no data", "no record", "not available", "not found",
    "无法找到", "没有找到", "未找到", "无相关记录", "查不到",
    "could you provide", "could you clarify", "can you specify",
    "请提供", "请确认", "需要更多信息", "需要你提供",
]

# Tool/API failure patterns — agent got no usable data due to errors
_ERROR_PATTERNS = [
    "api call failed", "verification failed", "invalid code",
    "error occurred", "request failed", "connection refused",
    "timed out", "timeout", "rate limit", "429",
    "500 internal server error", "502 bad gateway", "503 service unavailable",
    "tool execution failed", "function call error",
    "接口报错", "调用失败", "请求超时",
]


def _detect_data_status(response_text: str) -> str:
    """Detect whether AI found relevant data.
    Returns 'data_not_found' if refusal or error patterns are found."""
    lower = response_text.lower()
    for pattern in _REFUSAL_PATTERNS:
        if pattern in lower:
            return "data_not_found"
    for pattern in _ERROR_PATTERNS:
        if pattern in lower:
            return "data_not_found"
    return "data_found"


def _normalize_text(text: str) -> str:
    """Normalize text: strip markdown, split CamelCase, remove punctuation, lowercase"""
    text = _MD_BOLD_RE.sub("", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_CODE_RE.sub(lambda m: m.group().strip("`"), text)
    text = _MD_LINK_RE.sub(r"\1", text)
    # Split CamelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    # Separators -> spaces
    text = re.sub(r"[-;/]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _extract_response_items(text: str) -> list[str]:
    """Heuristically extract list items from free text (bullet/numbered/semicolon/comma separated)"""
    # 1. markdown bullet: "- item" / "* item" / "• item" / "1. item"
    #    Note: \*(?!\*) avoids misidentifying **bold** headings as * bullet
    bullet_re = re.compile(r"^\s*(?:[-•]\s|\*(?!\*)\s|\d+[.)]\s)\s*(.+)", re.MULTILINE)
    bullets = bullet_re.findall(text)
    if len(bullets) >= 2:
        items = [b.strip() for b in bullets if b.strip()]
    elif re.search(r"^\s*\|.+\|", text, re.MULTILINE):
        # 2. Markdown table: "| col1 | col2 | ..."
        #    Each row joins all non-index cells as one item to ensure keyword_match covers multi-column content
        table_rows = re.findall(r"^\s*\|(.+)\|", text, re.MULTILINE)
        items = []
        after_separator = False  # markdown table: header -> separator -> data rows
        for row in table_rows:
            cells = [c.strip() for c in row.split("|")]
            # Skip separator rows (| --- | --- |), mark subsequent rows as data rows
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                after_separator = True
                continue
            # Rows before separator are header rows, skip
            if not after_separator:
                continue
            non_empty = [c for c in cells if c]
            if not non_empty:
                continue
            # Remove pure numeric index columns, join remaining cells
            content_cells = [c for c in non_empty if not re.fullmatch(r"\d+", c)]
            if not content_cells:
                continue
            item = " ".join(content_cells)
            items.append(item)
    elif ";" in text:
        # 3. Semicolon separated (consistent with standard answer format)
        items = [s.strip() for s in text.split(";") if s.strip()]
    elif text.count(",") >= 2:
        # 4. Comma separated (only enabled with 3+ commas to avoid splitting sentences)
        items = [s.strip() for s in text.split(",") if s.strip()]
    else:
        # 5. Alternating key-value lines: common in categorized listing format
        #    Header
        #    Indicator1
        #    Value1
        #    Indicator2
        #    Value2
        # Heuristic: filter out empty/header/value lines, keep lines that look like indicator names
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        # Filter: pure numeric/unit value lines, header lines (numeric+parenthesis start or all-caps short headers)
        _VALUE_RE = re.compile(
            r"^[\d.,+\-]+\s*(%|mg|kg|cm|mm|bpm|mmHg|mmol|g/dL|mL|µg|IU|"
            r"mg/dL|mEq|mIU|ng|pg|fl|fL|U/L|cells|k/µL|M/µL|µmol|"
            r"mm/hr|sec|min|hours?|days?|years?|lbs?|in\.?|°[CF])?$",
            re.IGNORECASE,
        )
        _HEADER_RE = re.compile(
            r"^(\*{1,2})?(\d+[).]\s|#{1,4}\s|[A-Z][A-Z\s&]{5,}$)",
        )
        _SKIP_WORDS = {
            "indicator", "value", "unit", "name", "result", "range",
            "normal range", "reference", "status", "date", "category",
            "指标", "数值", "单位", "参考范围", "结果",
        }
        candidates = []
        for ln in lines:
            if _VALUE_RE.match(ln):
                continue
            if _HEADER_RE.match(ln):
                continue
            if ln.lower().strip("*#: ") in _SKIP_WORDS:
                continue
            # Skip sentences (containing verb/preposition phrases; indicator names are typically not this long)
            if len(ln) > 40 and any(w in ln.lower() for w in
                                    (" are ", " is ", " the ", " here ", " based on ",
                                     " following ", " recorded ", " below")):
                continue
            # At least 2 chars, no more than 80 chars (reasonable indicator name length)
            if 2 <= len(ln) <= 80:
                candidates.append(ln)
        # Only use when enough candidates are extracted (avoid false triggers)
        if len(candidates) >= 5:
            items = candidates
        else:
            return []
    # Remove short items, deduplicate
    seen = set()
    result = []
    for i in items:
        norm = i.lower().strip()
        if len(norm) >= 2 and norm not in seen:
            seen.add(norm)
            result.append(i)
    return result


def _keyword_match(generated: str, standard: str, threshold: float = 0.3) -> tuple[bool, float, str]:
    """Keyword matching: Chinese 2-gram / English word-level matching

    Returns: (is_match, ratio, detail)
    """
    norm_gen = _normalize_text(generated)
    norm_std = _normalize_text(standard)

    # Split by spaces
    space_tokens = [w for w in norm_std.split() if len(w) >= 2]

    if len(space_tokens) > 1:
        keywords = space_tokens
    elif len(norm_std) >= 2:
        # Chinese without spaces -> 2-gram
        keywords = [norm_std[i : i + 2] for i in range(len(norm_std) - 1)]
    else:
        hit = norm_std in norm_gen if norm_std else False
        return hit, 1.0 if hit else 0.0, f"Substring match: {'hit' if hit else 'miss'}"

    matched = sum(1 for w in keywords if w in norm_gen)
    ratio = matched / len(keywords)
    is_match = ratio >= threshold
    detail = f"Keywords {matched}/{len(keywords)} hit (ratio={ratio:.0%})"
    return is_match, ratio, detail


# ============================================================
# LLM Judge Prompt
# ============================================================

_TEXT_JUDGE_PROMPT = """你是一个健康知识图谱 QA 评估专家。请评估 AI 助手对用户健康数据查询的回答质量。

## 标准答案
{expected_value}

## 关键点（评分标准）
{key_points_formatted}

## 数据来源上下文
{source_context}

## AI 助手的回答
{response_text}

## 评估指引
评估 AI 回答是否准确覆盖了上述关键点:
1. 回答是否包含标准答案中的关键信息？
2. 信息是否与标准答案一致（不矛盾）？
3. 是否遗漏了重要信息？

评分 0.0~1.0（请大胆使用完整分值范围，不要保守聚集在 0.7-0.9）:
- 0.95~1.0: 所有关键点均正确覆盖，即使措辞、格式、详略与标准答案不同
- 0.80~0.94: 核心信息正确，仅遗漏次要细节或表述略有偏差
- 0.50~0.79: 部分关键点正确覆盖，但有明显遗漏或部分不一致
- 0.20~0.49: 仅包含少量相关信息，大部分关键点缺失
- 0.01~0.19: 几乎无相关信息，但有微弱关联
- 0.0: 完全错误、无相关内容、或拒绝回答

重要：如果 AI 的核心结论与标准答案一致，不要因为"回答冗余"或"格式不同"而扣分。
例如标准答案为 "Normal"，AI 回答 "Your result is Normal" 并附加了解释，应得 0.95+。

仅返回 JSON 对象:
```json
{{"score": 0.85, "explanation": "简短评分理由"}}
```"""


# ============================================================
# KgQaEvalInfo — config model
# ============================================================


class KgQaEvalInfo(BaseModel):
    """KG knowledge Q&A evaluation config — hybrid rule/LLM, routes by answer_type

    - numeric_value: rule-based number extraction + tolerance comparison (zero LLM)
    - boolean: rule-based yes/no signal word extraction (zero LLM)
    - list: rule-based set comparison + recall scoring (zero LLM)
    - text: LLM-as-Judge, using key_points as scoring criteria
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "kg_qa",
                    "answer_type": "numeric_value",
                    "expected_value": 120,
                    "key_points": ["收缩压"],
                    "number_tolerance": 0.01,
                },
                {
                    "evaluator": "kg_qa",
                    "answer_type": "text",
                    "expected_value": "患者最近一周血压趋势平稳",
                    "key_points": ["血压趋势", "平稳", "一周"],
                    "model": "gpt-4.1",
                },
                {
                    "evaluator": "kg_qa",
                    "answer_type": "list",
                    "expected_value": ["血红蛋白", "白细胞", "血小板"],
                    "key_points": ["血常规三项"],
                },
            ]
        },
    )

    evaluator: Literal["kg_qa"] = Field(default="kg_qa", description="Evaluator type")
    answer_type: Literal["numeric_value", "text", "list", "boolean"] = Field(
        description="Answer type — routes to different evaluation strategies: numeric_value, text, list, boolean",
    )

    @field_validator("answer_type", mode="before")
    @classmethod
    def _normalize_answer_type(cls, v: str) -> str:
        """兼容别名: numeric → numeric_value"""
        if v == "numeric":
            return "numeric_value"
        return v

    expected_value: Union[str, int, float, List[Union[str, int, float]]] = Field(description="Expected standard answer value")
    key_points: List[str] = Field(default_factory=list, description="Key scoring points (used as LLM scoring criteria for text type)")
    source_data: Optional[Dict[str, Any]] = Field(None, description="Data source context (optional, for LLM judge reference)")
    number_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        description="Numeric comparison relative error tolerance (default 0.01 = 1%, only for numeric_value)",
    )
    model: str = Field(default="gpt-4.1", description="LLM model (default gpt-4.1, only used for text type)")
    difficulty: Optional[str] = Field(None, description="Question difficulty tag (optional, for report analysis)")
    answer_format_hint: Optional[str] = Field(
        default="\n\nPlease end your response with a structured answer line: ANSWER: <value>",
        description="Format hint appended to user question (None to skip). "
        "Automatically concatenated to last strict_inputs during bench_item_to_test_case conversion.",
    )


# ============================================================
# KgQaEvalAgent
# ============================================================


class KgQaEvalAgent(AbstractEvalAgent, name="kg_qa", params_model=KgQaEvalInfo):
    """KG knowledge Q&A evaluator — hybrid rule/LLM, routes by answer_type"""

    _display_meta = {
        "icon": "M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125",
        "color": "#0ea5e9",
        "features": ["KG QA", "Hybrid Evaluation", "Partial Scoring"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.005,
    }

    def __init__(self, eval_config: KgQaEvalInfo, **kwargs: Any):
        super().__init__(eval_config, **kwargs)
        self.config: KgQaEvalInfo = eval_config
        self.model = eval_config.model
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: Optional[Dict[str, UsageMetadata]]) -> None:
        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        # 1. Extract AI response
        response_text = self._extract_response(memory_list)
        if not response_text:
            return EvalResult(result="fail", score=0.0, feedback="AI response is empty")

        # 2. Infrastructure error detection
        if "502 Bad Gateway" in response_text or "API 调用失败" in response_text:
            return EvalResult(result="error", score=0.0, feedback="system_error: API infrastructure failure")

        # 3. Route by answer_type
        answer_type = self.config.answer_type
        if answer_type == "numeric_value":
            score, detail = self._eval_numeric(response_text)
        elif answer_type == "boolean":
            score, detail = self._eval_boolean(response_text)
        elif answer_type == "list":
            score, detail = self._eval_list(response_text)
        elif answer_type == "text":
            score, detail = await self._eval_text(response_text)
        else:
            return EvalResult(result="fail", score=0.0, feedback=f"Unknown answer_type: {answer_type}")

        clipped = max(0.0, min(1.0, score))
        return EvalResult(
            result="scored",
            score=clipped,
            feedback=f"[{answer_type}] score={clipped:.3f} — {detail}",
            trace=EvalTrace(
                eval_detail={
                    "answer_type": answer_type,
                    "difficulty": self.config.difficulty,
                    "expected_value": self.config.expected_value,
                    "key_points": self.config.key_points,
                    "score": clipped,
                    "detail": detail,
                }
            ),
        )

    # ----------------------------------------------------------
    # Response extraction
    # ----------------------------------------------------------

    @staticmethod
    def _extract_response(memory_list: List[TestAgentMemory]) -> str:
        parts: List[str] = []
        for mem in memory_list:
            if mem.target_response is not None:
                text = mem.target_response.extract_text()
                if text:
                    parts.append(text)
        return "\n".join(parts)

    # ----------------------------------------------------------
    # numeric_value evaluation (zero LLM)
    # ----------------------------------------------------------

    def _eval_numeric(self, response_text: str) -> tuple[float, str]:
        expected = str(self.config.expected_value)
        tolerance = self.config.number_tolerance
        data_status = _detect_data_status(response_text)

        # Prefer extracting numbers from ANSWER: line, narrowing scope to reduce false matches
        answer_line = _extract_answer_line(response_text)
        extract_source = answer_line if answer_line is not None else response_text

        std_numbers = _NUMBER_RE.findall(_normalize_minus(expected))
        gen_numbers = _NUMBER_RE.findall(_normalize_minus(extract_source))

        # If no numbers in ANSWER line, fall back to full text search
        if answer_line is not None and not gen_numbers and std_numbers:
            gen_numbers = _NUMBER_RE.findall(_normalize_minus(response_text))

        if not std_numbers:
            # Standard answer has no numbers -> fallback to keyword
            _, ratio, detail = _keyword_match(response_text, expected)
            return ratio, f"numeric fallback: {detail} [{data_status}]"

        try:
            std_floats = [float(n.replace(",", "")) for n in std_numbers]
            gen_floats = [float(n.replace(",", "")) for n in gen_numbers]
        except ValueError:
            _, ratio, detail = _keyword_match(response_text, expected)
            return ratio, f"numeric parse fallback: {detail} [{data_status}]"

        matched = 0
        for std_num, std_str in zip(std_floats, std_numbers):
            for gen_num, gen_str in zip(gen_floats, gen_numbers):
                if _numbers_close_with_precision(std_num, gen_num, std_str, gen_str, tolerance):
                    matched += 1
                    break

        if matched == len(std_floats):
            score = 1.0
            detail = f"Number match {matched}/{len(std_floats)} (tolerance={tolerance:.0%}) [{data_status}]"
        elif data_status == "data_not_found":
            score = 0.0
            detail = f"Number match {matched}/{len(std_floats)} — AI did not find relevant data [{data_status}]"
        elif matched > 0:
            score = matched / len(std_floats)
            detail = f"Number match {matched}/{len(std_floats)} — partially correct [{data_status}]"
        else:
            # Found data but wrong answer -> 0.2 partial score (distinguish from not found)
            score = 0.2
            detail = f"Number match 0/{len(std_floats)} — found data but answer mismatch [{data_status}, wrong_answer]"
        return score, detail

    # ----------------------------------------------------------
    # boolean evaluation (zero LLM)
    # ----------------------------------------------------------

    def _eval_boolean(self, response_text: str) -> tuple[float, str]:
        expected = str(self.config.expected_value).strip().lower()

        # Prefer extracting from ANSWER: line to reduce body text interference
        answer_line = _extract_answer_line(response_text)
        eval_text = answer_line if answer_line is not None else response_text
        text_lower = eval_text.lower()

        expected_bool = expected in ("true", "yes", "是", "对", "正确", "有", "存在")

        # Positive/negative signal words (_word_boundary_match auto-selects strategy: short English uses word boundary, Chinese uses substring)
        pos_patterns = ["yes", "true", "correct", "是的", "对的", "正确", "确实", "有的", "存在"]
        neg_patterns = ["no", "not", "false", "never", "incorrect", "不是", "没有", "不对", "不正确", "不存在", "否"]

        pos_hits = sum(1 for p in pos_patterns if _word_boundary_match(p, text_lower))
        neg_hits = sum(1 for p in neg_patterns if _word_boundary_match(p, text_lower))

        if pos_hits == 0 and neg_hits == 0:
            return 0.0, "Cannot extract boolean value"

        detected_bool = pos_hits > neg_hits
        score = 1.0 if detected_bool == expected_bool else 0.0
        detail = f"Boolean match: expected={expected_bool}, detected={detected_bool} (pos={pos_hits}, neg={neg_hits})"
        return score, detail

    # ----------------------------------------------------------
    # list evaluation (zero LLM, recall scoring)
    # ----------------------------------------------------------

    def _eval_list(self, response_text: str) -> tuple[float, str]:
        expected_items = self.config.expected_value
        if isinstance(expected_items, str):
            expected_items = [s.strip() for s in expected_items.split(";") if s.strip()]

        if not expected_items:
            return 0.0, "Standard answer list is empty"

        # Prefer extracting list items from ANSWER: line
        answer_line = _extract_answer_line(response_text)
        eval_text = answer_line if answer_line is not None else response_text

        # Recall: how many expected items are hit by AI response
        found = 0
        for item in expected_items:
            is_match, _, _ = _keyword_match(eval_text, str(item), threshold=0.4)
            if is_match:
                found += 1
        recall = found / len(expected_items)

        # If ANSWER line recall=0 but full text may have content, fall back to full text
        if answer_line is not None and recall == 0 and len(expected_items) > 0:
            eval_text = response_text
            found = 0
            for item in expected_items:
                is_match, _, _ = _keyword_match(eval_text, str(item), threshold=0.4)
                if is_match:
                    found += 1
            recall = found / len(expected_items)

        # Precision: how many items listed by AI are expected
        generated_items = _extract_response_items(eval_text)
        if generated_items:
            correct_in_generated = 0
            for gen_item in generated_items:
                for exp_item in expected_items:
                    if _keyword_match(gen_item, str(exp_item), threshold=0.4)[0]:
                        correct_in_generated += 1
                        break
            precision = correct_in_generated / len(generated_items)
        else:
            # Cannot extract list items from response -> precision degrades to recall (backward compatible)
            precision = recall

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        n_gen = len(generated_items) if generated_items else -1
        detail = (f"List F1={f1:.0%} (P={precision:.0%}, R={recall:.0%}, "
                  f"expected {len(expected_items)} items, extracted {n_gen} items)")
        return f1, detail

    # ----------------------------------------------------------
    # text evaluation (LLM-as-Judge)
    # ----------------------------------------------------------

    async def _eval_text(self, response_text: str) -> tuple[float, str]:
        expected = str(self.config.expected_value)
        key_points = self.config.key_points

        # Prefer ANSWER: line as evaluation text, but keep full response for judge reference
        answer_line = _extract_answer_line(response_text)
        eval_text = answer_line if answer_line is not None else response_text

        # No key_points -> fallback to keyword matching
        if not key_points:
            _, ratio, detail = _keyword_match(eval_text, expected, threshold=0.3)
            return ratio, f"text fallback (no key_points): {detail}"

        key_points_formatted = "\n".join(f"- {kp}" for kp in key_points)
        source_context = "Not available"
        if self.config.source_data:
            source_context = json.dumps(self.config.source_data, ensure_ascii=False, indent=2)[:1000]

        prompt = _TEXT_JUDGE_PROMPT.format(
            expected_value=expected[:500],
            key_points_formatted=key_points_formatted,
            source_context=source_context,
            response_text=eval_text[:3000],
        )

        # LLM call, retry up to 3 times
        for attempt in range(3):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="你是精确的健康知识图谱 QA 评估专家。",
                    input=prompt,
                    max_tokens=500,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json(result.content)
                if parsed and "score" in parsed:
                    score = float(parsed["score"])
                    explanation = parsed.get("explanation", "")
                    return score, f"LLM judge: {explanation}"
            except Exception as e:
                logger.warning("[KgQaEval] LLM judge failed (attempt %d/3): %s", attempt + 1, e)

        # Retries exhausted -> fallback
        _, ratio, detail = _keyword_match(response_text, expected, threshold=0.3)
        return ratio, f"LLM judge retries exhausted, fallback: {detail}"

    # ----------------------------------------------------------
    # JSON parsing
    # ----------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Extract JSON object from LLM response"""
        # Try markdown code block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find { ... }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None
