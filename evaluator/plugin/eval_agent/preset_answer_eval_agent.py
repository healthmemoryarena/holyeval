"""
PresetAnswerEvalAgent — 预设答案准确率评估器

注册名称: "preset_answer"

将 AI 助手的回答与标准答案进行确定性比对，零 LLM 调用。
支持三种匹配模式：
  - number: 提取数字，允许相对误差容忍（默认 1%）
  - keyword: 关键词匹配
  - exact: 精确字符串匹配

适用场景：数据提取验证、预设问答、回归测试。
"""

import logging
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, SessionInfo, TestAgentMemory

logger = logging.getLogger(__name__)


# ============================================================
# 评估配置模型
# ============================================================


class PresetAnswerEvalInfo(BaseModel):
    """预设答案准确率评估 — 单题确定性比对，零 LLM

    将 AI 助手的回答与标准答案进行比对，支持数字容差匹配、关键词匹配和精确匹配。
    适用于数据提取验证、预设问答等有明确标准答案的场景。

    每个 TestCase 包含一个问题和一个标准答案，批量执行后通过 TestReport.pass_rate 计算准确率。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "数字匹配（默认），允许 1% 误差",
                    "evaluator": "preset_answer",
                    "standard_answer": "59.06",
                },
                {
                    "_comment": "关键词匹配",
                    "evaluator": "preset_answer",
                    "standard_answer": "建议就医",
                    "match_mode": "keyword",
                },
            ]
        },
    )

    evaluator: Literal["preset_answer"] = Field(description="评估器类型")
    model: Optional[str] = Field(None, description="预留字段（不使用 LLM）")
    standard_answer: str = Field(description="标准答案 — 数字型会自动提取比对，文本型按关键词匹配")
    match_mode: str = Field(
        default="number",
        description=(
            "匹配模式：\n"
            "  - number: 提取数字，允许相对误差容忍（默认）\n"
            "  - keyword: 关键词匹配\n"
            "  - exact: 精确字符串匹配"
        ),
    )
    number_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="数字匹配相对误差容忍度（默认 0.01 = 1%）",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="数字/关键词命中比例阈值（默认 0.5 = 50% 命中即正确）",
    )

# 数字提取正则：支持 1,234.56 / -10 / 0.5 等格式
# 注意：同时支持 Unicode 减号 U+2212（−）和 ASCII 连字符 -
_NUMBER_RE = re.compile(r"[\-\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _normalize_minus(text: str) -> str:
    """将 Unicode 减号（−, U+2212）替换为 ASCII 连字符（-），避免数字提取漏匹配"""
    return text.replace("\u2212", "-")


class PresetAnswerEvalAgent(AbstractEvalAgent, name="preset_answer", params_model=PresetAnswerEvalInfo):
    """预设答案准确率评估器 — 确定性比对，零 LLM"""

    _display_meta = {
        "icon": "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
        "color": "#06b6d4",
        "features": ["零 LLM", "精确匹配", "确定性评分"],
    }
    _cost_meta = {
        "est_cost_per_case": 0,  # 纯规则比对，无 LLM 调用
    }

    def __init__(self, eval_config: PresetAnswerEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.config: PresetAnswerEvalInfo = eval_config

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """从对话记忆中提取 AI 回答，与标准答案比对"""
        standard_answer = self.config.standard_answer

        # 从 memory 中提取所有 AI 回复文本
        generated_parts: List[str] = []
        for mem in memory_list:
            if mem.target_response is not None:
                text = mem.target_response.extract_text()
                if text:
                    generated_parts.append(text)

        generated_answer = "\n".join(generated_parts)

        if not generated_answer.strip():
            logger.warning("[PresetAnswerEval] AI 回复为空")
            return EvalResult(
                result="fail",
                score=0.0,
                feedback="AI 回复为空，无法比对",
            )

        # 502 / API 基础设施错误 → 标记为 system_error，不计入 Agent 能力评分
        if "502 Bad Gateway" in generated_answer or "API 调用失败" in generated_answer:
            logger.warning("[PresetAnswerEval] 检测到 API 基础设施错误，标记为 system_error")
            return EvalResult(
                result="error",
                score=0.0,
                feedback="system_error: API 基础设施故障 (502)，不计入评分",
            )

        logger.info(
            "[PresetAnswerEval] 标准答案: %s, 匹配模式: %s",
            standard_answer, self.config.match_mode,
        )
        logger.info(
            "[PresetAnswerEval] AI 回答: %s",
            generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer,
        )

        # 根据匹配模式比对
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
                feedback=f"未知匹配模式: {mode}",
            )

        score = 1.0 if is_correct else 0.0
        result = "pass" if is_correct else "fail"

        logger.info("[PresetAnswerEval] 结果: %s, 详情: %s", result, detail)

        return EvalResult(
            result=result,
            score=score,
            feedback=detail,
        )

    # ============================================================
    # 匹配方法
    # ============================================================

    @staticmethod
    def _match_number(
        generated: str,
        standard: str,
        tolerance: float = 0.01,
        threshold: float = 0.5,
    ) -> tuple[bool, str]:
        """数字匹配 — 提取数字，允许相对误差容忍

        Returns:
            (is_correct, detail_message)
        """
        std_numbers = _NUMBER_RE.findall(_normalize_minus(standard))
        gen_numbers = _NUMBER_RE.findall(_normalize_minus(generated))

        if not std_numbers:
            # 标准答案无数字，fallback 到关键词匹配
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
            f"数字匹配: {matched}/{len(std_floats)} 命中 "
            f"(比例={ratio:.0%}, 阈值={threshold:.0%}, 容差={tolerance:.0%})"
        )
        return is_correct, detail

    @staticmethod
    def _match_keyword(
        generated: str,
        standard: str,
        threshold: float = 0.3,
    ) -> tuple[bool, str]:
        """关键词匹配 — 提取标准答案关键词，检查命中率

        中文文本无空格分隔，自动切换为 2-gram 匹配。
        """

        def strip_markdown(text: str) -> str:
            """去除常见 markdown 格式符号"""
            # 去除加粗/斜体：**text** / *text* / __text__ / _text_
            text = re.sub(r"\*{1,3}|_{1,3}", "", text)
            # 去除标题 # ## ###
            text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
            # 去除行内代码 `code`
            text = re.sub(r"`[^`]*`", lambda m: m.group().strip("`"), text)
            # 去除链接 [text](url)
            text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
            return text

        def split_camel(text: str) -> str:
            """将 CamelCase 拆为空格分隔的小写词：DailyStepCount → daily step count"""
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
            text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
            return text

        def normalize(text: str) -> str:
            text = strip_markdown(text)
            text = split_camel(text)
            # 先将连字符/分号等分隔符替换为空格，避免缩写粘连
            text = re.sub(r"[-;/]", " ", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.lower().strip()

        norm_gen = normalize(generated)
        norm_std = normalize(standard)

        # 按空格拆词
        space_tokens = [w for w in norm_std.split() if len(w) >= 2]

        # 若拆出多个词（英文/混合文本），用词级匹配
        # 若只有单个 token（中文无空格），用 2-gram 匹配
        if len(space_tokens) > 1:
            keywords = space_tokens
        elif len(norm_std) >= 2:
            # 中文：滑动窗口取 2-gram
            keywords = [norm_std[i : i + 2] for i in range(len(norm_std) - 1)]
        else:
            is_correct = norm_std in norm_gen if norm_std else False
            detail = f"关键词为空，子串匹配: {'命中' if is_correct else '未命中'}"
            return is_correct, detail

        matched = sum(1 for w in keywords if w in norm_gen)
        ratio = matched / len(keywords)
        is_correct = ratio >= threshold

        detail = (
            f"关键词匹配: {matched}/{len(keywords)} 命中 "
            f"(比例={ratio:.0%}, 阈值={threshold:.0%})"
        )
        return is_correct, detail

    @staticmethod
    def _match_exact(generated: str, standard: str) -> tuple[bool, str]:
        """精确匹配 — 标准答案字符串是否出现在 AI 回答中"""
        is_correct = standard.strip() in generated
        detail = f"精确匹配: {'命中' if is_correct else '未命中'}"
        return is_correct, detail


# ============================================================
# 工具函数
# ============================================================


def _decimal_places(s: str) -> int:
    """从原始数字字符串中获取小数位数：'463.74' → 2, '25' → 0"""
    s = s.replace(",", "")
    if "." in s:
        return len(s.split(".")[-1])
    return 0


def _numbers_close(a: float, b: float, tolerance: float) -> bool:
    """判断两个数字是否在相对误差容忍范围内"""
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
    """先按原始容差比对；若失败，尝试对齐精度后比对。

    例: 标准答案 463.7415 (4位) vs AI 回答 463.74 (2位)
    → 将 463.7415 round 到 2 位 = 463.74 → 相等 → 通过
    """
    # 第一步：原始容差比对
    if _numbers_close(a, b, tolerance):
        return True

    # 第二步：精度对齐 — 将高精度数 round 到低精度数的位数后再比
    dp_a = _decimal_places(a_str)
    dp_b = _decimal_places(b_str)
    min_dp = min(dp_a, dp_b)

    rounded_a = round(a, min_dp)
    rounded_b = round(b, min_dp)
    return _numbers_close(rounded_a, rounded_b, tolerance)
