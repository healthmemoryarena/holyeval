"""
MedCalcEvalAgent — 基于 MedCalc-Bench 的医疗计算评估器

注册名称: "medcalc"

工作原理:
1. 从 TargetAgent 的 memory_list 中提取 AI 回复文本
2. 调用 LLM 从自然语言回复中提取结构化答案
3. 根据 output_type 进行类型化匹配：
   - decimal: 答案落在 [lower_limit, upper_limit] 区间内
   - integer: 四舍五入后精确匹配
   - date: MM/DD/YYYY 格式精确匹配
   - weeks_days: "X weeks, Y days" 标准化后精确匹配
4. 二元评分: pass(1.0) / fail(0.0)

参考: https://arxiv.org/abs/2406.12036
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Literal, Optional

from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, EvalTrace, SessionInfo, TestAgentMemory
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)


# ============================================================
# 评估配置模型
# ============================================================


class MedCalcEvalInfo(BaseModel):
    """MedCalc-Bench 医疗计算评估 — LLM 答案提取 + 类型化数值匹配

    基于 MedCalc-Bench 数据集（https://arxiv.org/abs/2406.12036），评估 LLM 的医疗计算能力。
    评估流程：LLM 从目标回复中提取答案 → 根据 output_type 进行类型化匹配：
      - decimal: 答案落在 [lower_limit, upper_limit] 区间内（±5% 容差）
      - integer: 四舍五入后精确匹配
      - date: MM/DD/YYYY 格式精确匹配
      - weeks_days: "X weeks, Y days" 标准化后精确匹配
    二元评分：pass(1.0) / fail(0.0)，与原版 MedCalc-Bench 准确率指标一致。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "medcalc",
                    "ground_truth": "25.2381",
                    "lower_limit": "23.97619",
                    "upper_limit": "26.50001",
                    "output_type": "decimal",
                },
                {
                    "evaluator": "medcalc",
                    "ground_truth": "18",
                    "output_type": "integer",
                },
            ]
        },
    )

    evaluator: Literal["medcalc"] = Field(description="评估器类型")
    ground_truth: str = Field(description="标准答案（字符串形式，保留精度）")
    lower_limit: str = Field("", description="容差下限（仅 decimal 类型使用）")
    upper_limit: str = Field("", description="容差上限（仅 decimal 类型使用）")
    output_type: Literal["decimal", "integer", "date", "weeks_days"] = Field(description="输出类型")
    explanation: str = Field("", description="标准答案解题步骤（调试用，不参与评分）")
    extractor_model: Optional[str] = Field("gpt-4.1", description="答案提取模型")

DEFAULT_MODEL = "gpt-4.1"

# ============================================================
# 答案提取 Prompt
# ============================================================

EXTRACTION_PROMPT = """You are a medical answer extractor. Given an AI assistant's response to a medical calculation question, extract ONLY the final numerical answer.

# AI Response
{response_text}

# Expected Output Type: {output_type}

# Instructions
Extract the final answer from the response above. Return a JSON object:

For decimal: {{"answer": "25.2381"}} (just the number, no units)
For integer: {{"answer": "18"}} (just the integer, no units)
For date: {{"answer": "03/15/2025"}} (MM/DD/YYYY format)
For weeks_days: {{"answer": "17 weeks, 4 days"}} (this exact format)

If the response does not contain a clear answer, return: {{"answer": "NONE"}}

Return ONLY the JSON object, no other text."""


class MedCalcEvalAgent(AbstractEvalAgent, name="medcalc", params_model=MedCalcEvalInfo):
    """医疗计算评估器 — LLM 答案提取 + 类型化数值匹配"""

    _display_meta = {
        "icon": (
            "M15.75 15.75V18m-7.5-6.75h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25v"
            "-.008zm0 2.25h.008v.008H8.25v-.008zm2.498-6.75h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007v-.008zm0"
            " 2.25h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007v-.008zm2.504-6.75h.008v.008h-.008v-.008zm0 2.25h"
            ".008v.008h-.008v-.008zm0 2.25h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008v-.008zm2.498-6.75h.008v"
            ".008h-.008v-.008zm0 2.25h.008v.008h-.008v-.008zM8.25 6h7.5v2.25h-7.5V6zM12 2.25c-1.892 0-3.758.11"
            "-5.593.322C5.307 2.7 4.5 3.65 4.5 4.757V19.5a2.25 2.25 0 002.25 2.25h10.5a2.25 2.25 0 002.25-2.25V4"
            ".757c0-1.108-.806-2.057-1.907-2.185A48.507 48.507 0 0012 2.25z"
        ),
        "color": "#8b5cf6",
        "features": ["MedCalc-Bench", "LLM 答案提取", "类型化匹配"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.003,  # 单次 gpt-4.1 答案提取调用
    }

    def __init__(self, eval_config: MedCalcEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.config: MedCalcEvalInfo = eval_config
        self.model = eval_config.extractor_model or DEFAULT_MODEL
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # 框架接口
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        try:
            # 1. 提取 AI 回复文本
            response_text = self._extract_response_text(memory_list)
            if not response_text:
                return EvalResult(result="fail", score=0.0, feedback="目标 AI 无回复，无法评估")

            logger.info(
                "[MedCalcEval] 开始评估 — output_type=%s, ground_truth=%s, model=%s",
                self.config.output_type,
                self.config.ground_truth,
                self.model,
            )

            # 2. LLM 提取答案
            extracted = await self._extract_answer(response_text)
            if extracted is None or extracted.upper() == "NONE":
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback=f"无法从 AI 回复中提取答案（output_type={self.config.output_type}）",
                    trace=EvalTrace(eval_detail={"response_preview": response_text[:500], "extracted": None}),
                )

            # 3. 类型化匹配
            is_correct, detail = self._match_answer(extracted)

            score = 1.0 if is_correct else 0.0
            result = "pass" if is_correct else "fail"

            feedback = (
                f"MedCalc 评估 ({self.config.output_type}): {result} — "
                f"提取答案={extracted}, 标准答案={self.config.ground_truth}, {detail}"
            )

            logger.info("[MedCalcEval] %s — extracted=%s, ground_truth=%s", result, extracted, self.config.ground_truth)

            return EvalResult(
                result=result,
                score=score,
                feedback=feedback,
                trace=EvalTrace(
                    eval_detail={
                        "extracted_answer": extracted,
                        "ground_truth": self.config.ground_truth,
                        "output_type": self.config.output_type,
                        "is_correct": is_correct,
                        "match_detail": detail,
                    }
                ),
            )

        except Exception as e:
            logger.error("[MedCalcEval] 评估过程出错: %s", e, exc_info=True)
            return EvalResult(result="fail", score=0.0, feedback=f"评估过程出错: {e}")

    # ----------------------------------------------------------
    # 回复提取
    # ----------------------------------------------------------

    @staticmethod
    def _extract_response_text(memory_list: list[TestAgentMemory]) -> str:
        """从 memory_list 中提取所有 AI 回复文本"""
        parts: List[str] = []
        for mem in memory_list:
            if mem.target_response is not None:
                text = mem.target_response.extract_text()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # LLM 答案提取
    # ----------------------------------------------------------

    async def _extract_answer(self, response_text: str) -> Optional[str]:
        """调用 LLM 从自然语言回复中提取结构化答案"""
        prompt = EXTRACTION_PROMPT.format(
            response_text=response_text[:3000],  # 截断避免过长
            output_type=self.config.output_type,
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a precise medical answer extractor. Return only JSON.",
                    input=prompt,
                    max_tokens=200,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json_response(result.content)
                if parsed and "answer" in parsed:
                    return str(parsed["answer"]).strip()

                logger.warning(
                    "[MedCalcEval] 提取输出格式异常 (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    result.content[:200],
                )
            except Exception as e:
                logger.warning("[MedCalcEval] 提取调用失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)

        return None

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """从 LLM 响应中提取 JSON"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    # ----------------------------------------------------------
    # 类型化匹配
    # ----------------------------------------------------------

    def _match_answer(self, extracted: str) -> tuple[bool, str]:
        """根据 output_type 路由到对应匹配方法"""
        output_type = self.config.output_type
        if output_type == "decimal":
            return self._match_decimal(extracted)
        elif output_type == "integer":
            return self._match_integer(extracted)
        elif output_type == "date":
            return self._match_date(extracted)
        elif output_type == "weeks_days":
            return self._match_weeks_days(extracted)
        else:
            return False, f"未知 output_type: {output_type}"

    def _match_decimal(self, extracted: str) -> tuple[bool, str]:
        """decimal 匹配 — 答案落在 [lower_limit, upper_limit] 区间内"""
        try:
            answer = float(extracted.replace(",", ""))
        except ValueError:
            return False, f"无法解析为数字: {extracted}"

        try:
            lower = float(self.config.lower_limit)
            upper = float(self.config.upper_limit)
        except ValueError:
            return False, f"容差范围配置异常: lower={self.config.lower_limit}, upper={self.config.upper_limit}"

        is_correct = lower <= answer <= upper
        detail = f"范围匹配: {answer} {'∈' if is_correct else '∉'} [{lower}, {upper}]"
        return is_correct, detail

    def _match_integer(self, extracted: str) -> tuple[bool, str]:
        """integer 匹配 — 四舍五入后精确匹配"""
        try:
            answer = round(float(extracted.replace(",", "")))
        except ValueError:
            return False, f"无法解析为数字: {extracted}"

        try:
            ground_truth = int(float(self.config.ground_truth))
        except ValueError:
            return False, f"标准答案非整数: {self.config.ground_truth}"

        is_correct = answer == ground_truth
        detail = f"整数匹配: {answer} {'==' if is_correct else '!='} {ground_truth}"
        return is_correct, detail

    def _match_date(self, extracted: str) -> tuple[bool, str]:
        """date 匹配 — 标准化 MM/DD/YYYY 后精确比对"""
        try:
            answer_date = datetime.strptime(extracted.strip(), "%m/%d/%Y")
            answer_normalized = answer_date.strftime("%-m/%-d/%Y")
        except ValueError:
            return False, f"无法解析日期: {extracted}"

        try:
            gt_date = datetime.strptime(self.config.ground_truth.strip(), "%m/%d/%Y")
            gt_normalized = gt_date.strftime("%-m/%-d/%Y")
        except ValueError:
            return False, f"标准答案日期格式异常: {self.config.ground_truth}"

        is_correct = answer_normalized == gt_normalized
        detail = f"日期匹配: {answer_normalized} {'==' if is_correct else '!='} {gt_normalized}"
        return is_correct, detail

    def _match_weeks_days(self, extracted: str) -> tuple[bool, str]:
        """weeks_days 匹配 — 标准化后精确比对"""
        answer_norm = self._normalize_weeks_days(extracted)
        gt_norm = self._normalize_weeks_days(self.config.ground_truth)

        if answer_norm is None:
            return False, f"无法解析周/天格式: {extracted}"
        if gt_norm is None:
            return False, f"标准答案周/天格式异常: {self.config.ground_truth}"

        is_correct = answer_norm == gt_norm
        detail = f"周天匹配: {answer_norm} {'==' if is_correct else '!='} {gt_norm}"
        return is_correct, detail

    @staticmethod
    def _normalize_weeks_days(text: str) -> Optional[tuple[int, int]]:
        """标准化 "X weeks, Y days" 为 (weeks, days) 元组"""
        text = text.lower().strip()
        match = re.search(r"(\d+)\s*weeks?\s*[,\s]*(\d+)\s*days?", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
