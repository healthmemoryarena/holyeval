"""
HealthBenchEvalAgent — 基于 HealthBench 原版 GRADER_TEMPLATE 的 rubric 评估器

注册名称: "healthbench"

工作原理:
1. 从 TestAgent / TargetAgent 的 memory_list + history 中提取完整对话
2. 对每条 rubric criterion，调用 grader LLM 判定是否满足（criteria_met: bool）
3. 计算得分：achieved_points / total_positive_points（与原版 calculate_score 一致）
4. clip 到 [0, 1]，结果为 scored（仅评分，不做 pass/fail 判定）

Grader Prompt 完全复用 HealthBench 原版（英文），保证评测结果可比。
参考: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

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
# 评估配置模型
# ============================================================


class HealthBenchRubric(BaseModel):
    """HealthBench rubric 单条评分标准

    对应 HealthBench 数据集中每条 rubric criterion。
    points 可为负值（表示不希望出现的行为），范围 -10 ~ +10。
    """

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(description="评分标准文本")
    points: float = Field(description="分值（可负，-10 ~ +10）")
    tags: List[str] = Field(default_factory=list, description="标签（如 axis:accuracy, level:example）")


class HealthBenchEvalInfo(BaseModel):
    """HealthBench rubric 评估 — 逐条 criterion 判定，LLM-as-Judge

    复用 HealthBench 原版 GRADER_TEMPLATE，由 grader 模型逐条判断 criterion 是否满足。
    分数计算：achieved_points / total_positive_points，与原版 calculate_score 完全一致。
    仅评分（result="scored"），不做 pass/fail 判定。

    适用于将 HealthBench 数据集转换后在 HolyEval 框架中运行评测。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "healthbench",
                    "rubrics": [
                        {"criterion": "Advises the patient to see a doctor", "points": 10, "tags": ["axis:accuracy"]},
                        {"criterion": "Is overly verbose", "points": -5, "tags": ["axis:communication_quality"]},
                    ],
                },
            ]
        },
    )

    evaluator: Literal["healthbench"] = Field(description="评估器类型")
    model: Optional[str] = Field("gpt-4.1", description="grader 模型（默认 gpt-4.1，与 HealthBench 原版一致）")
    rubrics: List[HealthBenchRubric] = Field(description="评分标准列表")


# ============================================================
# HealthBench 原版 GRADER_TEMPLATE
# 来源: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
# ============================================================

GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""


# ============================================================
# HealthBenchEvalAgent
# ============================================================


class HealthBenchEvalAgent(AbstractEvalAgent, name="healthbench", params_model=HealthBenchEvalInfo):
    """基于 HealthBench 原版 rubric 的评估器"""

    _display_meta = {
        "icon": (
            "M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313"
            "-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
        ),
        "color": "#ef4444",
        "features": ["HealthBench", "Rubric 评分", "并发评估"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.035,  # ~11 rubrics × gpt-4.1 grading, USD/case
    }

    def __init__(self, eval_config: HealthBenchEvalInfo, model: str | None = None, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.eval_config: HealthBenchEvalInfo = eval_config
        self.model = model or eval_config.model or DEFAULT_MODEL
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
            # 提取完整对话（含 history）
            conversation = self._build_conversation(memory_list, self.history)
            if not conversation:
                return EvalResult(result="fail", score=0.0, feedback="无对话记录，无法评估")

            rubrics = self.eval_config.rubrics
            if not rubrics:
                return EvalResult(result="fail", score=0.0, feedback="未配置 rubrics，无法评估")

            logger.info(
                "[HealthBenchEval] 开始评估 — %d 条 rubric, model=%s",
                len(rubrics),
                self.model,
            )

            # 并发评估所有 rubric
            conversation_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in conversation)
            grading_results = await asyncio.gather(*[
                self._grade_rubric_item(conversation_str, rubric) for rubric in rubrics
            ])

            # 计算分数（与原版 calculate_score 一致）
            score = self._calculate_score(rubrics, grading_results)
            if score is None:
                return EvalResult(result="fail", score=0.0, feedback="无正分 rubric 项，无法计算分数")

            clipped_score = max(0.0, min(1.0, score))

            # 构建详情
            rubric_details = []
            for rubric, result in zip(rubrics, grading_results):
                rubric_details.append({
                    "criterion": rubric.criterion,
                    "points": rubric.points,
                    "criteria_met": result.get("criteria_met", False),
                    "explanation": result.get("explanation", ""),
                    "tags": rubric.tags,
                })

            met_count = sum(1 for r in grading_results if r.get("criteria_met"))
            feedback = f"HealthBench rubric 评估: {met_count}/{len(rubrics)} 条标准满足, 得分={clipped_score:.3f}"

            logger.info(
                "[HealthBenchEval] 评估完成 — score=%.3f, scored (%d/%d met)",
                clipped_score,
                met_count,
                len(rubrics),
            )

            return EvalResult(
                result="scored",
                score=clipped_score,
                feedback=feedback,
                trace=EvalTrace(eval_detail={"rubric_results": rubric_details, "raw_score": score}),
            )

        except Exception as e:
            logger.error("[HealthBenchEval] 评估过程出错: %s", e, exc_info=True)
            return EvalResult(result="fail", score=0.0, feedback=f"评估过程出错: {e}")

    # ----------------------------------------------------------
    # 核心方法
    # ----------------------------------------------------------

    async def _grade_rubric_item(self, conversation_str: str, rubric: HealthBenchRubric) -> dict:
        """调用 grader LLM 评估单条 rubric criterion"""
        rubric_str = f"[{rubric.points}] {rubric.criterion}"
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation_str).replace(
            "<<rubric_item>>", rubric_str
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a rubric grader for healthcare AI evaluation.",
                    input=grader_prompt,
                    max_tokens=500,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json_response(result.content)
                if parsed is not None and "criteria_met" in parsed:
                    if isinstance(parsed["criteria_met"], bool):
                        return parsed

                logger.warning(
                    "[HealthBenchEval] grader 输出格式异常 (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    result.content[:200],
                )
            except Exception as e:
                logger.warning(
                    "[HealthBenchEval] grader 调用失败 (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

        # 重试耗尽，默认 criteria_met=False
        logger.error("[HealthBenchEval] grader 重试耗尽，默认 criteria_met=False: %s", rubric.criterion[:100])
        return {"explanation": "Grading failed after retries", "criteria_met": False}

    @staticmethod
    def _calculate_score(rubrics: List[HealthBenchRubric], grading_results: List[dict]) -> Optional[float]:
        """计算分数 — 与 HealthBench 原版 calculate_score 一致"""
        total_possible = sum(r.points for r in rubrics if r.points > 0)
        if total_possible == 0:
            return None
        achieved = sum(r.points for r, g in zip(rubrics, grading_results) if g.get("criteria_met"))
        return achieved / total_possible

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """从 grader 响应中提取 JSON（兼容 markdown 包裹）"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    # ----------------------------------------------------------
    # 对话提取
    # ----------------------------------------------------------

    @staticmethod
    def _build_conversation(
        memory_list: List[TestAgentMemory],
        history: List[BaseMessage] | None = None,
    ) -> List[Dict[str, str]]:
        """从 history + memory_list 提取完整对话"""
        conversation: List[Dict[str, str]] = []

        # 1. 历史对话（TestCase.history）
        if history:
            for msg in history:
                role = "user" if msg.type == "human" else "assistant"
                conversation.append({"role": role, "content": msg.content})

        # 2. 本轮对话（memory_list）
        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue

            user_text = mem.test_reaction.action.semantic_content
            if user_text:
                conversation.append({"role": "user", "content": user_text})

            if mem.target_response:
                target_text = mem.target_response.extract_text()
                if target_text:
                    conversation.append({"role": "assistant", "content": target_text})

        return conversation
