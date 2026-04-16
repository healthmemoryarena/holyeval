"""
EngagementEvalAgent — 用户参与度评估器 (LLM-as-Judge)

Registered name: "engagement"

判定虚拟用户在看到产品开场白后是否产生"主动参与"行为：
- engaged (1.0):     主动披露健康信息、提出具体问题、或表达使用意愿
- not_engaged (0.0): 敷衍回复、拒绝、质疑产品、未提供有效信息、或沉默([沉默])

用于开场白 A/B 测试，通过 pass_rate 统计"主动录入率"。
"""

import json
import logging
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage
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
# Judge Prompt
# ============================================================

_JUDGE_PROMPT = """你是一个用户行为分析师。以下是一款健康管理 APP 的开场对话。
产品先发了一句开场白，用户做出了回应。

# 开场白（产品发出）
{opening}

# 用户回应
{user_responses}

# 任务
请判断用户是否产生了"主动参与"行为：

- **engaged**: 用户主动描述了自己的健康状况、提出了具体问题、表达了使用意愿、或提供了个人健康相关信息
- **not_engaged**: 用户敷衍回复（如"嗯"、"好的"）、表示拒绝（如"不需要"、"不用了"）、质疑产品（如"你是谁"、"靠谱吗"）、未提供任何有效健康信息、或选择沉默（如"[沉默]"）

请严格按以下 JSON 格式输出，不要输出其他内容：
```json
{{"result": "engaged" | "not_engaged", "reason": "一句话说明判断依据"}}
```"""


# ============================================================
# Config model
# ============================================================


class EngagementEvalInfo(BaseModel):
    """Engagement 评估配置 — 用户参与度二元判定"""

    model_config = ConfigDict(extra="forbid")

    evaluator: Literal["engagement"] = Field(description="评估器类型")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="通过阈值（0.5 → engaged=pass, not_engaged=fail）",
    )
    model: Optional[str] = Field(None, description="Judge LLM 模型（默认 gpt-4.1）")


# ============================================================
# EvalAgent
# ============================================================


class EngagementEvalAgent(AbstractEvalAgent, name="engagement", params_model=EngagementEvalInfo):
    """用户参与度评估器 — LLM-as-Judge 二元判定"""

    _display_meta = {
        "icon": (
            "M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933"
            " 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
        ),
        "color": "#10b981",
        "features": ["LLM-as-Judge", "Binary", "Engagement"],
    }
    _cost_meta = {"est_cost_per_case": 0.005}

    def __init__(self, eval_config: EngagementEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.threshold = eval_config.threshold
        self.model = eval_config.model or DEFAULT_MODEL

    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """执行参与度评估"""
        try:
            # 1. 提取开场白（来自 history — 最后一条 AI 消息）
            opening = ""
            for msg in reversed(self.history):
                if isinstance(msg, AIMessage):
                    opening = msg.content
                    break

            if not opening:
                return EvalResult(
                    result="error",
                    score=0.0,
                    feedback="无法提取开场白（history 中无 AI 消息）",
                )

            # 2. 提取用户回应
            user_responses = []
            for mem in memory_list:
                if mem.test_reaction and mem.test_reaction.action:
                    content = mem.test_reaction.action.semantic_content
                    if content:
                        user_responses.append(content)

            if not user_responses:
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback="用户无任何回应 → not_engaged",
                )

            user_text = "\n".join(f"- {r}" for r in user_responses)

            # 3. 调用 Judge LLM
            prompt = _JUDGE_PROMPT.format(opening=opening, user_responses=user_text)

            result = await do_execute(
                model=self.model,
                system_prompt="You are a user behavior analyst. Output valid JSON only.",
                input=prompt,
                max_tokens=200,
            )

            # 4. 解析结果
            content = result.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            is_engaged = parsed.get("result") == "engaged"
            reason = parsed.get("reason", "")

            score = 1.0 if is_engaged else 0.0
            passed = score >= self.threshold

            return EvalResult(
                result="pass" if passed else "fail",
                score=score,
                feedback=f"{'engaged' if is_engaged else 'not_engaged'}: {reason}",
                trace=EvalTrace(
                    eval_detail={"judge_output": parsed, "opening": opening, "user_responses": user_responses},
                ),
            )

        except json.JSONDecodeError as e:
            logger.error("[EngagementEval] JSON 解析失败: %s", e)
            return EvalResult(
                result="error",
                score=0.0,
                feedback=f"Judge 输出解析失败: {e}",
            )
        except Exception as e:
            logger.error("[EngagementEval] 评估异常: %s", e, exc_info=True)
            return EvalResult(
                result="error",
                score=0.0,
                feedback=f"评估异常: {e}",
            )
