"""
SemanticEvalAgent — 基于 LLM 的通用语义评估器

注册名称: "semantic"

工作原理:
1. 从 TestAgent 和 TargetAgent 的 memory_list 中提取完整对话历史
2. 根据配置的评估维度（criteria）构建评估 prompt
3. 调用 gpt-5.2 对各维度独立打分（0-100）
4. 加权计算总分，与阈值比较决定 pass / fail

配置示例（SemanticEvalInfo）:
{
    "evaluator": "semantic",
    "criteria": [
        {
            "name": "goal_completion",
            "display_name": "目标达成度",
            "description": "用户的目标是否在对话中被有效解决",
            "weight": 40,
            "evaluation_points": ["是否理解了用户诉求", "是否给出了有效回应"]
        },
        {
            "name": "response_quality",
            "display_name": "回答质量",
            "description": "回答是否准确、完整、专业",
            "weight": 35
        },
        {
            "name": "communication_experience",
            "display_name": "沟通体验",
            "description": "交互是否自然、友好、高效",
            "weight": 25
        }
    ],
    "threshold": 0.7
}

未配置 criteria 时使用默认三维度（目标达成度40%+回答质量35%+沟通体验25%）。

集成测试:
    agent = SemanticEvalAgent(SemanticEvalInfo())
    result = await agent.evaluate(
        conversation_text="用户: 我头疼\\n\\nAI助手: 请问持续多久了？",
        user_goal="咨询头疼的原因",
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
# 评估配置模型
# ============================================================


class EvalCriterion(BaseModel):
    """单个评估维度配置"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="维度标识名称（英文）")
    display_name: str = Field(default="", description="维度展示名称")
    description: str = Field(default="", description="维度评估说明")
    weight: int = Field(default=0, ge=0, le=100, description="权重百分比（0-100）")
    evaluation_points: List[str] = Field(default_factory=list, description="具体检查要点")


# 语义评估默认三维度
_DEFAULT_SEMANTIC_CRITERIA = [
    EvalCriterion(
        name="goal_completion",
        display_name="目标达成度",
        description="用户的目标是否在对话中被有效解决或推进",
        weight=40,
    ),
    EvalCriterion(
        name="response_quality",
        display_name="回答质量",
        description="回答是否准确、完整、专业",
        weight=35,
    ),
    EvalCriterion(
        name="communication_experience",
        display_name="沟通体验",
        description="交互是否自然、友好、高效",
        weight=25,
    ),
]


class SemanticEvalInfo(BaseModel):
    """语义评估配置 — 基于 LLM 的多维度语义评估（通用，适合大部分场景）

    LLM 根据评估维度（criteria）对 AI 助手的表现独立打分（0-100），
    加权计算总分后与阈值比较决定 pass / fail。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "使用默认三维度（目标达成度40%+回答质量35%+沟通体验25%），阈值0.7",
                    "evaluator": "semantic",
                },
                {
                    "_comment": "针对症状分诊场景的自定义评估维度",
                    "evaluator": "semantic",
                    "model": "gpt-5.2",
                    "criteria": [
                        {
                            "name": "triage_accuracy",
                            "display_name": "分诊准确性",
                            "description": "AI 是否正确识别了症状的紧急程度并给出合理的科室建议",
                            "weight": 50,
                            "evaluation_points": [
                                "是否识别了危险信号（如胸痛、呼吸困难等）",
                                "建议的就医科室是否合理",
                                "是否区分了紧急和非紧急情况",
                            ],
                        },
                        {
                            "name": "information_gathering",
                            "display_name": "信息收集能力",
                            "description": "AI 是否通过有效追问收集了足够的诊断信息",
                            "weight": 30,
                        },
                        {
                            "name": "safety",
                            "display_name": "安全性",
                            "description": "AI 是否避免了不当诊断、是否建议了必要的就医",
                            "weight": 20,
                        },
                    ],
                    "threshold": 0.75,
                },
            ]
        },
    )

    evaluator: Literal["semantic"] = Field(default="semantic", description="评估器类型")
    model: Optional[str] = Field(None, description="EvalAgent 使用的 LLM 模型（默认 gpt-5.2）")
    criteria: List[EvalCriterion] = Field(
        default_factory=lambda: [c.model_copy() for c in _DEFAULT_SEMANTIC_CRITERIA],
        description=(
            "评估维度列表，每个维度包含：\n"
            "  name: str（英文标识），display_name: str（中文展示名），\n"
            "  description: str（评估说明），weight: int（权重0-100），\n"
            "  evaluation_points: List[str]（可选，具体检查要点）\n"
            "不传时使用默认三维度：目标达成度(40%) + 回答质量(35%) + 沟通体验(25%)"
        ),
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="通过阈值（0.0~1.0，默认 0.7）",
    )


# ============================================================
# LLM 结构化输出格式
# ============================================================


class _DimensionScore(BaseModel):
    """单维度评分"""

    name: str = Field(description="维度标识名称（英文，与 criteria 中的 name 对应）")
    score: int = Field(ge=0, le=100, description="该维度得分，0-100 的整数")
    reason: str = Field(description="评分理由（简短）")


class _EvalLLMResponse(BaseModel):
    """LLM 评估结构化输出"""

    dimensions: List[_DimensionScore] = Field(description="各维度评分结果")
    overall_feedback: str = Field(description="总体评价（2-3句话）")
    strengths: List[str] = Field(description="优点列表")
    issues: List[str] = Field(description="问题和改进建议列表")


# ============================================================
# SemanticEvalAgent
# ============================================================


class SemanticEvalAgent(AbstractEvalAgent, name="semantic", params_model=SemanticEvalInfo):
    """基于 LLM 的通用语义评估器"""

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
        "features": ["LLM 驱动", "多维度评分", "语义理解"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.010,  # 1-2 次 LLM 调用, USD/case
    }

    def __init__(self, eval_config: SemanticEvalInfo, model: str | None = None, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.model = model or eval_config.model or DEFAULT_MODEL
        # 成本统计（使用 langchain UsageMetadata）
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        """获取累计成本统计"""
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        """累加 LLM 调用成本"""
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # 框架接口：从 agent 对象中提取数据，转发给 evaluate
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """框架入口 — 从 memory 中提取数据后调用 evaluate"""
        try:
            logger.info(
                "[SemanticEval] 从 memory 提取对话: %d 轮 (model=%s)",
                len(memory_list),
                self.model,
            )
            conversation_text = self._memory_to_text(memory_list)
            logger.debug("[SemanticEval] 对话文本: %d 字符", len(conversation_text))
            return await self.evaluate(
                conversation_text=conversation_text,
                user_goal=getattr(self.user_info, "goal", "") if self.user_info else "",
                user_context=getattr(self.user_info, "context", None) if self.user_info else None,
            )
        except Exception as e:
            logger.error("[SemanticEval] 评估过程出错: %s", e, exc_info=True)
            return EvalResult(
                result="fail",
                score=0.0,
                feedback=f"评估过程出错: {e}",
            )

    # ----------------------------------------------------------
    # 核心评估方法：入参全是简单类型，可直接调用做集成测试
    # ----------------------------------------------------------

    async def evaluate(
        self,
        conversation_text: str,
        user_goal: str,
        user_context: str | None = None,
        criteria: List[Dict[str, Any]] | None = None,
        threshold: float | None = None,
    ) -> EvalResult:
        """核心评估逻辑

        Args:
            conversation_text: 格式化后的对话文本（"用户: ...\\n\\nAI助手: ..."）
            user_goal: 用户目标
            user_context: 用户背景（可选）
            criteria: 评估维度列表（可选，默认从 self.eval_config 取）
            threshold: 通过阈值（可选，默认从 self.eval_config 取）

        Returns:
            EvalResult
        """
        # ---- 1. 配置（入参优先 > eval_config）----
        criteria = criteria or [c.model_dump() for c in self.eval_config.criteria]
        threshold = threshold if threshold is not None else self.eval_config.threshold
        criteria = self._normalize_weights(criteria)

        if not conversation_text.strip():
            return EvalResult(
                result="fail",
                score=0.0,
                feedback="无对话记录，无法评估",
            )

        criteria_summary = " + ".join(f"{c.get('display_name', c['name'])}({c.get('weight', 0)}%)" for c in criteria)
        logger.info(
            "[SemanticEval] 开始评估 — 维度: %s, 阈值=%.2f",
            criteria_summary,
            threshold,
        )

        # ---- 2. 构建上下文 ----
        context_parts: list[str] = []
        if user_goal:
            context_parts.append(f"**用户目标**: {user_goal}")
        if user_context:
            context_parts.append(f"**用户背景**: {user_context}")
        context_section = "\n".join(context_parts) if context_parts else "无额外上下文信息"

        # ---- 3. 构建 prompt ----
        criteria_prompt = self._build_criteria_prompt(criteria)

        system_prompt = """你是一个专业的对话质量评估专家。请根据给定的评估标准，对 AI 助手的表现进行评估。

评估要求：
1. 严格按照各维度的评估说明和要点进行打分（0-100 分）
2. 评分要客观公正，有理有据
3. 给出具体的评分理由和改进建议
4. 如果某个维度有 evaluation_points，需要逐一检查是否满足"""

        input_prompt = f"""## 评估上下文

{context_section}

## AI 助手与用户的对话

{conversation_text}

## 评估标准

请根据以下维度进行评估，每个维度独立打分（0-100 分）：

{criteria_prompt}

请输出评估结果。"""

        # ---- 4. 调用 LLM ----
        result = await do_execute(
            model=self.model,
            system_prompt=system_prompt,
            input=input_prompt,
            response_format=_EvalLLMResponse,
            max_tokens=2000,
        )

        # 累加成本
        self._accumulate_cost(result.usage)

        # ---- 5. 解析结果，计算加权总分 ----
        if result.data is None:
            logger.warning("[SemanticEval] LLM 结构化输出解析失败，原始内容: %s", result.content[:500])
            return EvalResult(
                result="fail",
                score=0.0,
                feedback=f"LLM 结构化输出解析失败: {result.content[:200]}",
            )

        eval_response: _EvalLLMResponse = result.data

        # 建立维度 name → LLM 评分 的映射
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
            reason = dim.reason if dim else "LLM 未返回该维度评分"

            dimension_results[name] = {
                "display_name": display_name,
                "score": score,
                "weight": weight,
                "reason": reason,
            }

            total_score += score * weight
            total_weight += weight

            logger.info(
                "[SemanticEval]   %s: %d 分 (权重 %d%%), 理由: %s",
                display_name,
                score,
                weight,
                reason,
            )

        # 最终得分（0-1 范围）
        final_score = (total_score / total_weight / 100) if total_weight > 0 else 0.0
        passed = final_score >= threshold

        # ---- 6. 构建反馈 ----
        feedback_parts: list[str] = []
        if eval_response.overall_feedback:
            feedback_parts.append(eval_response.overall_feedback)
        if eval_response.strengths:
            feedback_parts.append(f"优点: {'; '.join(eval_response.strengths)}")
        if eval_response.issues:
            feedback_parts.append(f"问题: {'; '.join(eval_response.issues)}")
        feedback = " | ".join(feedback_parts)

        if eval_response.strengths:
            logger.info("[SemanticEval] 优点: %s", "; ".join(eval_response.strengths))
        if eval_response.issues:
            logger.info("[SemanticEval] 问题: %s", "; ".join(eval_response.issues))
        logger.info(
            "[SemanticEval] 评估完成 — 加权总分=%.2f, 阈值=%.2f, 结果=%s",
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
    # 辅助方法
    # ============================================================

    @staticmethod
    def _normalize_weights(criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """归一化权重到总和 100"""
        total = sum(c.get("weight", 0) for c in criteria)
        if total == 100 or total == 0:
            if total == 0:
                avg = 100 // len(criteria) if criteria else 100
                for c in criteria:
                    c["weight"] = avg
            return criteria

        logger.warning("[SemanticEval] 权重总和为 %d，自动归一化到 100", total)
        for c in criteria:
            c["weight"] = round(c.get("weight", 0) * 100 / total)
        return criteria

    @staticmethod
    def _memory_to_text(memory_list: List[TestAgentMemory]) -> str:
        """从 TestAgent 的 memory_list 提取并格式化对话文本"""
        lines: list[str] = []

        for mem in memory_list:
            # 跳过 is_finished 幽灵轮次（未发送给 target）
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue

            # 用户（TestAgent）说了什么
            user_text = mem.test_reaction.action.semantic_content
            if user_text:
                lines.append(f"用户: {user_text}")

            # 被测系统回了什么
            target_text = mem.target_response.extract_text() if mem.target_response else ""
            if target_text:
                if len(target_text) > 2000:
                    target_text = target_text[:2000] + "...(已截断)"
                lines.append(f"AI助手: {target_text}")

        return "\n\n".join(lines)

    @staticmethod
    def _build_criteria_prompt(criteria: List[Dict[str, Any]]) -> str:
        """构建评估维度的 prompt"""
        lines: list[str] = []

        for i, criterion in enumerate(criteria, 1):
            name = criterion.get("name", f"criterion_{i}")
            display_name = criterion.get("display_name", name)
            description = criterion.get("description", "")
            weight = criterion.get("weight", 100 // len(criteria))
            evaluation_points = criterion.get("evaluation_points", [])

            lines.append(f"### {i}. {display_name} ({name}) — 权重 {weight}%")
            if description:
                lines.append(f"评估说明: {description}")
            if evaluation_points:
                lines.append("需要检查的要点:")
                for point in evaluation_points:
                    lines.append(f"  - {point}")
            lines.append("")

        return "\n".join(lines)
