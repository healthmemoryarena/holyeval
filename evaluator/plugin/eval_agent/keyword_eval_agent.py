"""
KeywordEvalAgent — 关键词匹配评估器

基于规则的关键词匹配评估，快速、免费、确定性高。
适用于结构化检查场景，如验证对话中是否包含特定关键词。

配置示例（KeywordEvalInfo）:
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
# 评估配置模型
# ============================================================


class KeywordRule(BaseModel):
    """关键词匹配规则"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(description="规则标识")
    rule_type: str = Field(default="keyword_check", description="规则类型（当前仅 keyword_check）")
    keywords: List[str] = Field(description="匹配关键词列表")
    match_mode: str = Field(
        default="any",
        description=(
            "匹配模式：\n"
            "  - any: 命中任一关键词即通过\n"
            "  - all: 须命中所有关键词\n"
            "  - ratio: 命中数 >= min_matches 即通过"
        ),
    )
    weight: float = Field(default=1.0, ge=0.0, description="权重")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="通过阈值")
    target: str = Field(
        default="output",
        description="匹配目标：output（仅 AI 回复）| conversation（完整对话）",
    )
    min_matches: int = Field(default=1, ge=1, description="最少匹配数（ratio 模式）")


class KeywordEvalInfo(BaseModel):
    """关键词评估配置 — 基于规则的关键词匹配评估（快速、免费、确定性高）

    适用于结构化检查场景，如验证对话中是否包含特定关键词。
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

    evaluator: Literal["keyword"] = Field(description="评估器类型")
    model: Optional[str] = Field(None, description="预留字段（关键词评估不使用 LLM）")
    rules: List[KeywordRule] = Field(description="关键词匹配规则列表")
    pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="整体通过阈值（0.0~1.0，默认 0.7）",
    )


class KeywordEvalAgent(AbstractEvalAgent, name="keyword", params_model=KeywordEvalInfo):
    """关键词匹配评估器"""

    _display_meta = {
        "icon": (
            "M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872"
            " 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568"
            " 3z M6 6h.008v.008H6V6z"
        ),
        "color": "#f59e0b",
        "features": ["零 LLM", "关键词匹配", "规则驱动"],
    }
    _cost_meta = {
        "est_cost_per_case": 0,  # 纯规则匹配，无 LLM 调用
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
        """执行关键词评估"""
        try:
            if not self.rules:
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback="缺少 rules 配置",
                    trace={"error": "missing rules"},
                )

            # 提取对话历史
            conversation = self._extract_conversation(memory_list)

            if not conversation:
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback="无对话记录",
                    trace={"error": "no conversation"},
                )

            # 构建文本
            output_text = self._build_output_text(conversation)
            conversation_text = self._build_conversation_text(conversation)

            # 评估每条规则
            metrics = []
            for rule in self.rules:
                metric = self._evaluate_rule(rule, output_text, conversation_text)
                metrics.append(metric)

            # 计算加权总分
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

            # 构建反馈
            passed_count = sum(1 for m in metrics if m["passed"])
            failed_rules = [m["metric_name"] for m in metrics if not m["passed"]]

            feedback_parts = [f"总分: {overall_score:.2f}"]
            feedback_parts.append(f"通过: {passed_count}/{len(metrics)} 条规则")
            if failed_rules:
                feedback_parts.append(f"未通过: {', '.join(failed_rules)}")

            feedback = " | ".join(feedback_parts)

            logger.info(
                f"[KeywordEval] 评估完成: score={overall_score:.2f}, "
                f"passed={passed}, {len(metrics)} 条规则"
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
                feedback=f"评估过程出错: {str(e)}",
                trace={"error": str(e)},
            )

    def _extract_conversation(
        self, memory_list: list[TestAgentMemory]
    ) -> list[dict[str, str]]:
        """从 memory 提取对话历史"""
        conversation = []

        for memory in memory_list:
            # 跳过 is_finished 幽灵轮次（未发送给 target）
            if memory.test_reaction and memory.test_reaction.is_finished and memory.target_response is None:
                continue

            # 用户消息
            if memory.test_reaction and memory.test_reaction.action:
                content = memory.test_reaction.action.semantic_content
                if content:
                    conversation.append({"role": "user", "content": content})

            # AI 响应
            target_text = self._extract_target_text(memory.target_response)
            if target_text:
                conversation.append({"role": "assistant", "content": target_text})

        return conversation

    @staticmethod
    def _extract_target_text(target_reaction: Optional[TargetAgentReaction]) -> str:
        """从 TargetAgentReaction 提取文本"""
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
        """构建 AI 输出文本（仅 assistant 的回复）"""
        return "\n".join(
            msg["content"] for msg in conversation if msg["role"] == "assistant"
        )

    def _build_conversation_text(self, conversation: list[dict[str, str]]) -> str:
        """构建完整对话文本"""
        return "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation
        )

    def _evaluate_rule(
        self,
        rule: KeywordRule,
        output_text: str,
        conversation_text: str,
    ) -> dict[str, Any]:
        """评估单条规则"""
        if rule.rule_type == "keyword_check":
            return self._evaluate_keyword_check(rule, output_text, conversation_text)
        else:
            logger.warning(f"[KeywordEval] 未知规则类型: {rule.rule_type}")
            return {
                "metric_name": rule.rule_id,
                "score": 0.0,
                "passed": False,
                "weight": rule.weight,
                "reason": f"未知规则类型: {rule.rule_type}",
            }

    def _evaluate_keyword_check(
        self,
        rule: KeywordRule,
        output_text: str,
        conversation_text: str,
    ) -> dict[str, Any]:
        """评估关键词检查规则"""
        keywords = rule.keywords
        match_mode = rule.match_mode
        target = rule.target
        threshold = rule.threshold
        weight = rule.weight

        # 选择目标文本
        text = conversation_text if target == "conversation" else output_text

        # 匹配关键词
        matched = [kw for kw in keywords if kw in text]

        # 根据匹配模式计算分数
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
            "reason": f"匹配 {len(matched)}/{len(keywords)} 个关键词: {matched}",
            "details": {
                "matched_keywords": matched,
                "missing_keywords": [kw for kw in keywords if kw not in matched],
                "total_keywords": len(keywords),
                "match_mode": match_mode,
            },
        }
