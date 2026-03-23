"""
AutoTestAgent — 自动模式虚拟用户（LLM 驱动）

工作流程：
1. 如果当前轮次还有 strict_inputs 没消费完 → 直接发送强制输入
2. 否则 → 调用 LLM，让它扮演用户角色生成下一句话

LLM 会根据 goal / context / finish_condition 和对话历史来决定说什么、是否结束。
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, Field

from evaluator.core.interfaces.abstract_test_agent import AbstractTestAgent
from evaluator.core.schema import (
    AutoUserInfo,
    TargetAgentReaction,
    TestAgentAction,
    TestAgentReaction,
)
from evaluator.utils.llm import BasicMessage, do_execute

logger = logging.getLogger(__name__)


def _truncate(text: str | None, max_len: int = 80) -> str:
    """截断文本用于日志输出"""
    if text is None:
        return "(None)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# LLM 结构化输出格式
class _LLMReaction(BaseModel):
    content: str = Field(description="用户要说的话")
    reason: str = Field(description="用户这样说的原因（简短）")
    is_finished: bool = Field(description="用户目标是否已达成，达成则为 true")
    next_fuzzy_action: Optional[str] = Field(
        None,
        description="预测用户下一步可能的模糊行为，例如：'继续追问细节'、'表达疑虑'、'准备结束对话'、'切换话题'等",
    )


class AutoTestAgent(AbstractTestAgent, name="auto"):
    """自动模式虚拟用户 — 基于 LLM 驱动的对话模拟"""

    _config_model = "AutoUserInfo"  # evaluator.core.schema 中的展示用配置模型名

    _display_meta = {
        "icon": (
            "M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75"
            "-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25"
            " 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0"
            " 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0"
            " 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
        ),
        "color": "#a855f7",
        "features": ["LLM 驱动", "自主对话", "智能收敛"],
    }

    def __init__(self, user_info: AutoUserInfo, model: str = "gpt-4.1", **kwargs):
        super().__init__(user_info, **kwargs)
        self.model = model
        self.system_prompt = self._build_system_prompt()
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

    def _build_system_prompt(self) -> str:
        """拼 system prompt，告诉 LLM 它是谁、要干嘛、什么时候停"""
        lines = [
            "# 角色设定",
            "你正在扮演一个真实用户，与一个 AI 助手进行对话。",
            "你是用户，AI 助手是你的对话对象。你要像普通用户一样提问、回答问题、表达需求。",
            "",
            "# 重要提醒",
            "- 你是用户，不是 AI 助手！",
            "- 不要像 AI 助手那样提问或引导对话",
            "- 不要使用「你就回答...」「你先告诉我...」这类指令性语言",
            "- 当 AI 助手问你问题时，你应该回答，而不是反问",
            "",
            f"# 你的目标\n{self.user_info.goal}",
        ]
        if self.user_info.context:
            lines.append(f"\n# 你的背景信息\n{self.user_info.context}")
        if self.user_info.finish_condition:
            lines.append(f"\n# 结束条件\n{self.user_info.finish_condition}")
        else:
            lines.append("\n# 结束条件\n当你认为目标已经达成时，将 is_finished 设为 true。")
        lines.append("\n# 行为要求")
        lines.append("- 像真实用户一样说话，简洁自然")
        lines.append("- 回答 AI 助手的问题，而不是反问")
        lines.append("- 不要暴露你是 AI")
        lines.append(
            "\n同时，请预测你下一步可能的模糊行为（next_fuzzy_action），"
            "例如：'继续追问细节'、'表达疑虑'、'准备结束对话'、'切换话题'、'寻求确认'等。"
        )
        return "\n".join(lines)

    async def _generate_next_reaction(
        self, target_reaction: Optional[TargetAgentReaction]
    ) -> TestAgentReaction:
        """生成下一步用户反应"""

        # ---- 1. 如果有 strict_inputs 还没用完，直接发 ----
        strict_index = self.current_turn - 1  # current_turn 已在 do_generate 中 +1
        if strict_index < len(self.user_info.strict_inputs):
            forced_text = self.user_info.strict_inputs[strict_index]
            logger.info(
                "[TestAgent] Turn %d — 使用 strict_input[%d]: %s",
                self.current_turn, strict_index, _truncate(forced_text, 100),
            )
            return TestAgentReaction(
                action=TestAgentAction(type="semantic", semantic_content=forced_text),
                reason="强制输入",
                is_finished=False,
            )

        # ---- 2. 拼对话历史 ----
        # 注意 role 映射：LLM 扮演虚拟用户，所以虚拟用户说的是 assistant（LLM 自己），
        # AI 助手说的是 user（对方）。这样 LLM 才能正确认同虚拟用户角色。
        llm_history: list[BasicMessage] = []

        # 2a. 评测前的历史对话（role 翻转：原始 HumanMessage→assistant, AIMessage→user）
        for msg in self.history:
            flipped_role: str = "assistant" if isinstance(msg, HumanMessage) else "user"
            llm_history.append(BasicMessage(role=flipped_role, content=msg.content))

        # 2b. 本次评测的对话记忆（完整包含所有轮次）
        for mem in self.memory_list:
            # 虚拟用户（LLM 自己）说了什么 → assistant
            my_text = mem.test_reaction.action.semantic_content
            if my_text:
                llm_history.append(BasicMessage(role="assistant", content=my_text))

            # 被测系统（对方）回了什么 → user
            target_text = mem.target_response.extract_text() if mem.target_response else ""
            if target_text:
                llm_history.append(BasicMessage(role="user", content=target_text))

        # ---- 3. 确定 input ----
        if not llm_history:
            input_text = "请开始对话，说出你的第一句话。"
        else:
            input_text = "请继续对话。"

        logger.debug(
            "[TestAgent] Turn %d — 调用 LLM (model=%s), history=%d 条消息",
            self.current_turn, self.model, len(llm_history),
        )

        result = await do_execute(
            model=self.model,
            system_prompt=self.system_prompt,
            input=input_text,
            history_messages=llm_history if llm_history else None,
            response_format=_LLMReaction,
        )

        # 累加成本
        self._accumulate_cost(result.usage)

        # ---- 4. 拼结果 ----
        if result.data:
            logger.info(
                "[TestAgent] Turn %d — LLM 生成: %s (reason: %s, is_finished=%s, next_fuzzy=%s)",
                self.current_turn,
                _truncate(result.data.content, 100),
                result.data.reason,
                result.data.is_finished,
                result.data.next_fuzzy_action,
            )
            return TestAgentReaction(
                action=TestAgentAction(type="semantic", semantic_content=result.data.content),
                next_fuzzy_action=result.data.next_fuzzy_action,
                reason=result.data.reason,
                is_finished=result.data.is_finished,
                usage=result.usage,
            )

        # fallback：结构化解析失败，用原始文本
        logger.warning(
            "[TestAgent] Turn %d — 结构化解析失败，使用原始文本: %s",
            self.current_turn, _truncate(result.content, 100),
        )
        return TestAgentReaction(
            action=TestAgentAction(type="semantic", semantic_content=result.content),
            reason="LLM 原始输出（结构化解析失败）",
            is_finished=False,
            usage=result.usage,
        )
