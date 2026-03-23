"""
ManualTestAgent — 手动模式虚拟用户（脚本驱动）

按顺序发送 strict_inputs 中的预设输入，用完即结束。
零 LLM 调用、完全确定性、零成本。

适用场景：数据提取验证、预设问答、回归测试等单轮/固定轮次测试。
"""

import logging
from typing import Optional

from evaluator.core.interfaces.abstract_test_agent import AbstractTestAgent
from evaluator.core.schema import (
    ManualUserInfo,
    TargetAgentReaction,
    TestAgentAction,
    TestAgentReaction,
)

logger = logging.getLogger(__name__)


class ManualTestAgent(AbstractTestAgent, name="manual"):
    """手动模式虚拟用户 — 按序发送 strict_inputs，用完自动结束"""

    _config_model = "ManualUserInfo"

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621"
            " 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0"
            " 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
        ),
        "color": "#64748b",
        "features": ["零 LLM", "脚本驱动", "确定性执行"],
    }

    def __init__(self, user_info: ManualUserInfo, **kwargs):
        super().__init__(user_info, **kwargs)
        # 手动模式轮次由 strict_inputs 数量决定，忽略 UserInfo.max_turns
        self.max_turns = len(self.user_info.strict_inputs) + 1

    async def _generate_next_reaction(
        self, target_reaction: Optional[TargetAgentReaction]
    ) -> TestAgentReaction:
        """按序消费 strict_inputs，用完则结束对话"""
        idx = self.current_turn - 1  # current_turn 已在 do_generate 中 +1

        if idx < len(self.user_info.strict_inputs):
            text = self.user_info.strict_inputs[idx]
            logger.info(
                "[ManualTestAgent] Turn %d — strict_input[%d]: %s",
                self.current_turn,
                idx,
                text[:80] + "..." if len(text) > 80 else text,
            )
            return TestAgentReaction(
                action=TestAgentAction(type="semantic", semantic_content=text),
                reason="scripted input",
                is_finished=False,
            )

        # strict_inputs 全部消费完 → 结束
        logger.info(
            "[ManualTestAgent] Turn %d — all %d strict_inputs consumed, finishing",
            self.current_turn,
            len(self.user_info.strict_inputs),
        )
        return TestAgentReaction(
            action=TestAgentAction(type="semantic", semantic_content=""),
            reason="all strict_inputs consumed",
            is_finished=True,
        )
