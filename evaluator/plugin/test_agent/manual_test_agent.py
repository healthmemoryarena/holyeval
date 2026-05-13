"""
ManualTestAgent — Manual-mode virtual user (script-driven)

Sends preset inputs from strict_inputs in order, finishes when exhausted.
Zero LLM calls, fully deterministic, zero cost.

Use cases: data extraction validation, preset Q&A, regression tests, and other single/fixed-turn tests.
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
    """Manual-mode virtual user — sends strict_inputs in order, auto-finishes when exhausted"""

    _config_model = "ManualUserInfo"

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621"
            " 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0"
            " 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
        ),
        "color": "#64748b",
        "features": ["Zero LLM", "Script-driven", "Deterministic"],
    }

    def __init__(self, user_info: ManualUserInfo, **kwargs):
        super().__init__(user_info, **kwargs)
        # Manual mode turn count determined by strict_inputs length, ignores UserInfo.max_turns
        self.max_turns = len(self.user_info.strict_inputs) + 1

    async def _generate_next_reaction(
        self, target_reaction: Optional[TargetAgentReaction]
    ) -> TestAgentReaction:
        """Consume strict_inputs in order, end conversation when exhausted"""
        idx = self.current_turn - 1  # current_turn already incremented in do_generate

        if idx < len(self.user_info.strict_inputs):
            text = self.user_info.strict_inputs[idx]
            # Attachments are addressed by turn number (1-based). Pick the
            # ones that belong to the current turn; target agent decides
            # how to actually ship them to the system-under-test.
            turn_attachments = [
                a for a in getattr(self.user_info, "attachments", []) or []
                if a.turn == self.current_turn
            ]
            logger.info(
                "[ManualTestAgent] Turn %d — strict_input[%d]: %s%s",
                self.current_turn,
                idx,
                text[:80] + "..." if len(text) > 80 else text,
                f" (+{len(turn_attachments)} attachment(s))" if turn_attachments else "",
            )
            return TestAgentReaction(
                action=TestAgentAction(
                    type="semantic",
                    semantic_content=text,
                    attachments=turn_attachments,
                ),
                reason="scripted input",
                is_finished=False,
            )

        # All strict_inputs consumed -> finish
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
