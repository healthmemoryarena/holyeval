"""
AutoTestAgent — Auto-mode virtual user (LLM-driven)

Workflow:
1. If there are remaining strict_inputs for the current turn -> send forced input directly
2. Otherwise -> call LLM to generate the next user utterance in character

LLM decides what to say and whether to end based on goal / context / finish_condition and conversation history.
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
    """Truncate text for log output"""
    if text is None:
        return "(None)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# LLM structured output format
class _LLMReaction(BaseModel):
    content: str = Field(description="What the user wants to say")
    reason: str = Field(description="Why the user says this (brief)")
    is_finished: bool = Field(description="Whether the user's goal has been achieved; true if so")
    next_fuzzy_action: Optional[str] = Field(
        None,
        description="Predicted next fuzzy action the user might take, e.g.: 'ask for more details', 'express concerns', 'prepare to end conversation', 'switch topic', etc.",
    )


class AutoTestAgent(AbstractTestAgent, name="auto"):
    """Auto-mode virtual user — LLM-driven conversation simulation"""

    _config_model = "AutoUserInfo"  # Display config model name in evaluator.core.schema

    _display_meta = {
        "icon": (
            "M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75"
            "-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25"
            " 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0"
            " 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0"
            " 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
        ),
        "color": "#a855f7",
        "features": ["LLM-driven", "Autonomous", "Smart Convergence"],
    }

    def __init__(self, user_info: AutoUserInfo, model: str = "gpt-4.1", **kwargs):
        super().__init__(user_info, **kwargs)
        self.model = model
        self.system_prompt = self._build_system_prompt()
        # Cost tracking (using langchain UsageMetadata)
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        """Get cumulative cost statistics"""
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        """Accumulate LLM call cost"""
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    def _build_system_prompt(self) -> str:
        """Build system prompt telling the LLM who it is, what to do, and when to stop"""
        lines = [
            "# Role",
            "You are role-playing as a real user having a conversation with an AI assistant.",
            "You are the user, and the AI assistant is your conversation partner. Ask questions, answer questions, and express needs like a regular user would.",
            "",
            "# Important Reminders",
            "- You are the USER, not the AI assistant!",
            "- Do not ask questions or steer the conversation like an AI assistant would",
            "- Do not use commanding language like 'just answer...' or 'first tell me...'",
            "- When the AI assistant asks you a question, you should answer it, not ask a counter-question",
            "",
            f"# Your Goal\n{self.user_info.goal}",
        ]
        if self.user_info.context:
            lines.append(f"\n# Your Background\n{self.user_info.context}")
        if self.user_info.finish_condition:
            lines.append(f"\n# Finish Condition\n{self.user_info.finish_condition}")
        else:
            lines.append("\n# Finish Condition\nWhen you believe the goal has been achieved, set is_finished to true.")
        lines.append("\n# Behavior Requirements")
        lines.append("- Speak like a real user — concise and natural")
        lines.append("- Answer the AI assistant's questions instead of asking counter-questions")
        lines.append("- Do not reveal that you are an AI")
        lines.append(
            "\nAlso, predict your next possible fuzzy action (next_fuzzy_action), "
            "e.g.: 'ask for more details', 'express concerns', 'prepare to end conversation', 'switch topic', 'seek confirmation', etc."
        )
        return "\n".join(lines)

    async def _generate_next_reaction(
        self, target_reaction: Optional[TargetAgentReaction]
    ) -> TestAgentReaction:
        """Generate next user reaction"""

        # ---- 1. If there are remaining strict_inputs, send directly ----
        strict_index = self.current_turn - 1  # current_turn already incremented in do_generate
        if strict_index < len(self.user_info.strict_inputs):
            forced_text = self.user_info.strict_inputs[strict_index]
            logger.info(
                "[TestAgent] Turn %d — using strict_input[%d]: %s",
                self.current_turn, strict_index, _truncate(forced_text, 100),
            )
            return TestAgentReaction(
                action=TestAgentAction(type="semantic", semantic_content=forced_text),
                reason="forced input",
                is_finished=False,
            )

        # ---- 2. Build conversation history ----
        # Note role mapping: LLM plays the virtual user, so virtual user's words are assistant (LLM itself),
        # and the AI assistant's words are user (the other party). This lets the LLM correctly identify with the virtual user role.
        llm_history: list[BasicMessage] = []

        # 2a. Pre-evaluation history (role flipped: original HumanMessage->assistant, AIMessage->user)
        for msg in self.history:
            flipped_role: str = "assistant" if isinstance(msg, HumanMessage) else "user"
            llm_history.append(BasicMessage(role=flipped_role, content=msg.content))

        # 2b. Current evaluation conversation memory (all turns)
        for mem in self.memory_list:
            # What the virtual user (LLM itself) said -> assistant
            my_text = mem.test_reaction.action.semantic_content
            if my_text:
                llm_history.append(BasicMessage(role="assistant", content=my_text))

            # What the system under test (other party) replied -> user
            target_text = mem.target_response.extract_text() if mem.target_response else ""
            if target_text:
                llm_history.append(BasicMessage(role="user", content=target_text))

        # ---- 3. Determine input ----
        if not llm_history:
            input_text = "Please start the conversation with your first message."
        else:
            input_text = "Please continue the conversation."

        logger.debug(
            "[TestAgent] Turn %d — calling LLM (model=%s), history=%d messages",
            self.current_turn, self.model, len(llm_history),
        )

        result = await do_execute(
            model=self.model,
            system_prompt=self.system_prompt,
            input=input_text,
            history_messages=llm_history if llm_history else None,
            response_format=_LLMReaction,
        )

        # Accumulate cost
        self._accumulate_cost(result.usage)

        # ---- 4. Build result ----
        if result.data:
            logger.info(
                "[TestAgent] Turn %d — LLM generated: %s (reason: %s, is_finished=%s, next_fuzzy=%s)",
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

        # fallback: structured parsing failed, use raw text
        logger.warning(
            "[TestAgent] Turn %d — structured parsing failed, using raw text: %s",
            self.current_turn, _truncate(result.content, 100),
        )
        return TestAgentReaction(
            action=TestAgentAction(type="semantic", semantic_content=result.content),
            reason="LLM raw output (structured parsing failed)",
            is_finished=False,
            usage=result.usage,
        )
