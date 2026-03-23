"""
Eval-Only 模式测试

测试覆盖：
1. messages_to_memory 转换
2. do_eval_only 单条评测（TestCase + memory_list）
3. do_batch_eval 批量评测

运行：uv run pytest evaluator/tests/test_eval_only.py -v -s
"""

from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from evaluator.core.orchestrator import do_batch_eval, do_eval_only
from evaluator.core.schema import (
    TargetAgentReaction,
    TestAgentAction,
    TestAgentMemory,
    TestAgentReaction,
    TestCase,
)


# ============================================================
# Helper: messages → memory_list（上层转换逻辑）
# ============================================================


def messages_to_memory(messages: list) -> list[TestAgentMemory]:
    """将 [{role, content}] 转换为 TestAgentMemory 列表"""
    memory_list: list[TestAgentMemory] = []
    now = datetime.now()
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.type == "human":
            action = TestAgentAction(type="semantic", semantic_content=str(msg.content))
            reaction = TestAgentReaction(action=action, is_finished=False)
            target = None
            if i + 1 < len(messages) and messages[i + 1].type == "ai":
                target = TargetAgentReaction(type="message", message_list=[{"content": str(messages[i + 1].content)}])
                i += 2
            else:
                i += 1
            memory_list.append(TestAgentMemory(
                test_reaction=reaction, test_reaction_time=now,
                target_response=target, target_response_time=now if target else None,
            ))
        else:
            i += 1
    return memory_list


# ============================================================
# Fixtures
# ============================================================

_RAW_MESSAGES = [
    HumanMessage(content="我最近老是头疼"),
    AIMessage(content="请问持续多久了？"),
    HumanMessage(content="大概一周了"),
    AIMessage(content="建议去医院检查一下。"),
]


def _make_test_case(eval_config=None) -> TestCase:
    return TestCase(
        id="test_001",
        title="测试对话",
        user={"type": "manual", "strict_inputs": ["(eval-only)"]},
        eval=eval_config or {"evaluator": "keyword", "rules": [
            {"rule_id": "r1", "keywords": ["医院", "检查"], "match_mode": "any"},
        ], "pass_threshold": 0.5},
        tags=["测试"],
    )


# ============================================================
# 1. messages_to_memory 转换测试
# ============================================================


class TestMessagesToMemory:
    def test_basic_conversion(self):
        memory = messages_to_memory(_RAW_MESSAGES)
        assert len(memory) == 2

    def test_memory_content(self):
        msgs = [HumanMessage(content="你好"), AIMessage(content="你好，有什么可以帮您？")]
        memory = messages_to_memory(msgs)
        assert memory[0].test_reaction.action.semantic_content == "你好"
        assert memory[0].target_response.extract_text() == "你好，有什么可以帮您？"

    def test_trailing_user_message(self):
        msgs = [HumanMessage(content="你好"), AIMessage(content="嗯"), HumanMessage(content="再见")]
        memory = messages_to_memory(msgs)
        assert len(memory) == 2
        assert memory[1].target_response is None


# ============================================================
# 2. do_eval_only 端到端测试
# ============================================================


class TestDoEvalOnly:
    @pytest.mark.asyncio
    async def test_keyword_eval(self):
        import evaluator.plugin.eval_agent  # noqa: F401

        tc = _make_test_case()
        memory = messages_to_memory(_RAW_MESSAGES)
        result = await do_eval_only(tc, memory)

        assert result.id == "test_001"
        assert result.eval_type == "keyword"
        assert result.user_type == "eval_only"
        assert result.eval.score >= 0.0

    @pytest.mark.asyncio
    async def test_result_contains_trace(self):
        import evaluator.plugin.eval_agent  # noqa: F401

        tc = _make_test_case()
        memory = messages_to_memory(_RAW_MESSAGES)
        result = await do_eval_only(tc, memory)

        assert result.eval.trace is not None
        assert len(result.eval.trace.test_memory) == 2


# ============================================================
# 3. do_batch_eval 批量测试
# ============================================================


class TestDoBatchEval:
    @pytest.mark.asyncio
    async def test_batch_eval(self):
        import evaluator.plugin.eval_agent  # noqa: F401

        items = [
            (_make_test_case(), messages_to_memory(_RAW_MESSAGES)),
            (
                TestCase(
                    id="test_002", title="第二条",
                    user={"type": "manual", "strict_inputs": ["x"]},
                    eval={"evaluator": "keyword", "rules": [
                        {"rule_id": "r1", "keywords": ["医院"], "match_mode": "any"},
                    ], "pass_threshold": 0.5},
                ),
                messages_to_memory([HumanMessage(content="感冒了"), AIMessage(content="建议去医院看看。")]),
            ),
        ]

        report = await do_batch_eval(items, max_concurrency=2)
        assert len(report.cases) == 2
