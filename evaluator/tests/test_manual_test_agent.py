"""
ManualTestAgent 单元测试

测试覆盖：
1. strict_inputs 顺序消费
2. inputs 用完自动 is_finished=True
3. 空 strict_inputs 首轮即结束
4. max_turns 自动由 strict_inputs 数量决定
5. memory 正确记录与补全
6. 注册表可通过 "manual" 查找

运行：uv run pytest evaluator/tests/test_manual_test_agent.py -v -s
"""

import pytest

from evaluator.plugin.test_agent.manual_test_agent import ManualTestAgent
from evaluator.core.interfaces.abstract_test_agent import AbstractTestAgent
from evaluator.core.schema import (
    ManualUserInfo,
    TargetAgentReaction,
)


# ============================================================
# 工具函数
# ============================================================


def _make_user_info(**kwargs) -> ManualUserInfo:
    """快速构造 ManualUserInfo"""
    defaults = {"strict_inputs": ["默认输入"]}
    defaults.update(kwargs)
    return ManualUserInfo(**defaults)


def _make_target_reaction(text: str) -> TargetAgentReaction:
    """快速构造 TargetAgentReaction"""
    return TargetAgentReaction(
        type="message",
        message_list=[{"content": text}],
    )


# ============================================================
# 1. strict_inputs 顺序消费
# ============================================================


@pytest.mark.asyncio
async def test_strict_inputs_sequential():
    """strict_inputs 应按顺序逐条发送"""
    agent = ManualTestAgent(
        _make_user_info(strict_inputs=["你好", "我头疼", "已经三天了"])
    )

    r1 = await agent.do_generate(None)
    assert r1.action.semantic_content == "你好"
    assert r1.reason == "scripted input"
    assert r1.is_finished is False

    r2 = await agent.do_generate(_make_target_reaction("您好"))
    assert r2.action.semantic_content == "我头疼"
    assert r2.is_finished is False

    r3 = await agent.do_generate(_make_target_reaction("请问多久了？"))
    assert r3.action.semantic_content == "已经三天了"
    assert r3.is_finished is False


# ============================================================
# 2. inputs 用完自动结束
# ============================================================


@pytest.mark.asyncio
async def test_auto_finish_when_inputs_exhausted():
    """strict_inputs 全部消费后应自动 is_finished=True"""
    agent = ManualTestAgent(
        _make_user_info(strict_inputs=["唯一的问题"])
    )

    r1 = await agent.do_generate(None)
    assert r1.action.semantic_content == "唯一的问题"
    assert r1.is_finished is False

    # strict_inputs 已用完 → 自动结束
    r2 = await agent.do_generate(_make_target_reaction("AI 回复"))
    assert r2.is_finished is True
    assert "strict_inputs consumed" in r2.reason


# ============================================================
# 3. 空 strict_inputs 首轮即结束
# ============================================================


@pytest.mark.asyncio
async def test_empty_strict_inputs():
    """空 strict_inputs 应在首轮直接返回 is_finished=True"""
    agent = ManualTestAgent(
        _make_user_info(strict_inputs=[])
    )

    r1 = await agent.do_generate(None)
    assert r1.is_finished is True
    assert "strict_inputs consumed" in r1.reason


# ============================================================
# 4. max_turns 自动由 strict_inputs 数量决定
# ============================================================


@pytest.mark.asyncio
async def test_max_turns_auto_computed():
    """手动模式 max_turns 应自动等于 len(strict_inputs) + 1"""
    # ManualUserInfo 无 max_turns 字段，ManualTestAgent 自动根据 strict_inputs 计算
    agent = ManualTestAgent(
        _make_user_info(strict_inputs=["a", "b", "c"])
    )
    assert agent.max_turns == 4  # 3 inputs + 1 termination turn

    # 所有 3 条输入都应正常消费
    r1 = await agent.do_generate(None)
    assert r1.action.semantic_content == "a"
    assert r1.is_finished is False

    r2 = await agent.do_generate(_make_target_reaction("r1"))
    assert r2.action.semantic_content == "b"
    assert r2.is_finished is False

    r3 = await agent.do_generate(_make_target_reaction("r2"))
    assert r3.action.semantic_content == "c"
    assert r3.is_finished is False

    # Turn 4: 全部消费完 → 结束
    r4 = await agent.do_generate(_make_target_reaction("r3"))
    assert r4.is_finished is True


# ============================================================
# 5. memory 正确记录与补全
# ============================================================


@pytest.mark.asyncio
async def test_memory_management():
    """验证 memory_list 正确记录且 target_response 被补全"""
    agent = ManualTestAgent(
        _make_user_info(strict_inputs=["你好", "谢谢"])
    )

    r1 = await agent.do_generate(None)
    assert len(agent.memory_list) == 1
    assert agent.memory_list[0].test_reaction == r1
    assert agent.memory_list[0].target_response is None

    target_r1 = _make_target_reaction("您好")
    r2 = await agent.do_generate(target_r1)
    assert len(agent.memory_list) == 2
    # Turn 1 的 target_response 被补全
    assert agent.memory_list[0].target_response == target_r1
    assert agent.memory_list[0].target_response_time is not None
    # Turn 2 的 target_response 尚未补全
    assert agent.memory_list[1].target_response is None


# ============================================================
# 6. 注册表查找
# ============================================================


def test_registry_lookup():
    """应能通过 AbstractTestAgent.get('manual') 获取 ManualTestAgent"""
    import evaluator.plugin.test_agent  # noqa: F401 — 触发注册

    cls = AbstractTestAgent.get("manual")
    assert cls is ManualTestAgent


def test_registry_has():
    """注册表应包含 'manual' 和 'auto'"""
    import evaluator.plugin.test_agent  # noqa: F401

    assert AbstractTestAgent.has("manual")
    assert AbstractTestAgent.has("auto")
