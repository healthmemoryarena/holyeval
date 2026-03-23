"""
AutoTestAgent 集成测试 + 缺陷检测

测试覆盖：
1. strict_inputs 强制输入（不调 LLM，纯确定性）
2. max_turns 超限自动终止
3. max_turns=0 边界情况
4. strict_inputs 数量 > max_turns 时被截断
5. memory 自动管理（target_response 补全）
6. max_turns 超限时不写入额外 memory
7. 多轮对话 history 正确性（无重复）
8. LLM 开场白（首轮无 target_reaction）
9. LLM 多轮对话 + 目标达成终止

运行：uv run pytest evaluator/tests/test_auto_test_agent.py -v -s
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()

from evaluator.plugin.test_agent.auto_test_agent import AutoTestAgent
from evaluator.core.schema import (
    AutoUserInfo,
    TargetAgentReaction,
    TestAgentAction,
    TestAgentReaction,
)
from evaluator.utils.llm import BasicMessage


# ============================================================
# 工具函数
# ============================================================


def _make_user_info(**kwargs) -> AutoUserInfo:
    """快速构造 AutoUserInfo"""
    defaults = {
        "goal": "测试目标",
    }
    defaults.update(kwargs)
    return AutoUserInfo(**defaults)


def _make_target_reaction(text: str) -> TargetAgentReaction:
    """快速构造文本类型的 TargetAgentReaction"""
    return TargetAgentReaction(
        type="message",
        message_list=[{"content": text}],
    )


def _make_mock_do_execute(captured_histories: list):
    """构造一个 mock do_execute，会把每次调用的 history_messages 记录下来"""

    async def _mock(**kwargs):
        captured_histories.append(kwargs.get("history_messages"))
        result = MagicMock()
        result.data = MagicMock()
        result.data.content = "模拟回复"
        result.data.reason = "mock"
        result.data.is_finished = False
        result.data.next_fuzzy_action = None
        result.content = "模拟回复"
        result.usage = {}
        return result

    return _mock


# ============================================================
# 1. strict_inputs — 前 N 轮发送预设内容，不调 LLM
# ============================================================


@pytest.mark.asyncio
async def test_strict_inputs_basic():
    """strict_inputs 前 N 轮应直接发送预设内容，不调用 LLM"""
    agent = AutoTestAgent(
        _make_user_info(strict_inputs=["你好", "我头疼"], max_turns=5)
    )

    # Turn 1: 应发送 strict_inputs[0]
    r1 = await agent.do_generate(None)
    assert r1.action.semantic_content == "你好"
    assert r1.reason == "强制输入"
    assert r1.is_finished is False

    # Turn 2: 应发送 strict_inputs[1]
    target_r1 = _make_target_reaction("您好，请问有什么可以帮您？")
    r2 = await agent.do_generate(target_r1)
    assert r2.action.semantic_content == "我头疼"
    assert r2.reason == "强制输入"
    assert r2.is_finished is False

    print(f"\n[strict_inputs] turn1={r1.action.semantic_content!r}, turn2={r2.action.semantic_content!r}")


# ============================================================
# 2. max_turns 超限自动终止
# ============================================================


@pytest.mark.asyncio
async def test_max_turns_terminates():
    """超过 max_turns 应自动返回 is_finished=True"""
    agent = AutoTestAgent(
        _make_user_info(strict_inputs=["a", "b"], max_turns=2)
    )

    # Turn 1 & 2: 正常
    r1 = await agent.do_generate(None)
    assert r1.is_finished is False
    r2 = await agent.do_generate(_make_target_reaction("reply1"))
    assert r2.is_finished is False

    # Turn 3: 超过 max_turns=2，应自动终止
    r3 = await agent.do_generate(_make_target_reaction("reply2"))
    assert r3.is_finished is True
    assert "最大轮次" in r3.reason

    print(f"\n[max_turns] terminated at turn 3, reason={r3.reason}")


# ============================================================
# 3. max_turns=0 边界情况
# ============================================================


@pytest.mark.asyncio
async def test_max_turns_zero():
    """max_turns=0 应立即终止，首轮即返回 is_finished=True"""
    agent = AutoTestAgent(
        _make_user_info(max_turns=0, strict_inputs=["hello"])
    )

    assert agent.max_turns == 0, f"max_turns 应为 0，实际为 {agent.max_turns}"

    r1 = await agent.do_generate(None)
    assert r1.is_finished is True
    assert "最大轮次" in r1.reason

    print(f"\n[max_turns_zero] max_turns=0, 首轮即终止, reason={r1.reason}")


# ============================================================
# 4. strict_inputs 数量 > max_turns 时，多余的被截断
# ============================================================


@pytest.mark.asyncio
async def test_strict_inputs_exceeds_max_turns():
    """strict_inputs 数量 > max_turns 时，多余的 strict_inputs 永远不会被消费"""
    agent = AutoTestAgent(
        _make_user_info(strict_inputs=["a", "b", "c"], max_turns=1)
    )

    # Turn 1: 发送 strict_inputs[0]
    r1 = await agent.do_generate(None)
    assert r1.action.semantic_content == "a"

    # Turn 2: max_turns=1，应终止，strict_inputs[1] 和 [2] 被丢弃
    r2 = await agent.do_generate(_make_target_reaction("reply"))
    assert r2.is_finished is True

    print(f"\n[strict_exceeds_max] consumed 1/{len(agent.user_info.strict_inputs)} strict_inputs")


# ============================================================
# 5. memory 自动管理
# ============================================================


@pytest.mark.asyncio
async def test_memory_management():
    """验证 do_generate 正确管理 memory_list：记录新轮 + 补全上轮 target_response"""
    agent = AutoTestAgent(
        _make_user_info(strict_inputs=["你好", "我不舒服"], max_turns=5)
    )

    # Turn 1
    r1 = await agent.do_generate(None)
    assert len(agent.memory_list) == 1
    assert agent.memory_list[0].test_reaction == r1
    assert agent.memory_list[0].target_response is None  # 首轮无 target

    # Turn 2
    target_r1 = _make_target_reaction("有什么可以帮您？")
    r2 = await agent.do_generate(target_r1)
    assert len(agent.memory_list) == 2
    # Turn 1 的 memory 应被补全 target_response
    assert agent.memory_list[0].target_response == target_r1
    assert agent.memory_list[0].target_response_time is not None
    # Turn 2 的 memory
    assert agent.memory_list[1].test_reaction == r2
    assert agent.memory_list[1].target_response is None  # 还没补全

    print(f"\n[memory] {len(agent.memory_list)} memories, all correctly managed")


# ============================================================
# 6. max_turns 超限时不应写入 memory
# ============================================================


@pytest.mark.asyncio
async def test_max_turns_no_extra_memory():
    """max_turns 超限返回终止反应时，不应写入新的 memory 条目"""
    agent = AutoTestAgent(
        _make_user_info(strict_inputs=["a"], max_turns=1)
    )

    r1 = await agent.do_generate(None)
    assert len(agent.memory_list) == 1

    # Turn 2: 超限终止，memory 不应新增
    r2 = await agent.do_generate(_make_target_reaction("reply"))
    assert r2.is_finished is True
    assert len(agent.memory_list) == 1  # 不应新增

    # 但 Turn 1 的 target_response 应被补全
    assert agent.memory_list[0].target_response is not None

    print(f"\n[max_turns_memory] memory_list length stayed at {len(agent.memory_list)}")


# ============================================================
# 7. 多轮对话 history 正确性验证
# ============================================================


@pytest.mark.asyncio
async def test_history_no_duplication():
    """验证多轮对话时 LLM 收到的 history_messages 无重复"""
    captured_histories: list = []

    agent = AutoTestAgent(
        _make_user_info(
            goal="了解头疼原因",
            strict_inputs=["我头疼"],  # Turn 1 用 strict_input，不调 LLM
            max_turns=5,
        )
    )

    with patch(
        "evaluator.plugin.test_agent.auto_test_agent.do_execute",
        side_effect=_make_mock_do_execute(captured_histories),
    ):
        # Turn 1: strict_input，不调 LLM
        r1 = await agent.do_generate(None)
        assert r1.action.semantic_content == "我头疼"
        assert len(captured_histories) == 0  # strict_input 路径不调 LLM

        # Turn 2: strict_inputs 用完，走 LLM 路径
        target_r1 = _make_target_reaction("请问头疼多久了？")
        r2 = await agent.do_generate(target_r1)
        assert len(captured_histories) == 1

        history = captured_histories[0]
        assert history is not None

        user_msgs = [m for m in history if m.role == "user"]
        assistant_msgs = [m for m in history if m.role == "assistant"]

        print(f"\n[history_check] Turn 2 — LLM 收到的 history:")
        for i, msg in enumerate(history):
            print(f"  [{i}] {msg.role}: {msg.content!r}")

        assert len(user_msgs) == 1, f"预期 1 条 user 消息，实际 {len(user_msgs)}"
        assert len(assistant_msgs) == 1, f"预期 1 条 assistant 消息，实际 {len(assistant_msgs)}"


@pytest.mark.asyncio
async def test_history_correct_over_multiple_turns():
    """验证多轮对话 history 随轮次正确增长，无累积重复"""
    captured_histories: list = []

    agent = AutoTestAgent(
        _make_user_info(
            goal="了解头疼原因",
            strict_inputs=["我头疼"],  # Turn 1 用 strict_input
            max_turns=5,
        )
    )

    with patch(
        "evaluator.plugin.test_agent.auto_test_agent.do_execute",
        side_effect=_make_mock_do_execute(captured_histories),
    ):
        # Turn 1: strict_input
        await agent.do_generate(None)

        # Turn 2: LLM (第一次 mock 调用)
        await agent.do_generate(_make_target_reaction("回复1"))

        # Turn 3: LLM (第二次 mock 调用)
        await agent.do_generate(_make_target_reaction("回复2"))

        # Turn 4: LLM (第三次 mock 调用)
        await agent.do_generate(_make_target_reaction("回复3"))

    assert len(captured_histories) == 3

    print(f"\n[history_accumulation] 各轮 LLM 收到的 history 长度:")

    for turn_idx, history in enumerate(captured_histories, start=2):
        if history is None:
            continue
        user_count = sum(1 for m in history if m.role == "user")
        assistant_count = sum(1 for m in history if m.role == "assistant")
        expected_user = turn_idx - 1
        expected_assistant = turn_idx - 1
        print(
            f"  Turn {turn_idx}: user={user_count} assistant={assistant_count} "
            f"(预期 user={expected_user} assistant={expected_assistant})"
        )
        assert user_count == expected_user, f"Turn {turn_idx}: 预期 {expected_user} 条 user，实际 {user_count}"
        assert assistant_count == expected_assistant, f"Turn {turn_idx}: 预期 {expected_assistant} 条 assistant，实际 {assistant_count}"


# ============================================================
# 8. LLM 集成 — 首轮开场白
# ============================================================


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY 未设置")
async def test_llm_opening_message():
    """首轮无 target_reaction，LLM 应生成有意义的开场白"""
    agent = AutoTestAgent(
        _make_user_info(goal="咨询最近头疼的问题", context="30岁男性，程序员")
    )

    r = await agent.do_generate(None)
    assert r.action.type == "semantic"
    assert r.action.semantic_content, "开场白不应为空"
    assert isinstance(r.is_finished, bool)

    print(f"\n[llm_opening] content={r.action.semantic_content!r}")
    print(f"  reason={r.reason}, is_finished={r.is_finished}")


# ============================================================
# 9. LLM 集成 — 多轮对话 + 目标达成终止
# ============================================================


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY 未设置")
async def test_llm_multi_turn_goal_reached():
    """多轮对话，验证 LLM 能最终将 is_finished 设为 True"""
    agent = AutoTestAgent(
        _make_user_info(
            goal="询问明天北京的天气",
            max_turns=5,
            finish_condition="当得到具体的天气预报信息后结束对话",
        )
    )

    target_responses = [
        None,  # Turn 1: 开场
        _make_target_reaction("您好！明天北京晴转多云，最高25°C，最低15°C，建议带件薄外套。"),
    ]

    finished = False
    for i, target_r in enumerate(target_responses):
        r = await agent.do_generate(target_r)
        print(f"\n[multi_turn] Turn {i + 1}: {r.action.semantic_content[:80]!r}")
        print(f"  reason={r.reason}, is_finished={r.is_finished}")
        if r.is_finished:
            finished = True
            print(f"  → 对话在 Turn {i + 1} 正常结束")
            break

    if not finished:
        print(f"  → 对话未在预设轮次内结束，共 {len(agent.memory_list)} 轮")

    assert len(agent.memory_list) >= 1
