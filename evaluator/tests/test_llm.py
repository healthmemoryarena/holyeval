"""
do_execute 集成测试

测试覆盖：
1. 基础功能：普通模式、思考模式、工具调用、历史消息、BasicMessage 输入
2. 纯对话场景：多模型对比（GPT、Gemini、Claude、GLM）
3. 多模态场景：图片识别能力测试

运行：uv run pytest evaluator/tests/test_llm.py -v -s
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = [
    pytest.mark.api,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY 未设置，跳过 LLM 集成测试",
    ),
]

from langchain.tools import tool

from evaluator.utils.llm import BasicMessage, ExecuteResult, do_execute

SYSTEM_PROMPT = "你是一个简洁的助手，用一句话回答问题。"

# 测试图片：Wikipedia 上的一张猫咪照片（广泛缓存，可靠性高）
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"


# ============================================================
# 通用断言
# ============================================================


def _assert_valid_result(result: ExecuteResult, *, allow_data: bool = False) -> None:
    """验证返回结构完整"""
    assert isinstance(result, ExecuteResult)
    assert result.content, "content 不应为空"
    assert len(result.usage) > 0, "usage 不应为空"
    for model_name, meta in result.usage.items():
        assert meta["input_tokens"] > 0
        assert meta["output_tokens"] > 0
        assert meta["total_tokens"] > 0
    assert result.time_cost > 0
    if not allow_data:
        assert result.data is None


# ============================================================
# 1. 基础功能测试
# ============================================================


@pytest.mark.asyncio
async def test_normal_mode():
    """普通模式（thinking_level=None）"""
    result = await do_execute(model="gpt-5.2", system_prompt=SYSTEM_PROMPT, input="1+1等于几？")
    _assert_valid_result(result)
    print(f"\n[normal] content={result.content!r}, time={result.time_cost}ms")


@pytest.mark.asyncio
async def test_thinking_medium():
    """reasoning — medium"""
    result = await do_execute(
        model="gpt-5.2", system_prompt=SYSTEM_PROMPT, input="1+1等于几？", thinking_level="medium"
    )
    _assert_valid_result(result)
    print(f"\n[medium] content={result.content!r}, time={result.time_cost}ms")


@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘并返回结果"""
    return a * b


@pytest.mark.asyncio
async def test_tool_calling():
    """验证 agent 能正确调用工具并返回结果"""
    result = await do_execute(
        model="gpt-5.2",
        system_prompt="你是一个计算助手，必须使用工具完成计算。",
        input="请帮我算 17 乘以 23 等于多少？",
        tools=[multiply],
    )
    _assert_valid_result(result)
    # 17 * 23 = 391，回答中应包含正确结果
    assert "391" in result.content, f"回答中应包含 391，实际: {result.content!r}"
    print(f"\n[tool] content={result.content!r}, time={result.time_cost}ms")


@pytest.mark.asyncio
async def test_history_messages():
    """验证模型能理解历史消息上下文"""
    history = [
        BasicMessage(role="user", content="我叫小明。"),
        BasicMessage(role="assistant", content="你好小明！有什么可以帮你的？"),
    ]
    result = await do_execute(
        model="gpt-5.2",
        system_prompt=SYSTEM_PROMPT,
        input="我叫什么名字？",
        history_messages=history,
    )
    _assert_valid_result(result)
    assert "小明" in result.content, f"回答中应包含小明，实际: {result.content!r}"
    print(f"\n[history] content={result.content!r}, time={result.time_cost}ms")


@pytest.mark.asyncio
async def test_input_as_basic_message():
    """验证 input 传 BasicMessage 结构体时正常工作"""
    msg = BasicMessage(role="user", content="中国的首都是哪里？")
    result = await do_execute(
        model="gpt-5.2",
        system_prompt=SYSTEM_PROMPT,
        input=msg,
    )
    _assert_valid_result(result)
    assert "北京" in result.content, f"回答中应包含北京，实际: {result.content!r}"
    print(f"\n[basic_message] content={result.content!r}, time={result.time_cost}ms")


# ============================================================
# 2. 纯对话场景 — 多模型对比（问"你是什么模型？"）
# ============================================================


@pytest.mark.asyncio
async def test_chat_gpt52_normal():
    """纯对话 — GPT-5.2 普通模式"""
    result = await do_execute(
        model="gpt-5.2",
        system_prompt=SYSTEM_PROMPT,
        input="你是什么模型？",
    )
    _assert_valid_result(result)
    print(f"\n[gpt-5.2-normal] {result.content!r} | {result.time_cost}ms")


@pytest.mark.asyncio
async def test_chat_gpt52_medium():
    """纯对话 — GPT-5.2 思考模式"""
    result = await do_execute(
        model="gpt-5.2",
        system_prompt=SYSTEM_PROMPT,
        input="你是什么模型？",
        thinking_level="medium",
    )
    _assert_valid_result(result)
    print(f"\n[gpt-5.2-medium] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY 未设置，跳过 Gemini 测试",
)
@pytest.mark.asyncio
async def test_chat_gemini_flash():
    """纯对话 — Gemini 3 Flash"""
    result = await do_execute(
        model="gemini-3-flash",
        system_prompt=SYSTEM_PROMPT,
        input="你是什么模型？",
    )
    _assert_valid_result(result)
    print(f"\n[gemini-3-flash] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY 未设置，跳过 Gemini 测试",
)
@pytest.mark.asyncio
async def test_chat_gemini_pro():
    """纯对话 — Gemini 3 Pro"""
    result = await do_execute(
        model="gemini-3-pro",
        system_prompt=SYSTEM_PROMPT,
        input="你是什么模型？",
    )
    _assert_valid_result(result)
    print(f"\n[gemini-3-pro] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY 未设置，跳过 OpenRouter 测试",
)
@pytest.mark.asyncio
async def test_chat_claude_sonnet45():
    """纯对话 — Claude Sonnet 4.5（OpenRouter）"""
    result = await do_execute(
        model="anthropic/claude-sonnet-4.5",
        system_prompt=SYSTEM_PROMPT,
        input="你是什么模型？",
    )
    _assert_valid_result(result)
    print(f"\n[claude-sonnet-4.5] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY 未设置，跳过 OpenRouter 测试",
)
@pytest.mark.asyncio
async def test_chat_glm5():
    """纯对话 — 智谱 GLM-5（OpenRouter）"""
    result = await do_execute(
        model="z-ai/glm-5",
        system_prompt=SYSTEM_PROMPT,
        input="你是什么模型？",
    )
    _assert_valid_result(result)
    print(f"\n[glm-5] {result.content!r} | {result.time_cost}ms")


# ============================================================
# 3. 多模态场景 — 图片识别（问"这张图片里是什么？"）
# ============================================================


@pytest.mark.asyncio
async def test_vision_gpt52_normal():
    """多模态 — GPT-5.2 普通模式"""
    msg = BasicMessage(
        role="user",
        content=[
            {"type": "text", "text": "这张图片里是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    )
    result = await do_execute(
        model="gpt-5.2",
        system_prompt="你是一个图片描述助手。",
        input=msg,
    )
    _assert_valid_result(result)
    print(f"\n[vision-gpt-5.2-normal] {result.content!r} | {result.time_cost}ms")


@pytest.mark.asyncio
async def test_vision_gpt52_medium():
    """多模态 — GPT-5.2 思考模式"""
    msg = BasicMessage(
        role="user",
        content=[
            {"type": "text", "text": "这张图片里是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    )
    result = await do_execute(
        model="gpt-5.2",
        system_prompt="你是一个图片描述助手。",
        input=msg,
        thinking_level="medium",
    )
    _assert_valid_result(result)
    print(f"\n[vision-gpt-5.2-medium] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY 未设置，跳过 Gemini 测试",
)
@pytest.mark.asyncio
async def test_vision_gemini_flash():
    """多模态 — Gemini 3 Flash"""
    msg = BasicMessage(
        role="user",
        content=[
            {"type": "text", "text": "这张图片里是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    )
    result = await do_execute(
        model="gemini-3-flash",
        system_prompt="你是一个图片描述助手。",
        input=msg,
    )
    _assert_valid_result(result)
    print(f"\n[vision-gemini-3-flash] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY 未设置，跳过 Gemini 测试",
)
@pytest.mark.asyncio
async def test_vision_gemini_pro():
    """多模态 — Gemini 3 Pro"""
    msg = BasicMessage(
        role="user",
        content=[
            {"type": "text", "text": "这张图片里是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    )
    result = await do_execute(
        model="gemini-3-pro",
        system_prompt="你是一个图片描述助手。",
        input=msg,
    )
    _assert_valid_result(result)
    print(f"\n[vision-gemini-3-pro] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY 未设置，跳过 OpenRouter 测试",
)
@pytest.mark.asyncio
async def test_vision_claude_sonnet45():
    """多模态 — Claude Sonnet 4.5（OpenRouter）"""
    msg = BasicMessage(
        role="user",
        content=[
            {"type": "text", "text": "这张图片里是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    )
    result = await do_execute(
        model="anthropic/claude-sonnet-4.5",
        system_prompt="你是一个图片描述助手。",
        input=msg,
    )
    _assert_valid_result(result)
    print(f"\n[vision-claude-sonnet-4.5] {result.content!r} | {result.time_cost}ms")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY 未设置，跳过 OpenRouter 测试",
)
@pytest.mark.asyncio
async def test_vision_glm5():
    """多模态 — 智谱 GLM-5（OpenRouter）"""
    msg = BasicMessage(
        role="user",
        content=[
            {"type": "text", "text": "这张图片里是什么？用一句话回答。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ],
    )
    result = await do_execute(
        model="z-ai/glm-5",
        system_prompt="你是一个图片描述助手。",
        input=msg,
    )
    _assert_valid_result(result)
    print(f"\n[vision-glm-5] {result.content!r} | {result.time_cost}ms")
