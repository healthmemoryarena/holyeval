"""
LlmApiTargetAgent — 统一调用大模型 API

注册名称: "llm_api"

基于 evaluator/utils/llm.py 的 do_execute 接口，支持所有已接入的模型提供商
（OpenAI、Google Gemini 等），自动维护多轮对话历史，兼容 TestCase.history 字段。

用户级参数（LlmApiTargetInfo）：
    model: 模型名称（必填，如 gpt-5.2 / gemini-3-pro）
    system_prompt: 系统提示词（可选）

基础设施参数（从环境变量读取，由 do_execute / init_chat_model 管理）：
    OPENAI_API_KEY / GOOGLE_API_KEY 等
"""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    SessionInfo,
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.utils.llm import BasicMessage, do_execute

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = "你是一个AI助手，请根据用户的问题提供有用的回答。"

# benchmark/data/ 目录（从 evaluator/plugin/target_agent/ 向上 3 级 → 项目根 → benchmark/data/）
_BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"


def _load_tool_group(tool_group: str, tool_context: dict[str, Any]) -> tuple[list[BaseTool], type | None]:
    """从 benchmark/data/ 下加载工具组模块

    tool_group 格式: "{benchmark}/{tool_name}" 如 "thetagen/retrieve"
    解析为: benchmark/data/{benchmark}/tools/{tool_name}.py

    模块须导出 get_tools(**tool_context) -> list[BaseTool]
    可选导出 ToolContext 类（用于 langgraph context_schema，消除 Pydantic 序列化警告）

    Returns:
        (tools, context_class) — context_class 为 None 时 fallback 到 dict
    """
    import importlib.util

    parts = tool_group.split("/")
    if len(parts) != 2:
        raise ValueError(f"tool_group 格式应为 '{{benchmark}}/{{tool_name}}'，收到: {tool_group!r}")

    benchmark, tool_name = parts
    file_path = _BENCHMARK_DATA_DIR / benchmark / "tools" / f"{tool_name}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"工具组文件不存在: {file_path}")

    spec = importlib.util.spec_from_file_location(f"tool_group.{benchmark}.{tool_name}", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_tools_fn = getattr(module, "get_tools", None)
    if get_tools_fn is None:
        raise AttributeError(f"工具组 {tool_group!r} 未导出 get_tools() 函数")

    # 提取 ToolContext 类（可选，用于 context_schema）
    context_class = getattr(module, "ToolContext", None)

    return get_tools_fn(**tool_context), context_class


class LlmApiTargetInfo(BaseModel):
    """LLM API 被测目标配置 — 通过 do_execute 统一调用大模型"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"_comment": "最小配置，仅需 model", "type": "llm_api", "model": "gpt-5.4"},
                {
                    "_comment": "指定系统提示词",
                    "type": "llm_api",
                    "model": "gemini-3-pro-preview",
                    "system_prompt": "你是一个专业的健康助手。",
                },
                {
                    "_comment": "带工具组",
                    "type": "llm_api",
                    "model": "gpt-5.4",
                    "tool_group": "thetagen/retrieve",
                    "tool_context": {"user_email": "user110@demo"},
                },
            ],
        },
    )
    type: Literal["llm_api"] = Field(description="目标类型")
    model: Literal[
        "gpt-5.4",
        "gpt-5.2",
        "gpt-4.1",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "minimax/minimax-m2.5",
        "z-ai/glm-5",
    ] = Field(description="模型名称")
    system_prompt: Optional[str] = Field(None, description="系统提示词（不填时使用默认提示）")
    tool_group: Optional[str] = Field(
        None,
        description="工具组路径（格式: '{benchmark}/tools/{name}'，如 'thetagen/retrieve'，对应 benchmark/data/ 下的 .py 文件）",
    )
    tool_context: Optional[Dict[str, Any]] = Field(
        None, description="工具组运行时上下文（传递给 get_tools()，如 user_email 等）"
    )
    thinking_level: Optional[str] = Field(
        None, description='思考级别 — OpenAI: "low"/"medium"/"high"; Gemini: budget token 数（如 "8192"）'
    )


class LlmApiTargetAgent(AbstractTargetAgent, name="llm_api", params_model=LlmApiTargetInfo):
    """统一调用大模型 API"""

    _display_meta = {
        "icon": (
            "M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12"
            " 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0"
            " 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z"
        ),
        "color": "#6366f1",
        "features": ["OpenAI", "Gemini", "多模型支持"],
    }
    _cost_meta = {
        "est_input_tokens": 200,  # 预估单次调用输入 token（含 system prompt + 用户消息）
        "est_output_tokens": 600,  # 预估单次调用输出 token
    }

    def __init__(self, target_config: LlmApiTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: LlmApiTargetInfo = target_config
        self.model: str = target_config.model

        # 将 history (List[BaseMessage]) 转换为 BasicMessage 列表，供 do_execute 使用
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # 加载工具组（可选）
        self._tools: list[BaseTool] | None = None
        self._tool_context_typed: Any = None  # 类型化的 tool_context（消除 Pydantic 序列化警告）
        self._tool_context_schema: type | None = None
        if target_config.tool_group:
            self._tools, context_class = _load_tool_group(target_config.tool_group, target_config.tool_context or {})
            # 将 dict 转为工具模块声明的 ToolContext 类型（如有）
            if context_class is not None and target_config.tool_context:
                self._tool_context_typed = context_class(**target_config.tool_context)
                self._tool_context_schema = context_class
            else:
                self._tool_context_typed = target_config.tool_context
                self._tool_context_schema = None
            logger.info(
                "[LlmApiTargetAgent] Loaded tool_group=%r with %d tools",
                target_config.tool_group,
                len(self._tools),
            )

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

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        """将用户输入通过 do_execute 发送给大模型，返回模型响应"""
        if test_action is None:
            # 首轮无用户输入，返回欢迎语
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "你好，有什么可以帮你的？"}],
            )

        # 提取用户输入
        user_text = self._extract_user_input(test_action)
        system_prompt = self.config.system_prompt or _DEFAULT_SYSTEM_PROMPT

        logger.debug(
            "[LlmApiTargetAgent] Calling %s, history=%d messages",
            self.config.model,
            len(self._conversation),
        )

        # 调用 do_execute：history_messages 为之前的对话，input 为当前用户输入
        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=user_text,
            history_messages=self._conversation if self._conversation else None,
            tools=self._tools,
            thinking_level=self.config.thinking_level,
            tool_context=self._tool_context_typed or self.config.tool_context,
            tool_context_schema=self._tool_context_schema,
        )

        assistant_content = result.content

        # 累加成本
        self._accumulate_cost(result.usage)

        # 追加本轮对话到历史
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=assistant_content))

        logger.debug(
            "[LlmApiTargetAgent] Response: %d chars",
            len(assistant_content),
        )

        # 序列化 tool_calls（ToolCallRecord → dict）
        tc_dicts = [tc.model_dump() for tc in result.tool_calls] if result.tool_calls else None

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": assistant_content}],
            tool_calls=tc_dicts,
            usage=result.usage,
        )

    def get_session_info(self) -> SessionInfo:
        """当 system_prompt 注入了用户健康档案时，通知 EvalAgent 可按"有用户数据"规则评测"""
        has_user_data = bool(self.config.system_prompt and len(self.config.system_prompt) > 200)
        return SessionInfo(has_user_data=has_user_data)

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        """从 TestAgentAction 提取用户输入"""
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        else:
            return str(test_action.custom_content) if test_action.custom_content else ""
