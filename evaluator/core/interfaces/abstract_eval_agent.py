"""
AbstractEvalAgent — 评估代理抽象基类

职责：
- 根据对话历史评估测试效果
- 返回评估结果（pass/fail、分数、反馈）

特点：
- 有状态实体（携带评估配置 + 测试上下文）
- 提供统一的 run() 接口
- 子类只需实现 run() 即可

注册机制（__init_subclass__）：
    class SemanticEvalAgent(AbstractEvalAgent, name="semantic"):
        ...
    查询: AbstractEvalAgent.get("semantic")

生命周期：
- 创建：通过 EvalInfo 初始化，同时注入 history/user_info/case_id
- 运行：对话循环结束后，调用 run(memory_list, session_info) 执行评估
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

from evaluator.core.schema import EvalResult, SessionInfo, TestAgentMemory

if TYPE_CHECKING:
    from evaluator.core.schema import EvalInfo, UserInfo
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class AbstractEvalAgent(ABC):
    """评估代理抽象基类"""

    _registry: ClassVar[Dict[str, Type["AbstractEvalAgent"]]] = {}
    _params_registry: ClassVar[Dict[str, type]] = {}  # name → EvalInfo config model

    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        params_model: type | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            if name in AbstractEvalAgent._registry:
                logger.warning(
                    "EvalAgent %r 重复注册，已覆盖: %s -> %s",
                    name,
                    AbstractEvalAgent._registry[name].__name__,
                    cls.__name__,
                )
            AbstractEvalAgent._registry[name] = cls
            if params_model is not None:
                AbstractEvalAgent._params_registry[name] = params_model

    @classmethod
    def get(cls, name: str) -> Type["AbstractEvalAgent"]:
        """根据名称获取已注册的 EvalAgent 类"""
        if name not in cls._registry:
            registered = list(cls._registry.keys())
            raise ValueError(f"未注册的 EvalAgent 类型: {name!r}，已注册: {registered or '(空)'}")
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> Dict[str, Type["AbstractEvalAgent"]]:
        """获取所有已注册的 EvalAgent 类"""
        return dict(cls._registry)

    @classmethod
    def has(cls, name: str) -> bool:
        """检查指定名称的 EvalAgent 是否已注册"""
        return name in cls._registry

    def __init__(
        self,
        eval_config: EvalInfo,
        *,
        history: List[BaseMessage] | None = None,
        user_info: UserInfo | None = None,
        case_id: str = "",
    ):
        self.eval_config = eval_config
        self.history: List[BaseMessage] = history or []
        self.user_info = user_info
        self.case_id = case_id

    @abstractmethod
    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """执行评估（子类实现）

        Args:
            memory_list: 测试代理记忆列表（包含完整对话：用户行为 + 目标系统响应）
            session_info: 目标系统会话信息（认证数据等，可选）

        Returns:
            EvalResult: 评估结果
        """
        ...
