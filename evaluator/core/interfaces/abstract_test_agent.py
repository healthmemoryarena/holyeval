"""
AbstractTestAgent — 测试代理（虚拟用户）抽象基类

职责：
- 模拟真实用户的行为
- 根据 UserInfo 的 goal 生成用户反应
- 自动管理测试代理侧的对话记忆
- 判断目标是否达成（终止条件）

注册机制（__init_subclass__）：
    class AutoTestAgent(AbstractTestAgent, name="auto"):       # LLM 驱动
    class ManualTestAgent(AbstractTestAgent, name="manual"):   # 脚本驱动
    查询: AbstractTestAgent.get("auto") / AbstractTestAgent.get("manual")

生命周期：
- 创建：通过 UserInfo 初始化
- 运行：在对话循环中，do_generate 被反复调用
- 终止：当 goal 达成或超过最大轮次
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, Type

from evaluator.core.schema import (
    TargetAgentReaction,
    TestAgentAction,
    TestAgentMemory,
    TestAgentReaction,
    UserInfo,
)
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class AbstractTestAgent(ABC):
    """测试代理（虚拟用户）抽象基类"""

    _registry: ClassVar[Dict[str, Type["AbstractTestAgent"]]] = {}

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            if name in AbstractTestAgent._registry:
                logger.warning(
                    "TestAgent %r 重复注册，已覆盖: %s -> %s",
                    name,
                    AbstractTestAgent._registry[name].__name__,
                    cls.__name__,
                )
            AbstractTestAgent._registry[name] = cls

    @classmethod
    def get(cls, name: str) -> Type["AbstractTestAgent"]:
        """根据名称获取已注册的 TestAgent 类"""
        if name not in cls._registry:
            registered = list(cls._registry.keys())
            raise ValueError(f"未注册的 TestAgent 类型: {name!r}，已注册: {registered or '(空)'}")
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> Dict[str, Type["AbstractTestAgent"]]:
        """获取所有已注册的 TestAgent 类"""
        return dict(cls._registry)

    @classmethod
    def has(cls, name: str) -> bool:
        """检查指定名称的 TestAgent 是否已注册"""
        return name in cls._registry

    def __init__(self, user_info: UserInfo, history: list[BaseMessage] | None = None):
        self.user_info = user_info
        self.history: list[BaseMessage] = history or []
        self.memory_list: List[TestAgentMemory] = []
        self.current_turn = 0
        # max_turns 仅 AutoUserInfo 拥有，ManualUserInfo 由子类自行覆盖
        _max = getattr(user_info, "max_turns", None)
        self.max_turns = _max if _max is not None else 5

    @abstractmethod
    async def _generate_next_reaction(
        self, target_reaction: Optional[TargetAgentReaction]
    ) -> TestAgentReaction:
        """生成下一步测试代理反应（子类实现）

        实现要点：
        1. 根据 user_info.goal 判断是否达成目标
        2. 根据 user_info.context 决定行为风格和知识背景
        3. 优先消费 user_info.strict_inputs 中的强制输入
        4. 返回 TestAgentReaction（包含 action 和 is_finished）

        Args:
            target_reaction: 被测系统的上一次响应，首次调用时为 None

        Returns:
            TestAgentReaction
        """
        ...

    async def do_generate(
        self, target_reaction: Optional[TargetAgentReaction]
    ) -> TestAgentReaction:
        """对外暴露的生成方法（框架实现，子类不需要重写）

        职责：
        1. 自动管理 memory（补全上一轮的 target_response）
        2. 检查轮次限制
        3. 调用 _generate_next_reaction
        4. 记录新一轮 memory
        """
        self.current_turn += 1

        # 补全上一轮 memory 的 target_response
        if self.memory_list and target_reaction is not None:
            last_memory = self.memory_list[-1]
            last_memory.target_response = target_reaction
            last_memory.target_response_time = datetime.now()

        # 检查轮次限制
        if self.current_turn > self.max_turns:
            return TestAgentReaction(
                action=TestAgentAction(type="semantic", semantic_content=""),
                reason=f"达到最大轮次限制 {self.max_turns}",
                is_finished=True,
            )

        # 生成测试代理反应
        test_reaction = await self._generate_next_reaction(target_reaction)

        # 记录新 memory
        self.memory_list.append(
            TestAgentMemory(
                test_reaction=test_reaction,
                test_reaction_time=datetime.now(),
            )
        )

        return test_reaction
