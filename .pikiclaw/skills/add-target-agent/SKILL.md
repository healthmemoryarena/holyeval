---
name: add-target-agent
description: Scaffold a new TargetAgent plugin (config + implementation + registration).
argument-hint: [agent-name]
---

# Add TargetAgent Plugin

Create a new TargetAgent plugin with all required boilerplate: Pydantic config, implementation class, and plugin registration.

## Framework Protection（必读）

本 skill 仅通过**插件扩展点**添加功能，**严禁修改框架核心逻辑**。文件按修改权限分为三级：

### 🟢 Plugin 层 — 自由新建

| 文件 | 操作 |
|------|------|
| `evaluator/plugin/target_agent/<name>_target_agent.py` | 创建新文件（plugin 实现） |
| `evaluator/plugin/target_agent/__init__.py` | **仅追加**: 添加 import + `__all__` 条目 + docstring 映射。不得删除/修改已有内容 |

### 🟡 Schema 扩展点 — 仅限追加

`evaluator/core/schema.py` 是 Pydantic Discriminated Union 的类型注册文件。由于 Pydantic v2 要求 Union 成员在定义时静态列举，新增配置类型必须在此文件中追加。

**允许的操作（纯追加，不改已有代码）**：
1. 在 `TargetInfo` Union 定义**之前**添加新的 `*TargetInfo` 配置类
2. 在 `TargetInfo = Annotated[..., Discriminator("type")]` 中追加新类型
3. 更新文件顶部 docstring 的 `TargetInfo` 列表

**禁止的操作**：
- 修改任何已有的类定义（字段、默认值、validator 等）
- 修改 Union 的构建逻辑或 Discriminator 配置
- 修改其他 section 的任何代码（UserInfo, EvalInfo, TestCase, TestResult 等）

### 🔴 框架核心 — 严禁修改

以下文件为框架核心，任何修改都可能破坏全局功能：

- `evaluator/core/orchestrator.py` — 编排引擎
- `evaluator/core/bench_schema.py` — Benchmark 数据模型（merge_target / bench_item_to_test_case 等）
- `evaluator/core/interfaces/abstract_*.py` — 抽象基类
- `evaluator/utils/*.py` — 通用工具层（llm, benchmark_reader, report_reader, agent_inspector, config）
- `benchmark/basic_runner.py` — 跑分执行器
- `web/` — Web UI（通过 agent_inspector 自动适配新 plugin）

> **如果你发现需要修改 🔴 文件才能完成需求，请停下来通知用户** — 这通常意味着需求理解有误，或框架需要由维护者升级扩展点。

## Auto-Adaptation

完成以下步骤后，Web UI / CLI 会自动适配新 plugin：
- **Config Schema**: `agent_inspector` 从 `TargetInfo` Discriminated Union 自动派生 config map，无需手动维护
- **merge_target()**: 使用 `TypeAdapter(TargetInfo)` 自动路由，新 target 类型的 per-case overrides 开箱即用
- **展示元数据**: 新 plugin 默认使用通用图标/颜色，可通过 `_display_meta` 类属性自定义（可选）

## Workflow

### Step 1: Gather Requirements

Ask the user (via AskUserQuestion) for the following if not provided in $ARGUMENTS:

1. **Plugin name** (snake_case, e.g. `openai_api`) — used as:
   - `__init_subclass__` registration name
   - `type: Literal["<name>"]` discriminator value
   - File name: `evaluator/plugin/target_agent/<name>_target_agent.py`
   - Class name: `<PascalCase>TargetAgent`

2. **Target system description** — what system this agent connects to and how (HTTP API, SDK, CLI, etc.)

3. **Config fields** — what parameters the user needs to provide in test case JSON (beyond the standard `type` field)

4. **Connection lifecycle** — whether the agent needs session management / cleanup

### Step 2: Validate Constraints

Before generating code, verify:

- [ ] Name is unique — must NOT conflict with existing registered names. Check dynamically by reading `evaluator/plugin/target_agent/__init__.py`.
- [ ] Name is valid snake_case identifier (lowercase, underscores only, no leading digits)
- [ ] Config field names don't shadow Pydantic reserved names

### Step 3: Modify `evaluator/core/schema.py`

Add a new Pydantic config model. Follow these rules:

- Place it in the "被测目标配置" section, **before** the `TargetInfo` definition
- Use `model_config = ConfigDict(extra="forbid")`
- Include `type: Literal["<name>"]` as discriminator field
- Add `json_schema_extra.examples` with at least one minimal and one full example
- Add docstring in Chinese explaining the target system

Then update the `TargetInfo` union (append new type):

```python
TargetInfo = Annotated[
    ThetaApiTargetInfo | LlmApiTargetInfo | <NewTargetInfo>,
    Discriminator("type"),
]
```

Also update the module-level docstring's TargetInfo section to include the new target type.

> **Important**: 将新类型加入 `TargetInfo` Union 后：
> - `agent_inspector` 会自动发现其 config schema 并在 Web UI 展示
> - `merge_target()` 会自动支持新类型的 per-case overrides 合并
> - 无需修改 `agent_inspector.py` 或 `bench_schema.py`

### Step 4: Create Implementation File

Create `evaluator/plugin/target_agent/<name>_target_agent.py` following this template:

```python
"""
<PascalCase>TargetAgent — <中文描述>

注册名称: "<name>"

<详细说明连接方式和交互协议>
"""

import logging
from typing import Optional

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    TargetAgentReaction,
    TestAgentAction,
    <NewTargetInfo>,
)

logger = logging.getLogger(__name__)


class <PascalCase>TargetAgent(AbstractTargetAgent, name="<name>"):
    """<中文一句话描述>"""

    # 可选：自定义 Web UI 展示元数据（不声明则使用默认图标/颜色）
    # _display_meta = {"color": "#6366f1", "features": ["HTTP API", "自定义协议"]}

    # 费用预估元数据 — 声明单次调用 token 量，实际费用 = tokens × 用户选择模型定价
    _cost_meta = {
        "est_input_tokens": 200,   # 预估单次调用输入 token
        "est_output_tokens": 600,  # 预估单次调用输出 token
    }

    def __init__(self, target_config: <NewTargetInfo>):
        super().__init__(target_config)
        self.config: <NewTargetInfo> = target_config
        # Initialize connection state here

    async def _generate_next_reaction(
        self, test_action: Optional[TestAgentAction]
    ) -> TargetAgentReaction:
        """将用户输入发送到被测系统，返回系统响应

        Args:
            test_action: 用户动作（首轮可能为 None）

        Returns:
            TargetAgentReaction: 被测系统响应
        """
        # 1. Extract user message from test_action
        user_text = ""
        if test_action and test_action.semantic_content:
            user_text = test_action.semantic_content

        # 2. Call target system
        # response = await self._call_system(user_text)

        # 3. Build and return reaction
        return TargetAgentReaction(
            type="message",
            message_list=[{"content": "TODO: implement"}],
        )

    async def cleanup(self):
        """释放连接资源（对话结束后由 orchestrator 调用）"""
        # Close HTTP sessions, SDK clients, etc.
        pass
```

Key patterns to follow (from `theta_api_target_agent.py`):
- **Lazy initialization**: First call can trigger auth/setup
- **Response format**: Return `TargetAgentReaction(type="message", message_list=[{"content": "..."}])` for text responses
- **Cleanup**: Implement `async cleanup()` if holding network connections or sessions
- **Config from env**: Use `.env` for infrastructure params (base_url, timeout), keep only business params in config model
- **Error handling**: Log errors clearly, raise exceptions that orchestrator can catch

### Step 5: Register Plugin

Edit `evaluator/plugin/target_agent/__init__.py`:
- Add import for the new class
- Add class name to `__all__`
- Update the module docstring to include the new registration mapping

### Step 6: Verify

Run the following to verify:
```bash
# 插件注册检查
uv run python -c "import evaluator.plugin.target_agent; from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; print(AbstractTargetAgent.get_all())"

# Web UI schema 自动发现检查
uv run python -c "from evaluator.utils.agent_inspector import list_target_agents; print([(a.name, list(a.config_schema.get('properties', {}).keys())) for a in list_target_agents()])"

# Lint
ruff check evaluator/core/schema.py evaluator/plugin/target_agent/
ruff format evaluator/core/schema.py evaluator/plugin/target_agent/
```

### Step 7: Update Documentation

新增 TargetAgent 后，更新以下文档中的被测系统列表/表格，保持信息同步：

| 文件 | 更新内容 |
|------|---------|
| `web/guides/develop-target-agent.md` | 「现有被测系统」表格追加新行 + 「关键文件」表格追加参考实现 |
| `web/guides/overview.md` | 「对接服务系统」章节补充说明（如有新的接入模式） |
| `CLAUDE.md` | Plugin System 表格中 TargetAgent 行追加新名称 |
| `README.md` | 「已注册插件」表格 TargetAgent 区域追加新行 |

> 每个文件只需追加一行到已有表格，不要修改其他内容。

## Optional: Custom Display & Cost Metadata

新 plugin 默认使用灰色图标和空特性标签。若需自定义 Web UI 展示，在 plugin 类上声明 `_display_meta`:

```python
class <PascalCase>TargetAgent(AbstractTargetAgent, name="<name>"):
    _display_meta = {
        "icon": "M5.25 14.25h13.5m...",  # heroicons SVG path (24x24 viewBox)
        "color": "#6366f1",               # CSS 颜色值
        "features": ["HTTP API", "自定义协议"],  # 特性标签
    }
```

费用预估通过 `_cost_meta` 声明，前端自动读取并结合用户选择的模型定价计算预估费用：

```python
    _cost_meta = {
        "est_input_tokens": 200,   # 预估单次调用输入 token（含 system prompt + 上下文）
        "est_output_tokens": 600,  # 预估单次调用输出 token
    }
```

- LLM 类 target（如 `llm_api`）：声明合理的 token 估算值
- 外部 API 类 target（如 `theta_api`）：token 填 0（不按 token 计费）

`_display_meta` 和 `_cost_meta` 中的字段均为可选，未指定的使用默认值。

## Reference: Existing TargetAgent Plugins

| Name | Config | File | Description |
|------|--------|------|-------------|
| `theta_api` | `ThetaApiTargetInfo` | `theta_api_target_agent.py` | Theta Health HTTP API (email auth, polling-based chat) |
| `llm_api` | `LlmApiTargetInfo` | `llm_api_target_agent.py` | Generic LLM API (OpenAI/Gemini via do_execute) |

## TargetAgentReaction Types

The response must be one of:
- `type="message"` + `message_list=[{"content": "..."}]` — text response (most common)
- `type="gui"` + `gui_snapshots=["..."]` — GUI screenshot response
- `type="custom"` + `custom_content={...}` — custom structured response
