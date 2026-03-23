# 开发被测系统（TargetAgent）

> **在 Claude Code 中使用**: 输入 `/add-target-agent <agent-name>`，后跟你的被测系统描述（如 API 地址、认证方式、协议细节），Claude 会自动生成配置、实现和注册代码。

## 概述

TargetAgent 封装被测系统的接入层，负责将虚拟用户的输入发送给目标系统并获取响应。每种被测系统（HTTP API、SDK、GUI 等）对应一个 TargetAgent 插件。

## 现有被测系统

| 名称 | 说明 |
|------|------|
| `theta_api` | Theta Health HTTP API，独立管理 session/token |
| `llm_api` | 通用 LLM API（OpenAI / Gemini），基于 `do_execute` 统一 LLM 层 |

## 开发步骤

### 1. 确定接入方式

- **HTTP API** — 最常见，通过 HTTP 请求与目标系统交互
- **SDK/Client** — 使用目标系统提供的 Python SDK
- **CLI** — 通过命令行调用目标系统
- **LLM API** — 直接调用 LLM 提供商的 API

### 2. 定义配置模型

在 `evaluator/core/schema.py` 中新增配置：

```python
class MyTargetInfo(BaseModel):
    """我的被测系统配置"""
    model_config = ConfigDict(extra="forbid")

    type: Literal["my_target"] = "my_target"
    # 连接参数（建议基础设施参数放 .env，业务参数放配置）
```

更新 `TargetInfo` 联合类型以包含新配置。

### 3. 实现 TargetAgent

在 `evaluator/plugin/target_agent/` 创建实现文件：

```python
class MyTargetAgent(AbstractTargetAgent, name="my_target"):
    async def _generate_next_reaction(self, test_action):
        # 1. 提取用户输入 test_action.semantic_content
        # 2. 调用目标系统
        # 3. 返回 TargetAgentReaction(type="message", message_list=[...])

    async def cleanup(self):
        # 释放连接资源
```

### 4. 注册插件

在 `evaluator/plugin/target_agent/__init__.py` 中导入新类。

### 5. 验证

```bash
python -c "from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; print(AbstractTargetAgent.get_all())"
ruff check evaluator/core/schema.py evaluator/plugin/target_agent/
```

## 关键设计模式

- **懒初始化**: 首次调用时触发认证/建立连接
- **响应格式**: `TargetAgentReaction(type="message", message_list=[{"content": "..."}])`
- **资源清理**: 实现 `cleanup()` 方法释放网络连接
- **环境变量**: 基础设施参数（base_url、timeout）放 `.env`，业务参数放配置模型

## 关键文件

| 文件 | 说明 |
|------|------|
| `evaluator/core/schema.py` | 配置模型定义 |
| `evaluator/core/interfaces/abstract_target_agent.py` | 抽象接口 |
| `evaluator/plugin/target_agent/` | 插件实现目录 |
| `evaluator/plugin/target_agent/theta_api_target_agent.py` | HTTP API 参考实现 |
| `evaluator/plugin/target_agent/llm_api_target_agent.py` | LLM API 参考实现 |
