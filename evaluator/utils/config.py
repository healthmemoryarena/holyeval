"""
config — 配置管理模块

从 .env 文件和环境变量读取配置。
"""

import os
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv

# 加载 .env 文件（本地开发用）
load_dotenv()

# Docker 部署时从配置中心加载（有 CONFIG_SERVER 环境变量时生效）
try:
    from evaluator.utils.remote_config import load_remote_config

    load_remote_config()
except Exception:
    pass


@lru_cache(maxsize=128)
def get_config(key: str, default: Any = None) -> Any:
    """
    获取配置值

    Args:
        key: 配置键名（支持点分隔格式，如 'theta_api.base_url'）
        default: 默认值

    Returns:
        配置值
    """
    # 将点分隔格式转换为环境变量格式（大写 + 下划线）
    env_key = key.upper().replace(".", "_")
    return os.environ.get(env_key, default)


def get_agent_llm_timeout() -> int:
    """获取 Agent LLM 调用超时时间（秒），默认 840s（14分钟）"""
    return int(get_config("AGENT_LLM_TIMEOUT", 840))


def get_theta_api_config() -> dict[str, Any]:
    """获取 Theta API 相关配置"""
    return {
        "base_url": get_config("THETA_API_BASE_URL", "https://test-mcp.thetahealth.ai"),
        "timeout": 180,
        "poll_interval": 1.0,
        "poll_timeout": 300,
        "chat_api_path": get_config("THETA_API_CHAT_PATH", "/api/v1/holywell/chat/create_message_v2"),
        "indicator_api_path": "/api/v1/holywell/health-indicator/watch",
    }


def get_theta_smart_api_config() -> dict[str, Any]:
    """获取 Theta Smart API 相关配置"""
    return {
        "base_url": get_config("THETA_SMART_API_BASE_URL", "http://localhost:8199"),
        "timeout": 180,
        "poll_interval": 1.0,
        "poll_timeout": 300,
    }
