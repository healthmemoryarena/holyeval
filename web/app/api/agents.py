"""Agent metadata API"""

import aiohttp
from fastapi import APIRouter

from evaluator.utils.config import get_config
from web.app.models.responses import AgentInfo
from web.app.services.agent_inspector import list_eval_agents, list_target_agents, list_test_agents

router = APIRouter(tags=["agents"])


@router.get("/agents/target", response_model=list[AgentInfo])
async def get_target_agents():
    return list_target_agents()


@router.get("/agents/eval", response_model=list[AgentInfo])
async def get_eval_agents():
    return list_eval_agents()


@router.get("/agents/test", response_model=list[AgentInfo])
async def get_test_agents():
    return list_test_agents()


@router.get("/agents/hermes/status")
async def get_hermes_status():
    """探测 Hermes Agent 状态和当前模型（从 config.yaml 读取真实模型名）"""
    from pathlib import Path

    base_url = get_config("HERMES_API_BASE_URL", "http://127.0.0.1:8642").rstrip("/")

    # 从 ~/.hermes/config.yaml 读取真实模型名
    model = None
    try:
        import yaml
        config_path = Path.home() / ".hermes" / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                model = model_cfg.get("default") or model_cfg.get("model")
            elif isinstance(model_cfg, str):
                model = model_cfg
    except Exception:
        pass

    # 探测 API 可达性
    ready = False
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            async with session.get(f"{base_url}/health") as resp:
                ready = resp.status == 200
    except Exception:
        pass

    return {"ready": ready, "model": model, "base_url": base_url}
