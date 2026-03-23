"""Agent metadata API"""

from fastapi import APIRouter

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
