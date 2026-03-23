"""Agents API tests"""

import pytest


@pytest.mark.asyncio
async def test_list_target_agents(client):
    """GET /api/agents/target returns TargetAgent list"""
    resp = await client.get("/api/agents/target")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2  # llm_api + theta_api
    names = {a["name"] for a in data}
    assert "llm_api" in names
    assert "theta_api" in names


@pytest.mark.asyncio
async def test_list_eval_agents(client):
    """GET /api/agents/eval returns EvalAgent list"""
    resp = await client.get("/api/agents/eval")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 5  # semantic, indicator, keyword, preset_answer, healthbench
    names = {a["name"] for a in data}
    assert "semantic" in names
    assert "healthbench" in names


@pytest.mark.asyncio
async def test_list_test_agents(client):
    """GET /api/agents/test returns TestAgent list"""
    resp = await client.get("/api/agents/test")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2  # auto + manual
    names = {a["name"] for a in data}
    assert "auto" in names
    assert "manual" in names


@pytest.mark.asyncio
async def test_agent_info_fields(client):
    """Verify AgentInfo field completeness"""
    resp = await client.get("/api/agents/eval")
    assert resp.status_code == 200
    for agent in resp.json():
        assert "name" in agent
        assert "class_name" in agent
        assert "icon" in agent
        assert "color" in agent
        assert "features" in agent
        assert isinstance(agent["features"], list)
