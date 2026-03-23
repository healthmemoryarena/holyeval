"""Guides API 测试"""

import pytest


@pytest.mark.asyncio
async def test_list_guides(client):
    """GET /api/guides 返回指南列表"""
    resp = await client.get("/api/guides")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    names = {g["name"] for g in data}
    assert "develop-eval-agent" in names
    assert "run-benchmark" in names


@pytest.mark.asyncio
async def test_get_guide(client):
    """GET /api/guides/{name} 返回指南内容"""
    resp = await client.get("/api/guides/develop-eval-agent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "develop-eval-agent"
    assert "content" in data
    assert len(data["content"]) > 0


@pytest.mark.asyncio
async def test_get_guide_not_found(client):
    """GET /api/guides/{name} — 不存在时返回 404"""
    resp = await client.get("/api/guides/nonexistent-guide")
    assert resp.status_code == 404
