"""Guides API tests"""

import pytest


@pytest.mark.asyncio
async def test_list_guides(client):
    """GET /api/guides returns guide list"""
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
    """GET /api/guides/{name} returns guide content"""
    resp = await client.get("/api/guides/develop-eval-agent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "develop-eval-agent"
    assert "content" in data
    assert len(data["content"]) > 0


@pytest.mark.asyncio
async def test_get_guide_not_found(client):
    """GET /api/guides/{name} returns 404 when not found"""
    resp = await client.get("/api/guides/nonexistent-guide")
    assert resp.status_code == 404
