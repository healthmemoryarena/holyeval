"""Reports API tests"""

import pytest

from evaluator.core.schema import ReportEntry


@pytest.mark.asyncio
async def test_list_reports(client, monkeypatch):
    """GET /api/reports returns report list"""
    mock_reports = [
        ReportEntry(benchmark="healthbench", dataset="sample", date="20260212_143012", filename="sample_20260212_143012.json"),
    ]
    monkeypatch.setattr("web.app.api.reports.list_reports", lambda: mock_reports)
    resp = await client.get("/api/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert "reports" in data
    assert len(data["reports"]) == 1
    assert data["reports"][0]["benchmark"] == "healthbench"


@pytest.mark.asyncio
async def test_list_reports_empty(client, monkeypatch):
    """GET /api/reports returns empty list when no reports exist"""
    monkeypatch.setattr("web.app.api.reports.list_reports", lambda: [])
    resp = await client.get("/api/reports")
    assert resp.status_code == 200
    assert resp.json()["reports"] == []


@pytest.mark.asyncio
async def test_get_report_content(client, monkeypatch):
    """GET /api/reports/{bm}/{fn} returns report content"""
    mock_content = {
        "id": "session-001",
        "benchmark": "healthbench",
        "dataset": "sample",
        "total": 10,
        "pass_count": 8,
        "cases": [{"id": "c1", "score": 0.85}],
    }
    monkeypatch.setattr("web.app.api.reports.get_report_content", lambda bm, fn: mock_content)
    resp = await client.get("/api/reports/healthbench/sample_20260212_143012.json")
    assert resp.status_code == 200
    assert resp.json()["total"] == 10


@pytest.mark.asyncio
async def test_get_report_not_found(client, monkeypatch):
    """GET /api/reports/{bm}/{fn} returns 404 when not found"""

    def _raise(bm, fn):
        raise FileNotFoundError("Report not found")

    monkeypatch.setattr("web.app.api.reports.get_report_content", _raise)
    resp = await client.get("/api/reports/healthbench/nonexistent.json")
    assert resp.status_code == 404
