"""Benchmark API 测试"""

import pytest

from evaluator.core.schema import BenchmarkSummary, CaseSummary, DatasetDetail, DatasetInfo


@pytest.mark.asyncio
async def test_list_benchmarks(client, monkeypatch):
    """GET /api/benchmarks 返回 benchmark 列表"""
    mock_data = [
        BenchmarkSummary(
            name="healthbench",
            description="医疗 AI 评测",
            datasets=[DatasetInfo(name="sample", case_count=100, file_size_kb=512.0)],
        )
    ]
    monkeypatch.setattr("web.app.api.benchmarks.list_benchmarks", lambda: mock_data)
    resp = await client.get("/api/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "healthbench"
    assert data[0]["datasets"][0]["case_count"] == 100


@pytest.mark.asyncio
async def test_get_dataset_detail(client, monkeypatch):
    """GET /api/benchmarks/{bm}/{ds} 返回 dataset 详情"""
    mock_detail = DatasetDetail(
        benchmark="healthbench",
        dataset="sample",
        case_count=100,
        cases_preview=[{"id": "c1", "title": "test"}],
        case_summaries=[CaseSummary(id="c1", title="test", user_type="manual", target_type="llm_api", evaluator="healthbench")],
        tag_distribution={"cardio": 10},
        readme="# HealthBench",
    )
    monkeypatch.setattr("web.app.api.benchmarks.get_dataset_detail", lambda bm, ds, preview_limit=10: mock_detail)
    resp = await client.get("/api/benchmarks/healthbench/sample")
    assert resp.status_code == 200
    data = resp.json()
    assert data["benchmark"] == "healthbench"
    assert data["case_count"] == 100
    assert len(data["case_summaries"]) == 1


@pytest.mark.asyncio
async def test_get_dataset_detail_not_found(client, monkeypatch):
    """GET /api/benchmarks/{bm}/{ds} — 不存在时返回 404"""

    def _raise(bm, ds, preview_limit=10):
        raise FileNotFoundError("数据集不存在")

    monkeypatch.setattr("web.app.api.benchmarks.get_dataset_detail", _raise)
    resp = await client.get("/api/benchmarks/nonexistent/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_case_by_id(client, monkeypatch):
    """GET /api/benchmarks/{bm}/{ds}/cases/{id} 返回单条 case"""
    mock_case = {"id": "hb_001", "title": "头痛咨询", "user": {"type": "manual"}}
    monkeypatch.setattr("web.app.api.benchmarks.get_case_by_id", lambda bm, ds, cid: mock_case)
    resp = await client.get("/api/benchmarks/healthbench/sample/cases/hb_001")
    assert resp.status_code == 200
    assert resp.json()["id"] == "hb_001"


@pytest.mark.asyncio
async def test_get_case_by_id_not_found(client, monkeypatch):
    """GET /api/benchmarks/{bm}/{ds}/cases/{id} — 不存在时返回 404"""

    def _raise(bm, ds, cid):
        raise KeyError("用例不存在")

    monkeypatch.setattr("web.app.api.benchmarks.get_case_by_id", _raise)
    resp = await client.get("/api/benchmarks/healthbench/sample/cases/nonexistent")
    assert resp.status_code == 404
