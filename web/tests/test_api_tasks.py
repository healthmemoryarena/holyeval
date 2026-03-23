"""Tasks API 测试"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluator.core.schema import BenchmarkSummary, TargetFieldSpec, TargetSpec
from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetInfo


@dataclass
class _MockSession:
    """轻量 BatchSession mock"""

    id: str = "test-session-001"
    total: int = 10
    completed: int = 5
    cancel_event: MagicMock = field(default_factory=MagicMock)
    contexts: dict = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "total": self.total,
            "completed": self.completed,
            "cancelled": False,
            "created_at": "2026-02-12T14:30:00",
            "cases": {},
        }

    def cancel(self):
        pass


def _default_runtime_target():
    return LlmApiTargetInfo(type="llm_api", model="gpt-4.1")


@dataclass
class _MockEntry:
    task_id: str = "test-session-001"
    benchmark: str = "healthbench"
    dataset: str = "sample"
    runtime_target: LlmApiTargetInfo = field(default_factory=_default_runtime_target)
    session: _MockSession = field(default_factory=_MockSession)
    asyncio_task: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"
    report_path: str | None = None
    error: str | None = None


def _make_mock_manager(entries: list[_MockEntry] | None = None):
    """构造 TaskManager mock"""
    entries = entries or []
    mgr = MagicMock()
    mgr.list_tasks.return_value = entries
    mgr.get_task.side_effect = lambda tid: next((e for e in entries if e.task_id == tid), None)

    def _snapshot(tid):
        entry = next((e for e in entries if e.task_id == tid), None)
        if not entry:
            raise KeyError(f"任务不存在: {tid}")
        snap = entry.session.snapshot()
        snap["status"] = entry.status
        snap["benchmark"] = entry.benchmark
        snap["dataset"] = entry.dataset
        snap["runtime_target"] = entry.runtime_target.model_dump(mode="json")
        snap["created_at"] = entry.created_at.isoformat()
        snap["report_path"] = entry.report_path
        snap["stats_by_tag"] = {"difficulty:direct": {"avg_score": 0.9, "total": 2}}
        snap["report_summary"] = {"avg_score": 0.9, "stats_by_tag": {"difficulty:direct": {"avg_score": 0.9, "count": 2}}}
        return snap

    mgr.get_snapshot.side_effect = _snapshot
    mgr.cancel_task.side_effect = lambda tid: (_ for _ in ()).throw(KeyError(f"任务不存在: {tid}")) if not any(
        e.task_id == tid for e in entries
    ) else None
    return mgr


@pytest.mark.asyncio
async def test_list_tasks_empty(client, monkeypatch):
    """GET /api/tasks — 无任务时返回空列表"""
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager())
    resp = await client.get("/api/tasks")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_tasks_with_entries(client, monkeypatch):
    """GET /api/tasks — 有任务时返回列表"""
    entry = _MockEntry()
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager([entry]))
    resp = await client.get("/api/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["task_id"] == "test-session-001"
    assert data[0]["benchmark"] == "healthbench"
    assert data[0]["status"] == "running"


@pytest.mark.asyncio
async def test_get_task_detail(client, monkeypatch):
    """GET /api/tasks/{id} 返回任务详情"""
    entry = _MockEntry()
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager([entry]))
    resp = await client.get(f"/api/tasks/{entry.task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == entry.task_id
    assert data["total"] == 10
    assert data["completed"] == 5
    assert data["stats_by_tag"]["difficulty:direct"]["avg_score"] == 0.9
    assert data["report_summary"]["stats_by_tag"]["difficulty:direct"]["count"] == 2


@pytest.mark.asyncio
async def test_get_task_not_found(client, monkeypatch):
    """GET /api/tasks/{id} — 不存在时返回 404"""
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager())
    resp = await client.get("/api/tasks/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_task(client, monkeypatch):
    """POST /api/tasks/{id}/cancel 取消任务"""
    entry = _MockEntry()
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager([entry]))
    resp = await client.post(f"/api/tasks/{entry.task_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelling"


@pytest.mark.asyncio
async def test_cancel_task_not_found(client, monkeypatch):
    """POST /api/tasks/{id}/cancel — 不存在时返回 404"""
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager())
    resp = await client.post("/api/tasks/nonexistent/cancel")
    assert resp.status_code == 404


# ============================================================
# POST /api/tasks — 创建任务
# ============================================================


def _mock_benchmarks():
    """返回 mock 的 benchmark 列表"""
    return [
        BenchmarkSummary(
            name="healthbench",
            description="",
            datasets=[],
            target=[
                TargetSpec(
                    type="llm_api",
                    fields={
                        "model": TargetFieldSpec(default="gpt-4.1", editable=True, required=True),
                        "system_prompt": TargetFieldSpec(default=None, editable=True),
                    },
                )
            ],
        )
    ]


@pytest.mark.asyncio
async def test_create_task_success(client, monkeypatch):
    """POST /api/tasks 创建任务成功"""
    entry = _MockEntry(status="running")

    async def _mock_create(**kwargs):
        return entry

    mgr = _make_mock_manager([entry])
    mgr.create_task = _mock_create
    monkeypatch.setattr("web.app.api.tasks.task_manager", mgr)
    monkeypatch.setattr("web.app.api.tasks.list_benchmarks", _mock_benchmarks)

    resp = await client.post("/api/tasks", json={
        "benchmark": "healthbench",
        "dataset": "sample",
        "target": {"type": "llm_api", "model": "gpt-4.1"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "test-session-001"
    assert data["benchmark"] == "healthbench"
    assert data["status"] == "running"
    assert data["target_type"] == "llm_api"
    assert data["target_model"] == "gpt-4.1"


@pytest.mark.asyncio
async def test_create_task_invalid_benchmark(client, monkeypatch):
    """POST /api/tasks — benchmark 不存在返回 400"""

    async def _mock_create(**kwargs):
        raise FileNotFoundError("评测类型不存在: bad")

    mgr = _make_mock_manager()
    mgr.create_task = _mock_create
    monkeypatch.setattr("web.app.api.tasks.task_manager", mgr)
    monkeypatch.setattr("web.app.api.tasks.list_benchmarks", _mock_benchmarks)

    resp = await client.post("/api/tasks", json={
        "benchmark": "bad",
        "dataset": "sample",
        "target": {"type": "llm_api", "model": "gpt-4.1"},
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_task_required_field_missing(client, monkeypatch):
    """POST /api/tasks — required 字段无默认值时返回 400"""
    # 构造一个 model 无默认值且 required 的 spec
    no_default_benchmarks = lambda: [
        BenchmarkSummary(
            name="healthbench",
            description="",
            datasets=[],
            target=[
                TargetSpec(
                    type="llm_api",
                    fields={"model": TargetFieldSpec(default=None, editable=True, required=True)},
                )
            ],
        )
    ]
    mgr = _make_mock_manager()
    monkeypatch.setattr("web.app.api.tasks.task_manager", mgr)
    monkeypatch.setattr("web.app.api.tasks.list_benchmarks", no_default_benchmarks)

    resp = await client.post("/api/tasks", json={
        "benchmark": "healthbench",
        "dataset": "sample",
    })
    assert resp.status_code == 400
    assert "必填字段" in resp.json()["detail"]


# ============================================================
# GET /api/tasks/{task_id}/cases/{case_id} — 查看用例结果
# ============================================================


@dataclass
class _MockCaseContext:
    """轻量 CaseContext mock"""
    status: MagicMock = field(default_factory=lambda: MagicMock(value="completed"))
    result: Any = None


@pytest.mark.asyncio
async def test_get_case_result_no_result(client, monkeypatch):
    """GET /api/tasks/{id}/cases/{id} — 运行中，无结果"""
    ctx = _MockCaseContext(status=MagicMock(value="dialogue"), result=None)
    session = _MockSession(contexts={"case_001": ctx})
    entry = _MockEntry(session=session)
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager([entry]))

    resp = await client.get(f"/api/tasks/{entry.task_id}/cases/case_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "dialogue"
    assert data["result"] is None


@pytest.mark.asyncio
async def test_get_case_result_with_result(client, monkeypatch):
    """GET /api/tasks/{id}/cases/{id} — 已完成，有结果"""
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {"id": "case_001", "eval": {"result": "pass", "score": 0.9}}
    ctx = _MockCaseContext(status=MagicMock(value="completed"), result=mock_result)
    session = _MockSession(contexts={"case_001": ctx})
    entry = _MockEntry(session=session)
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager([entry]))

    resp = await client.get(f"/api/tasks/{entry.task_id}/cases/case_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["eval"]["result"] == "pass"


@pytest.mark.asyncio
async def test_get_case_result_task_not_found(client, monkeypatch):
    """GET /api/tasks/{id}/cases/{id} — 任务不存在返回 404"""
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager())
    resp = await client.get("/api/tasks/nonexistent/cases/case_001")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_case_result_case_not_found(client, monkeypatch):
    """GET /api/tasks/{id}/cases/{id} — 用例不存在返回 404"""
    entry = _MockEntry()
    monkeypatch.setattr("web.app.api.tasks.task_manager", _make_mock_manager([entry]))
    resp = await client.get(f"/api/tasks/{entry.task_id}/cases/nonexistent")
    assert resp.status_code == 404
