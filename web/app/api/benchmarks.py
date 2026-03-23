"""Benchmark 数据 API"""

from typing import Any

from fastapi import APIRouter, HTTPException

from web.app.models.responses import BenchmarkSummary, DatasetDetail
from web.app.services.benchmark_reader import get_case_by_id, get_dataset_detail, list_benchmarks
from web.app.services.prepare_manager import prepare_manager

router = APIRouter(tags=["benchmarks"])


@router.get("/benchmarks", response_model=list[BenchmarkSummary])
async def get_benchmarks():
    benchmarks = list_benchmarks()
    # 注入准备状态
    for b in benchmarks:
        entry = prepare_manager.get_status(b.name)
        if entry:
            b.prepare_status = entry.status
            b.prepare_error = entry.error
    return benchmarks


@router.get("/prepare-status")
async def get_prepare_status():
    """获取所有 benchmark 的准备脚本状态"""
    result = {}
    for name, entry in prepare_manager.get_all_statuses().items():
        result[name] = {
            "status": entry.status,
            "module": entry.module_path,
            "started_at": entry.started_at.isoformat(),
            "finished_at": entry.finished_at.isoformat() if entry.finished_at else None,
            "error": entry.error,
        }
    return result


@router.get("/benchmarks/{benchmark}/{dataset}", response_model=DatasetDetail)
async def get_benchmark_dataset(benchmark: str, dataset: str, preview_limit: int = 10):
    try:
        return get_dataset_detail(benchmark, dataset, preview_limit=preview_limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/benchmarks/{benchmark}/{dataset}/cases/{case_id}")
async def get_benchmark_case(benchmark: str, dataset: str, case_id: str) -> dict[str, Any]:
    try:
        return get_case_by_id(benchmark, dataset, case_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
