"""Task execution / progress / cancellation / checkpoint API"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from evaluator.core.bench_schema import find_target_spec, resolve_runtime_target
from evaluator.utils.benchmark_reader import list_benchmarks
from evaluator.utils.checkpoint import CheckpointManager
from web.app.models.responses import CheckpointSummary, EvalCheckRequest, EvalCheckResponse, EvalOnlyRequest, EvalOnlyResponse, TaskCreateRequest, TaskDetail, TaskSummary
from web.app.services.prepare_manager import prepare_manager
from web.app.services.task_manager import task_manager

router = APIRouter(tags=["tasks"])


@router.post("/tasks", response_model=TaskSummary)
async def create_task(req: TaskCreateRequest):
    # Check if benchmark is being prepared
    if prepare_manager.is_preparing(req.benchmark):
        raise HTTPException(status_code=409, detail=f"Benchmark '{req.benchmark}' is preparing data, please try again later")

    # Find benchmark metadata to get TargetSpec
    benchmarks = list_benchmarks()
    bm = next((b for b in benchmarks if b.name == req.benchmark), None)
    if not bm:
        raise HTTPException(status_code=400, detail=f"Benchmark not found: '{req.benchmark}'")

    try:
        spec = find_target_spec(bm.target, req.target_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Frontend target dict used as cli_overrides (only editable fields take effect)
    cli_overrides = req.target
    try:
        resolve_runtime_target(spec, cli_overrides)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        entry = await task_manager.create_task(
            benchmark=req.benchmark,
            dataset=req.dataset,
            spec=spec,
            cli_overrides=cli_overrides,
            ids=req.ids,
            limit=req.limit,
            max_concurrency=req.max_concurrency,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TaskSummary(
        task_id=entry.task_id,
        benchmark=entry.benchmark,
        dataset=entry.dataset,
        target_type=entry.runtime_target.type,
        target_model=getattr(entry.runtime_target, "model", None),
        status=entry.status,
        total=entry.session.total,
        completed=entry.session.completed,
        created_at=entry.created_at,
    )


@router.post("/eval-check", response_model=EvalCheckResponse)
async def check_eval(req: EvalCheckRequest):
    """Validate submitted results against dataset (without starting evaluation)"""
    if prepare_manager.is_preparing(req.benchmark):
        raise HTTPException(status_code=409, detail=f"Benchmark '{req.benchmark}' is preparing data, please try again later")

    try:
        validation = task_manager.check_eval(
            benchmark=req.benchmark,
            dataset=req.dataset,
            results=req.results,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return EvalCheckResponse(**validation)


@router.post("/eval-only", response_model=EvalOnlyResponse)
async def create_eval_only_task(req: EvalOnlyRequest):
    """Create eval-only task -- accept external call results and run evaluation only (with validation)"""
    if prepare_manager.is_preparing(req.benchmark):
        raise HTTPException(status_code=409, detail=f"Benchmark '{req.benchmark}' is preparing data, please try again later")

    try:
        entry, validation = await task_manager.create_eval_task(
            benchmark=req.benchmark,
            dataset=req.dataset,
            results=req.results,
            max_concurrency=req.max_concurrency,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return EvalOnlyResponse(
        task_id=entry.task_id,
        status=entry.status,
        total=entry.total,
        dataset_total=validation["dataset_total"],
        submitted=validation["submitted"],
        matched=validation["matched"],
        missed_ids=validation["missed_ids"],
        extra_ids=validation["extra_ids"],
        created_at=entry.created_at,
    )


@router.get("/tasks", response_model=list[TaskSummary])
async def list_tasks():
    return [
        TaskSummary(
            task_id=e.task_id,
            benchmark=e.benchmark,
            dataset=e.dataset,
            target_type=e.runtime_target.type if e.runtime_target else None,
            target_model=getattr(e.runtime_target, "model", None) if e.runtime_target else None,
            status=e.status,
            total=e.session.total if e.session else e.total,
            completed=e.session.completed if e.session else e.completed,
            created_at=e.created_at,
        )
        for e in task_manager.list_tasks()
    ]


@router.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str):
    try:
        snapshot = task_manager.get_snapshot(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskDetail(
        task_id=snapshot["id"],
        benchmark=snapshot["benchmark"],
        dataset=snapshot["dataset"],
        runtime_target=snapshot["runtime_target"],
        status=snapshot["status"],
        total=snapshot["total"],
        completed=snapshot["completed"],
        cancelled=snapshot["cancelled"],
        created_at=snapshot["created_at"],
        cases=snapshot["cases"],
        stats_by_tag=snapshot.get("stats_by_tag"),
        report_path=snapshot.get("report_path"),
        report_summary=snapshot.get("report_summary"),
    )


@router.get("/tasks/{task_id}/cases/{case_id}")
async def get_case_result(task_id: str, case_id: str):
    """Get real-time result for a single case (from running CaseContext or completed result)"""
    from web.app.services.task_manager import read_live_file

    entry = task_manager.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if not entry.session:
        raise HTTPException(status_code=410, detail="Task completed, view results in reports")
    ctx = entry.session.contexts.get(case_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    if ctx.result:
        return {"status": ctx.status.value, "result": ctx.result.model_dump(mode="json")}

    # Running case -- read conversation data from live file
    live_data = read_live_file(task_id, case_id)
    if live_data:
        return {"status": ctx.status.value, "result": live_data}

    return {"status": ctx.status.value, "result": None}


@router.get("/tasks/{task_id}/results")
async def get_task_results(
    task_id: str,
    page: int = 1,
    page_size: int = 20,
    tag: str | None = None,
):
    """Get paginated evaluation results for a task (from memory or report file)"""
    entry = task_manager.get_task(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Collect results list
    results: list[dict] = []
    if entry.eval_results:
        results = [r.model_dump(mode="json") for r in entry.eval_results]
    elif entry.session and entry.status == "completed":
        # Collect from session contexts
        for ctx in entry.session.contexts.values():
            if ctx.result:
                results.append(ctx.result.model_dump(mode="json"))
        # Merge resumed results
        for r in entry.resumed_results:
            results.append(r.model_dump(mode="json"))
    elif entry.report_path:
        # Read from report file
        import json
        from pathlib import Path

        try:
            with open(entry.report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            results = report_data.get("cases", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    if not results:
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}

    # Filter by tag
    if tag:
        results = [r for r in results if tag in r.get("tags", [])]

    total = len(results)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    items = results[start : start + page_size]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    try:
        task_manager.cancel_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return {"status": "cancelling", "task_id": task_id}


# ==================== Checkpoint API ====================


@router.get("/checkpoints", response_model=list[CheckpointSummary])
async def list_checkpoints(benchmark: str | None = None):
    """List resumable checkpoints"""
    checkpoints = task_manager.list_checkpoints(benchmark)
    return [
        CheckpointSummary(
            session_id=cp.session_id,
            benchmark=cp.benchmark,
            dataset=cp.dataset,
            target_type=cp.target_type,
            case_count=len(cp.case_ids),
            completed_count=CheckpointManager.completed_count(cp.session_id),
            started_at=cp.started_at,
        )
        for cp in checkpoints
    ]


@router.post("/checkpoints/{session_id}/resume", response_model=TaskSummary)
async def resume_checkpoint(session_id: str):
    """Resume a checkpoint task"""
    try:
        meta, _ = CheckpointManager.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {session_id}")

    # Find TargetSpec
    benchmarks = list_benchmarks()
    bm = next((b for b in benchmarks if b.name == meta.benchmark), None)
    if not bm:
        raise HTTPException(status_code=400, detail=f"Benchmark not found: '{meta.benchmark}'")

    try:
        spec = find_target_spec(bm.target, meta.target_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        entry = await task_manager.resume_task(session_id, spec)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TaskSummary(
        task_id=entry.task_id,
        benchmark=entry.benchmark,
        dataset=entry.dataset,
        target_type=entry.runtime_target.type,
        target_model=getattr(entry.runtime_target, "model", None),
        status=entry.status,
        total=entry.session.total + len(entry.resumed_results),
        completed=len(entry.resumed_results),
        created_at=entry.created_at,
    )


@router.delete("/checkpoints/{session_id}")
async def delete_checkpoint(session_id: str):
    """Delete a checkpoint (abandon resume)"""
    mgr = CheckpointManager(session_id)
    mgr.cleanup()
    return {"status": "deleted", "session_id": session_id}
