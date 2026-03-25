"""TaskManager — manages benchmark task lifecycle (BenchItem + BenchReport)"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluator.core.bench_schema import (
    ApiCallResult,
    api_result_to_eval_items,
    bench_item_to_test_case,
    build_bench_report,
    resolve_runtime_target,
)
from evaluator.core.orchestrator import BatchSession, CaseContext, do_batch_eval
from evaluator.core.schema import EvalResult, TargetInfo, TargetSpec, TestCost, TestResult
from evaluator.utils.benchmark_reader import _read_metadata, filter_bench_items, load_bench_items, resolve_data_path
from evaluator.utils.checkpoint import CheckpointManager, CheckpointMeta
from evaluator.utils.report_reader import save_bench_report

LIVE_DIR = Path("benchmark/report/.live")

logger = logging.getLogger(__name__)


@dataclass
class TaskEntry:
    """Task entry — contains BatchSession and runtime state"""

    task_id: str
    # Checkpoint ID this task writes to / reuses:
    # - New task: same as task_id
    # - Resumed task: reuses old checkpoint session_id (usually different from task_id)
    checkpoint_session_id: str
    benchmark: str
    dataset: str
    runtime_target: TargetInfo | None = None  # no target in eval-only mode
    session: BatchSession | None = None  # no BatchSession in eval-only mode
    asyncio_task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running | completed | cancelled | error
    report_path: str | None = None
    error: str | None = None
    # checkpoint-related
    cli_overrides: dict | None = None
    data_file_hash: str = ""
    resumed_results: list[TestResult] = field(default_factory=list)
    # eval-only progress tracking
    total: int = 0
    completed: int = 0
    eval_results: list[TestResult] = field(default_factory=list)

    def _release_heavy_data(self) -> None:
        """Release heavy objects, keep only lightweight metadata for API queries"""
        self.session = None
        self.eval_results = []
        self.resumed_results = []
        self.asyncio_task = None


class TaskManager:
    """In-memory task manager — based on BenchItem and BenchReport"""

    def __init__(self):
        self._tasks: dict[str, TaskEntry] = {}

    async def create_task(
        self,
        benchmark: str,
        dataset: str,
        spec: TargetSpec,
        cli_overrides: dict | None = None,
        ids: str | None = None,
        limit: int | None = None,
        max_concurrency: int = 3,
    ) -> TaskEntry:
        """Create and start a benchmark task

        Args:
            benchmark: benchmark type (e.g. "healthbench")
            dataset: dataset name (e.g. "sample")
            spec: TargetSpec from metadata.json
            cli_overrides: target override params from frontend (only editable fields apply)
            ids: comma-separated IDs (run only specified cases)
            limit: max number of cases
            max_concurrency: max concurrency

        Returns:
            TaskEntry
        """
        # 1. Load BenchItems (with $ref resolution)
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)
        items = filter_bench_items(items, ids=ids, limit=limit)

        if not items:
            raise ValueError(f"No runnable cases after filtering ({benchmark}/{dataset})")

        # 2. Build global runtime target (for logging and reports)
        runtime_target = resolve_runtime_target(spec, cli_overrides)
        logger.info(
            "Creating task: %s/%s, target=%s, model=%s, %d cases",
            benchmark,
            dataset,
            runtime_target.type,
            getattr(runtime_target, "model", "N/A"),
            len(items),
        )

        # 3. Convert BenchItem -> TestCase (with three-layer target merge)
        test_cases = []
        for item in items:
            try:
                test_case = bench_item_to_test_case(item, spec, cli_overrides)
                test_cases.append(test_case)
            except Exception as e:
                logger.error("Case %s conversion failed: %s", item.id, e)
                raise ValueError(f"Case {item.id} conversion failed: {e}") from e

        # 4. Create BatchSession
        session = BatchSession(
            cases=test_cases,
            max_concurrency=max_concurrency,
        )

        # 5. Create TaskEntry
        data_file_hash = CheckpointManager.compute_data_hash(path)
        entry = TaskEntry(
            task_id=session.id,
            checkpoint_session_id=session.id,
            benchmark=benchmark,
            dataset=dataset,
            runtime_target=runtime_target,
            session=session,
            cli_overrides=cli_overrides,
            data_file_hash=data_file_hash,
        )
        self._tasks[session.id] = entry

        # 6. Run in background
        entry.asyncio_task = asyncio.create_task(
            self._run_session(entry, max_concurrency)
        )

        logger.info("Task created: %s (%s/%s, %d cases)", session.id, benchmark, dataset, len(test_cases))
        return entry

    async def resume_task(
        self,
        session_id: str,
        spec: TargetSpec,
    ) -> TaskEntry:
        """Resume an incomplete checkpoint task

        Args:
            session_id: checkpoint session ID
            spec: TargetSpec from metadata.json

        Returns:
            TaskEntry
        """
        meta, completed_results = CheckpointManager.load(session_id)

        # Verify data file hash
        path = resolve_data_path(meta.benchmark, meta.dataset)
        current_hash = CheckpointManager.compute_data_hash(path)
        if meta.data_file_hash and current_hash != meta.data_file_hash:
            logger.warning("Data file changed (hash: %s -> %s)", meta.data_file_hash, current_hash)

        # Filter completed cases (re-run cancelled ones)
        cancelled_ids = {r.id for r in completed_results if r.eval.feedback == "用例被取消"}
        completed_ids = {r.id for r in completed_results} - cancelled_ids
        completed_results = [r for r in completed_results if r.id not in cancelled_ids]
        remaining_ids = set(meta.case_ids) - completed_ids

        logger.info(
            "Resuming task: %s/%s (completed %d/%d, remaining %d)",
            meta.benchmark, meta.dataset, len(completed_ids), len(meta.case_ids), len(remaining_ids),
        )

        # Load remaining cases (with $ref resolution)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)
        remaining_items = [item for item in items if item.id in remaining_ids]

        if not remaining_items:
            raise ValueError("All cases already completed, nothing to resume")

        runtime_target = resolve_runtime_target(spec, meta.cli_overrides)
        test_cases = []
        for item in remaining_items:
            try:
                test_cases.append(bench_item_to_test_case(item, spec, meta.cli_overrides))
            except Exception as e:
                logger.error("Case %s conversion failed: %s", item.id, e)
                raise ValueError(f"Case {item.id} conversion failed: {e}") from e

        # Create BatchSession (new ID)
        session = BatchSession(
            cases=test_cases,
            max_concurrency=meta.max_concurrency,
        )

        entry = TaskEntry(
            task_id=session.id,
            checkpoint_session_id=session_id,
            benchmark=meta.benchmark,
            dataset=meta.dataset,
            runtime_target=runtime_target,
            session=session,
            cli_overrides=meta.cli_overrides,
            data_file_hash=current_hash,
            resumed_results=completed_results,
        )
        self._tasks[session.id] = entry

        # Run in background (append results using original checkpoint session_id)
        entry.asyncio_task = asyncio.create_task(
            self._run_session(entry, meta.max_concurrency, checkpoint_session_id=session_id)
        )

        logger.info(
            "Resumed task created: %s (original checkpoint %s, %d remaining cases)",
            session.id, session_id, len(test_cases),
        )
        return entry

    def check_eval(
        self,
        benchmark: str,
        dataset: str,
        results: list[ApiCallResult],
    ) -> dict:
        """Validate submitted results against dataset (does not start evaluation)

        Returns:
            validation_info dict
        """
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        try:
            items = load_bench_items(path, params=params)
        except ValueError:
            logger.exception("数据集加载失败: %s/%s", benchmark, dataset)
            raise ValueError("数据集加载异常，请联系管理员") from None

        if not items:
            raise ValueError(f"Dataset is empty ({benchmark}/{dataset})")

        dataset_ids = {item.id for item in items}
        submitted_ids = {r.id for r in results}
        matched_ids = dataset_ids & submitted_ids
        extra_ids = sorted(submitted_ids - dataset_ids)
        missed_ids = sorted(dataset_ids - submitted_ids)

        # Tag distribution stats (matched cases only)
        matched_items = [item for item in items if item.id in matched_ids]
        tag_counts: dict[str, int] = {}
        for item in matched_items:
            for tag in item.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "benchmark": benchmark,
            "dataset": dataset,
            "benchmark_description": metadata.get("description", ""),
            "dataset_total": len(dataset_ids),
            "submitted": len(submitted_ids),
            "matched": len(matched_ids),
            "extra_ids": extra_ids[:50],
            "missed_ids": missed_ids[:50],
            "tag_distribution": tag_counts,
        }

    async def create_eval_task(
        self,
        benchmark: str,
        dataset: str,
        results: list[ApiCallResult],
        max_concurrency: int = 5,
    ) -> tuple["TaskEntry", dict]:
        """Create eval-only task — runs evaluation only, no dialogue loop

        Returns:
            (TaskEntry, validation_info)
        """
        import uuid

        # 1. Reuse check_eval validation
        validation_info = self.check_eval(benchmark, dataset, results)

        matched_ids = set()
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        try:
            items = load_bench_items(path, params=params)
        except ValueError:
            logger.exception("数据集加载失败: %s/%s", benchmark, dataset)
            raise ValueError("数据集加载异常，请联系管理员") from None
        dataset_ids = {item.id for item in items}
        submitted_ids = {r.id for r in results}
        matched_ids = dataset_ids & submitted_ids

        if not matched_ids:
            raise ValueError(
                f"No matching cases: {len(submitted_ids)} submitted, {len(dataset_ids)} in dataset, 0 IDs matched"
            )

        # 2. Convert (matched only)
        matched_results = [r for r in results if r.id in matched_ids]
        eval_items = api_result_to_eval_items(matched_results, items)

        # Unmatched cases (in dataset but not submitted), scored as 0
        missed_items = [item for item in items if item.id not in matched_ids]

        logger.info(
            "Creating eval-only task: %s/%s, submitted %d, matched %d, missing %d (scored 0)",
            benchmark, dataset, len(results), len(matched_ids), len(missed_items),
        )

        # 3. Create TaskEntry (total = full dataset count)
        task_id = f"eval_{uuid.uuid4().hex[:12]}"
        entry = TaskEntry(
            task_id=task_id,
            checkpoint_session_id=task_id,
            benchmark=benchmark,
            dataset=dataset,
            total=len(items),
        )
        self._tasks[task_id] = entry

        # 4. Run in background
        entry.asyncio_task = asyncio.create_task(
            self._run_eval_session(entry, eval_items, max_concurrency, missed_items=missed_items)
        )

        return entry, validation_info

    async def _run_eval_session(
        self,
        entry: TaskEntry,
        eval_items: list,
        max_concurrency: int,
        *,
        missed_items: list | None = None,
    ) -> None:
        """Run eval-only batch evaluation in background and save report

        missed_items: unmatched BenchItems from dataset, scored 0 and merged into final report
        """
        try:
            started_at = datetime.now()

            def _on_eval_progress(result: TestResult) -> None:
                entry.eval_results.append(result)
                entry.completed = len(entry.eval_results)

            report = await do_batch_eval(
                eval_items, max_concurrency=max_concurrency, on_progress=_on_eval_progress,
            )
            finished_at = datetime.now()

            all_results = list(report.cases)

            # Generate zero-score results for unsubmitted cases
            if missed_items:
                for item in missed_items:
                    zero_result = TestResult(
                        id=item.id,
                        title=item.title or item.id,
                        user_type="eval_only",
                        target_type="eval_only",
                        eval_type=item.eval.evaluator if item.eval else "",
                        eval=EvalResult(result="fail", score=0.0, feedback="Not submitted, scored 0"),
                        cost=TestCost(),
                        start=finished_at,
                        end=finished_at,
                        tags=item.tags or [],
                        eval_config=item.eval.model_dump(exclude={"evaluator"}) if item.eval else None,
                    )
                    all_results.append(zero_result)
                logger.info("Added %d unsubmitted cases (scored 0)", len(missed_items))

            entry.eval_results = all_results
            entry.completed = len(all_results)

            # Build and save BenchReport
            bench_report = build_bench_report(
                test_results=all_results,
                benchmark_name=entry.benchmark,
                dataset_name=entry.dataset,
                runtime_target=None,
                max_concurrency=max_concurrency,
                started_at=started_at,
                finished_at=finished_at,
            )

            report_path = save_bench_report(
                bench_report, benchmark=entry.benchmark, dataset=entry.dataset
            )
            entry.report_path = str(report_path)
            entry.status = "completed"

            logger.info(
                "Eval-only task completed: %s (evaluated %d + unsubmitted %d = %d cases, avg_score=%.2f)",
                entry.task_id, len(report.cases), len(missed_items or []),
                len(all_results), bench_report.avg_score,
            )
            entry._release_heavy_data()

        except Exception as e:
            logger.error("Eval-only task failed: %s - %s", entry.task_id, e, exc_info=True)
            entry.status = "error"
            entry.error = str(e)
            entry._release_heavy_data()

    async def _run_session(
        self, entry: TaskEntry, max_concurrency: int, *, checkpoint_session_id: str | None = None
    ) -> None:
        """Run BatchSession in background and save BenchReport"""
        try:
            # Create/reuse checkpoint
            is_resume = checkpoint_session_id is not None
            cp_id = checkpoint_session_id or entry.task_id
            mgr = CheckpointManager(cp_id)

            if not is_resume:
                # New task: save checkpoint metadata
                meta = CheckpointMeta(
                    session_id=cp_id,
                    benchmark=entry.benchmark,
                    dataset=entry.dataset,
                    target_type=entry.runtime_target.type,
                    cli_overrides=entry.cli_overrides,
                    runtime_target=entry.runtime_target.model_dump(mode="json"),
                    case_ids=[c.id for c in entry.session.cases],
                    max_concurrency=max_concurrency,
                    started_at=datetime.now().isoformat(),
                    data_file_hash=entry.data_file_hash,
                )
                mgr.save_meta(meta)

            # Set progress callback (append checkpoint + clean up live files)
            original_on_progress = entry.session.on_progress

            def _checkpoint_callback(session: BatchSession, case_id: str) -> None:
                ctx = session.contexts.get(case_id)
                if ctx and ctx.result:
                    mgr.append_result(ctx.result)
                # Case completed, delete corresponding live file
                _delete_live_file(entry.task_id, case_id)
                if original_on_progress:
                    original_on_progress(session, case_id)

            entry.session.on_progress = _checkpoint_callback

            # Set per-turn dialogue callback (write live files)
            for ctx in entry.session.contexts.values():
                ctx.on_turn = lambda c, tid=entry.task_id: _write_live_file(tid, c)

            started_at = datetime.now()
            test_report = await entry.session.run()
            finished_at = datetime.now()

            # Merge resumed results
            all_results = list(entry.resumed_results) + list(test_report.cases)

            # Build BenchReport
            bench_report = build_bench_report(
                test_results=all_results,
                benchmark_name=entry.benchmark,
                dataset_name=entry.dataset,
                runtime_target=entry.runtime_target,
                max_concurrency=max_concurrency,
                started_at=started_at,
                finished_at=finished_at,
            )

            # Save report
            report_path = save_bench_report(
                bench_report, benchmark=entry.benchmark, dataset=entry.dataset
            )
            entry.report_path = str(report_path)

            # Clean up live directory
            _cleanup_live_dir(entry.task_id)

            if entry.session.cancel_event.is_set():
                entry.status = "cancelled"
                # Keep checkpoint on cancel to allow future resume
            else:
                entry.status = "completed"
                mgr.cleanup()
            entry._release_heavy_data()

        except Exception as e:
            # Keep checkpoint on error to allow future resume
            _cleanup_live_dir(entry.task_id)
            logger.error("Task execution failed: %s - %s (checkpoint preserved)", entry.task_id, e, exc_info=True)
            entry.status = "error"
            entry.error = str(e)
            entry._release_heavy_data()

    def cancel_task(self, task_id: str) -> None:
        """Cancel specified task"""
        entry = self._tasks.get(task_id)
        if not entry:
            raise KeyError(f"Task not found: {task_id}")
        if entry.session:
            entry.session.cancel()
        logger.info("Task cancellation requested: %s", task_id)

    def get_task(self, task_id: str) -> TaskEntry | None:
        """Get task entry"""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskEntry]:
        """List all tasks"""
        return list(self._tasks.values())

    def get_snapshot(self, task_id: str) -> dict[str, Any]:
        """Get task snapshot (JSON serializable)"""
        entry = self._tasks.get(task_id)
        if not entry:
            raise KeyError(f"Task not found: {task_id}")

        # Session already released (completed/error/cancelled or eval-only)
        if entry.session is None:
            # Rebuild cases summary from report file (no cases in memory after session release)
            cases, max_concurrency, runtime_target = _load_report_cases(entry.report_path)
            snap: dict[str, Any] = {
                "id": entry.task_id,
                "status": entry.status,
                "benchmark": entry.benchmark,
                "dataset": entry.dataset,
                "runtime_target": runtime_target,
                "created_at": entry.created_at.isoformat(),
                "report_path": entry.report_path,
                "total": entry.total or len(cases),
                "completed": entry.completed or len(cases),
                "cancelled": False,
                "cases": cases,
                "error": entry.error,
                "max_concurrency": max_concurrency,
            }
            # Stats grouped by tag (eval-only tasks computed from eval_results)
            if entry.eval_results:
                sbt = _compute_test_result_tag_stats(entry.eval_results)
                snap["stats_by_tag"] = sbt
                # report_summary — hma-web depends on this field to determine evaluation completion
                all_scores = [r.eval.score for r in entry.eval_results]
                snap["report_summary"] = {
                    "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                    "stats_by_tag": {tag: {"avg_score": s["avg_score"], "count": s["total"]} for tag, s in sbt.items()},
                }
            elif entry.report_path:
                stats_by_tag, report_summary = _load_report_summary(entry.report_path)
                snap["stats_by_tag"] = stats_by_tag
                if report_summary:
                    snap["report_summary"] = report_summary
            else:
                snap["stats_by_tag"] = {}
            return snap

        snapshot = entry.session.snapshot()
        snapshot["status"] = entry.status
        snapshot["benchmark"] = entry.benchmark
        snapshot["dataset"] = entry.dataset
        snapshot["runtime_target"] = entry.runtime_target.model_dump(mode="json") if entry.runtime_target else None
        snapshot["created_at"] = entry.created_at.isoformat()
        snapshot["report_path"] = entry.report_path
        snapshot["max_concurrency"] = entry.session.max_concurrency or None
        if entry.error:
            snapshot["error"] = entry.error
        # Resumed task: include previously completed count
        if entry.resumed_results:
            snapshot["resumed_count"] = len(entry.resumed_results)

        # Real-time tag-grouped stats (computed from completed cases)
        snapshot["stats_by_tag"] = _compute_snapshot_tag_stats(snapshot.get("cases", {}))

        return snapshot

    def list_checkpoints(self, benchmark: str | None = None) -> list[CheckpointMeta]:
        """List resumable checkpoints"""
        # Exclude checkpoints for currently running tasks
        # Note: for resumed tasks task_id != checkpoint_session_id, filter by checkpoint_session_id
        running_checkpoint_ids = {
            e.checkpoint_session_id for e in self._tasks.values() if e.status == "running"
        }
        checkpoints = CheckpointManager.find_checkpoints(benchmark=benchmark)
        return [cp for cp in checkpoints if cp.session_id not in running_checkpoint_ids]

    def cancel_all(self) -> None:
        """Cancel all active tasks on shutdown"""
        for entry in self._tasks.values():
            if entry.status == "running" and entry.session:
                entry.session.cancel()


def _compute_snapshot_tag_stats(cases: dict[str, dict]) -> dict[str, dict]:
    """Compute tag-grouped statistics from snapshot cases"""
    tag_data: dict[str, list[dict]] = {}
    for info in cases.values():
        if info.get("eval_result") is None:
            continue
        for tag in info.get("tags", []):
            tag_data.setdefault(tag, []).append(info)

    stats: dict[str, dict] = {}
    for tag, items in tag_data.items():
        tag_pass = sum(1 for i in items if i.get("eval_result") == "pass")
        tag_fail = sum(1 for i in items if i.get("eval_result") == "fail")
        tag_judged = tag_pass + tag_fail
        scores = [i["score"] for i in items if i.get("score") is not None]
        stats[tag] = {
            "total": len(items),
            "pass_count": tag_pass,
            "fail_count": tag_fail,
            "pass_rate": tag_pass / tag_judged if tag_judged else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }
    return stats


def _compute_test_result_tag_stats(results: list[TestResult]) -> dict[str, dict]:
    """Compute tag-grouped statistics from TestResult list"""
    tag_data: dict[str, list[TestResult]] = {}
    for result in results:
        for tag in result.tags:
            tag_data.setdefault(tag, []).append(result)

    stats: dict[str, dict] = {}
    for tag, items in tag_data.items():
        tag_pass = sum(1 for item in items if item.eval.result == "pass")
        tag_fail = sum(1 for item in items if item.eval.result == "fail")
        tag_judged = tag_pass + tag_fail
        stats[tag] = {
            "total": len(items),
            "pass_count": tag_pass,
            "fail_count": tag_fail,
            "pass_rate": tag_pass / tag_judged if tag_judged else 0.0,
            "avg_score": sum(item.eval.score for item in items) / len(items) if items else 0.0,
        }
    return stats


def _load_report_cases(report_path: str | None) -> tuple[dict[str, dict], int | None, dict | None]:
    """Rebuild cases summary dict from report file, used for snapshots after session release.

    Returns: (cases_dict, max_concurrency, runtime_target)
    """
    if not report_path:
        return {}, None, None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, None, None

    cases: dict[str, dict] = {}
    for c in report_data.get("cases", []):
        cid = c.get("id", "")
        ev = c.get("eval") or {}
        cases[cid] = {
            "status": "completed",
            "title": c.get("title", ""),
            "user_type": c.get("user_type", ""),
            "target_type": c.get("target_type", ""),
            "eval_type": c.get("eval_type", ""),
            "score": ev.get("score"),
            "eval_result": ev.get("result"),
            "cost": c.get("cost"),
            "tags": c.get("tags", []),
            "turn": 0,
        }
    max_concurrency = report_data.get("max_concurrency")
    runtime_target = report_data.get("runtime_target")
    return cases, max_concurrency, runtime_target


def _load_report_summary(report_path: str | None) -> tuple[dict[str, dict], dict[str, Any] | None]:
    """Read stats_by_tag / report_summary from report file as fallback when not in memory"""
    if not report_path:
        return {}, None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, None

    raw_stats = report_data.get("stats_by_tag")
    if not isinstance(raw_stats, dict):
        return {}, None

    stats_by_tag = {
        str(tag): value
        for tag, value in raw_stats.items()
        if isinstance(value, dict)
    }
    report_summary = {
        "avg_score": float(report_data.get("avg_score") or 0.0),
        "stats_by_tag": {
            tag: {
                "avg_score": float(value.get("avg_score") or 0.0),
                "count": int(value.get("total") or value.get("count") or 0),
            }
            for tag, value in stats_by_tag.items()
        },
    }
    return stats_by_tag, report_summary


# ==================== Live file management ====================


def _write_live_file(task_id: str, ctx: CaseContext) -> None:
    """Write current dialogue state to file after each turn"""
    task_dir = LIVE_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    test_memory = [m.model_dump(mode="json") for m in ctx.test_agent.memory_list] if ctx.test_agent else []
    target_memory = [m.model_dump(mode="json") for m in ctx.target_agent.memory_list] if ctx.target_agent else []

    # Preloaded history (HealthBench multi-turn etc.), aligned with report format
    history = []
    if ctx.test_agent and ctx.test_agent.history:
        history = [{"type": m.type, "content": m.content} for m in ctx.test_agent.history]

    data = {
        "id": ctx.case_id,
        "eval": {
            "result": None,
            "score": None,
            "feedback": None,
            "trace": {
                "history": history,
                "test_memory": test_memory,
                "target_memory": target_memory,
            },
        },
        "_live": True,
    }

    file_path = task_dir / f"{ctx.case_id}.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False, default=str))


def read_live_file(task_id: str, case_id: str) -> dict | None:
    """Read live file for a running case"""
    file_path = LIVE_DIR / task_id / f"{case_id}.json"
    if file_path.exists():
        try:
            return json.loads(file_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _delete_live_file(task_id: str, case_id: str) -> None:
    """Delete live file after case completion"""
    file_path = LIVE_DIR / task_id / f"{case_id}.json"
    file_path.unlink(missing_ok=True)


def _cleanup_live_dir(task_id: str) -> None:
    """Clean up entire live directory after task ends"""
    task_dir = LIVE_DIR / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)


# Global singleton
task_manager = TaskManager()
