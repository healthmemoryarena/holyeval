"""TaskManager — 管理跑分任务生命周期（基于 BenchItem + BenchReport）"""

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
    """任务条目 — 包含 BatchSession 和运行时状态"""

    task_id: str
    # 当前任务实际写入/复用的检查点 ID：
    # - 新任务: 与 task_id 相同
    # - 恢复任务: 复用旧检查点 session_id（通常与 task_id 不同）
    checkpoint_session_id: str
    benchmark: str
    dataset: str
    runtime_target: TargetInfo | None = None  # eval-only 模式无 target
    session: BatchSession | None = None  # eval-only 模式无 BatchSession
    asyncio_task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running | completed | cancelled | error
    report_path: str | None = None
    error: str | None = None
    # 检查点相关
    cli_overrides: dict | None = None
    data_file_hash: str = ""
    resumed_results: list[TestResult] = field(default_factory=list)
    # eval-only 进度跟踪
    total: int = 0
    completed: int = 0
    eval_results: list[TestResult] = field(default_factory=list)

    def _release_heavy_data(self) -> None:
        """释放重对象，仅保留轻量元数据供 API 查询"""
        self.session = None
        self.eval_results = []
        self.resumed_results = []
        self.asyncio_task = None


class TaskManager:
    """内存中的任务管理器 — 基于 BenchItem 和 BenchReport"""

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
        """创建并启动跑分任务

        Args:
            benchmark: 评测类型（如 "healthbench"）
            dataset: 数据集名称（如 "sample"）
            spec: metadata.json 中的 TargetSpec
            cli_overrides: 前端传入的 target 覆盖参数（仅 editable 字段生效）
            ids: 逗号分隔的 ID（仅运行指定用例）
            limit: 最大用例数
            max_concurrency: 最大并发数

        Returns:
            TaskEntry
        """
        # 1. 加载 BenchItem（含 $ref 解析）
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)
        items = filter_bench_items(items, ids=ids, limit=limit)

        if not items:
            raise ValueError(f"过滤后无可执行用例 ({benchmark}/{dataset})")

        # 2. 构建全局 runtime target（用于日志和报告）
        runtime_target = resolve_runtime_target(spec, cli_overrides)
        logger.info(
            "创建任务: %s/%s, target=%s, model=%s, %d 条用例",
            benchmark,
            dataset,
            runtime_target.type,
            getattr(runtime_target, "model", "N/A"),
            len(items),
        )

        # 3. 转换 BenchItem → TestCase（含三层 target 合并）
        test_cases = []
        for item in items:
            try:
                test_case = bench_item_to_test_case(item, spec, cli_overrides)
                test_cases.append(test_case)
            except Exception as e:
                logger.error("用例 %s 转换失败: %s", item.id, e)
                raise ValueError(f"用例 {item.id} 转换失败: {e}") from e

        # 4. 创建 BatchSession
        session = BatchSession(
            cases=test_cases,
            max_concurrency=max_concurrency,
        )

        # 5. 创建 TaskEntry
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

        # 6. 后台运行
        entry.asyncio_task = asyncio.create_task(
            self._run_session(entry, max_concurrency)
        )

        logger.info("任务已创建: %s (%s/%s, %d 条用例)", session.id, benchmark, dataset, len(test_cases))
        return entry

    async def resume_task(
        self,
        session_id: str,
        spec: TargetSpec,
    ) -> TaskEntry:
        """恢复未完成的检查点任务

        Args:
            session_id: 检查点会话 ID
            spec: metadata.json 中的 TargetSpec

        Returns:
            TaskEntry
        """
        meta, completed_results = CheckpointManager.load(session_id)

        # 验证数据文件哈希
        path = resolve_data_path(meta.benchmark, meta.dataset)
        current_hash = CheckpointManager.compute_data_hash(path)
        if meta.data_file_hash and current_hash != meta.data_file_hash:
            logger.warning("数据文件已变更（hash: %s → %s）", meta.data_file_hash, current_hash)

        # 过滤已完成用例（被取消的重跑）
        cancelled_ids = {r.id for r in completed_results if r.eval.feedback == "用例被取消"}
        completed_ids = {r.id for r in completed_results} - cancelled_ids
        completed_results = [r for r in completed_results if r.id not in cancelled_ids]
        remaining_ids = set(meta.case_ids) - completed_ids

        logger.info(
            "恢复任务: %s/%s（已完成 %d/%d，剩余 %d）",
            meta.benchmark, meta.dataset, len(completed_ids), len(meta.case_ids), len(remaining_ids),
        )

        # 加载剩余用例（含 $ref 解析）
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)
        remaining_items = [item for item in items if item.id in remaining_ids]

        if not remaining_items:
            raise ValueError("所有用例已完成，无需恢复")

        runtime_target = resolve_runtime_target(spec, meta.cli_overrides)
        test_cases = []
        for item in remaining_items:
            try:
                test_cases.append(bench_item_to_test_case(item, spec, meta.cli_overrides))
            except Exception as e:
                logger.error("用例 %s 转换失败: %s", item.id, e)
                raise ValueError(f"用例 {item.id} 转换失败: {e}") from e

        # 创建 BatchSession（新 ID）
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

        # 后台运行（使用原检查点 session_id 继续追加结果）
        entry.asyncio_task = asyncio.create_task(
            self._run_session(entry, meta.max_concurrency, checkpoint_session_id=session_id)
        )

        logger.info(
            "恢复任务已创建: %s（原检查点 %s，剩余 %d 条用例）",
            session.id, session_id, len(test_cases),
        )
        return entry

    def check_eval(
        self,
        benchmark: str,
        dataset: str,
        results: list[ApiCallResult],
    ) -> dict:
        """校验提交结果与数据集的匹配情况（不启动评测）

        Returns:
            validation_info dict
        """
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)

        if not items:
            raise ValueError(f"数据集为空 ({benchmark}/{dataset})")

        dataset_ids = {item.id for item in items}
        submitted_ids = {r.id for r in results}
        matched_ids = dataset_ids & submitted_ids
        extra_ids = sorted(submitted_ids - dataset_ids)
        missed_ids = sorted(dataset_ids - submitted_ids)

        # 按 tag 统计难度分布（仅匹配的）
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
        """创建 eval-only 评测任务 — 只跑评测，不走对话循环

        Returns:
            (TaskEntry, validation_info)
        """
        import uuid

        # 1. 复用 check_eval 校验
        validation_info = self.check_eval(benchmark, dataset, results)

        matched_ids = set()
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)
        dataset_ids = {item.id for item in items}
        submitted_ids = {r.id for r in results}
        matched_ids = dataset_ids & submitted_ids

        if not matched_ids:
            raise ValueError(
                f"无匹配用例: 提交 {len(submitted_ids)} 条，数据集 {len(dataset_ids)} 条，0 条 ID 匹配"
            )

        # 2. 转换（仅匹配的）
        matched_results = [r for r in results if r.id in matched_ids]
        eval_items = api_result_to_eval_items(matched_results, items)

        # 未匹配的用例（数据集中有但用户未提交的），后续记 0 分
        missed_items = [item for item in items if item.id not in matched_ids]

        logger.info(
            "创建 eval-only 任务: %s/%s, 提交 %d, 匹配 %d, 缺失 %d (记 0 分)",
            benchmark, dataset, len(results), len(matched_ids), len(missed_items),
        )

        # 3. 创建 TaskEntry（total = 数据集全量）
        task_id = f"eval_{uuid.uuid4().hex[:12]}"
        entry = TaskEntry(
            task_id=task_id,
            checkpoint_session_id=task_id,
            benchmark=benchmark,
            dataset=dataset,
            total=len(items),
        )
        self._tasks[task_id] = entry

        # 4. 后台运行
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
        """后台执行 eval-only 批量评测并保存报告

        missed_items: 数据集中未匹配的 BenchItem，记 0 分合并到最终报告
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

            # 为未提交的用例生成 0 分结果
            if missed_items:
                for item in missed_items:
                    zero_result = TestResult(
                        id=item.id,
                        title=item.title or item.id,
                        user_type="eval_only",
                        target_type="eval_only",
                        eval_type=item.eval.evaluator if item.eval else "",
                        eval=EvalResult(result="fail", score=0.0, feedback="未提交，记 0 分"),
                        cost=TestCost(),
                        start=finished_at,
                        end=finished_at,
                        tags=item.tags or [],
                        eval_config=item.eval.model_dump(exclude={"evaluator"}) if item.eval else None,
                    )
                    all_results.append(zero_result)
                logger.info("补充 %d 条未提交用例（0 分）", len(missed_items))

            entry.eval_results = all_results
            entry.completed = len(all_results)

            # 构建并保存 BenchReport
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
                "eval-only 任务完成: %s (评测 %d + 未提交 %d = %d 条, avg_score=%.2f)",
                entry.task_id, len(report.cases), len(missed_items or []),
                len(all_results), bench_report.avg_score,
            )
            entry._release_heavy_data()

        except Exception as e:
            logger.error("eval-only 任务失败: %s - %s", entry.task_id, e, exc_info=True)
            entry.status = "error"
            entry.error = str(e)
            entry._release_heavy_data()

    async def _run_session(
        self, entry: TaskEntry, max_concurrency: int, *, checkpoint_session_id: str | None = None
    ) -> None:
        """后台执行 BatchSession 并保存 BenchReport"""
        try:
            # 创建/复用检查点
            is_resume = checkpoint_session_id is not None
            cp_id = checkpoint_session_id or entry.task_id
            mgr = CheckpointManager(cp_id)

            if not is_resume:
                # 新任务：保存检查点元数据
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

            # 设置进度回调（追加检查点 + 清理 live 文件）
            original_on_progress = entry.session.on_progress

            def _checkpoint_callback(session: BatchSession, case_id: str) -> None:
                ctx = session.contexts.get(case_id)
                if ctx and ctx.result:
                    mgr.append_result(ctx.result)
                # 用例完成，删除对应 live 文件
                _delete_live_file(entry.task_id, case_id)
                if original_on_progress:
                    original_on_progress(session, case_id)

            entry.session.on_progress = _checkpoint_callback

            # 设置每轮对话回调（写 live 文件）
            for ctx in entry.session.contexts.values():
                ctx.on_turn = lambda c, tid=entry.task_id: _write_live_file(tid, c)

            started_at = datetime.now()
            test_report = await entry.session.run()
            finished_at = datetime.now()

            # 合并恢复的结果
            all_results = list(entry.resumed_results) + list(test_report.cases)

            # 构建 BenchReport
            bench_report = build_bench_report(
                test_results=all_results,
                benchmark_name=entry.benchmark,
                dataset_name=entry.dataset,
                runtime_target=entry.runtime_target,
                max_concurrency=max_concurrency,
                started_at=started_at,
                finished_at=finished_at,
            )

            # 保存报告
            report_path = save_bench_report(
                bench_report, benchmark=entry.benchmark, dataset=entry.dataset
            )
            entry.report_path = str(report_path)

            # 清理 live 目录
            _cleanup_live_dir(entry.task_id)

            if entry.session.cancel_event.is_set():
                entry.status = "cancelled"
                # 取消时不清理检查点，允许后续恢复
            else:
                entry.status = "completed"
                mgr.cleanup()
            entry._release_heavy_data()

        except Exception as e:
            # 异常时不清理检查点，保留用于恢复
            _cleanup_live_dir(entry.task_id)
            logger.error("任务执行失败: %s - %s（检查点已保留）", entry.task_id, e, exc_info=True)
            entry.status = "error"
            entry.error = str(e)
            entry._release_heavy_data()

    def cancel_task(self, task_id: str) -> None:
        """取消指定任务"""
        entry = self._tasks.get(task_id)
        if not entry:
            raise KeyError(f"任务不存在: {task_id}")
        if entry.session:
            entry.session.cancel()
        logger.info("任务已请求取消: %s", task_id)

    def get_task(self, task_id: str) -> TaskEntry | None:
        """获取任务条目"""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskEntry]:
        """列出所有任务"""
        return list(self._tasks.values())

    def get_snapshot(self, task_id: str) -> dict[str, Any]:
        """获取任务快照（JSON 可序列化）"""
        entry = self._tasks.get(task_id)
        if not entry:
            raise KeyError(f"任务不存在: {task_id}")

        # eval-only 任务无 BatchSession
        if entry.session is None:
            snap: dict[str, Any] = {
                "id": entry.task_id,
                "status": entry.status,
                "benchmark": entry.benchmark,
                "dataset": entry.dataset,
                "runtime_target": None,
                "created_at": entry.created_at.isoformat(),
                "report_path": entry.report_path,
                "total": entry.total,
                "completed": entry.completed,
                "cancelled": False,
                "cases": {},
                "error": entry.error,
                "max_concurrency": None,
            }
            # 按 tag 分组统计（eval-only 任务从 eval_results 计算）
            if entry.eval_results:
                sbt = _compute_test_result_tag_stats(entry.eval_results)
                snap["stats_by_tag"] = sbt
                # report_summary — hma-web 依赖此字段判断评测完成
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
        # 恢复任务：补充已完成数量
        if entry.resumed_results:
            snapshot["resumed_count"] = len(entry.resumed_results)

        # 实时按 tag 分组统计（从已完成用例计算）
        snapshot["stats_by_tag"] = _compute_snapshot_tag_stats(snapshot.get("cases", {}))

        return snapshot

    def list_checkpoints(self, benchmark: str | None = None) -> list[CheckpointMeta]:
        """列出可恢复的检查点"""
        # 排除当前正在运行的任务对应的检查点
        # 注意：恢复任务时 task_id != checkpoint_session_id，应按 checkpoint_session_id 过滤
        running_checkpoint_ids = {
            e.checkpoint_session_id for e in self._tasks.values() if e.status == "running"
        }
        checkpoints = CheckpointManager.find_checkpoints(benchmark=benchmark)
        return [cp for cp in checkpoints if cp.session_id not in running_checkpoint_ids]

    def cancel_all(self) -> None:
        """关闭时取消所有活跃任务"""
        for entry in self._tasks.values():
            if entry.status == "running" and entry.session:
                entry.session.cancel()


def _compute_snapshot_tag_stats(cases: dict[str, dict]) -> dict[str, dict]:
    """从 snapshot cases 中计算按 tag 分组的统计信息"""
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
    """从 TestResult 列表计算按 tag 分组的统计信息"""
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


def _load_report_summary(report_path: str | None) -> tuple[dict[str, dict], dict[str, Any] | None]:
    """从报告文件读取 stats_by_tag / report_summary，作为内存态缺失时的兜底"""
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


# ==================== Live 文件管理 ====================


def _write_live_file(task_id: str, ctx: CaseContext) -> None:
    """每轮对话完成后将当前对话状态写入文件"""
    task_dir = LIVE_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    test_memory = [m.model_dump(mode="json") for m in ctx.test_agent.memory_list] if ctx.test_agent else []
    target_memory = [m.model_dump(mode="json") for m in ctx.target_agent.memory_list] if ctx.target_agent else []

    data = {
        "id": ctx.case_id,
        "eval": {
            "result": None,
            "score": None,
            "feedback": None,
            "trace": {
                "test_memory": test_memory,
                "target_memory": target_memory,
            },
        },
        "_live": True,
    }

    file_path = task_dir / f"{ctx.case_id}.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False, default=str))


def read_live_file(task_id: str, case_id: str) -> dict | None:
    """读取执行中用例的 live 文件"""
    file_path = LIVE_DIR / task_id / f"{case_id}.json"
    if file_path.exists():
        try:
            return json.loads(file_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _delete_live_file(task_id: str, case_id: str) -> None:
    """用例完成后删除对应 live 文件"""
    file_path = LIVE_DIR / task_id / f"{case_id}.json"
    file_path.unlink(missing_ok=True)


def _cleanup_live_dir(task_id: str) -> None:
    """任务结束后清理整个 live 目录"""
    task_dir = LIVE_DIR / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)


# 全局单例
task_manager = TaskManager()
