"""PrepareManager — manages background execution of benchmark prepare scripts

Benchmarks declaring a "prepare" field (Python module path) in metadata.json
are automatically executed asynchronously on Web UI startup. Benchmarks are
unavailable for task creation while preparation is in progress.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sentry_sdk

from evaluator.utils.benchmark_reader import _DATA_DIR, _read_metadata

logger = logging.getLogger(__name__)

# Hang 时长上限：超过则 kill 子进程并标记 error。可通过 env 覆盖。
PREPARE_TIMEOUT_SECONDS = int(os.environ.get("PREPARE_TIMEOUT_SECONDS", "600"))


def _report_to_sentry(
    entry: "PrepareEntry",
    *,
    kind: str,
    detail: str,
    exc: BaseException | None = None,
) -> None:
    """把 prepare 失败/超时主动上报到 Sentry。

    若 sentry_sdk 未初始化（缺 DSN），capture_* 调用会被 SDK 静默丢弃，无副作用。
    """
    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("prepare.benchmark", entry.benchmark)
            scope.set_tag("prepare.kind", kind)
            scope.set_extra("prepare.module_path", entry.module_path)
            scope.set_extra("prepare.started_at", entry.started_at.isoformat())
            scope.set_extra("prepare.detail", detail[-2000:])
            if exc is not None:
                sentry_sdk.capture_exception(exc)
            else:
                sentry_sdk.capture_message(
                    f"prepare {kind}: {entry.benchmark} - {detail[:200]}",
                    level="error",
                )
    except Exception:
        logger.exception("Sentry capture failed (ignored)")


@dataclass
class PrepareEntry:
    """Preparation status for a single benchmark"""

    benchmark: str
    module_path: str  # e.g. "generator.theta_benchmark.build_datasets"
    status: str = "running"  # running | completed | error
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    error: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class PrepareManager:
    """Manages benchmark prepare scripts (executed on Web startup, process-isolated)"""

    def __init__(self) -> None:
        self._entries: dict[str, PrepareEntry] = {}

    # ==================== Query ====================

    def get_status(self, benchmark: str) -> PrepareEntry | None:
        return self._entries.get(benchmark)

    def get_all_statuses(self) -> dict[str, PrepareEntry]:
        return dict(self._entries)

    def is_preparing(self, benchmark: str) -> bool:
        entry = self._entries.get(benchmark)
        return entry is not None and entry.status == "running"

    # ==================== Startup ====================

    async def start_all(self) -> None:
        """Scan all benchmarks and start scripts with a prepare field"""
        if not _DATA_DIR.is_dir():
            return

        for bench_dir in sorted(_DATA_DIR.iterdir()):
            if not bench_dir.is_dir() or bench_dir.name.startswith((".", "_")):
                continue
            metadata = _read_metadata(bench_dir)
            prepare_module = metadata.get("prepare")
            if not prepare_module:
                continue
            self._start_one(bench_dir.name, prepare_module)

    def _start_one(self, benchmark: str, module_path: str) -> None:
        entry = PrepareEntry(benchmark=benchmark, module_path=module_path)
        self._entries[benchmark] = entry
        entry._task = asyncio.create_task(self._run_script(entry))
        logger.info("Starting prepare script: %s (%s)", benchmark, module_path)

    async def _run_script(self, entry: PrepareEntry) -> None:
        """Execute prepare script as a subprocess"""
        try:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                entry.module_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[3]),  # project root
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=PREPARE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                # 子进程 hang 超时：kill + 收尾，状态置 error，并上报 Sentry
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                except asyncio.TimeoutError:
                    stdout, stderr = b"", b""

                entry.finished_at = datetime.now()
                elapsed = (entry.finished_at - entry.started_at).total_seconds()
                entry.status = "error"
                err_tail = (stderr.decode("utf-8", errors="replace")[-1000:] if stderr else "").strip()
                out_tail = (stdout.decode("utf-8", errors="replace")[-1000:] if stdout else "").strip()
                entry.error = (
                    f"timeout after {PREPARE_TIMEOUT_SECONDS}s\n"
                    f"--- stderr tail ---\n{err_tail or '(empty)'}\n"
                    f"--- stdout tail ---\n{out_tail or '(empty)'}"
                )
                logger.error(
                    "Prepare script timeout: %s (%.1fs, limit=%ds) - stderr_tail=%r",
                    entry.benchmark, elapsed, PREPARE_TIMEOUT_SECONDS, err_tail[-300:],
                )
                _report_to_sentry(entry, kind="timeout", detail=entry.error)
                return

            entry.finished_at = datetime.now()
            elapsed = (entry.finished_at - entry.started_at).total_seconds()

            if proc.returncode == 0:
                entry.status = "completed"
                logger.info("Prepare script done: %s (%.1fs)", entry.benchmark, elapsed)
                if stdout:
                    for line in stdout.decode("utf-8", errors="replace").strip().splitlines()[-5:]:
                        logger.debug("[%s stdout] %s", entry.benchmark, line)
            else:
                entry.status = "error"
                err_msg = stderr.decode(errors="replace")[-1000:] if stderr else f"exit code {proc.returncode}"
                entry.error = err_msg
                logger.error(
                    "Prepare script failed: %s (exit=%d, %.1fs) - %s",
                    entry.benchmark, proc.returncode, elapsed, err_msg,
                )
                _report_to_sentry(
                    entry,
                    kind="non-zero-exit",
                    detail=f"exit={proc.returncode}\n{err_msg}",
                )

        except Exception as e:
            entry.finished_at = datetime.now()
            entry.status = "error"
            entry.error = str(e)
            logger.error("Prepare script exception: %s - %s", entry.benchmark, e, exc_info=True)
            _report_to_sentry(entry, kind="exception", detail=str(e), exc=e)

    # ==================== Shutdown ====================

    def cancel_all(self) -> None:
        """Cancel all running prepare scripts on shutdown"""
        for entry in self._entries.values():
            if entry.status == "running" and entry._task:
                entry._task.cancel()


# Global singleton
prepare_manager = PrepareManager()
