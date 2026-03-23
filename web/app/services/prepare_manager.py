"""PrepareManager — manages background execution of benchmark prepare scripts

Benchmarks declaring a "prepare" field (Python module path) in metadata.json
are automatically executed asynchronously on Web UI startup. Benchmarks are
unavailable for task creation while preparation is in progress.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from evaluator.utils.benchmark_reader import _DATA_DIR, _read_metadata

logger = logging.getLogger(__name__)


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
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                entry.module_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[3]),  # project root
            )
            stdout, stderr = await proc.communicate()

            entry.finished_at = datetime.now()
            elapsed = (entry.finished_at - entry.started_at).total_seconds()

            if proc.returncode == 0:
                entry.status = "completed"
                logger.info("Prepare script done: %s (%.1fs)", entry.benchmark, elapsed)
                if stdout:
                    for line in stdout.decode().strip().splitlines()[-5:]:
                        logger.debug("[%s stdout] %s", entry.benchmark, line)
            else:
                entry.status = "error"
                err_msg = stderr.decode()[-1000:] if stderr else f"exit code {proc.returncode}"
                entry.error = err_msg
                logger.error("Prepare script failed: %s (exit=%d) - %s", entry.benchmark, proc.returncode, err_msg)

        except Exception as e:
            entry.finished_at = datetime.now()
            entry.status = "error"
            entry.error = str(e)
            logger.error("Prepare script exception: %s - %s", entry.benchmark, e, exc_info=True)

    # ==================== Shutdown ====================

    def cancel_all(self) -> None:
        """Cancel all running prepare scripts on shutdown"""
        for entry in self._entries.values():
            if entry.status == "running" and entry._task:
                entry._task.cancel()


# Global singleton
prepare_manager = PrepareManager()
