"""PrepareManager — 管理 benchmark 准备脚本的后台执行

metadata.json 中声明 "prepare" 字段（Python 模块路径）的 benchmark，
在 Web UI 启动时自动异步执行准备脚本。准备期间 benchmark 不可用于创建任务。
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
    """单个 benchmark 的准备状态"""

    benchmark: str
    module_path: str  # e.g. "generator.theta_benchmark.build_datasets"
    status: str = "running"  # running | completed | error
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    error: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class PrepareManager:
    """管理 benchmark 准备脚本（Web 启动时执行，进程隔离）"""

    def __init__(self) -> None:
        self._entries: dict[str, PrepareEntry] = {}

    # ==================== 查询 ====================

    def get_status(self, benchmark: str) -> PrepareEntry | None:
        return self._entries.get(benchmark)

    def get_all_statuses(self) -> dict[str, PrepareEntry]:
        return dict(self._entries)

    def is_preparing(self, benchmark: str) -> bool:
        entry = self._entries.get(benchmark)
        return entry is not None and entry.status == "running"

    # ==================== 启动 ====================

    async def start_all(self) -> None:
        """扫描所有 benchmark，启动含 prepare 字段的脚本"""
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
        logger.info("启动准备脚本: %s (%s)", benchmark, module_path)

    async def _run_script(self, entry: PrepareEntry) -> None:
        """以子进程方式执行准备脚本"""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                entry.module_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[3]),  # 项目根目录
            )
            stdout, stderr = await proc.communicate()

            entry.finished_at = datetime.now()
            elapsed = (entry.finished_at - entry.started_at).total_seconds()

            if proc.returncode == 0:
                entry.status = "completed"
                logger.info("准备脚本完成: %s (耗时 %.1fs)", entry.benchmark, elapsed)
                if stdout:
                    for line in stdout.decode().strip().splitlines()[-5:]:
                        logger.debug("[%s stdout] %s", entry.benchmark, line)
            else:
                entry.status = "error"
                err_msg = stderr.decode()[-1000:] if stderr else f"exit code {proc.returncode}"
                entry.error = err_msg
                logger.error("准备脚本失败: %s (exit=%d) - %s", entry.benchmark, proc.returncode, err_msg)

        except Exception as e:
            entry.finished_at = datetime.now()
            entry.status = "error"
            entry.error = str(e)
            logger.error("准备脚本异常: %s - %s", entry.benchmark, e, exc_info=True)

    # ==================== 关闭 ====================

    def cancel_all(self) -> None:
        """关闭时取消所有运行中的准备脚本"""
        for entry in self._entries.values():
            if entry.status == "running" and entry._task:
                entry._task.cancel()


# 全局单例
prepare_manager = PrepareManager()
