"""Task Registry — 文件级任务注册表

CLI 和 Web 共享的任务发现机制。每个运行中或已完成的任务在 .tasks/ 目录写入一个 JSON 文件，
Web API 通过扫描该目录发现 CLI 启动的任务。

文件格式: benchmark/report/.tasks/{task_id}.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "report" / ".tasks"


class TaskRegistryEntry(BaseModel):
    """任务注册表条目"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="任务 ID")
    checkpoint_session_id: str = Field(description="检查点 session ID")
    benchmark: str = Field(description="评测类型")
    dataset: str = Field(description="数据集名称")
    runtime_target: dict[str, Any] | None = Field(None, description="运行时 target 配置")
    status: str = Field("running", description="任务状态 (running/completed/cancelled/error)")
    created_at: str = Field(description="创建时间 (ISO)")
    source: str = Field("cli", description="任务来源 (cli/web)")
    total: int = Field(0, description="总用例数")
    max_concurrency: int = Field(0, description="最大并发数")
    report_path: str | None = Field(None, description="报告文件路径")
    error: str | None = Field(None, description="错误信息")
    pid: int | None = Field(None, description="进程 PID（用于存活检测）")


def write_task_entry(entry: TaskRegistryEntry) -> None:
    """原子写入任务注册表条目"""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    data = entry.model_dump(mode="json")
    fd, tmp_path = tempfile.mkstemp(dir=TASKS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, TASKS_DIR / f"{entry.task_id}.json")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_task_entry(task_id: str) -> TaskRegistryEntry | None:
    """读取单个任务注册表条目"""
    path = TASKS_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return TaskRegistryEntry.model_validate_json(f.read())
    except Exception as e:
        logger.warning("无法解析任务注册表 %s: %s", path.name, e)
        return None


def list_task_entries(source: str | None = None) -> list[TaskRegistryEntry]:
    """列出所有任务注册表条目，按创建时间倒序"""
    if not TASKS_DIR.is_dir():
        return []

    entries: list[TaskRegistryEntry] = []
    for path in TASKS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = TaskRegistryEntry.model_validate_json(f.read())
            if source and entry.source != source:
                continue
            entries.append(entry)
        except Exception as e:
            logger.warning("无法解析任务注册表 %s: %s", path.name, e)

    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def cleanup_task_entry(task_id: str) -> None:
    """删除任务注册表条目"""
    path = TASKS_DIR / f"{task_id}.json"
    path.unlink(missing_ok=True)


def is_process_alive(pid: int) -> bool:
    """检查进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
