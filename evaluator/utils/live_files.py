"""Live file 管理 — 实时对话数据的文件持久化

每个运行中的任务对应 .live/{task_id}/ 目录，
每个进行中的用例对应一个 {case_id}.json 文件，每轮对话后更新。

CLI 和 Web 共享此模块，确保两种启动方式的 live file 格式一致。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator.core.orchestrator import CaseContext

LIVE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "report" / ".live"


def write_live_file(task_id: str, ctx: CaseContext) -> None:
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


def write_live_init(task_id: str, case_ids: list[str]) -> None:
    """Write initial live files for all cases (marks them as 'init' before dialogue starts)"""
    task_dir = LIVE_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        data = {
            "id": case_id,
            "eval": {
                "result": None,
                "score": None,
                "feedback": None,
                "trace": {"history": [], "test_memory": [], "target_memory": []},
            },
            "_live": True,
            "_init": True,
        }
        file_path = task_dir / f"{case_id}.json"
        file_path.write_text(json.dumps(data, ensure_ascii=False))


def read_live_file(task_id: str, case_id: str) -> dict | None:
    """Read live file for a running case"""
    file_path = LIVE_DIR / task_id / f"{case_id}.json"
    if file_path.exists():
        try:
            return json.loads(file_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def delete_live_file(task_id: str, case_id: str) -> None:
    """Delete live file after case completion"""
    file_path = LIVE_DIR / task_id / f"{case_id}.json"
    file_path.unlink(missing_ok=True)


def cleanup_live_dir(task_id: str) -> None:
    """Clean up entire live directory after task ends"""
    task_dir = LIVE_DIR / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)
