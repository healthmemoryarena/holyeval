"""TaskManager 单元测试"""

from unittest.mock import MagicMock

from evaluator.utils.checkpoint import CheckpointMeta
from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetInfo
from web.app.services.task_manager import TaskEntry, TaskManager


def _make_meta(session_id: str) -> CheckpointMeta:
    return CheckpointMeta(
        session_id=session_id,
        benchmark="healthbench",
        dataset="sample",
        target_type="llm_api",
        cli_overrides={"model": "gpt-4.1"},
        runtime_target={"type": "llm_api", "model": "gpt-4.1"},
        case_ids=["case_1"],
        max_concurrency=3,
        started_at="2026-03-05T10:00:00",
        data_file_hash="abcdef1234567890",
    )


def _make_entry(task_id: str, checkpoint_session_id: str, status: str = "running") -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        checkpoint_session_id=checkpoint_session_id,
        benchmark="healthbench",
        dataset="sample",
        runtime_target=LlmApiTargetInfo(type="llm_api", model="gpt-4.1"),
        session=MagicMock(),
        status=status,
    )


def test_list_checkpoints_excludes_running_task_checkpoint(monkeypatch):
    """运行中的新任务应从检查点列表中过滤"""
    mgr = TaskManager()
    mgr._tasks["task_1"] = _make_entry(task_id="task_1", checkpoint_session_id="task_1")

    monkeypatch.setattr(
        "web.app.services.task_manager.CheckpointManager.find_checkpoints",
        lambda benchmark=None: [_make_meta("task_1"), _make_meta("task_2")],
    )

    result = mgr.list_checkpoints()
    assert [cp.session_id for cp in result] == ["task_2"]


def test_list_checkpoints_excludes_running_resumed_checkpoint(monkeypatch):
    """运行中的恢复任务应按 checkpoint_session_id 过滤，而非 task_id"""
    mgr = TaskManager()
    # 恢复任务: task_id(新) != checkpoint_session_id(旧)
    mgr._tasks["task_new"] = _make_entry(task_id="task_new", checkpoint_session_id="cp_old")

    monkeypatch.setattr(
        "web.app.services.task_manager.CheckpointManager.find_checkpoints",
        lambda benchmark=None: [_make_meta("cp_old"), _make_meta("cp_other")],
    )

    result = mgr.list_checkpoints()
    assert [cp.session_id for cp in result] == ["cp_other"]
