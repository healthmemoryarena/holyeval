"""测试 checkpoint — 检查点管理器的保存/加载/追加/清理/容错"""

from datetime import datetime

import pytest

from evaluator.core.schema import EvalResult, TestCost, TestResult
from evaluator.utils.checkpoint import CheckpointManager, CheckpointMeta


# ============================================================
# 辅助
# ============================================================


def _make_meta(**overrides) -> CheckpointMeta:
    defaults = {
        "session_id": "abc12345",
        "benchmark": "healthbench",
        "dataset": "sample",
        "target_type": "llm_api",
        "cli_overrides": {"model": "gpt-4.1"},
        "runtime_target": {"type": "llm_api", "model": "gpt-4.1"},
        "case_ids": ["case_1", "case_2", "case_3"],
        "max_concurrency": 3,
        "started_at": datetime.now().isoformat(),
        "data_file_hash": "abcdef1234567890",
    }
    defaults.update(overrides)
    return CheckpointMeta(**defaults)


def _make_result(case_id: str, score: float = 0.8, feedback: str = "OK") -> TestResult:
    now = datetime.now()
    return TestResult(
        id=case_id,
        eval=EvalResult(result="pass" if score >= 0.5 else "fail", score=score, feedback=feedback),
        cost=TestCost(),
        start=now,
        end=now,
    )


# ============================================================
# save_meta / load
# ============================================================


def test_save_and_load_meta(tmp_path):
    """保存元数据后可正确加载"""
    meta = _make_meta(session_id="test01")
    mgr = CheckpointManager("test01", checkpoint_dir=tmp_path)
    mgr.save_meta(meta)

    loaded_meta, results = CheckpointManager.load("test01", checkpoint_dir=tmp_path)
    assert loaded_meta.session_id == "test01"
    assert loaded_meta.benchmark == "healthbench"
    assert loaded_meta.case_ids == ["case_1", "case_2", "case_3"]
    assert results == []


def test_save_meta_atomic(tmp_path):
    """meta.json 应使用原子写入（os.replace）"""
    mgr = CheckpointManager("atomic", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="atomic"))

    meta_path = tmp_path / "atomic.meta.json"
    assert meta_path.exists()
    # 不应有残留的 .tmp 文件
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


# ============================================================
# append_result / load
# ============================================================


def test_append_and_load_results(tmp_path):
    """追加多条结果后可全部加载"""
    mgr = CheckpointManager("test02", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="test02"))

    r1 = _make_result("case_1", score=0.9)
    r2 = _make_result("case_2", score=0.7)
    r3 = _make_result("case_3", score=0.5)
    mgr.append_result(r1)
    mgr.append_result(r2)
    mgr.append_result(r3)

    _, results = CheckpointManager.load("test02", checkpoint_dir=tmp_path)
    assert len(results) == 3
    assert results[0].id == "case_1"
    assert results[1].id == "case_2"
    assert results[2].id == "case_3"
    assert results[0].eval.score == 0.9


def test_append_result_creates_file(tmp_path):
    """首次追加时自动创建 results.jsonl"""
    mgr = CheckpointManager("test03", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="test03"))

    results_path = tmp_path / "test03.results.jsonl"
    assert not results_path.exists()

    mgr.append_result(_make_result("case_1"))
    assert results_path.exists()


# ============================================================
# 容错: 截断的 JSONL
# ============================================================


def test_corrupted_last_line(tmp_path):
    """末行截断时跳过该行，正常加载其余结果"""
    mgr = CheckpointManager("test04", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="test04"))

    mgr.append_result(_make_result("case_1"))
    mgr.append_result(_make_result("case_2"))

    # 手动追加一行截断的 JSON
    results_path = tmp_path / "test04.results.jsonl"
    with open(results_path, "a") as f:
        f.write('{"id": "case_3", "eval": {"result": "pass", "scor\n')

    _, results = CheckpointManager.load("test04", checkpoint_dir=tmp_path)
    assert len(results) == 2  # 只有前两条有效


def test_missing_results_file(tmp_path):
    """meta.json 存在但 results.jsonl 不存在 → 返回空列表"""
    mgr = CheckpointManager("test05", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="test05"))

    meta, results = CheckpointManager.load("test05", checkpoint_dir=tmp_path)
    assert meta.session_id == "test05"
    assert results == []


# ============================================================
# cleanup
# ============================================================


def test_cleanup(tmp_path):
    """清理后两个文件都应被删除"""
    mgr = CheckpointManager("test06", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="test06"))
    mgr.append_result(_make_result("case_1"))

    assert (tmp_path / "test06.meta.json").exists()
    assert (tmp_path / "test06.results.jsonl").exists()

    mgr.cleanup()

    assert not (tmp_path / "test06.meta.json").exists()
    assert not (tmp_path / "test06.results.jsonl").exists()


def test_cleanup_idempotent(tmp_path):
    """重复清理不报错"""
    mgr = CheckpointManager("test07", checkpoint_dir=tmp_path)
    mgr.cleanup()  # 文件不存在也不报错
    mgr.cleanup()


# ============================================================
# find_checkpoints
# ============================================================


def test_find_checkpoints_all(tmp_path):
    """查找所有检查点"""
    mgr1 = CheckpointManager("cp1", checkpoint_dir=tmp_path)
    mgr1.save_meta(_make_meta(session_id="cp1", benchmark="healthbench", started_at="2026-03-03T10:00:00"))
    mgr2 = CheckpointManager("cp2", checkpoint_dir=tmp_path)
    mgr2.save_meta(_make_meta(session_id="cp2", benchmark="medcalc", started_at="2026-03-03T11:00:00"))

    result = CheckpointManager.find_checkpoints(checkpoint_dir=tmp_path)
    assert len(result) == 2
    # 按时间倒序
    assert result[0].session_id == "cp2"
    assert result[1].session_id == "cp1"


def test_find_checkpoints_filter_benchmark(tmp_path):
    """按 benchmark 过滤"""
    mgr1 = CheckpointManager("cp1", checkpoint_dir=tmp_path)
    mgr1.save_meta(_make_meta(session_id="cp1", benchmark="healthbench"))
    mgr2 = CheckpointManager("cp2", checkpoint_dir=tmp_path)
    mgr2.save_meta(_make_meta(session_id="cp2", benchmark="medcalc"))

    result = CheckpointManager.find_checkpoints(benchmark="medcalc", checkpoint_dir=tmp_path)
    assert len(result) == 1
    assert result[0].benchmark == "medcalc"


def test_find_checkpoints_filter_dataset(tmp_path):
    """按 dataset 过滤"""
    mgr1 = CheckpointManager("cp1", checkpoint_dir=tmp_path)
    mgr1.save_meta(_make_meta(session_id="cp1", dataset="sample"))
    mgr2 = CheckpointManager("cp2", checkpoint_dir=tmp_path)
    mgr2.save_meta(_make_meta(session_id="cp2", dataset="full"))

    result = CheckpointManager.find_checkpoints(dataset="full", checkpoint_dir=tmp_path)
    assert len(result) == 1
    assert result[0].dataset == "full"


def test_find_checkpoints_empty_dir(tmp_path):
    """空目录返回空列表"""
    assert CheckpointManager.find_checkpoints(checkpoint_dir=tmp_path) == []


def test_find_checkpoints_nonexistent_dir(tmp_path):
    """目录不存在返回空列表"""
    assert CheckpointManager.find_checkpoints(checkpoint_dir=tmp_path / "nonexistent") == []


# ============================================================
# completed_count
# ============================================================


def test_completed_count(tmp_path):
    """统计已完成用例数"""
    mgr = CheckpointManager("test08", checkpoint_dir=tmp_path)
    mgr.save_meta(_make_meta(session_id="test08"))
    mgr.append_result(_make_result("case_1"))
    mgr.append_result(_make_result("case_2"))

    count = CheckpointManager.completed_count("test08", checkpoint_dir=tmp_path)
    assert count == 2


def test_completed_count_no_results(tmp_path):
    """无 results 文件时返回 0"""
    count = CheckpointManager.completed_count("nonexistent", checkpoint_dir=tmp_path)
    assert count == 0


# ============================================================
# compute_data_hash
# ============================================================


def test_data_hash_deterministic(tmp_path):
    """相同内容哈希一致"""
    f = tmp_path / "data.jsonl"
    f.write_text('{"id": "1"}\n{"id": "2"}\n', encoding="utf-8")

    h1 = CheckpointManager.compute_data_hash(f)
    h2 = CheckpointManager.compute_data_hash(f)
    assert h1 == h2
    assert len(h1) == 16  # SHA-256 前 16 位


def test_data_hash_changes_on_modification(tmp_path):
    """内容变更后哈希不同"""
    f = tmp_path / "data.jsonl"
    f.write_text('{"id": "1"}\n', encoding="utf-8")
    h1 = CheckpointManager.compute_data_hash(f)

    f.write_text('{"id": "1"}\n{"id": "2"}\n', encoding="utf-8")
    h2 = CheckpointManager.compute_data_hash(f)
    assert h1 != h2


# ============================================================
# load 异常
# ============================================================


def test_load_nonexistent_meta(tmp_path):
    """meta.json 不存在 → FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        CheckpointManager.load("nonexistent", checkpoint_dir=tmp_path)
