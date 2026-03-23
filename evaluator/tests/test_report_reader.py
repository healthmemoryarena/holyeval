"""测试 report_reader — 报告列表 + 内容读取"""

import json

import pytest

from evaluator.core.bench_schema import BenchReport
from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetInfo
from evaluator.utils.report_reader import (
    _REPORT_FILENAME_RE,
    get_report_content,
    list_reports,
    save_bench_report,
)


# ============================================================
# 辅助: 构造 tmp report 目录
# ============================================================


def _make_report_dir(tmp_path, reports=None):
    """
    构造 benchmark/report 目录结构。

    reports = {
        "healthbench": ["sample_20260213_143012.json", "sample_20260212.json"],
        "extraction": ["simple_20260213_120000.json"],
    }
    """
    reports = reports or {
        "healthbench": ["sample_20260213_143012.json"],
    }
    for bench, files in reports.items():
        bench_dir = tmp_path / bench
        bench_dir.mkdir(parents=True)
        for fname in files:
            content = {"benchmark_name": bench, "dataset_name": fname.split("_")[0], "cases": []}
            (bench_dir / fname).write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")

    return tmp_path


# ============================================================
# _REPORT_FILENAME_RE
# ============================================================


def test_regex_timestamp_seconds():
    """匹配秒级精度: sample_20260213_143012.json"""
    m = _REPORT_FILENAME_RE.match("sample_20260213_143012.json")
    assert m is not None
    assert m.group(1) == "sample"
    assert m.group(2) == "20260213_143012"


def test_regex_timestamp_day():
    """匹配日级精度: sample_20260213.json"""
    m = _REPORT_FILENAME_RE.match("sample_20260213.json")
    assert m is not None
    assert m.group(1) == "sample"
    assert m.group(2) == "20260213"


def test_regex_no_match():
    """非报告文件不匹配"""
    assert _REPORT_FILENAME_RE.match("metadata.json") is None
    assert _REPORT_FILENAME_RE.match("readme.md") is None


def test_regex_complex_dataset_name():
    """dataset 名称含下划线"""
    m = _REPORT_FILENAME_RE.match("my_dataset_20260213_143012.json")
    assert m is not None
    assert m.group(1) == "my_dataset"
    assert m.group(2) == "20260213_143012"


# ============================================================
# list_reports
# ============================================================


def test_list_reports_basic(tmp_path, monkeypatch):
    """列出报告"""
    _make_report_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    result = list_reports()
    assert len(result) == 1
    r = result[0]
    assert r.benchmark == "healthbench"
    assert r.dataset == "sample"
    assert r.date == "20260213_143012"
    assert r.filename == "sample_20260213_143012.json"


def test_list_reports_empty(tmp_path, monkeypatch):
    """空目录"""
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)
    assert list_reports() == []


def test_list_reports_nonexistent_dir(tmp_path, monkeypatch):
    """report 目录不存在"""
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path / "nonexistent")
    assert list_reports() == []


def test_list_reports_multiple(tmp_path, monkeypatch):
    """多个 benchmark + 多个报告"""
    _make_report_dir(tmp_path, {
        "healthbench": ["sample_20260213_143012.json", "sample_20260212_100000.json"],
        "extraction": ["simple_20260213_120000.json"],
    })
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    result = list_reports()
    assert len(result) == 3
    benchmarks = {r.benchmark for r in result}
    assert benchmarks == {"healthbench", "extraction"}


def test_list_reports_skip_hidden(tmp_path, monkeypatch):
    """跳过 . 和 _ 开头的目录"""
    _make_report_dir(tmp_path)
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "_cache").mkdir()
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    result = list_reports()
    benchmarks = {r.benchmark for r in result}
    assert ".hidden" not in benchmarks
    assert "_cache" not in benchmarks


def test_list_reports_skip_invalid_filenames(tmp_path, monkeypatch):
    """跳过不符合命名规范的文件"""
    bench_dir = tmp_path / "test"
    bench_dir.mkdir()
    (bench_dir / "valid_20260213_143012.json").write_text("{}", encoding="utf-8")
    (bench_dir / "invalid_name.json").write_text("{}", encoding="utf-8")
    (bench_dir / "readme.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    result = list_reports()
    assert len(result) == 1
    assert result[0].filename == "valid_20260213_143012.json"


# ============================================================
# get_report_content
# ============================================================


def test_get_report_content_basic(tmp_path, monkeypatch):
    """读取报告内容"""
    _make_report_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    content = get_report_content("healthbench", "sample_20260213_143012.json")
    assert content["benchmark_name"] == "healthbench"


def test_get_report_content_not_found(tmp_path, monkeypatch):
    """报告不存在 → FileNotFoundError"""
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="报告不存在"):
        get_report_content("healthbench", "nonexistent.json")


# ============================================================
# save_bench_report
# ============================================================


def test_save_bench_report_basic(tmp_path, monkeypatch):
    """保存报告并验证文件内容"""
    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    report = BenchReport(
        benchmark_name="test",
        dataset_name="sample",
        runtime_target=LlmApiTargetInfo(type="llm_api", model="gpt-4.1"),
        cases=[],
        pass_count=0,
        fail_count=0,
        pass_rate=0.0,
        avg_score=0.0,
        total_duration_seconds=0.0,
    )
    path = save_bench_report(report, "test", "sample")

    assert path.exists()
    assert path.suffix == ".json"
    assert path.parent.name == "test"

    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["benchmark_name"] == "test"
    assert content["dataset_name"] == "sample"


def test_save_bench_report_filename_format(tmp_path, monkeypatch):
    """文件名格式: {dataset}_{target_label}_{YYYYMMDD_HHmmss}.json"""
    import re

    monkeypatch.setattr("evaluator.utils.report_reader._REPORT_DIR", tmp_path)

    report = BenchReport(
        benchmark_name="x",
        dataset_name="y",
        runtime_target=LlmApiTargetInfo(type="llm_api", model="gpt-4.1"),
        cases=[],
        pass_count=0,
        fail_count=0,
        pass_rate=0.0,
        avg_score=0.0,
        total_duration_seconds=0.0,
    )
    path = save_bench_report(report, "x", "y")
    assert re.match(r"y_gpt-4.1_\d{8}_\d{6}\.json", path.name)
