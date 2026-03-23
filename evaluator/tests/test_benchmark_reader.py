"""测试 benchmark_reader — 目录结构读取 + metadata 解析"""

import json

import pytest

from evaluator.utils.benchmark_reader import (
    _extract_title,
    _read_metadata_description,
    _resolve_refs,
    get_case_by_id,
    get_dataset_detail,
    list_benchmarks,
    load_bench_items,
    load_benchmark,
    resolve_data_path,
)


# ============================================================
# 辅助: 构造 tmp benchmark/data 目录
# ============================================================


def _make_data_dir(tmp_path, benchmarks=None):
    """
    构造 benchmark/data 目录结构。

    benchmarks = {
        "mybench": {
            "description": "...",
            "datasets": {
                "sample": [{"id": "t1", ...}, ...]
            }
        }
    }
    """
    benchmarks = benchmarks or {
        "mybench": {
            "description": "测试描述",
            "datasets": {
                "sample": [
                    {"id": "t1", "user": {"goal": "g", "type": "auto", "max_turns": 3}, "eval": {"evaluator": "semantic"}, "tags": ["tag1"]},
                    {"id": "t2", "user": {"type": "auto", "goal": "g2", "strict_inputs": ["你好"]}, "eval": {"evaluator": "keyword"}, "tags": ["tag1", "tag2"]},
                ],
            },
        }
    }

    for bench_name, bench_config in benchmarks.items():
        bench_dir = tmp_path / bench_name
        bench_dir.mkdir(parents=True)

        # metadata.json
        meta = {"description": bench_config.get("description", "")}
        (bench_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        # JSONL files
        for ds_name, items in bench_config.get("datasets", {}).items():
            lines = [json.dumps(item, ensure_ascii=False) for item in items]
            (bench_dir / f"{ds_name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return tmp_path


# ============================================================
# _read_metadata_description
# ============================================================


def test_read_metadata_description(tmp_path):
    """从 metadata.json 读取 description"""
    _make_data_dir(tmp_path)
    desc = _read_metadata_description(tmp_path / "mybench")
    assert desc == "测试描述"


def test_read_metadata_description_missing(tmp_path):
    """metadata.json 不存在 → 空字符串"""
    assert _read_metadata_description(tmp_path) == ""


def test_read_metadata_description_no_field(tmp_path):
    """metadata.json 无 description 字段 → 空字符串"""
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert _read_metadata_description(tmp_path) == ""


# ============================================================
# list_benchmarks
# ============================================================


def test_list_benchmarks_basic(tmp_path, monkeypatch):
    """列出 benchmark 及其 dataset"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    result = list_benchmarks()
    assert len(result) == 1
    bm = result[0]
    assert bm.name == "mybench"
    assert bm.description == "测试描述"
    assert len(bm.datasets) == 1
    assert bm.datasets[0].name == "sample"
    assert bm.datasets[0].case_count == 2


def test_list_benchmarks_empty(tmp_path, monkeypatch):
    """空目录 → 空列表"""
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)
    assert list_benchmarks() == []


def test_list_benchmarks_skip_hidden(tmp_path, monkeypatch):
    """跳过 . 和 _ 开头的目录"""
    _make_data_dir(tmp_path)
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "_internal").mkdir()
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    result = list_benchmarks()
    names = [b.name for b in result]
    assert ".hidden" not in names
    assert "_internal" not in names


def test_list_benchmarks_multiple(tmp_path, monkeypatch):
    """多个 benchmark"""
    _make_data_dir(tmp_path, {
        "alpha": {"description": "A", "datasets": {"d1": [{"id": "a1", "user": {"type": "auto", "goal": "g", "max_turns": 1}, "eval": {"evaluator": "semantic"}}]}},
        "beta": {"description": "B", "datasets": {"d2": [{"id": "b1", "user": {"type": "auto", "goal": "g", "max_turns": 1}, "eval": {"evaluator": "semantic"}}]}},
    })
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    result = list_benchmarks()
    names = sorted(b.name for b in result)
    assert names == ["alpha", "beta"]


# ============================================================
# get_dataset_detail
# ============================================================


def test_get_dataset_detail_basic(tmp_path, monkeypatch):
    """获取 dataset 详情"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    detail = get_dataset_detail("mybench", "sample")
    assert detail.benchmark == "mybench"
    assert detail.dataset == "sample"
    assert detail.case_count == 2
    assert len(detail.cases_preview) == 2
    assert len(detail.case_summaries) == 2
    assert detail.description == "测试描述"


def test_get_dataset_detail_tags(tmp_path, monkeypatch):
    """标签分布统计"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    detail = get_dataset_detail("mybench", "sample")
    assert detail.tag_distribution["tag1"] == 2
    assert detail.tag_distribution["tag2"] == 1


def test_get_dataset_detail_case_summaries(tmp_path, monkeypatch):
    """CaseSummary 字段提取"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    detail = get_dataset_detail("mybench", "sample")
    s0 = detail.case_summaries[0]
    assert s0.id == "t1"
    assert s0.user_type == "auto"
    assert s0.evaluator == "semantic"


def test_get_dataset_detail_not_found(tmp_path, monkeypatch):
    """dataset 不存在"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        get_dataset_detail("mybench", "nonexistent")


def test_get_dataset_detail_preview_limit(tmp_path, monkeypatch):
    """preview_limit 截断"""
    items = [{"id": f"t{i}", "user": {"type": "auto", "goal": "g", "max_turns": 1}, "eval": {"evaluator": "semantic"}} for i in range(20)]
    _make_data_dir(tmp_path, {"bench": {"description": "", "datasets": {"big": items}}})
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    detail = get_dataset_detail("bench", "big", preview_limit=5)
    assert len(detail.cases_preview) == 5
    assert detail.case_count == 20
    assert len(detail.case_summaries) == 20


# ============================================================
# get_case_by_id
# ============================================================


def test_get_case_by_id_found(tmp_path, monkeypatch):
    """按 ID 查找 case"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    case = get_case_by_id("mybench", "sample", "t2")
    assert case["id"] == "t2"


def test_get_case_by_id_not_found(tmp_path, monkeypatch):
    """ID 不存在 → KeyError"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(KeyError, match="用例不存在"):
        get_case_by_id("mybench", "sample", "nonexistent")


def test_get_case_by_id_dataset_not_found(tmp_path, monkeypatch):
    """dataset 不存在 → FileNotFoundError"""
    _make_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        get_case_by_id("mybench", "bad_dataset", "t1")


# ============================================================
# _extract_title
# ============================================================


def test_extract_title_from_title_field():
    """从 title 字段提取标题"""
    obj = {"title": "HealthBench — headache_migraine"}
    assert _extract_title(obj) == "HealthBench — headache_migraine"


def test_extract_title_from_strict_inputs():
    """从 strict_inputs 提取标题"""
    obj = {"user": {"strict_inputs": ["测试输入"]}}
    assert _extract_title(obj) == "测试输入"


def test_extract_title_from_goal():
    """从 goal 提取标题"""
    obj = {"user": {"goal": "咨询头痛问题"}}
    assert _extract_title(obj) == "咨询头痛问题"


def test_extract_title_empty():
    """无可提取字段 → 空字符串"""
    assert _extract_title({}) == ""


def test_extract_title_truncated():
    """长文本截断到 100 字符"""
    obj = {"user": {"goal": "A" * 200}}
    assert len(_extract_title(obj)) == 100


# ============================================================
# _resolve_refs
# ============================================================


def test_resolve_refs_basic():
    """$ref 正常替换"""
    params = {"my_history": [{"role": "user", "content": "hello"}]}
    raw = {"id": "t1", "history": {"$ref": "my_history"}, "title": "test"}
    result = _resolve_refs(raw, params)
    assert result["history"] == [{"role": "user", "content": "hello"}]
    assert result["id"] == "t1"  # 其他字段不受影响


def test_resolve_refs_missing_key(caplog):
    """$ref 引用不存在的 key → 警告 + 保持原值"""
    params = {"existing": []}
    raw = {"id": "t1", "history": {"$ref": "nonexistent"}}
    result = _resolve_refs(raw, params)
    assert result["history"] == {"$ref": "nonexistent"}  # 保持原值
    assert "nonexistent" in caplog.text


def test_resolve_refs_no_params():
    """params 为空 → 直接返回，不处理"""
    raw = {"id": "t1", "history": {"$ref": "anything"}}
    result = _resolve_refs(raw, {})
    assert result["history"] == {"$ref": "anything"}


def test_resolve_refs_not_ref_dict():
    """普通 dict（非 $ref）不受影响"""
    params = {"key": "value"}
    raw = {"id": "t1", "eval": {"evaluator": "semantic", "threshold": 0.8}}
    result = _resolve_refs(raw, params)
    assert result["eval"] == {"evaluator": "semantic", "threshold": 0.8}


def test_resolve_refs_multiple_keys_dict():
    """含 $ref 但有其他键的 dict 不替换"""
    params = {"my_history": []}
    raw = {"id": "t1", "history": {"$ref": "my_history", "extra": "field"}}
    result = _resolve_refs(raw, params)
    assert result["history"] == {"$ref": "my_history", "extra": "field"}  # 不替换


# ============================================================
# load_bench_items（含 $ref 集成测试）
# ============================================================


def test_load_bench_items_with_params(tmp_path):
    """load_bench_items 传入 params 时解析 $ref"""
    params = {"shared_history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    items_data = [
        {"id": "t1", "title": "A", "user": {"type": "manual", "strict_inputs": ["q1"]}, "eval": {"evaluator": "semantic"}, "history": {"$ref": "shared_history"}},
        {"id": "t2", "title": "B", "user": {"type": "manual", "strict_inputs": ["q2"]}, "eval": {"evaluator": "semantic"}, "history": {"$ref": "shared_history"}},
    ]
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item) for item in items_data), encoding="utf-8")

    items = load_bench_items(jsonl_path, params=params)
    assert len(items) == 2
    assert items[0].history == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert items[1].history == items[0].history


def test_load_benchmark_with_params(tmp_path, monkeypatch):
    """load_benchmark 自动从 metadata.json 读取 params 并解析 $ref"""
    metadata = {
        "description": "test",
        "target": {"type": "llm_api", "fields": {"model": {"default": "gpt-4.1", "editable": True, "required": True}}},
        "params": {
            "user_a_history": [{"role": "user", "content": "I have diabetes"}, {"role": "assistant", "content": "Noted"}],
        },
    }
    items_data = [
        {"id": "t1", "title": "A", "user": {"type": "manual", "strict_inputs": ["q1"]}, "eval": {"evaluator": "semantic"}, "history": {"$ref": "user_a_history"}},
    ]
    _make_bench_data_dir(tmp_path, name="refbench", datasets={"sample": items_data}, metadata=metadata)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    bm = load_benchmark("refbench", "sample")
    assert bm.items[0].history == [{"role": "user", "content": "I have diabetes"}, {"role": "assistant", "content": "Noted"}]


# ============================================================
# load_bench_items
# ============================================================


def _make_bench_data_dir(tmp_path, name="mybench", datasets=None, metadata=None):
    """在 tmp_path 下创建 benchmark 数据目录，返回 bench_dir"""
    bench_dir = tmp_path / name
    bench_dir.mkdir()

    meta = metadata or {"description": "test desc", "target": {"type": "llm_api", "fields": {"model": {"default": "gpt-4.1", "editable": True, "required": True}}}}
    (bench_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    datasets = datasets or {
        "sample": [
            {"id": "t1", "title": "A", "user": {"type": "auto", "goal": "g1", "max_turns": 3}, "eval": {"evaluator": "semantic"}},
            {"id": "t2", "title": "B", "user": {"type": "auto", "goal": "g2", "max_turns": 3}, "eval": {"evaluator": "semantic"}},
        ]
    }
    for ds_name, items in datasets.items():
        lines = [json.dumps(item, ensure_ascii=False) for item in items]
        (bench_dir / f"{ds_name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return bench_dir


def test_load_bench_items_basic(tmp_path):
    """正常加载 JSONL"""
    bench_dir = _make_bench_data_dir(tmp_path)
    items = load_bench_items(bench_dir / "sample.jsonl")
    assert len(items) == 2
    assert items[0].id == "t1"
    assert items[1].id == "t2"


def test_load_bench_items_skip_empty_lines(tmp_path):
    """跳过空行"""
    bench_dir = _make_bench_data_dir(tmp_path)
    jsonl_path = bench_dir / "sample.jsonl"
    content = jsonl_path.read_text(encoding="utf-8")
    jsonl_path.write_text("\n\n" + content + "\n\n", encoding="utf-8")

    items = load_bench_items(jsonl_path)
    assert len(items) == 2


def test_load_bench_items_file_not_found():
    """文件不存在"""
    with pytest.raises(FileNotFoundError):
        load_bench_items("/nonexistent/path.jsonl")


def test_load_bench_items_invalid_json(tmp_path):
    """JSON 格式错误"""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="解析失败"):
        load_bench_items(bad_file)


# ============================================================
# load_benchmark
# ============================================================


def test_load_benchmark_basic(tmp_path, monkeypatch):
    """正常加载 BenchMark"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    bm = load_benchmark("mybench", "sample")
    assert bm.name == "mybench/sample"
    assert bm.description == "test desc"
    assert bm.total_count == 2
    assert len(bm.target) == 1
    assert bm.target[0].type == "llm_api"
    assert "model" in bm.target[0].fields


def test_load_benchmark_no_metadata(tmp_path, monkeypatch):
    """metadata.json 不存在时，description 为空，target 为 None"""
    bench_dir = tmp_path / "nobench"
    bench_dir.mkdir()
    items = [{"id": "t1", "title": "A", "user": {"type": "auto", "goal": "g", "max_turns": 1}, "eval": {"evaluator": "semantic"}}]
    (bench_dir / "s.jsonl").write_text(json.dumps(items[0], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    bm = load_benchmark("nobench", "s")
    assert bm.description == ""
    assert bm.target == []


def test_load_benchmark_invalid_benchmark(tmp_path, monkeypatch):
    """不存在的 benchmark 类型"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="评测类型不存在"):
        load_benchmark("nonexistent", "sample")


def test_load_benchmark_invalid_dataset(tmp_path, monkeypatch):
    """不存在的 dataset"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="数据集不存在"):
        load_benchmark("mybench", "nonexistent")


def test_load_benchmark_error_lists_available(tmp_path, monkeypatch):
    """错误信息中列出可用的选项"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="mybench"):
        load_benchmark("nonexistent", "x")

    with pytest.raises(FileNotFoundError, match="sample"):
        load_benchmark("mybench", "nonexistent")


# ============================================================
# resolve_data_path
# ============================================================


def test_resolve_data_path_basic(tmp_path, monkeypatch):
    """正常解析路径"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    path = resolve_data_path("mybench", "sample")
    assert path.name == "sample.jsonl"
    assert path.exists()


def test_resolve_data_path_invalid_benchmark(tmp_path, monkeypatch):
    """不存在的 benchmark"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="评测类型不存在"):
        resolve_data_path("bad", "sample")


def test_resolve_data_path_invalid_dataset(tmp_path, monkeypatch):
    """不存在的 dataset"""
    _make_bench_data_dir(tmp_path)
    monkeypatch.setattr("evaluator.utils.benchmark_reader._DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="数据集不存在"):
        resolve_data_path("mybench", "bad")
