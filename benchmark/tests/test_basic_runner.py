"""测试 basic_runner — 过滤逻辑（纯逻辑，不涉及 LLM）

resolve_runtime_target / resolve_effective_target 的测试在 test_bench_schema.py 中。
"""

import pytest

from evaluator.utils.benchmark_reader import filter_bench_items
from evaluator.core.bench_schema import BenchAutoUserInfo, BenchItem
from evaluator.plugin.eval_agent.semantic_eval_agent import SemanticEvalInfo


# ============================================================
# 辅助: 构造 BenchItem
# ============================================================


def _item(id_: str) -> BenchItem:
    return BenchItem(
        id=id_,
        title=f"Title {id_}",
        user=BenchAutoUserInfo(goal="goal", max_turns=3),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )


# ============================================================
# filter_bench_items
# ============================================================


def test_filter_no_filters():
    """无过滤条件 → 原样返回"""
    items = [_item("a"), _item("b"), _item("c")]
    result = filter_bench_items(items)
    assert len(result) == 3


def test_filter_by_ids():
    """按 ID 过滤"""
    items = [_item("a"), _item("b"), _item("c")]
    result = filter_bench_items(items, ids="a,c")
    assert [i.id for i in result] == ["a", "c"]


def test_filter_by_ids_preserves_order():
    """ID 过滤保持原列表顺序"""
    items = [_item("c"), _item("a"), _item("b")]
    result = filter_bench_items(items, ids="b,c")
    assert [i.id for i in result] == ["c", "b"]


def test_filter_by_ids_with_spaces():
    """ID 字符串中有空格"""
    items = [_item("a"), _item("b")]
    result = filter_bench_items(items, ids=" a , b ")
    assert len(result) == 2


def test_filter_by_ids_unmatched():
    """部分 ID 未匹配（不报错，只跳过）"""
    items = [_item("a"), _item("b")]
    result = filter_bench_items(items, ids="a,nonexistent")
    assert [i.id for i in result] == ["a"]


def test_filter_by_limit():
    """按数量限制"""
    items = [_item("a"), _item("b"), _item("c")]
    result = filter_bench_items(items, limit=2)
    assert len(result) == 2
    assert [i.id for i in result] == ["a", "b"]


def test_filter_by_limit_larger_than_total():
    """limit 大于总数 → 返回全部"""
    items = [_item("a"), _item("b")]
    result = filter_bench_items(items, limit=100)
    assert len(result) == 2


def test_filter_ids_then_limit():
    """先 ID 后 limit"""
    items = [_item("a"), _item("b"), _item("c"), _item("d")]
    result = filter_bench_items(items, ids="a,b,c", limit=2)
    assert [i.id for i in result] == ["a", "b"]


def test_filter_empty_items():
    """空列表"""
    result = filter_bench_items([], ids="a", limit=5)
    assert result == []
