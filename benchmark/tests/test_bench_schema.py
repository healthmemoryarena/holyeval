"""测试 BenchSchema — 三层 target 合并逻辑和 BenchItem → TestCase 转换"""

import pytest

from evaluator.core.bench_schema import (
    BenchAutoUserInfo,
    BenchItem,
    bench_item_to_test_case,
    build_bench_report,
    find_target_spec,
    resolve_effective_target,
    resolve_runtime_target,
)
from evaluator.core.schema import (
    TargetFieldSpec,
    TargetSpec,
    TestCase,
    TestResult,
)
from evaluator.plugin.eval_agent.semantic_eval_agent import SemanticEvalInfo
from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetInfo
from evaluator.plugin.target_agent.theta_api_target_agent import ThetaApiTargetInfo


# ============================================================
# 辅助: 常用 TargetSpec
# ============================================================

LLM_SPEC = TargetSpec(
    type="llm_api",
    fields={
        "model": TargetFieldSpec(default="gpt-4.1", editable=True, required=True),
        "system_prompt": TargetFieldSpec(default=None, editable=True),
    },
)

THETA_SPEC = TargetSpec(
    type="theta_api",
    fields={
        "email": TargetFieldSpec(default="default@placeholder.com", editable=False, required=True),
    },
)


# ============================================================
# 测试 resolve_effective_target — 三层合并
# ============================================================


def test_resolve_spec_defaults_only():
    """无 CLI 和 case 覆盖 → 使用 spec 默认值"""
    target = resolve_effective_target(LLM_SPEC)
    assert target.type == "llm_api"
    assert target.model == "gpt-4.1"  # type: ignore
    assert target.system_prompt is None  # type: ignore


def test_resolve_cli_overrides_editable():
    """CLI 覆盖 editable 字段"""
    target = resolve_effective_target(LLM_SPEC, cli_overrides={"model": "gemini-3-pro-preview"})
    assert target.model == "gemini-3-pro-preview"  # type: ignore


def test_resolve_cli_overrides_non_editable_ignored():
    """CLI 覆盖 non-editable 字段被忽略"""
    target = resolve_effective_target(THETA_SPEC, cli_overrides={"email": "hacker@evil.com"})
    assert target.email == "default@placeholder.com"  # type: ignore (non-editable, 忽略 CLI)


def test_resolve_case_overrides_any_field():
    """case-level override 可修改任意字段（含 non-editable）"""
    target = resolve_effective_target(
        THETA_SPEC,
        case_overrides={"email": "case@test.com"},
    )
    assert target.email == "case@test.com"  # type: ignore


def test_resolve_priority_cli_over_case():
    """CLI > case override > spec default"""
    target = resolve_effective_target(
        LLM_SPEC,
        cli_overrides={"model": "gpt-5.2"},
        case_overrides={"model": "gpt-4.1", "system_prompt": "case prompt"},
    )
    assert target.model == "gpt-5.2"  # type: ignore  — CLI 优先
    assert target.system_prompt == "case prompt"  # type: ignore  — case 补充（CLI 未指定）


def test_resolve_case_overrides_none():
    """case_overrides 为 None 时使用 spec 默认值"""
    target = resolve_effective_target(
        LLM_SPEC,
        case_overrides=None,
    )
    assert target.type == "llm_api"
    assert target.model == "gpt-4.1"  # type: ignore


def test_resolve_required_missing_raises():
    """必填字段缺少值时抛出异常"""
    spec = TargetSpec(
        type="llm_api",
        fields={"model": TargetFieldSpec(default=None, editable=True, required=True)},
    )
    with pytest.raises(ValueError, match="必填字段"):
        resolve_effective_target(spec)


# ============================================================
# 测试 bench_item_to_test_case（使用 spec + cli_overrides）
# ============================================================


def test_bench_item_to_test_case_no_override():
    """无 override 时，使用 spec 默认值"""
    item = BenchItem(
        id="test_001",
        title="测试用例",
        user=BenchAutoUserInfo(goal="测试目标", max_turns=5, target_overrides={}),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )
    test_case = bench_item_to_test_case(item, LLM_SPEC)
    assert isinstance(test_case, TestCase)
    assert test_case.target.type == "llm_api"
    assert test_case.target.model == "gpt-4.1"  # type: ignore


def test_bench_item_to_test_case_with_case_override():
    """case-level override 补充 system_prompt"""
    item = BenchItem(
        id="test_002",
        title="测试用例",
        user=BenchAutoUserInfo(
            goal="测试目标",
            max_turns=5,
            target_overrides={"llm_api": {"system_prompt": "你是专家"}},
        ),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )
    test_case = bench_item_to_test_case(item, LLM_SPEC)
    assert test_case.target.system_prompt == "你是专家"  # type: ignore


def test_bench_item_to_test_case_cli_wins_over_case():
    """CLI 覆盖优先于 case-level override"""
    item = BenchItem(
        id="test_003",
        title="测试用例",
        user=BenchAutoUserInfo(
            goal="测试目标",
            max_turns=5,
            target_overrides={"llm_api": {"model": "gpt-4.1", "system_prompt": "case prompt"}},
        ),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )
    test_case = bench_item_to_test_case(item, LLM_SPEC, cli_overrides={"model": "gpt-5.2"})
    assert test_case.target.model == "gpt-5.2"  # type: ignore — CLI 优先
    assert test_case.target.system_prompt == "case prompt"  # type: ignore — case 补充


def test_bench_item_to_test_case_theta_locked():
    """theta_api: email non-editable → CLI 无法覆盖，case override 可以"""
    item = BenchItem(
        id="test_004",
        title="测试用例",
        user=BenchAutoUserInfo(
            goal="测试目标",
            max_turns=5,
            target_overrides={"theta_api": {"email": "case@test.com"}},
        ),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )
    test_case = bench_item_to_test_case(item, THETA_SPEC, cli_overrides={"email": "cli@evil.com"})
    assert test_case.target.email == "case@test.com"  # type: ignore — case 优先，CLI 被忽略（non-editable）


def test_target_overrides_backward_compat_list():
    """旧 list 格式的 target_overrides 自动转为 dict"""
    item = BenchItem(
        id="test_005",
        title="兼容测试",
        user=BenchAutoUserInfo(
            goal="测试目标",
            max_turns=5,
            target_overrides=[{"type": "llm_api", "system_prompt": "旧格式"}],
        ),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )
    assert isinstance(item.user.target_overrides, dict)
    assert "llm_api" in item.user.target_overrides
    assert item.user.target_overrides["llm_api"] == {"system_prompt": "旧格式"}

    test_case = bench_item_to_test_case(item, LLM_SPEC)
    assert test_case.target.system_prompt == "旧格式"  # type: ignore


def test_target_overrides_type_mismatch_via_dict():
    """dict 格式下，不匹配的 target_type 自然返回 None"""
    item = BenchItem(
        id="test_006",
        title="不匹配测试",
        user=BenchAutoUserInfo(
            goal="测试目标",
            max_turns=5,
            target_overrides={"theta_api": {"email": "other@test.com"}},
        ),
        eval=SemanticEvalInfo(evaluator="semantic"),
    )
    test_case = bench_item_to_test_case(item, LLM_SPEC)
    assert test_case.target.type == "llm_api"
    assert test_case.target.model == "gpt-4.1"  # type: ignore — 无匹配 override，使用 spec 默认值


# ============================================================
# 测试 build_bench_report
# ============================================================


def test_build_bench_report_basic():
    """基础报告构建"""
    from datetime import datetime

    from evaluator.core.schema import EvalResult, TestCost

    from datetime import timedelta

    now = datetime.now()
    results = [
        TestResult(
            id="c1",
            eval=EvalResult(result="pass", score=0.8, feedback="Good"),
            cost=TestCost(),
            start=now,
            end=now + timedelta(seconds=1.0),
            tags=["tag1", "tag2"],
        ),
        TestResult(
            id="c2",
            eval=EvalResult(result="fail", score=0.5, feedback="Bad"),
            cost=TestCost(),
            start=now,
            end=now + timedelta(seconds=2.0),
            tags=["tag1"],
        ),
        TestResult(
            id="c3",
            eval=EvalResult(result="pass", score=0.9, feedback="Excellent"),
            cost=TestCost(),
            start=now,
            end=now + timedelta(seconds=1.5),
            tags=["tag2"],
        ),
    ]
    runtime_target = LlmApiTargetInfo(type="llm_api", model="gpt-4.1")

    report = build_bench_report(
        test_results=results,
        benchmark_name="healthbench",
        dataset_name="sample",
        runtime_target=runtime_target,
    )

    assert report.benchmark_name == "healthbench"
    assert report.dataset_name == "sample"
    assert report.runtime_target.type == "llm_api"
    assert len(report.cases) == 3
    assert report.pass_count == 2
    assert report.fail_count == 1
    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.avg_score == pytest.approx((0.8 + 0.5 + 0.9) / 3)
    assert report.total_duration_seconds == pytest.approx(4.5)


def test_build_bench_report_stats_by_tag():
    """按 tag 统计"""
    from datetime import datetime, timedelta

    from evaluator.core.schema import EvalResult, TestCost

    now = datetime.now()
    results = [
        TestResult(
            id="c1",
            eval=EvalResult(result="pass", score=0.8, feedback=""),
            cost=TestCost(),
            start=now,
            end=now + timedelta(seconds=1.0),
            tags=["tag1", "tag2"],
        ),
        TestResult(
            id="c2",
            eval=EvalResult(result="fail", score=0.4, feedback=""),
            cost=TestCost(),
            start=now,
            end=now + timedelta(seconds=1.0),
            tags=["tag1"],
        ),
        TestResult(
            id="c3",
            eval=EvalResult(result="pass", score=1.0, feedback=""),
            cost=TestCost(),
            start=now,
            end=now + timedelta(seconds=1.0),
            tags=["tag2"],
        ),
    ]
    runtime_target = LlmApiTargetInfo(type="llm_api", model="gpt-4.1")

    report = build_bench_report(
        test_results=results,
        benchmark_name="test",
        dataset_name="test",
        runtime_target=runtime_target,
    )

    assert "tag1" in report.stats_by_tag
    assert "tag2" in report.stats_by_tag

    tag1_stats = report.stats_by_tag["tag1"]
    assert tag1_stats["total"] == 2
    assert tag1_stats["pass_count"] == 1
    assert tag1_stats["fail_count"] == 1
    assert tag1_stats["pass_rate"] == pytest.approx(0.5)
    assert tag1_stats["avg_score"] == pytest.approx((0.8 + 0.4) / 2)

    tag2_stats = report.stats_by_tag["tag2"]
    assert tag2_stats["total"] == 2
    assert tag2_stats["pass_count"] == 2
    assert tag2_stats["pass_rate"] == pytest.approx(1.0)
    assert tag2_stats["avg_score"] == pytest.approx((0.8 + 1.0) / 2)


# ============================================================
# 测试 find_target_spec — 从列表中查找 spec
# ============================================================


def test_find_target_spec_single():
    """单个 spec，不指定 type → 返回唯一 spec"""
    spec = find_target_spec([LLM_SPEC])
    assert spec.type == "llm_api"


def test_find_target_spec_by_type():
    """多个 spec，指定 type → 返回匹配的 spec"""
    spec = find_target_spec([LLM_SPEC, THETA_SPEC], "theta_api")
    assert spec.type == "theta_api"


def test_find_target_spec_default_first():
    """多个 spec，不指定 type → 返回第一个"""
    spec = find_target_spec([THETA_SPEC, LLM_SPEC])
    assert spec.type == "theta_api"


def test_find_target_spec_empty_raises():
    """空列表 → 抛出 ValueError"""
    with pytest.raises(ValueError, match="未定义 target"):
        find_target_spec([])


def test_find_target_spec_type_not_found_raises():
    """指定的 type 不在列表中 → 抛出 ValueError"""
    with pytest.raises(ValueError, match="未找到 target type"):
        find_target_spec([LLM_SPEC], "theta_api")
