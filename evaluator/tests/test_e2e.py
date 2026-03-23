"""
端到端集成测试 — JSONL → do_single_test / do_batch_test（需要真实 API Key）

支持两种运行方式：
1. 单条用例（参数化）:
   uv run pytest evaluator/tests/test_e2e.py -v -s -k "llm_004"
   uv run pytest evaluator/tests/test_e2e.py -v -s -k "headache_001"
2. 批量执行 + 聚合报告:
   uv run pytest evaluator/tests/test_e2e.py -v -s -k "batch"

运行全部：uv run pytest evaluator/tests/test_e2e.py -v -s
跳过：uv run pytest evaluator/tests/ -m "not api"
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import pytest
from dotenv import load_dotenv

# 模块级标记：整个文件的测试都需要真实 API，普通单测时跳过
pytestmark = pytest.mark.api

load_dotenv()

import evaluator.plugin.test_agent  # noqa: F401 — 触发插件注册
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.eval_agent  # noqa: F401

from evaluator.core.orchestrator import BatchSession, CaseContext, CaseStatus, do_batch_test, do_single_test
from evaluator.core.schema import TestCase, TestReport, TestResult

JSONL_PATH = Path(__file__).parent / "fixtures" / "test_cases.jsonl"
RESULT_PATH = Path(__file__).parent / "results" / "latest_test_cases_result.json"

_skip_no_key = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY 未设置")


def load_cases_from_jsonl(path: Path) -> List[TestCase]:
    """从 JSONL 文件加载测试用例列表"""
    cases: List[TestCase] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(TestCase(**data))
    return cases


# 预加载所有用例，用于参数化
_ALL_CASES: Dict[str, TestCase] = {c.id: c for c in load_cases_from_jsonl(JSONL_PATH)}


# ============================================================
# 1. 单条用例参数化测试
# ============================================================


@_skip_no_key
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", list(_ALL_CASES.keys()), ids=list(_ALL_CASES.keys()))
async def test_single_case(case_id: str):
    """单条用例端到端测试 — 可用 -k 过滤运行指定用例"""
    case = _ALL_CASES[case_id]
    result: TestResult = await do_single_test(case)

    # 基础断言
    assert result.eval.result in ("pass", "fail")
    assert 0.0 <= result.eval.score <= 1.0
    assert result.eval.feedback
    assert result.start <= result.end

    # 携带 history 的用例：验证 trace 中保留了历史
    if case.history:
        assert result.eval.trace is not None
        assert len(result.eval.trace.history) == len(case.history)

    # llm_api target：验证 target 成本
    if case.target.type == "llm_api":
        assert result.cost.target is not None, "llm_api 目标应有 target 成本"

    # 打印摘要
    print(f"\n[{case_id}] {result.eval.result} (score={result.eval.score:.2f})")
    if result.eval.feedback:
        print(f"  feedback: {result.eval.feedback[:200]}")


# ============================================================
# 2. 批量执行 + 聚合报告
# ============================================================


@_skip_no_key
@pytest.mark.asyncio
async def test_batch_from_jsonl():
    """批量执行全部用例，验证 TestReport 聚合指标 + 进度跟踪"""
    cases = list(_ALL_CASES.values())

    # 进度回调 — 实时打印每个 case 完成状态
    def _on_progress(session: BatchSession, case_id: str):
        ctx = session.contexts[case_id]
        status_icon = {"completed": "✅", "error": "❌", "cancelled": "⏹️"}.get(ctx.status.value, "⏳")
        score_text = f"score={ctx.result.eval.score:.2f}" if ctx.result else ""
        print(f"  {status_icon} [{session.completed}/{session.total}] {case_id} → {ctx.status.value} {score_text}")

    # 使用 BatchSession 直接运行，验证完整的进度跟踪能力
    session = BatchSession(cases, on_progress=_on_progress)

    # 验证初始状态
    assert session.total == len(cases)
    assert session.completed == 0
    assert all(ctx.status == CaseStatus.PENDING for ctx in session.contexts.values())

    print(f"\n{'=' * 60}")
    print(f"开始批量执行: {session.total} 个用例 (session={session.id})")
    print(f"{'=' * 60}")

    report = await session.run()

    # 验证执行后状态
    assert session.completed == len(cases)
    for ctx in session.contexts.values():
        assert ctx.status in (CaseStatus.COMPLETED, CaseStatus.ERROR)
        assert ctx.result is not None

    # snapshot 快照验证
    snap = session.snapshot()
    assert snap["total"] == len(cases)
    assert snap["completed"] == len(cases)
    assert not snap["cancelled"]
    assert len(snap["cases"]) == len(cases)

    # 结构完整性
    assert isinstance(report, TestReport)
    assert len(report.cases) == len(cases)

    # 报告聚合指标
    assert report.pass_count + report.fail_count == len(cases)
    assert 0.0 <= report.pass_rate <= 1.0
    assert 0.0 <= report.avg_score <= 1.0
    assert report.total_duration_seconds > 0

    # 结果落文件
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n结果已写入: {RESULT_PATH}")

    # 打印摘要
    print(f"{'=' * 60}")
    print(f"端到端测试报告: {report.pass_count} pass / {report.fail_count} fail")
    print(f"pass_rate={report.pass_rate:.1%}, avg_score={report.avg_score:.2f}")
    print(f"total_duration={report.total_duration_seconds:.1f}s")
    for cid, ctx in session.contexts.items():
        r = ctx.result
        mem_count = len(ctx.test_agent.memory_list) if ctx.test_agent else 0
        print(f"  [{cid}] {r.eval.result} (score={r.eval.score:.2f}, turns={ctx.turn}, memory={mem_count})")
    print(f"{'=' * 60}")
