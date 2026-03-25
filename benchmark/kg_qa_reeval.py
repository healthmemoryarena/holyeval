"""
kg_qa_reeval.py — 对已有报告用更新后的 kg_qa 评估器重新评分

从已有的 BenchReport JSON 中提取对话记录，用当前版本的 KgQaEvalAgent 重新评分，
输出新旧分数对比和汇总统计。

用法:
    # 重评单个报告
    python -m benchmark.kg_qa_reeval /path/to/report.json

    # 重评整个目录
    python -m benchmark.kg_qa_reeval /path/to/results_dir/

    # 限制数量 + 只看分数变化的
    python -m benchmark.kg_qa_reeval /path/to/report.json --limit 10 --changed-only

    # 保存重评结果到新 JSON
    python -m benchmark.kg_qa_reeval /path/to/report.json --output /path/to/reeval_report.json
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluator.core.schema import EvalResult, SessionInfo, TestAgentMemory
from evaluator.plugin.eval_agent.kg_qa_eval_agent import KgQaEvalAgent, KgQaEvalInfo

logger = logging.getLogger(__name__)


def _extract_ai_response(case: dict) -> str:
    """从 case 的 test_memory 中提取最后一轮 AI 回复文本"""
    test_memory = case.get("eval", {}).get("trace", {}).get("test_memory", [])
    for mem in reversed(test_memory):
        target_response = mem.get("target_response")
        if target_response:
            msg_list = target_response.get("message_list", [])
            if msg_list:
                return "\n".join(m.get("content", "") for m in msg_list if m.get("content"))
    return ""


def _build_memory_list(case: dict) -> list[TestAgentMemory]:
    """从 case 重建 TestAgentMemory 列表"""
    test_memory = case.get("eval", {}).get("trace", {}).get("test_memory", [])
    result = []
    for mem in test_memory:
        result.append(TestAgentMemory.model_validate(mem))
    return result


def _extract_eval_config(case: dict) -> dict | None:
    """从 case 提取 eval 配置（兼容新旧报告格式）"""
    # 新格式: 顶层 eval_config
    eval_config = case.get("eval_config")
    if eval_config:
        return eval_config
    # 旧格式: meta.eval_config
    meta = case.get("meta", {})
    if meta:
        return meta.get("eval_config")
    return None


async def reeval_single_case(case: dict) -> dict:
    """对单个 case 重新评分，返回对比结果"""
    case_id = case.get("id", "unknown")
    old_score = case.get("eval", {}).get("score", 0.0)
    old_feedback = case.get("eval", {}).get("feedback", "")

    # 提取 eval 配置
    eval_config = _extract_eval_config(case)
    if not eval_config:
        return {
            "id": case_id,
            "old_score": old_score,
            "new_score": None,
            "error": "无法提取 eval_config",
        }

    # 重建 eval agent
    try:
        eval_info = KgQaEvalInfo.model_validate({"evaluator": "kg_qa", **eval_config})
    except Exception as e:
        return {
            "id": case_id,
            "old_score": old_score,
            "new_score": None,
            "error": f"eval_config 解析失败: {e}",
        }

    agent = KgQaEvalAgent(eval_info)

    # 重建 memory list
    memory_list = _build_memory_list(case)
    if not memory_list:
        return {
            "id": case_id,
            "old_score": old_score,
            "new_score": None,
            "error": "test_memory 为空",
        }

    # 重新评分
    try:
        eval_result: EvalResult = await agent.run(memory_list, session_info=None)
        new_score = eval_result.score
        new_feedback = eval_result.feedback
    except Exception as e:
        return {
            "id": case_id,
            "old_score": old_score,
            "new_score": None,
            "error": f"评估失败: {e}",
        }

    return {
        "id": case_id,
        "title": case.get("title", ""),
        "answer_type": eval_config.get("answer_type", ""),
        "difficulty": eval_config.get("difficulty", ""),
        "old_score": old_score,
        "new_score": new_score,
        "delta": round(new_score - old_score, 4),
        "old_feedback": old_feedback,
        "new_feedback": new_feedback,
    }


async def reeval_report(
    report_path: Path,
    *,
    limit: int | None = None,
    concurrency: int = 10,
    changed_only: bool = False,
    write_back: bool = False,
) -> dict[str, Any]:
    """对单个报告文件重评"""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("cases", [])
    if limit:
        cases = cases[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(case: dict) -> dict:
        async with sem:
            return await reeval_single_case(case)

    tasks = [_eval_one(c) for c in cases]
    results = await asyncio.gather(*tasks)

    # --write-back: 把新分数写回原始报告 JSON
    if write_back:
        new_score_map = {r["id"]: r for r in results if r.get("new_score") is not None}
        for case in report.get("cases", []):
            cid = case.get("id", "")
            r = new_score_map.get(cid)
            if r:
                case["eval"]["score"] = r["new_score"]
                case["eval"]["feedback"] = r.get("new_feedback", "")
        # 更新报告级别均分
        all_scores = [c["eval"]["score"] for c in report.get("cases", []) if c.get("eval", {}).get("score") is not None]
        if all_scores:
            report["avg_score"] = round(sum(all_scores) / len(all_scores), 4)
        # 写回文件
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"已写回: {report_path.name}")

    # 统计
    total = len(results)
    errors = [r for r in results if r.get("error")]
    valid = [r for r in results if r.get("new_score") is not None]

    if changed_only:
        valid = [r for r in valid if abs(r["delta"]) > 0.001]

    old_avg = sum(r["old_score"] for r in valid) / len(valid) if valid else 0
    new_avg = sum(r["new_score"] for r in valid) / len(valid) if valid else 0

    # 按 answer_type 分组统计
    by_type: dict[str, dict] = {}
    for r in [r for r in results if r.get("new_score") is not None]:
        at = r.get("answer_type", "unknown")
        if at not in by_type:
            by_type[at] = {"count": 0, "old_sum": 0.0, "new_sum": 0.0, "changed": 0}
        by_type[at]["count"] += 1
        by_type[at]["old_sum"] += r["old_score"]
        by_type[at]["new_sum"] += r["new_score"]
        if abs(r["delta"]) > 0.001:
            by_type[at]["changed"] += 1

    # 按 difficulty 分组统计
    by_diff: dict[str, dict] = {}
    for r in [r for r in results if r.get("new_score") is not None]:
        d = r.get("difficulty", "unknown") or "unknown"
        if d not in by_diff:
            by_diff[d] = {"count": 0, "old_sum": 0.0, "new_sum": 0.0}
        by_diff[d]["count"] += 1
        by_diff[d]["old_sum"] += r["old_score"]
        by_diff[d]["new_sum"] += r["new_score"]

    return {
        "file": report_path.name,
        "total": total,
        "errors": len(errors),
        "valid": len(valid),
        "old_avg": round(old_avg, 4),
        "new_avg": round(new_avg, 4),
        "delta_avg": round(new_avg - old_avg, 4),
        "by_answer_type": {
            k: {
                "count": v["count"],
                "old_avg": round(v["old_sum"] / v["count"], 4),
                "new_avg": round(v["new_sum"] / v["count"], 4),
                "delta": round((v["new_sum"] - v["old_sum"]) / v["count"], 4),
                "changed": v["changed"],
            }
            for k, v in sorted(by_type.items())
        },
        "by_difficulty": {
            k: {
                "count": v["count"],
                "old_avg": round(v["old_sum"] / v["count"], 4),
                "new_avg": round(v["new_sum"] / v["count"], 4),
                "delta": round((v["new_sum"] - v["old_sum"]) / v["count"], 4),
            }
            for k, v in sorted(by_diff.items())
        },
        "cases": valid if changed_only else results,
    }


def print_summary(summary: dict) -> None:
    """打印重评摘要"""
    print(f"\n{'='*70}")
    print(f"  {summary['file']}")
    print(f"{'='*70}")
    print(f"  总计: {summary['total']} cases, 有效: {summary['valid']}, 错误: {summary['errors']}")
    print(f"  旧均分: {summary['old_avg']:.4f}  →  新均分: {summary['new_avg']:.4f}  (Δ = {summary['delta_avg']:+.4f})")

    if summary["by_answer_type"]:
        print(f"\n  按 answer_type:")
        for k, v in summary["by_answer_type"].items():
            print(f"    {k:16s}  n={v['count']:3d}  {v['old_avg']:.3f} → {v['new_avg']:.3f}  (Δ={v['delta']:+.4f}, changed={v['changed']})")

    if summary["by_difficulty"]:
        print(f"\n  按 difficulty:")
        for k, v in summary["by_difficulty"].items():
            print(f"    {k:20s}  n={v['count']:3d}  {v['old_avg']:.3f} → {v['new_avg']:.3f}  (Δ={v['delta']:+.4f})")

    # 分数变化最大的 case
    cases_with_delta = [c for c in summary["cases"] if c.get("delta") and abs(c["delta"]) > 0.001]
    if cases_with_delta:
        cases_with_delta.sort(key=lambda x: abs(x["delta"]), reverse=True)
        print(f"\n  分数变化 Top 10:")
        for c in cases_with_delta[:10]:
            print(f"    {c['id']:40s}  {c['old_score']:.2f} → {c['new_score']:.2f}  (Δ={c['delta']:+.3f})  [{c['answer_type']}]")
    print()


async def main():
    parser = argparse.ArgumentParser(description="kg_qa 重评工具")
    parser.add_argument("path", help="报告 JSON 文件或目录")
    parser.add_argument("--limit", type=int, default=None, help="每个报告最多评估 N 个 case")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--changed-only", action="store_true", help="只显示分数有变化的 case")
    parser.add_argument("--output", type=str, default=None, help="保存重评结果到 JSON")
    parser.add_argument("--write-back", action="store_true", help="将新分数写回原始报告 JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    path = Path(args.path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            print(f"目录 {path} 中没有 JSON 文件")
            return
    else:
        print(f"路径不存在: {path}")
        return

    all_summaries = []
    for f in files:
        if f.name in ("metadata.json", "README.md"):
            continue
        print(f"正在重评: {f.name} ...", flush=True)
        summary = await reeval_report(
            f,
            limit=args.limit,
            concurrency=args.concurrency,
            changed_only=args.changed_only,
            write_back=args.write_back,
        )
        print_summary(summary)
        all_summaries.append(summary)

    # 总汇总
    if len(all_summaries) > 1:
        print(f"\n{'#'*70}")
        print(f"  汇总: {len(all_summaries)} 个报告")
        print(f"{'#'*70}")
        for s in all_summaries:
            print(f"  {s['file']:60s}  {s['old_avg']:.3f} → {s['new_avg']:.3f}  (Δ={s['delta_avg']:+.4f})")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
