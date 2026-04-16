"""
虚拟用户开场白对比分析器

读取 BenchReport JSON，按 opening_id 分组统计录入率（pass_rate），
并按对抗维度做交叉分析。

用法:
    python -m generator.virtual_user.analyzer benchmark/report/virtual_user/round1_xxx.json
    python -m generator.virtual_user.analyzer report.json --opening 03
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# 对抗维度列表（用于交叉分析）
_ADVERSARIAL_DIMS = ["disclosure", "attitude", "cognition", "logic", "expression"]


def _extract_tag(tags: list[str], prefix: str) -> str | None:
    """从 tags 列表中提取指定前缀的值"""
    for tag in tags:
        if tag.startswith(f"{prefix}:"):
            return tag.split(":", 1)[1]
    return None


def _compute_rate(cases: list[dict]) -> tuple[int, int, float]:
    """计算 pass 数、总数、pass_rate"""
    total = len(cases)
    if total == 0:
        return 0, 0, 0.0
    passed = sum(1 for c in cases if c.get("eval", {}).get("result") == "pass")
    return passed, total, passed / total


def analyze(report_path: str | Path, focus_opening: str | None = None) -> dict:
    """分析报告

    Args:
        report_path: BenchReport JSON 路径
        focus_opening: 仅分析指定 opening_id（None = 全部）

    Returns:
        分析结果 dict
    """
    report_path = Path(report_path)
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("cases", [])
    if not cases:
        logger.warning("报告中无用例结果")
        return {}

    # 按 opening 分组
    by_opening: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        tags = case.get("tags", [])
        oid = _extract_tag(tags, "opening")
        if oid:
            by_opening[oid].append(case)

    # 总体统计
    result: dict = {"total_cases": len(cases), "openings": {}}

    for oid in sorted(by_opening.keys()):
        if focus_opening and oid != focus_opening:
            continue

        opening_cases = by_opening[oid]
        passed, total, rate = _compute_rate(opening_cases)

        opening_result: dict = {
            "pass_count": passed,
            "total": total,
            "pass_rate": round(rate, 4),
            "cross_analysis": {},
        }

        # 交叉分析：每个对抗维度
        for dim in _ADVERSARIAL_DIMS:
            by_dim: dict[str, list[dict]] = defaultdict(list)
            for case in opening_cases:
                tags = case.get("tags", [])
                val = _extract_tag(tags, dim)
                if val:
                    by_dim[val].append(case)

            dim_stats = {}
            for val in sorted(by_dim.keys()):
                p, t, r = _compute_rate(by_dim[val])
                dim_stats[val] = {"pass_count": p, "total": t, "pass_rate": round(r, 4)}
            opening_result["cross_analysis"][dim] = dim_stats

        result["openings"][oid] = opening_result

    return result


def print_report(result: dict) -> None:
    """将分析结果打印为终端表格"""
    print()
    print("=" * 60)
    print(f"  开场白 A/B 对比报告 — 虚拟用户（N={result.get('total_cases', 0)}）")
    print("=" * 60)
    print()

    openings = result.get("openings", {})
    if not openings:
        print("  (无数据)")
        return

    # 总览表
    print(f"{'开场白':<15} {'录入率':>8} {'样本数':>10}")
    print("-" * 40)
    for oid, data in openings.items():
        rate_pct = f"{data['pass_rate'] * 100:.0f}%"
        sample = f"{data['pass_count']}/{data['total']}"
        print(f"opening_{oid:<8} {rate_pct:>8} {sample:>10}")
    print()

    # 交叉分析（逐 opening）
    for oid, data in openings.items():
        cross = data.get("cross_analysis", {})
        if not cross:
            continue
        print(f"按对抗维度交叉分析（opening_{oid}）:")
        for dim, dim_stats in cross.items():
            for val, stats in dim_stats.items():
                rate_pct = f"{stats['pass_rate'] * 100:.0f}%"
                sample = f"({stats['pass_count']}/{stats['total']})"
                print(f"  {dim}={val:<30} {rate_pct:>6}  {sample}")
            print()


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="虚拟用户开场白对比分析器")
    parser.add_argument("report", help="BenchReport JSON 路径")
    parser.add_argument("--opening", default=None, help="仅分析指定 opening_id")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = analyze(args.report, focus_opening=args.opening)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
