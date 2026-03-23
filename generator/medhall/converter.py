"""
MedHall → HolyEval 数据转换器

将 data_gen.py 生成的原始 JSONL 转换为 HolyEval BenchItem JSONL，
使其可通过 `python -m benchmark.basic_runner medhall sample` 执行。

转换映射:
  patient_context + question  →  strict_inputs[0]（提问内容）
  patient_context             →  eval.context（幻觉检测上下文）
  category                    →  eval.categories（单类型列表）
  topic + category            →  tags

用法:
  python -m generator.medhall.converter raw_data.jsonl benchmark/data/medhall/sample.jsonl
  python -m generator.medhall.converter raw_data.jsonl output.jsonl --limit 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 单条转换
# ============================================================


def _convert_single(entry: dict[str, Any], index: int) -> Optional[dict[str, Any]]:
    """将单条原始生成数据转换为 HolyEval BenchItem dict"""
    patient_context = entry.get("patient_context", "").strip()
    question = entry.get("question", "").strip()
    category = entry.get("category", "factual").strip()
    topic = entry.get("topic", "").strip()
    difficulty = entry.get("difficulty", "medium").strip()

    if not patient_context:
        logger.warning("跳过无 patient_context 的条目 #%d", index)
        return None
    if not question:
        logger.warning("跳过无 question 的条目 #%d", index)
        return None

    # 构建用户输入：患者背景 + 问题
    user_input = f"{patient_context}\n\n{question}"

    # 幻觉类型 → 中文标签
    _category_label = {
        "factual": "事实幻觉",
        "contextual": "上下文幻觉",
        "citation": "引用幻觉",
    }

    tags = [f"type:{_category_label.get(category, category)}", f"difficulty:{difficulty}"]
    if topic:
        tags.append(f"topic:{topic}")

    bench_item: dict[str, Any] = {
        "id": f"mh_{category[:3]}_{index:04d}",
        "title": f"医疗幻觉检测 — {_category_label.get(category, category)} — {topic or '医疗场景'}",
        "description": f"检测 AI 回复中的{_category_label.get(category, category)}，难度: {difficulty}",
        "user": {
            "type": "manual",
            "strict_inputs": [user_input],
        },
        "eval": {
            "evaluator": "hallucination",
            "categories": [category],
            "context": patient_context,
        },
        "tags": tags,
    }

    return bench_item


# ============================================================
# 批量转换
# ============================================================


def convert(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """将 MedHall 原始 JSONL 转换为 HolyEval BenchItem JSONL

    Args:
        input_path:  data_gen.py 输出的原始 JSONL 路径
        output_path: 输出 HolyEval BenchItem JSONL 路径
        limit:       最大转换条数（None 表示全部）

    Returns:
        成功转换的用例数
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if limit is not None and converted >= limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("第 %d 行 JSON 解析失败: %s", i + 1, e)
                skipped += 1
                continue

            bench_item = _convert_single(entry, i)
            if bench_item is None:
                skipped += 1
                continue

            fout.write(json.dumps(bench_item, ensure_ascii=False) + "\n")
            converted += 1

    logger.info("转换完成: %d 条成功, %d 条跳过, 输出: %s", converted, skipped, output_path)
    return converted


# ============================================================
# CLI 入口
# ============================================================


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="将 MedHall 原始 JSONL 转换为 HolyEval BenchItem JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m generator.medhall.converter generator/medhall/raw_data.jsonl benchmark/data/medhall/sample.jsonl\n"
            "  python -m generator.medhall.converter raw_data.jsonl output.jsonl --limit 50\n"
        ),
    )
    parser.add_argument("input", help="原始 JSONL 文件路径（data_gen.py 输出）")
    parser.add_argument("output", help="输出 HolyEval BenchItem JSONL 文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最大转换条数")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    count = convert(args.input, args.output, limit=args.limit)
    print(f"转换完成: {count} 条 BenchItem → {args.output}")


if __name__ == "__main__":
    main()
