"""
MemoryArena → HolyEval 数据转换器

将 HuggingFace 上的 MemoryArena 数据集（ZexueHe/memoryarena）转换为 HolyEval BenchItem JSONL。
MemoryArena 评测 Agent 跨会话记忆能力，包含 5 个领域：
  - bundled_shopping: 多步商品选购
  - progressive_search: 渐进式信息检索
  - group_travel_planner: 多人行程规划
  - formal_reasoning_math: 数学推理
  - formal_reasoning_phys: 物理推理

转换映射:
  questions[]      →  strict_inputs（逐条发送给被测系统）
  answers[]        →  eval.ground_truths（JSON 序列化后逐子任务比对）
  backgrounds[]    →  合并到 questions 中（仅 formal_reasoning）
  base_person      →  合并到首条 question 中（仅 group_travel_planner）
  domain           →  tags + eval.domain

BenchItem 不含 target 字段（由运行时 --target-type/--target-model 决定）。

用法:
  # 安装依赖: uv pip install datasets
  python -m generator.memoryarena.converter --output-dir benchmark/data/memoryarena/

数据集来源:
  https://huggingface.co/datasets/ZexueHe/memoryarena
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# HuggingFace 数据集配置名
CONFIGS = [
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
]

# sample 抽样比例（总计 ~10 条，按领域数量等比，每领域至少 1 条）
SAMPLE_COUNTS = {
    "bundled_shopping": 2,
    "group_travel_planner": 3,
    "progressive_search": 3,
    "formal_reasoning_math": 1,
    "formal_reasoning_phys": 1,
}

# 领域前缀（用于 id 生成）
DOMAIN_PREFIX = {
    "bundled_shopping": "ma_shop",
    "progressive_search": "ma_search",
    "group_travel_planner": "ma_travel",
    "formal_reasoning_math": "ma_math",
    "formal_reasoning_phys": "ma_phys",
}


def _format_base_person(bp: Dict[str, Any]) -> str:
    """将 group_travel_planner 的 base_person 格式化为可读文本"""
    lines = [
        f"Current Travel Plan for {bp.get('name', 'Person 1')}:",
        f"Original Query: {bp.get('query', '')}",
        "",
        "Daily Itinerary:",
    ]
    for plan in bp.get("daily_plans", []):
        lines.append(f"  Day {plan.get('days', '?')}:")
        lines.append(f"    City: {plan.get('current_city', '-')}")
        lines.append(f"    Transportation: {plan.get('transportation', '-')}")
        lines.append(f"    Breakfast: {plan.get('breakfast', '-')}")
        lines.append(f"    Lunch: {plan.get('lunch', '-')}")
        lines.append(f"    Dinner: {plan.get('dinner', '-')}")
        lines.append(f"    Attraction: {plan.get('attraction', '-')}")
        lines.append(f"    Accommodation: {plan.get('accommodation', '-')}")
    return "\n".join(lines)


def _serialize_answer(answer: Any) -> str:
    """将答案序列化为 JSON 字符串"""
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False)


def _convert_shopping(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """转换 bundled_shopping 条目"""
    questions = entry.get("questions", [])
    answers = entry.get("answers", [])
    entry_id = entry.get("id", 0)
    category = entry.get("category", "unknown")

    if not questions or not answers:
        return None

    return {
        "id": f"{DOMAIN_PREFIX['bundled_shopping']}_{entry_id}",
        "title": f"MemoryArena Shopping — {category}",
        "description": f"多步商品选购任务（{len(questions)} 个子任务）",
        "user": {
            "type": "manual",
            "strict_inputs": list(questions),
        },
        "eval": {
            "evaluator": "memoryarena",
            "domain": "bundled_shopping",
            "ground_truths": [_serialize_answer(a) for a in answers],
        },
        "tags": ["domain:bundled_shopping", f"category:{category}"],
    }


def _convert_search(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """转换 progressive_search 条目"""
    questions = entry.get("questions", [])
    answers = entry.get("answers", [])
    entry_id = entry.get("id", 0)

    if not questions or not answers:
        return None

    return {
        "id": f"{DOMAIN_PREFIX['progressive_search']}_{entry_id}",
        "title": f"MemoryArena Search — {len(questions)} queries",
        "description": f"渐进式信息检索（{len(questions)} 个子查询）",
        "user": {
            "type": "manual",
            "strict_inputs": list(questions),
        },
        "eval": {
            "evaluator": "memoryarena",
            "domain": "progressive_search",
            "ground_truths": [_serialize_answer(a) for a in answers],
        },
        "tags": ["domain:progressive_search"],
    }


def _convert_travel(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """转换 group_travel_planner 条目"""
    questions = entry.get("questions", [])
    answers = entry.get("answers", [])
    base_person = entry.get("base_person", {})
    entry_id = entry.get("id", 0)

    if not questions or not answers:
        return None

    # 将 base_person 作为上下文合并到第一条 question
    strict_inputs = []
    if base_person and questions:
        preamble = _format_base_person(base_person)
        strict_inputs.append(f"{preamble}\n\n---\n\n{questions[0]}")
        strict_inputs.extend(questions[1:])
    else:
        strict_inputs = list(questions)

    return {
        "id": f"{DOMAIN_PREFIX['group_travel_planner']}_{entry_id}",
        "title": f"MemoryArena Travel — {len(questions)} travelers",
        "description": f"多人行程规划（{len(questions)} 个旅客约束）",
        "user": {
            "type": "manual",
            "strict_inputs": strict_inputs,
        },
        "eval": {
            "evaluator": "memoryarena",
            "domain": "group_travel_planner",
            "ground_truths": [_serialize_answer(a) for a in answers],
        },
        "tags": ["domain:group_travel_planner"],
    }


def _convert_reasoning(entry: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
    """转换 formal_reasoning_math / formal_reasoning_phys 条目"""
    questions = entry.get("questions", [])
    answers = entry.get("answers", [])
    backgrounds = entry.get("backgrounds", [])
    entry_id = entry.get("id", 0)
    paper_name = entry.get("paper_name", "")

    if not questions or not answers:
        return None

    # 将 background 合并到对应 question 中
    strict_inputs = []
    for i, q in enumerate(questions):
        if i < len(backgrounds) and backgrounds[i]:
            strict_inputs.append(f"Background:\n{backgrounds[i]}\n\nQuestion:\n{q}")
        else:
            strict_inputs.append(q)

    domain_label = "Math" if domain == "formal_reasoning_math" else "Physics"
    prefix = DOMAIN_PREFIX[domain]

    return {
        "id": f"{prefix}_{entry_id}",
        "title": f"MemoryArena {domain_label} — {paper_name}"
        if paper_name
        else f"MemoryArena {domain_label} #{entry_id}",
        "description": f"{domain_label} 推理（{len(questions)} 个子问题）",
        "user": {
            "type": "manual",
            "strict_inputs": strict_inputs,
        },
        "eval": {
            "evaluator": "memoryarena",
            "domain": domain,
            "ground_truths": [_serialize_answer(a) for a in answers],
        },
        "tags": [f"domain:{domain}"],
    }


CONVERTERS = {
    "bundled_shopping": _convert_shopping,
    "progressive_search": _convert_search,
    "group_travel_planner": _convert_travel,
    "formal_reasoning_math": lambda e: _convert_reasoning(e, "formal_reasoning_math"),
    "formal_reasoning_phys": lambda e: _convert_reasoning(e, "formal_reasoning_phys"),
}


def convert(output_dir: str | Path, sample_seed: int = 42) -> Dict[str, int]:
    """从 HuggingFace 下载 MemoryArena 并转换为 BenchItem JSONL

    生成:
      - full.jsonl: 全量数据
      - sample.jsonl: 按领域等比抽样 ~100 条

    Returns:
        每个配置的转换条数 dict
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("需要安装 datasets 库: uv pip install datasets")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items: List[Dict[str, Any]] = []
    items_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    stats: Dict[str, int] = {}

    for config in CONFIGS:
        logger.info("下载并转换 %s ...", config)
        ds = load_dataset("ZexueHe/memoryarena", config, split="test")
        converter_fn = CONVERTERS[config]
        domain_items = []

        for entry in ds:
            item = converter_fn(dict(entry))
            if item is not None:
                domain_items.append(item)

        all_items.extend(domain_items)
        items_by_domain[config] = domain_items
        stats[config] = len(domain_items)
        logger.info("  %s: %d 条转换成功", config, len(domain_items))

    # 写入 full.jsonl
    full_path = output_dir / "full.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("full.jsonl: %d 条 → %s", len(all_items), full_path)

    # 抽样 sample.jsonl
    rng = random.Random(sample_seed)
    sample_items: List[Dict[str, Any]] = []
    for config in CONFIGS:
        domain_items = items_by_domain[config]
        n = min(SAMPLE_COUNTS.get(config, 0), len(domain_items))
        sampled = rng.sample(domain_items, n)
        sample_items.extend(sampled)

    # 打乱顺序
    rng.shuffle(sample_items)

    sample_path = output_dir / "sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for item in sample_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("sample.jsonl: %d 条 → %s", len(sample_items), sample_path)

    stats["total"] = len(all_items)
    stats["sample"] = len(sample_items)
    return stats


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载 MemoryArena 数据集并转换为 HolyEval BenchItem JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m generator.memoryarena.converter\n"
            "  python -m generator.memoryarena.converter --output-dir benchmark/data/memoryarena/\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/data/memoryarena",
        help="输出目录（默认 benchmark/data/memoryarena/）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    stats = convert(args.output_dir)
    print("\n转换完成:")
    for k, v in stats.items():
        print(f"  {k}: {v} 条")


if __name__ == "__main__":
    main()
