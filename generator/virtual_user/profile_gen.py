"""
虚拟用户画像生成器 — 用 LLM 批量生成肥胖患者画像

输出 profiles.jsonl，每行一个 JSON 对象:
{
    "profile_id": "obesity_001",
    "age": 35,
    "gender": "female",
    "bmi": 32.5,
    "occupation": "小学教师",
    "comorbidities": ["高血压前期"],
    "motivation": "体检报告异常，医生建议减重",
    "background": "尝试过节食和跑步..."
}

用法:
    python -m generator.virtual_user.profile_gen --count 20 --output generator/virtual_user/profiles.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a medical data specialist creating realistic virtual patient profiles for product testing. "
    "Output valid JSON only."
)

_GENERATION_PROMPT = """请生成 {count} 个**肥胖相关**的虚拟患者画像，用于测试一款慢病管理 APP 的开场白效果。

要求：
1. 覆盖多样性：年龄（20~65）、性别、BMI 范围（28~45）、职业、合并症、减肥动机
2. 画像要真实可信，包含具体的生活细节
3. 每个画像的 background 应包含 2~3 句话，描述该患者的减肥经历或健康困扰

输出 JSON 数组，每个对象包含：
- "profile_id": 字符串，格式 "obesity_XXX"（三位数字，从 {start_index} 开始）
- "age": 整数
- "gender": "male" 或 "female"
- "bmi": 浮点数（保留一位小数）
- "occupation": 字符串（中文）
- "comorbidities": 字符串列表（如 ["高血压", "2型糖尿病"]，可为空列表）
- "motivation": 字符串（中文，一句话描述减肥动机）
- "background": 字符串（中文，2~3 句话描述该患者的具体情况）

Return ONLY the JSON array, no other text."""


async def _generate_batch(count: int, start_index: int) -> list[dict]:
    """调用 LLM 生成一批画像"""
    from evaluator.utils.llm import do_execute

    prompt = _GENERATION_PROMPT.format(count=count, start_index=start_index)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await do_execute(
                model="gpt-4.1",
                system_prompt=_SYSTEM_PROMPT,
                input=prompt,
                max_tokens=4000,
            )
            content = result.content.strip()

            # 去掉 markdown 包裹
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            items = json.loads(content)
            if isinstance(items, list):
                logger.info("[ProfileGen] 生成 %d 个画像", len(items))
                return items

            logger.warning("[ProfileGen] 格式异常 (attempt %d/%d)", attempt + 1, max_retries)
        except json.JSONDecodeError as e:
            logger.warning("[ProfileGen] JSON 解析失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)
        except Exception as e:
            logger.warning("[ProfileGen] 生成失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)

    logger.error("[ProfileGen] 生成失败，重试耗尽")
    return []


async def generate(output_path: str | Path, count: int = 20) -> int:
    """批量生成虚拟患者画像

    Args:
        output_path: 输出 JSONL 路径
        count: 生成数量

    Returns:
        成功生成的画像数
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 分批生成（每批最多 15 个，避免超出 token 限制）
    batch_size = 15
    tasks = []
    start = 1
    remaining = count
    while remaining > 0:
        this_batch = min(remaining, batch_size)
        tasks.append(_generate_batch(this_batch, start))
        start += this_batch
        remaining -= this_batch

    logger.info("[ProfileGen] 启动 %d 个生成任务...", len(tasks))
    results = await asyncio.gather(*tasks)

    total = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for items in results:
            for item in items:
                if isinstance(item, dict):
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total += 1

    logger.info("[ProfileGen] 生成完成: %d 个画像 → %s", total, output_path)
    return total


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="用 LLM 批量生成肥胖患者虚拟画像")
    parser.add_argument("--count", type=int, default=20, help="生成数量（默认 20）")
    parser.add_argument(
        "--output",
        default="generator/virtual_user/profiles.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    total = asyncio.run(generate(args.output, args.count))
    print(f"生成完成: {total} 个画像 → {args.output}")


if __name__ == "__main__":
    main()
