"""
为 medhall JSONL 注入 ground_truth_data — 从 thetagendata 读取真实健康记录

用法:
  python -m generator.medhall.enrich_ground_truth benchmark/data/medhall/theta.jsonl \
    --thetagendata-dir /Users/admin/thetagendata

功能:
  1. 读取现有 theta.jsonl 的每条 BenchItem
  2. 从 user.target_overrides.theta_api.email 提取用户邮箱
  3. 从问题文本提取日期（relevant_dates），按 case 裁剪数据范围
  4. 有日期 → exam ±30 天 + device ±7 天；无日期 → 全量 exam，跳过 device
  5. 注入 eval.ground_truth_data 字段
  6. 原地更新 JSONL
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

from generator.medhall.ground_truth_builder import build_ground_truth, extract_dates_from_text, get_exam_date_range

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _email_to_dir(email: str, base_dir: Path) -> Path | None:
    """user119@demo → /base_dir/user119_AT_demo/"""
    dirname = email.replace("@", "_AT_")
    d = base_dir / dirname
    return d if d.is_dir() else None


def enrich(jsonl_path: Path, thetagendata_dir: Path) -> int:
    """对 JSONL 文件注入 ground_truth_data，原地更新

    Returns:
        enriched case count
    """
    enriched = 0
    skipped = 0

    # 缓存用户目录和全量时间窗口（避免重复读取 exam_data.json）
    user_dir_cache: dict[str, Path | None] = {}
    exam_range_cache: dict[str, tuple[str, str]] = {}

    # 写入临时文件，成功后替换原文件
    tmp_fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=jsonl_path.parent,
        suffix=".jsonl.tmp",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = Path(tmp_fd.name)

    try:
        with open(jsonl_path, encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                case_id = item.get("id", "?")

                # 提取 email
                email = (
                    item.get("user", {}).get("target_overrides", {}).get("theta_api", {}).get("email", "")
                )
                if not email:
                    logger.warning("[%s] no theta_api email, skipping", case_id)
                    tmp_fd.write(json.dumps(item, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

                # 查找用户目录（缓存）
                if email not in user_dir_cache:
                    user_dir_cache[email] = _email_to_dir(email, thetagendata_dir)
                user_dir = user_dir_cache[email]
                if user_dir is None:
                    logger.warning("[%s] thetagendata dir not found for %s, skipping", case_id, email)
                    tmp_fd.write(json.dumps(item, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

                # 全量 exam 时间窗口（缓存）
                if email not in exam_range_cache:
                    exam_range_cache[email] = get_exam_date_range(user_dir)
                start, end = exam_range_cache[email]

                # 从问题提取日期
                question = (item.get("user", {}).get("strict_inputs") or [""])[0]
                relevant_dates = extract_dates_from_text(question)

                # 按 case 构建（有日期 → 裁剪 exam + device；无日期 → 全量 exam，跳过 device）
                if relevant_dates:
                    gt_data = build_ground_truth(
                        user_dir, device_start=start, device_end=end, relevant_dates=relevant_dates
                    )
                else:
                    gt_data = build_ground_truth(user_dir)

                logger.info(
                    "[%s] %s: %d chars (dates=%s)", case_id, email, len(gt_data), relevant_dates or "none"
                )

                item.setdefault("eval", {})["ground_truth_data"] = gt_data
                tmp_fd.write(json.dumps(item, ensure_ascii=False) + "\n")
                enriched += 1

        tmp_fd.close()
        tmp_path.replace(jsonl_path)
        logger.info("Done: %d enriched, %d skipped. File updated: %s", enriched, skipped, jsonl_path)

    except Exception:
        tmp_fd.close()
        tmp_path.unlink(missing_ok=True)
        raise

    return enriched


def main():
    parser = argparse.ArgumentParser(description="为 medhall JSONL 注入 ground_truth_data")
    parser.add_argument("jsonl_path", type=Path, help="theta.jsonl 文件路径")
    parser.add_argument(
        "--thetagendata-dir",
        type=Path,
        default=Path("/Users/admin/thetagendata"),
        help="thetagendata 根目录（默认 /Users/admin/thetagendata）",
    )
    args = parser.parse_args()

    if not args.jsonl_path.exists():
        parser.error(f"JSONL file not found: {args.jsonl_path}")
    if not args.thetagendata_dir.is_dir():
        parser.error(f"thetagendata dir not found: {args.thetagendata_dir}")

    enrich(args.jsonl_path, args.thetagendata_dir)


if __name__ == "__main__":
    main()
