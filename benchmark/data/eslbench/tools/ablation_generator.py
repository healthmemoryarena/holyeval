"""
ablation_generator.py — 基于同一批用户生成 ablation study 数据集

在 full_sample 用户（user110-114）的 device_data.json 上做扰动（删天/加噪），
复用同样的题目，只重算受影响的 ground truth（expected_value）。

扰动类型:
  - del10pct / del20pct / del30pct: 随机删除 10%/20%/30% 的 device-days
  - noise10pct / noise20pct: 对 device values 注入 ±10%/±20% 高斯噪声

用法:
    # 生成所有 5 种扰动
    python -m benchmark.data.eslbench.tools.ablation_generator \\
        --output-dir benchmark/data/eslbench/ --seed 42

    # 生成单种扰动
    python -m benchmark.data.eslbench.tools.ablation_generator \\
        --perturbation del10pct --output-dir benchmark/data/eslbench/ --seed 42
"""

import argparse
import copy
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import ijson
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

# sample.jsonl 中的用户
DEFAULT_USERS = ["user100", "user101", "user102", "user103", "user104"]

# 扰动配置: name → (type, param)
PERTURBATIONS = {
    "del10pct": ("delete", 0.10),
    "del20pct": ("delete", 0.20),
    "del30pct": ("delete", 0.30),
    "noise10pct": ("noise", 0.10),
    "noise20pct": ("noise", 0.20),
}

# 依赖 device_data 的 data_type，需要重算 GT
DEVICE_DEPENDENT_TYPES = {
    "device",
    "device_day_to_indicators",
    "device_timeseries",
    "two_window_delta",
    "three_phase_means",
    "event_window_peak_value",
    "event_window_min_value",
    "event_window_median_value",
    "event_window_stddev_value",
    "event_window_range",
    "event_window_peak_date",
    "peak_date",
}


# ============================================================
# DeviceIndex: 从 device_data.json 构建高效查询索引
# ============================================================


class DeviceIndex:
    """device_data.json 的内存索引"""

    def __init__(self, device_data: list[dict]):
        # {date_str: {indicator_key: (value_num, indicator_name)}}
        self._by_date: dict[str, dict[str, tuple[float | None, str]]] = {}
        # {indicator_key: [(date_str, value_num)]} — 按日期排序
        self._by_indicator: dict[str, list[tuple[str, float | None]]] = {}

        for day in device_data:
            date = day["date"]
            day_map: dict[str, tuple[float | None, str]] = {}
            for ind in day.get("indicators", []):
                key = ind.get("indicator_key", "")
                name = ind.get("indicator_name", "")
                val = ind.get("value_num") if ind.get("value_num") is not None else ind.get("value")
                if isinstance(val, (int, float)):
                    day_map[key] = (float(val), name)
                else:
                    day_map[key] = (None, name)

                if key not in self._by_indicator:
                    self._by_indicator[key] = []
                self._by_indicator[key].append((date, float(val) if isinstance(val, (int, float)) else None))
            self._by_date[date] = day_map

        # 按日期排序
        for key in self._by_indicator:
            self._by_indicator[key].sort(key=lambda x: x[0])

    def get_value(self, date: str, indicator_key: str) -> float | None:
        """获取指定日期指定 indicator 的值"""
        day = self._by_date.get(date, {})
        entry = day.get(indicator_key)
        return entry[0] if entry else None

    def get_indicators_on_date(self, date: str) -> list[str]:
        """获取指定日期所有 indicator 名称（sorted）"""
        day = self._by_date.get(date, {})
        return sorted(entry[1] for entry in day.values() if entry[0] is not None)

    def get_indicator_keys_on_date(self, date: str) -> set[str]:
        """获取指定日期所有 indicator_key"""
        return set(self._by_date.get(date, {}).keys())

    def get_values_in_range(
        self, indicator_key: str, start: str, end: str
    ) -> list[tuple[str, float]]:
        """获取指定 indicator 在日期范围内的 (date, value) 列表（仅 non-None）"""
        entries = self._by_indicator.get(indicator_key, [])
        result = []
        for date, val in entries:
            if start <= date <= end and val is not None:
                result.append((date, val))
        return result

    def has_date(self, date: str) -> bool:
        return date in self._by_date


# ============================================================
# 数据扰动函数
# ============================================================


def perturb_delete(device_data: list[dict], ratio: float, seed: int) -> list[dict]:
    """随机删除 ratio 比例的日期记录"""
    rng = np.random.default_rng(seed)
    n = len(device_data)
    n_delete = int(n * ratio)
    delete_indices = set(rng.choice(n, size=n_delete, replace=False))
    return [day for i, day in enumerate(device_data) if i not in delete_indices]


def perturb_noise(device_data: list[dict], sigma_pct: float, seed: int) -> list[dict]:
    """对所有 numeric indicator values 添加高斯噪声

    value_new = value * (1 + N(0, sigma_pct))
    """
    rng = np.random.default_rng(seed)
    result = []
    for day in device_data:
        new_day = copy.deepcopy(day)
        for ind in new_day.get("indicators", []):
            val = ind.get("value_num") if ind.get("value_num") is not None else ind.get("value")
            if isinstance(val, (int, float)) and val != 0:
                noise_factor = 1.0 + rng.normal(0, sigma_pct)
                new_val = round(val * noise_factor, 4)
                ind["value"] = new_val
                ind["value_num"] = new_val
        result.append(new_day)
    return result


# ============================================================
# GT 重算函数
# ============================================================


def _values_or_none(idx: DeviceIndex, indicator_key: str, start: str, end: str):
    """获取窗口内的值列表，不足则返回 None"""
    pairs = idx.get_values_in_range(indicator_key, start, end)
    if len(pairs) < 1:
        return None, 0
    values = [v for _, v in pairs]
    return values, len(values)


def _name_to_key(indicator_name: str) -> str:
    """indicator_name → indicator_key (如 'FastingBloodGlucose-FBG' → 'fastingbloodglucose_fbg')"""
    return indicator_name.lower().replace("-", "_").replace(" ", "")


def recompute_device_value(idx: DeviceIndex, dr: dict) -> dict:
    """device: 单值查找"""
    date = dr["date"]
    indicator_key = dr.get("indicator_key") or _name_to_key(dr.get("indicator_name", ""))

    val = idx.get_value(date, indicator_key)

    if val is None:
        # 日期可能被删除了
        if not idx.has_date(date):
            return {"expected_value": None, "unanswerable": True,
                    "reason": f"日期 {date} 数据缺失"}
        return {"expected_value": None, "unanswerable": True,
                "reason": f"指标 {dr.get('indicator_name', '')} 在 {date} 无数据"}

    return {"expected_value": round(val, 4)}


def recompute_day_indicators(idx: DeviceIndex, dr: dict) -> dict:
    """device_day_to_indicators: 列出指定日期的 indicator 名称"""
    date = dr["date"]
    if not idx.has_date(date):
        return {"expected_value": [], "unanswerable": True,
                "reason": f"日期 {date} 数据缺失"}
    names = idx.get_indicators_on_date(date)
    # 如果有 group filter，需要过滤
    # data_record 里可能有 indicator_group_filter
    group_filter = dr.get("indicator_group_filter")
    if group_filter and names:
        names = _filter_by_group(names, group_filter)
    return {"expected_value": sorted(names)}


def _filter_by_group(indicator_names: list[str], group: str) -> list[str]:
    """按指标分组过滤"""
    _GROUP_MAP = {
        "sleep": ["DeepSleep", "LightSleep", "REM", "SleepEfficiency", "SleepLatency",
                   "SleepRecovery", "TotalSleepTime", "WakeAfterSleepOnset", "NumberofAwakenings"],
        "cardio": ["HeartRate", "RestingHeartRate", "MaximumHeartRate", "HRV", "RMSSD",
                    "BloodPressure", "Systolic", "Diastolic", "HeartRateRecovery", "LF/HF"],
        "activity": ["Step", "Distance", "Exercise", "ActiveCalories", "ActiveEnergy",
                      "MET", "Training", "VO2Max", "Vigorous", "Moderate", "Light", "Sedentary",
                      "Standing", "Lying"],
        "oxygen": ["SpO2", "LowestSpO2", "MeanSpO2"],
        "metabolic": ["Glucose", "FBG", "MeanGlucose", "BMR", "TDEE", "Calories", "Energy"],
        "weight": ["BodyWeight", "BMI", "BodyFat", "MuscleMass", "VisceralFat"],
    }
    keywords = _GROUP_MAP.get(group, [])
    if not keywords:
        return indicator_names
    return [n for n in indicator_names if any(kw.lower() in n.lower() for kw in keywords)]


def _recompute_window_stat(idx: DeviceIndex, dr: dict, stat: str) -> dict:
    """通用窗口统计"""
    indicator_key = dr.get("indicator_key", "")
    rng = dr.get("range", {})
    start = rng.get("start", "")
    end = rng.get("end", "")

    values, n = _values_or_none(idx, indicator_key, start, end)
    if values is None or n < 1:
        return {"expected_value": None, "unanswerable": True, "n_points": 0,
                "reason": f"窗口 {start}~{end} 内无数据"}

    pairs = idx.get_values_in_range(indicator_key, start, end)

    if stat == "mean":
        return {"expected_value": round(statistics.mean(values), 4), "n_points": n}
    elif stat == "max":
        return {"expected_value": round(max(values), 4), "n_points": n}
    elif stat == "min":
        return {"expected_value": round(min(values), 4), "n_points": n}
    elif stat == "median":
        return {"expected_value": round(statistics.median(values), 4), "n_points": n}
    elif stat == "stddev":
        if n < 2:
            return {"expected_value": 0.0, "n_points": n}
        return {"expected_value": round(statistics.stdev(values), 4), "n_points": n}
    elif stat == "range":
        return {"expected_value": round(max(values) - min(values), 4), "n_points": n}
    elif stat == "argmax_date":
        max_val = max(values)
        for date, val in pairs:
            if val == max_val:
                return {"expected_value": date, "n_points": n}
    return {"expected_value": None}


def recompute_window_mean(idx: DeviceIndex, dr: dict) -> dict:
    """device_timeseries: 窗口均值"""
    return _recompute_window_stat(idx, dr, "mean")


def recompute_peak_date(idx: DeviceIndex, dr: dict) -> dict:
    """peak_date: 窗口内最大值的日期"""
    indicator_key = dr.get("indicator_key", "")
    rng = dr.get("range", {})
    start = rng.get("start", "")
    end = rng.get("end", "")
    pairs = idx.get_values_in_range(indicator_key, start, end)
    if not pairs:
        return {"expected_value": None, "unanswerable": True, "n_points": 0}
    max_pair = max(pairs, key=lambda x: x[1])
    return {"expected_value": max_pair[0], "n_points": len(pairs)}


def recompute_two_window_delta(idx: DeviceIndex, dr: dict) -> dict:
    """two_window_delta: mean(window2) - mean(window1)"""
    indicator_key = dr.get("indicator_key", "")
    w1 = dr.get("window1", {})
    w2 = dr.get("window2", {})

    vals1, n1 = _values_or_none(idx, indicator_key, w1.get("start", ""), w1.get("end", ""))
    vals2, n2 = _values_or_none(idx, indicator_key, w2.get("start", ""), w2.get("end", ""))

    if vals1 is None or vals2 is None:
        return {"expected_value": None, "unanswerable": True,
                "reason": f"窗口数据不足 (w1={n1}, w2={n2})"}

    delta = statistics.mean(vals2) - statistics.mean(vals1)
    return {"expected_value": round(delta, 4), "n_points_w1": n1, "n_points_w2": n2}


def recompute_three_phase(idx: DeviceIndex, dr: dict) -> dict:
    """three_phase_means: [mean_pre, mean_during, mean_post]"""
    indicator_key = dr.get("indicator_key", "")
    phases = [
        dr.get("pre_range", {}),
        dr.get("during_range", {}),
        dr.get("post_range", {}),
    ]
    means = []
    for phase in phases:
        vals, n = _values_or_none(idx, indicator_key, phase.get("start", ""), phase.get("end", ""))
        if vals is None:
            return {"expected_value": None, "unanswerable": True,
                    "reason": "某个 phase 数据不足"}
        means.append(round(statistics.mean(vals), 4))
    return {"expected_value": means}


# dispatch table
RECOMPUTE_DISPATCH = {
    "device": recompute_device_value,
    "device_day_to_indicators": recompute_day_indicators,
    "device_timeseries": recompute_window_mean,
    "two_window_delta": recompute_two_window_delta,
    "three_phase_means": recompute_three_phase,
    "event_window_peak_value": lambda idx, dr: _recompute_window_stat(idx, dr, "max"),
    "event_window_min_value": lambda idx, dr: _recompute_window_stat(idx, dr, "min"),
    "event_window_median_value": lambda idx, dr: _recompute_window_stat(idx, dr, "median"),
    "event_window_stddev_value": lambda idx, dr: _recompute_window_stat(idx, dr, "stddev"),
    "event_window_range": lambda idx, dr: _recompute_window_stat(idx, dr, "range"),
    "event_window_peak_date": lambda idx, dr: _recompute_window_stat(idx, dr, "argmax_date"),
    "peak_date": recompute_peak_date,
}


# ============================================================
# 核心: 对单个用户做扰动 + GT 重算
# ============================================================


def _load_device_data_safe(path: Path) -> list[dict]:
    """安全加载 device_data.json，只保留 ablation 需要的字段

    原始 device_data 每天的 indicators 包含大量 generation_metadata（占 90%+ 体积），
    ablation 只需要 date, indicator_key, indicator_name, value_num/value, unit。
    流式解析 + 字段裁剪大幅降低内存占用。
    """
    def _strip_day(day: dict) -> dict:
        """只保留 ablation 需要的字段，并将 Decimal→float"""
        stripped = {"date": str(day["date"]), "indicators": []}
        for ind in day.get("indicators", []):
            v = ind.get("value_num") if ind.get("value_num") is not None else ind.get("value")
            fv = float(v) if v is not None else None
            stripped["indicators"].append({
                "indicator_key": str(ind.get("indicator_key", "")),
                "indicator_name": str(ind.get("indicator_name", "")),
                "value": fv,
                "value_num": fv,
                "unit": str(ind.get("unit", "")),
            })
        return stripped

    # 先尝试标准 JSON，如果失败用 ijson 流式读取
    items = []
    try:
        with open(path, "rb") as f:
            for day in ijson.items(f, "item"):
                items.append(_strip_day(day))
    except Exception:
        if not items:
            raise ValueError(f"无法从 {path} 读取任何有效数据")
        logger.warning(f"  流式读取恢复 {len(items)} 天数据（文件部分损坏）")

    return items


def load_user_data(user_id: str) -> tuple[list[dict], list[dict]]:
    """加载单个用户的 device_data 和 queries（只加载一次）"""
    user_dir = DATA_DIR / f"{user_id}_AT_demo"

    logger.info(f"  加载 {user_id} device_data.json ...")
    device_data = _load_device_data_safe(user_dir / "device_data.json")
    logger.info(f"  {user_id}: {len(device_data)} days")

    with open(user_dir / "kg_evaluation_queries.json", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("evaluation_queries", data) if isinstance(data, dict) else data

    return device_data, queries


def recompute_gt(
    device_data: list[dict],
    queries: list[dict],
    perturbation_type: str,
    perturbation_param: float,
    seed: int,
    user_id: str = "",
) -> dict[str, dict]:
    """对已加载的 device_data 做扰动并重算 GT

    Returns: {query_id: updated_ground_truth_patch}
    """
    # 1. 扰动
    if perturbation_type == "delete":
        perturbed = perturb_delete(device_data, perturbation_param, seed)
        logger.info(f"  {user_id}: 删除 {len(device_data) - len(perturbed)} days ({perturbation_param:.0%})")
    elif perturbation_type == "noise":
        perturbed = perturb_noise(device_data, perturbation_param, seed)
        logger.info(f"  {user_id}: 注入 ±{perturbation_param:.0%} 高斯噪声")
    else:
        raise ValueError(f"未知扰动类型: {perturbation_type}")

    # 2. 建立扰动后索引
    idx = DeviceIndex(perturbed)

    # 3. 重算 GT
    patches: dict[str, dict] = {}
    for q in queries:
        qid = q["query_id"]
        gt = q["ground_truth"]
        data_type = gt["source_data"]["data_type"]

        if data_type not in DEVICE_DEPENDENT_TYPES:
            continue  # 不受影响，跳过

        dr = gt["source_data"]["data_record"]
        recompute_fn = RECOMPUTE_DISPATCH.get(data_type)
        if recompute_fn is None:
            logger.warning(f"  未实现重算: {data_type} (query {qid})")
            continue

        result = recompute_fn(idx, dr)
        patches[qid] = result

    logger.info(f"  {user_id}: 重算 {len(patches)} 题")
    return patches


# ============================================================
# 生成 ablation JSONL
# ============================================================


def generate_ablation_jsonl(
    sample_jsonl_path: Path,
    perturbation_name: str,
    users: list[str],
    seed: int,
    output_path: Path,
) -> dict:
    """生成单个 ablation JSONL

    逐用户加载 device_data（避免 OOM），处理完立即释放。
    """
    ptype, pparam = PERTURBATIONS[perturbation_name]

    # 对每个用户做扰动 + GT 重算（逐用户加载，避免同时占用过多内存）
    all_patches: dict[str, dict] = {}
    for uid in users:
        device_data, queries = load_user_data(uid)
        # 每个用户用不同的 seed 保证不同用户的扰动不同
        user_seed = seed + hash(uid) % (2**31)
        patches = recompute_gt(device_data, queries, ptype, pparam, user_seed, uid)
        all_patches.update(patches)
        del device_data  # 释放内存

    # 读取原始 sample.jsonl
    items = []
    with open(sample_jsonl_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    # 应用 patches
    stats = {"total": len(items), "patched": 0, "unanswerable": 0, "unchanged": 0}
    output_items = []

    for item in items:
        qid = item["id"]
        new_item = copy.deepcopy(item)

        if qid in all_patches:
            patch = all_patches[qid]
            new_item["eval"]["expected_value"] = patch["expected_value"]

            # 更新 key_points 中的 n_points 信息
            if "n_points" in patch:
                _update_key_points_n(new_item["eval"], patch["n_points"])

            if patch.get("unanswerable"):
                # 标记为不可回答
                new_item["eval"]["unanswerable"] = True
                new_item["eval"]["unanswerable_reason"] = patch.get("reason", "数据缺失")
                stats["unanswerable"] += 1

            stats["patched"] += 1
        else:
            stats["unchanged"] += 1

        output_items.append(new_item)

    # 写出
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return stats


def _update_key_points_n(eval_config: dict, n_points: int):
    """更新 key_points 中的数据点数信息"""
    kps = eval_config.get("key_points", [])
    for i, kp in enumerate(kps):
        if "n_points" in str(kp) or "数据点" in str(kp):
            kps[i] = f"n_points: {n_points}"
            return


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="生成 ablation study 数据集")
    parser.add_argument(
        "--users", nargs="+", default=DEFAULT_USERS,
        help="用户 ID 列表（默认: user110-114）",
    )
    parser.add_argument(
        "--perturbation", type=str, default=None,
        choices=list(PERTURBATIONS.keys()),
        help="指定单种扰动（默认: 生成全部 5 种）",
    )
    parser.add_argument(
        "--sample-jsonl", type=str,
        default=str(Path(__file__).resolve().parent.parent / "sample.jsonl"),
        help="原始 sample.jsonl 路径",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    sample_path = Path(args.sample_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perturbations = [args.perturbation] if args.perturbation else list(PERTURBATIONS.keys())

    for pname in perturbations:
        output_path = output_dir / f"ablation_{pname}_sample.jsonl"
        print(f"\n生成 {pname} → {output_path.name} ...", flush=True)

        stats = generate_ablation_jsonl(
            sample_jsonl_path=sample_path,
            perturbation_name=pname,
            users=args.users,
            seed=args.seed,
            output_path=output_path,
        )

        print(f"  总计: {stats['total']} 题")
        print(f"  重算: {stats['patched']} 题 (unanswerable: {stats['unanswerable']})")
        print(f"  未变: {stats['unchanged']} 题")

    print("\n完成!")


if __name__ == "__main__":
    main()
