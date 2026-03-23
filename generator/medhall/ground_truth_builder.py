"""
ThetagenData → ground_truth_data 文本构建器

从 thetagendata 目录读取用户健康档案（profile / exam / events / device_data），
生成紧凑的人可读文本格式，嵌入 BenchItem eval.ground_truth_data 字段。

用法:
    from generator.medhall.ground_truth_builder import build_ground_truth, get_exam_date_range, extract_dates_from_text

    start, end = get_exam_date_range(Path("/path/user119_AT_demo"))
    dates = extract_dates_from_text("2021年11月的体检里，我的收缩压是多少？")
    text = build_ground_truth(Path("/path/user119_AT_demo"), device_start=start, device_end=end, relevant_dates=dates)
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# 保留的事件类型（含补充剂、饮食变化、健康事件）
_RELEVANT_EVENT_TYPES = {"health_event", "diet_change"}

# 日期提取正则
_DATE_YMD = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?")
_DATE_YM = re.compile(r"(\d{4})年(\d{1,2})月")
_DATE_Y = re.compile(r"(\d{4})年")
_DATE_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")


def extract_dates_from_text(text: str) -> list[str]:
    """从中文问题文本提取日期，返回去重排序的 YYYY-MM-DD 列表"""
    dates: set[str] = set()

    # 优先匹配完整日期（年月日）
    for m in _DATE_YMD.finditer(text):
        dates.add(f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}")

    # ISO 格式
    for m in _DATE_ISO.finditer(text):
        dates.add(m.group(1))

    # 年月（排除已被年月日匹配的）
    for m in _DATE_YM.finditer(text):
        d = f"{m.group(1)}-{int(m.group(2)):02d}-01"
        # 如果该年月已有更精确的日期，跳过
        prefix = f"{m.group(1)}-{int(m.group(2)):02d}-"
        if not any(existing.startswith(prefix) for existing in dates):
            dates.add(d)

    # 仅年份（排除已被年月匹配的）
    for m in _DATE_Y.finditer(text):
        prefix = f"{m.group(1)}-"
        if not any(existing.startswith(prefix) for existing in dates):
            dates.add(f"{m.group(1)}-01-01")

    return sorted(dates)


def get_exam_date_range(user_dir: Path, buffer_days: int = 30) -> tuple[str, str]:
    """从 exam_data.json 推算设备数据时间窗口

    Returns:
        (start_date, end_date) — YYYY-MM-DD 格式，含前后 buffer_days 缓冲
    """
    exam_path = user_dir / "exam_data.json"
    if not exam_path.exists():
        return ("", "")

    exams = json.loads(exam_path.read_text(encoding="utf-8"))
    if not exams:
        return ("", "")

    dates = sorted(e.get("exam_date", "") for e in exams if e.get("exam_date"))
    if not dates:
        return ("", "")

    start = datetime.strptime(dates[0], "%Y-%m-%d") - timedelta(days=buffer_days)
    end = datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=buffer_days)
    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def _is_date_near_any(date_str: str, relevant_dates: list[str], buffer_days: int = 7) -> bool:
    """检查日期是否靠近任一 relevant_date ± buffer_days"""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return False
    for rd in relevant_dates:
        rd_dt = datetime.strptime(rd, "%Y-%m-%d")
        if abs((d - rd_dt).days) <= buffer_days:
            return True
    return False


def _is_exam_near_dates(exam_date: str, relevant_dates: list[str], buffer_days: int = 30) -> bool:
    """检查 exam_date 是否在任一 relevant_date ± buffer_days 范围内"""
    try:
        ed = datetime.strptime(exam_date, "%Y-%m-%d")
    except ValueError:
        return True  # 无法解析则保留
    for rd in relevant_dates:
        rd_dt = datetime.strptime(rd, "%Y-%m-%d")
        if abs((ed - rd_dt).days) <= buffer_days:
            return True
    return False


def build_ground_truth(
    user_dir: str | Path,
    device_start: str | None = None,
    device_end: str | None = None,
    relevant_dates: list[str] | None = None,
) -> str:
    """从 thetagendata 用户目录构建 ground_truth_data 文本

    Args:
        user_dir: thetagendata 用户目录路径
        device_start: device_data 全量起始日期（YYYY-MM-DD）
        device_end: device_data 全量结束日期（YYYY-MM-DD）
        relevant_dates: 问题引用的日期列表（YYYY-MM-DD），用于裁剪 exam 和 device_data

    行为:
        - relevant_dates 非空 → exam 只保留 ±30 天内的记录，device_data 缩为 ±7 天窗口
        - relevant_dates 为空 → exam 全量保留，跳过 device_data
    """
    user_dir = Path(user_dir)
    sections: list[str] = []

    # 1. Profile（始终全量）
    profile_path = user_dir / "profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        sections.append(_format_profile(profile))

    # 2. Health Events（始终全量，已按类型过滤）
    events_path = user_dir / "events.json"
    if events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8"))
        sections.append(_format_events(events))

    # 3. Exam Records（有 relevant_dates 则裁剪）
    exam_path = user_dir / "exam_data.json"
    if exam_path.exists():
        exams = json.loads(exam_path.read_text(encoding="utf-8"))
        sections.append(_format_exams(exams, relevant_dates=relevant_dates))

    # 4. Device Data（仅 relevant_dates 非空时加载，按各日期 ±7 天独立过滤）
    if relevant_dates and device_start and device_end:
        device_section = _load_and_format_device_data(user_dir, device_start, device_end, relevant_dates)
        if device_section:
            sections.append(device_section)

    return "\n\n".join(sections)


# ------------------------------------------------------------------
# 格式化函数
# ------------------------------------------------------------------


def _format_profile(profile: dict) -> str:
    demo = profile.get("demographics", {})
    hp = profile.get("health_profile", {})
    phys = hp.get("physical_measurements", {})

    lines = ["## User Profile"]
    parts = []
    if demo.get("age"):
        parts.append(f"Age: {demo['age']}")
    if demo.get("gender"):
        parts.append(f"Gender: {demo['gender']}")
    if demo.get("occupation"):
        parts.append(f"Occupation: {demo['occupation']}")
    if parts:
        lines.append(", ".join(parts))

    if phys.get("height") or phys.get("weight"):
        hw = []
        if phys.get("height"):
            hw.append(f"Height: {phys['height']} cm")
        if phys.get("weight"):
            hw.append(f"Weight: {phys['weight']} kg")
        lines.append(", ".join(hw))

    if hp.get("summary"):
        lines.append(f"Health Summary: {hp['summary']}")

    # 家族史
    fh = hp.get("family_history", [])
    if fh:
        lines.append("Family History:")
        for item in fh:
            lines.append(f"  - {item.get('relative', '?')}: {item.get('condition', '?')}")

    # 过敏
    allergy = hp.get("allergies_and_intolerances", [])
    if allergy:
        lines.append(f"Allergies: {', '.join(str(a) for a in allergy)}")
    else:
        lines.append("Allergies: None")

    # 慢性病
    chronic = hp.get("chronic_conditions", [])
    if chronic:
        lines.append(f"Chronic Conditions: {', '.join(str(c) for c in chronic)}")
    else:
        lines.append("Chronic Conditions: None")

    return "\n".join(lines)


def _format_events(events: list) -> str:
    lines = ["## Health Events"]
    relevant = [e for e in events if e.get("event_type") in _RELEVANT_EVENT_TYPES]
    if not relevant:
        lines.append("No significant health events recorded.")
        return "\n".join(lines)

    relevant.sort(key=lambda e: e.get("start_date", ""))
    for ev in relevant:
        date = ev.get("start_date", "?")
        name = ev.get("event_name", "")
        desc = (ev.get("description", "") or "")[:200]
        dur = ev.get("duration_days")
        dur_str = f" ({dur} days)" if dur else ""
        lines.append(f"{date} | {name}{dur_str} — {desc}")

    return "\n".join(lines)


def _format_exams(exams: list, relevant_dates: list[str] | None = None) -> str:
    lines = ["## Exam Records"]
    included = 0
    for exam in exams:
        date = exam.get("exam_date", "?")
        # 如果有 relevant_dates，只保留时间窗口内的 exam
        if relevant_dates and date != "?" and not _is_exam_near_dates(date, relevant_dates):
            continue
        etype = exam.get("exam_type", "")
        location = exam.get("exam_location", "")
        lines.append(f"\n### {date} ({etype}, {location})")

        indicators = exam.get("indicators", {})
        if isinstance(indicators, dict):
            for _name, ind in sorted(indicators.items()):
                lines.append(_format_indicator(ind))
        elif isinstance(indicators, list):
            for ind in indicators:
                lines.append(_format_indicator(ind))
        included += 1

    if relevant_dates:
        logger.info("Exams: %d/%d included (relevant_dates=%s)", included, len(exams), relevant_dates)
    return "\n".join(lines)


def _format_indicator(ind: dict) -> str:
    name = ind.get("indicator_name", "?")
    val = ind.get("value", "")
    unit = ind.get("unit", "")
    ref = ind.get("reference_range", "")
    status = ind.get("status", "")
    ref_str = f" [ref: {ref}]" if ref else ""
    status_str = f" ({status})" if status else ""
    return f"{name}: {val} {unit}{ref_str}{status_str}"


def _load_and_format_device_data(
    user_dir: Path, start_date: str, end_date: str, relevant_dates: list[str] | None = None
) -> str | None:
    """尝试从 device_data.json 加载；失败则回退到 timeline.json"""
    device_path = user_dir / "device_data.json"
    if device_path.exists():
        try:
            device_data = json.loads(device_path.read_text(encoding="utf-8"))
            return _format_device_data(device_data, start_date, end_date, relevant_dates)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("device_data.json corrupt for %s (%s), falling back to timeline.json", user_dir.name, exc)

    # Fallback: 从 timeline.json 提取 device_indicator
    timeline_path = user_dir / "timeline.json"
    if not timeline_path.exists():
        return None
    try:
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("timeline.json also unreadable for %s: %s", user_dir.name, exc)
        return None
    return _format_device_data_from_timeline(timeline, start_date, end_date, relevant_dates)


def _format_device_data_from_timeline(
    timeline: dict, start_date: str, end_date: str, relevant_dates: list[str] | None = None
) -> str | None:
    """从 timeline.json 的 device_indicator entries 构建设备数据文本"""
    from collections import defaultdict

    days: dict[str, list[dict]] = defaultdict(list)
    for entry in timeline.get("entries", []):
        if entry.get("entry_type") != "device_indicator":
            continue
        date = entry.get("time", "")[:10]
        if not date or not (start_date <= date <= end_date):
            continue
        if relevant_dates and not _is_date_near_any(date, relevant_dates):
            continue
        days[date].append(entry)

    if not days:
        return None

    lines = [f"## Device Data (Daily Records, {start_date} ~ {end_date})"]
    for date in sorted(days):
        lines.append(f"\n### {date}")
        for ind in days[date]:
            name = ind.get("indicator", "?")
            val = ind.get("value", "")
            unit = ind.get("unit", "")
            lines.append(f"{name}: {val} {unit}")

    logger.info("Device data (from timeline): %d days, %d chars", len(days), sum(len(line) for line in lines))
    return "\n".join(lines)


def _format_device_data(
    device_data: list, start_date: str, end_date: str, relevant_dates: list[str] | None = None
) -> str | None:
    """过滤并格式化 device_data，仅保留时间窗口内的每日记录"""
    filtered = [d for d in device_data if start_date <= d.get("date", "") <= end_date]
    if relevant_dates:
        filtered = [d for d in filtered if _is_date_near_any(d.get("date", ""), relevant_dates)]
    if not filtered:
        return None

    lines = [f"## Device Data (Daily Records, {start_date} ~ {end_date})"]
    for day in filtered:
        date = day.get("date", "?")
        lines.append(f"\n### {date}")
        indicators = day.get("indicators", [])
        for ind in indicators:
            name = ind.get("indicator_name", "?")
            val = ind.get("value", "")
            unit = ind.get("unit", "")
            lines.append(f"{name}: {val} {unit}")

    logger.info("Device data: %d days, %d chars", len(filtered), sum(len(line) for line in lines))
    return "\n".join(lines)
