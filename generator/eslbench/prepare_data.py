"""
ESLBench 数据准备脚本

Web UI 启动时通过 PrepareManager 自动执行，完成以下步骤:
1. 从 HuggingFace ($THETAGEN_HF_REPO, 默认 cailiang/thetagen) 下载用户数据（基于 manifest.json 增量更新）
2. 为每个用户创建独立 DuckDB（benchmark/data/eslbench/.data/{user_dir}/user.duckdb）

与 thetagen/prepare_data 的区别: 不生成 JSONL 评测集（eslbench 已有自己的 JSONL 数据文件）。

目录名即邮箱（_AT_ 替换 @），如 user110_AT_demo → user110@demo。
自动发现用户目录，无需硬编码用户列表。

变更检测: 通过 manifest.json 的 version + batch checksum 实现增量更新，
只下载新增或变更的批次，避免全量重下载。

用法:
    python -m generator.eslbench.prepare_data
    python -m generator.eslbench.prepare_data --force   # 强制重建
"""

import argparse
import json
import os
from pathlib import Path

HF_REPO = os.getenv("THETAGEN_HF_REPO", "cailiang/thetagen")

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data" / "eslbench"
DATA_DIR = BENCHMARK_DIR / ".data"
LOCAL_MANIFEST_FILE = DATA_DIR / ".manifest.json"


def _dir_to_email(dir_name: str) -> str:
    """目录名 → 邮箱: user110_AT_demo → user110@demo"""
    return dir_name.replace("_AT_", "@")


def _discover_user_dirs() -> list[Path]:
    """自动发现 .data/ 下的用户目录（排除隐藏目录）"""
    if not DATA_DIR.is_dir():
        return []
    return sorted(d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))


def _find_timeline(user_dir: Path) -> Path | None:
    """在用户目录中查找 timeline.json（兼容不同命名约定）"""
    candidates = list(user_dir.glob("*_timeline.json"))
    return candidates[0] if candidates else None


# ==================== Step 1: HuggingFace 下载（manifest 驱动） ====================


def _load_local_manifest() -> dict:
    """读取本地 manifest（不存在则返回空 dict）"""
    if LOCAL_MANIFEST_FILE.exists():
        return json.loads(LOCAL_MANIFEST_FILE.read_text(encoding="utf-8"))
    return {}


def _save_local_manifest(manifest: dict) -> None:
    """保存 manifest 到本地"""
    LOCAL_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _fetch_remote_manifest() -> dict | None:
    """从 HuggingFace 拉取 manifest.json（单文件，极快）"""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="manifest.json", repo_type="dataset")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[eslbench] 警告: 无法获取远端 manifest ({e})")
        return None


def _check_download_complete() -> bool:
    """检查是否有至少一个用户数据已下载完整"""
    user_dirs = _discover_user_dirs()
    if not user_dirs:
        return False
    for d in user_dirs:
        if not (d / "kg_evaluation_queries.json").exists():
            return False
    return True


def download_from_hf(force: bool = False) -> tuple[Path, bool]:
    """基于 manifest.json 增量下载用户数据到 .data/ 目录。返回 (data_dir, data_changed)"""
    import shutil

    from huggingface_hub import snapshot_download

    # 拉取远端 manifest
    remote_manifest = _fetch_remote_manifest()
    if remote_manifest is None:
        if _check_download_complete() and not force:
            print("[eslbench] 无法连接远端, 使用本地已有数据")
            return DATA_DIR, False
        print("[eslbench] 错误: 无法获取远端 manifest 且本地无数据")
        return DATA_DIR, False

    local_manifest = _load_local_manifest()

    # 快速判断: version 相同且非强制 → 无变更
    if not force and remote_manifest.get("version") == local_manifest.get("version"):
        if _check_download_complete():
            print("[eslbench] manifest version 一致, 跳过下载")
            return DATA_DIR, False

    # 找出需要下载的批次（新增或 checksum 变更）
    batches_to_download: list[str] = []
    remote_batches = remote_manifest.get("batches", {})
    local_batches = local_manifest.get("batches", {})

    for batch_id, batch_info in remote_batches.items():
        local_batch = local_batches.get(batch_id)
        if force or not local_batch or local_batch.get("checksum") != batch_info.get("checksum"):
            batches_to_download.append(batch_id)

    if not batches_to_download and _check_download_complete():
        print("[eslbench] 所有批次 checksum 一致, 跳过下载")
        _save_local_manifest(remote_manifest)
        return DATA_DIR, False

    # 逐批次下载
    total_copied = 0
    for batch_id in batches_to_download:
        batch_info = remote_batches[batch_id]
        user_count = batch_info.get("user_count", "?")
        print(f"[eslbench] 下载批次 {batch_id} ({user_count} 个用户)...")

        snapshot_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"data/{batch_id}/*/*.json"],
        )
        snapshot_batch = Path(snapshot_dir) / "data" / batch_id

        if not snapshot_batch.exists():
            print(f"[eslbench] 警告: 批次 {batch_id} 下载后目录不存在, 跳过")
            continue

        # 复制到 .data/{user_dir}/ 扁平结构（去掉 batch 层级）
        copied = 0
        for src_dir in sorted(snapshot_batch.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("."):
                continue
            dst_dir = DATA_DIR / src_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_dir.iterdir():
                if src_file.is_file() and src_file.suffix == ".json":
                    shutil.copy2(str(src_file), str(dst_dir / src_file.name))
                    copied += 1
        total_copied += copied
        print(f"[eslbench] 批次 {batch_id}: {copied} 个文件")

    # 保存 manifest
    _save_local_manifest(remote_manifest)
    print(f"[eslbench] 下载完成: {len(batches_to_download)} 个批次, {total_copied} 个文件")
    return DATA_DIR, total_copied > 0


# ==================== Step 2: Per-user DuckDB ====================


def _bulk_insert_ndjson(con: "duckdb.DuckDBPyConnection", table: str, columns: list[str], rows: list[dict]) -> None:
    """通过临时 NDJSON 文件批量导入数据到 DuckDB（比 executemany 快 200 倍以上）"""
    import tempfile

    if not rows:
        return

    col_select = ", ".join(columns)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False, encoding="utf-8")
    try:
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.close()
        con.execute(f"INSERT INTO {table} SELECT {col_select} FROM read_ndjson_auto('{tmp.name}')")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def create_user_duckdb(user_dir: Path, force: bool = False) -> None:
    """为单个用户创建 DuckDB 数据库"""
    import duckdb

    db_path = user_dir / "user.duckdb"
    dir_name = user_dir.name
    email = _dir_to_email(dir_name)

    if db_path.exists() and not force:
        return

    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))

    # 建表
    con.execute("""
        CREATE TABLE device_indicators (
            user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,
            device_type VARCHAR, value VARCHAR, unit VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE exam_indicators (
            user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,
            exam_type VARCHAR, exam_location VARCHAR, value VARCHAR, unit VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE events (
            user_id VARCHAR, time TIMESTAMP, end_time TIMESTAMP,
            event_id VARCHAR, event_type VARCHAR, event_name VARCHAR,
            start_date DATE, duration_days INTEGER,
            interrupted BOOLEAN, interruption_date DATE
        )
    """)
    con.execute("""
        CREATE TABLE event_indicators (
            user_id VARCHAR, event_name VARCHAR, start_date DATE,
            indicator_name VARCHAR, indicator_key VARCHAR,
            expected_change VARCHAR, impact_level VARCHAR
        )
    """)

    # 加载 timeline（自动查找 *_timeline.json）
    tl_path = _find_timeline(user_dir)
    if tl_path:
        with open(tl_path, encoding="utf-8") as f:
            data = json.load(f)

        device_rows, exam_rows, event_rows = [], [], []
        for e in data.get("entries", []):
            if e["entry_type"] == "device_indicator":
                raw_val = e["value"]
                device_rows.append(
                    {
                        "user_id": email,
                        "time": e["time"],
                        "indicator": e["indicator"],
                        "device_type": e.get("device_type"),
                        "value": None if raw_val is None else str(raw_val),
                        "unit": e.get("unit"),
                    }
                )
            elif e["entry_type"] == "exam_indicator":
                raw_val = e["value"]
                exam_rows.append(
                    {
                        "user_id": email,
                        "time": e["time"],
                        "indicator": e["indicator"],
                        "exam_type": e.get("exam_type"),
                        "exam_location": e.get("exam_location"),
                        "value": None if raw_val is None else str(raw_val),
                        "unit": e.get("unit"),
                    }
                )
            elif e["entry_type"] == "event":
                ev = e["event"]
                event_rows.append(
                    {
                        "user_id": email,
                        "time": e["time"],
                        "end_time": e.get("end_time"),
                        "event_id": ev["event_id"],
                        "event_type": ev["event_type"],
                        "event_name": ev["event_name"],
                        "start_date": ev["start_date"],
                        "duration_days": ev.get("duration_days"),
                        "interrupted": ev.get("interrupted"),
                        "interruption_date": ev.get("interruption_date"),
                    }
                )

        _bulk_insert_ndjson(
            con,
            "device_indicators",
            ["user_id", "time::TIMESTAMP AS time", "indicator", "device_type", "value", "unit"],
            device_rows,
        )
        _bulk_insert_ndjson(
            con,
            "exam_indicators",
            ["user_id", "time::TIMESTAMP AS time", "indicator", "exam_type", "exam_location", "value", "unit"],
            exam_rows,
        )
        _bulk_insert_ndjson(
            con,
            "events",
            [
                "user_id",
                "time::TIMESTAMP AS time",
                "end_time::TIMESTAMP AS end_time",
                "event_id",
                "event_type",
                "event_name",
                "start_date::DATE AS start_date",
                "duration_days::INTEGER AS duration_days",
                "interrupted::BOOLEAN AS interrupted",
                "interruption_date::DATE AS interruption_date",
            ],
            event_rows,
        )

    # 加载 event_indicators（从 events.json 展平 affected_indicators）
    events_json_path = user_dir / "events.json"
    if events_json_path.exists():
        with open(events_json_path, encoding="utf-8") as f:
            events_data = json.load(f)

        ei_rows = []
        if isinstance(events_data, list):
            for ev in events_data:
                if not isinstance(ev, dict):
                    continue
                ev_name = ev.get("event_name", "")
                ev_start = ev.get("start_date")
                for ai in ev.get("affected_indicators", []):
                    if not isinstance(ai, dict):
                        continue
                    ei_rows.append(
                        {
                            "user_id": email,
                            "event_name": ev_name,
                            "start_date": ev_start,
                            "indicator_name": ai.get("indicator_name", ""),
                            "indicator_key": ai.get("indicator_key", ""),
                            "expected_change": ai.get("expected_change"),
                            "impact_level": ai.get("impact_level"),
                        }
                    )

        _bulk_insert_ndjson(
            con,
            "event_indicators",
            [
                "user_id",
                "event_name",
                "start_date::DATE AS start_date",
                "indicator_name",
                "indicator_key",
                "expected_change",
                "impact_level",
            ],
            ei_rows,
        )

    # 创建索引
    con.execute("CREATE INDEX idx_device_time ON device_indicators (user_id, time)")
    con.execute("CREATE INDEX idx_device_ind ON device_indicators (indicator)")
    con.execute("CREATE INDEX idx_exam_time ON exam_indicators (user_id, time)")
    con.execute("CREATE INDEX idx_event_user ON events (user_id)")
    con.execute("CREATE INDEX idx_ei_event ON event_indicators (event_name)")
    con.execute("CREATE INDEX idx_ei_indicator ON event_indicators (indicator_key)")

    con.close()


def build_all_duckdb(force: bool = False) -> None:
    """为所有用户创建独立 DuckDB"""
    user_dirs = _discover_user_dirs()
    created = 0
    skipped = 0

    for user_dir in user_dirs:
        db_path = user_dir / "user.duckdb"
        if db_path.exists() and not force:
            skipped += 1
            continue

        create_user_duckdb(user_dir, force=force)
        created += 1

    print(f"[eslbench] DuckDB 创建完成: {created} 个新建, {skipped} 个已存在跳过")


# ==================== Main ====================


def _check_local_complete() -> bool:
    """快速本地完整性检查: manifest + 用户数据 + DuckDB 全部就绪"""
    if not LOCAL_MANIFEST_FILE.exists():
        return False
    user_dirs = _discover_user_dirs()
    if not user_dirs:
        return False
    for d in user_dirs:
        if not (d / "kg_evaluation_queries.json").exists():
            return False
        if not (d / "user.duckdb").exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="ESLBench 数据准备")
    parser.add_argument("--force", action="store_true", help="强制重建（忽略已存在的文件）")
    args = parser.parse_args()

    force = args.force

    print("=" * 60)
    print("ESLBench 数据准备")
    print("=" * 60)

    # 快速路径: 本地数据完整 + manifest version 一致 → 秒退
    if not force and _check_local_complete():
        remote_manifest = _fetch_remote_manifest()
        local_manifest = _load_local_manifest()
        if remote_manifest and remote_manifest.get("version") == local_manifest.get("version"):
            print("[eslbench] 本地数据完整且 manifest version 一致, 跳过准备")
            print("=" * 60)
            return

    # Step 1: 从 HuggingFace 下载（检测远端变更）
    print("\n--- Step 1: 下载 HuggingFace 数据 ---")
    _, data_changed = download_from_hf(force=force)

    # 远端数据有变更时，级联重建 DuckDB
    rebuild = force or data_changed
    if data_changed and not force:
        print("[eslbench] 远端数据有更新, 将级联重建 DuckDB")

    # Step 2: 创建每用户 DuckDB
    print("\n--- Step 2: 创建每用户 DuckDB ---")
    build_all_duckdb(force=rebuild)

    # 汇总
    print("\n" + "=" * 60)
    print("准备完成!")
    print(f"  数据目录: {DATA_DIR}/")
    user_dirs = _discover_user_dirs()
    db_count = sum(1 for d in user_dirs if (d / "user.duckdb").exists())
    print(f"  用户数据库: {db_count}/{len(user_dirs)} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()
