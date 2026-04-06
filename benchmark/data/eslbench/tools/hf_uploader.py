"""
hf_uploader.py — 上传用户数据到 HuggingFace 数据集

只上传安全文件（profile.json, exam_data.json, timeline.json），
仅替换明确的 ID 字段（profile_id, user_id, query_id 前缀），
绝不做全局文本替换，避免污染数据字段。

用法:
    python hf_uploader.py \
        --repo cailiang/thetagen \
        --batch 202604 \
        --user-dir /path/to/user_data \
        --dir-name user5022_AT_demo
"""

import argparse
import glob
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

def _id_match(val, raw_id) -> bool:
    """比较 JSON 中的值与 raw_id，兼容 str/int 混合类型。"""
    if val is None or raw_id is None:
        return False
    return str(val) == str(raw_id)


def _sanitize_profile(data: dict, raw_id, dir_name: str) -> dict:
    """仅替换 metadata.profile_id，不触碰其他字段。"""
    meta = data.get("metadata")
    if isinstance(meta, dict) and _id_match(meta.get("profile_id"), raw_id):
        meta["profile_id"] = dir_name
    return data


def _sanitize_timeline(data: dict, raw_id, dir_name: str) -> dict:
    """仅替换顶层 user_id，不递归替换 entries 内的任何值。"""
    if _id_match(data.get("user_id"), raw_id):
        data["user_id"] = dir_name
    return data


def _sanitize_kg_queries(data: dict, raw_id, dir_name: str) -> dict:
    """仅替换顶层 user_id 和 query_id 前缀（<id>_Qxxx → <dir_name>_Qxxx）。"""
    if _id_match(data.get("user_id"), raw_id):
        data["user_id"] = dir_name
    raw_id_str = str(raw_id) if raw_id is not None else None
    for q in data.get("evaluation_queries", []):
        qid = q.get("query_id", "")
        if raw_id_str and isinstance(qid, str):
            prefix, sep, suffix = qid.partition("_")
            if sep and prefix == raw_id_str:
                q["query_id"] = f"{dir_name}_{suffix}"
    return data


def _write_and_upload(api, obj, filename, tmpdir, repo_id, path_in_repo):
    """写入临时文件并上传到 HF。"""
    out = os.path.join(tmpdir, filename)
    with open(out, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  上传: {filename} -> {path_in_repo}")
    api.upload_file(path_or_fileobj=out, path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type="dataset")


def upload_user(repo_id: str, batch: str, user_dir: str, dir_name: str, kg_query_only: bool = False) -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    user_dir = os.path.realpath(user_dir)

    # 从 profile.json 提取原始 ID（可能是数字或字符串）
    raw_id = None
    profile_path = os.path.join(user_dir, "profile.json")
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        raw_id = profile.get("metadata", {}).get("profile_id")

    tmpdir = tempfile.mkdtemp()
    try:
        if not kg_query_only:
            # profile.json — 仅替换 metadata.profile_id
            if os.path.exists(profile_path):
                with open(profile_path) as f:
                    data = json.load(f)
                data = _sanitize_profile(data, raw_id, dir_name)
                _write_and_upload(api, data, "profile.json", tmpdir, repo_id,
                                  f"data/{batch}/{dir_name}/profile.json")

            # exam_data.json — 原样上传，不做任何替换
            exam_path = os.path.join(user_dir, "exam_data.json")
            if os.path.exists(exam_path):
                path_in_repo = f"data/{batch}/{dir_name}/exam_data.json"
                print(f"  上传: exam_data.json -> {path_in_repo}")
                api.upload_file(path_or_fileobj=exam_path, path_in_repo=path_in_repo,
                                repo_id=repo_id, repo_type="dataset")

            # timeline.json — 仅替换顶层 user_id，统一文件名
            tl_candidates = glob.glob(os.path.join(user_dir, "*_timeline.json"))
            tl_path = tl_candidates[0] if tl_candidates else os.path.join(user_dir, "timeline.json")
            if os.path.exists(tl_path):
                with open(tl_path) as f:
                    tl_data = json.load(f)
                if isinstance(tl_data, dict):
                    tl_data = _sanitize_timeline(tl_data, raw_id, dir_name)
                _write_and_upload(api, tl_data, "timeline.json", tmpdir, repo_id,
                                  f"data/{batch}/{dir_name}/timeline.json")
            else:
                print("  警告: timeline.json 不存在，跳过")

        # kg_evaluation_queries.json — 仅在 --kg-query-only 模式下上传
        kq_path = os.path.join(user_dir, "kg_evaluation_queries.json")
        if kg_query_only and os.path.exists(kq_path):
            with open(kq_path) as f:
                kq_data = json.load(f)
            if isinstance(kq_data, dict):
                kq_data = _sanitize_kg_queries(kq_data, raw_id, dir_name)
            _write_and_upload(api, kq_data, "kg_evaluation_queries.json", tmpdir, repo_id,
                              f"data/{batch}/{dir_name}/kg_evaluation_queries.json")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("  -> 上传完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上传用户数据到 HF 数据集")
    parser.add_argument("--repo", required=True, help="HF repo ID")
    parser.add_argument("--batch", required=True, help="批次 (如 202604)")
    parser.add_argument("--user-dir", required=True, help="用户数据目录")
    parser.add_argument("--dir-name", required=True, help="目标目录名 (如 user5022_AT_demo)")
    parser.add_argument("--kg-query-only", action="store_true", help="仅上传 kg_evaluation_queries.json")
    args = parser.parse_args()

    upload_user(args.repo, args.batch, args.user_dir, args.dir_name, kg_query_only=args.kg_query_only)
