"""
hf_uploader.py — 上传用户数据到 HuggingFace 数据集

只上传安全文件（profile.json, exam_data.json, timeline.json），
自动将数字 ID 替换为 email 格式的目录名。

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


def upload_user(repo_id: str, batch: str, user_dir: str, dir_name: str) -> None:
    api = HfApi()
    user_dir = os.path.realpath(user_dir)

    # 从 profile.json 提取原始数字 ID
    user_id_raw = None
    profile_path = os.path.join(user_dir, "profile.json")
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        user_id_raw = profile.get("metadata", {}).get("profile_id")

    tmpdir = tempfile.mkdtemp()
    try:
        # profile.json — 替换 profile_id
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                data = json.load(f)
            if user_id_raw and data.get("metadata", {}).get("profile_id") == user_id_raw:
                data["metadata"]["profile_id"] = dir_name
            out = os.path.join(tmpdir, "profile.json")
            with open(out, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            path_in_repo = f"data/{batch}/{dir_name}/profile.json"
            print(f"  上传: profile.json -> {path_in_repo}")
            api.upload_file(path_or_fileobj=out, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="dataset")

        # exam_data.json — 原样上传
        exam_path = os.path.join(user_dir, "exam_data.json")
        if os.path.exists(exam_path):
            path_in_repo = f"data/{batch}/{dir_name}/exam_data.json"
            print(f"  上传: exam_data.json -> {path_in_repo}")
            api.upload_file(path_or_fileobj=exam_path, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="dataset")

        # timeline.json — 替换 user_id，统一文件名
        tl_candidates = glob.glob(os.path.join(user_dir, "*_timeline.json"))
        tl_path = tl_candidates[0] if tl_candidates else os.path.join(user_dir, "timeline.json")
        if os.path.exists(tl_path):
            with open(tl_path) as f:
                tl_data = json.load(f)
            if isinstance(tl_data, dict) and user_id_raw and tl_data.get("user_id") == user_id_raw:
                tl_data["user_id"] = dir_name
            out = os.path.join(tmpdir, "timeline.json")
            with open(out, "w") as f:
                json.dump(tl_data, f, ensure_ascii=False, indent=2)
            path_in_repo = f"data/{batch}/{dir_name}/timeline.json"
            print(f"  上传: timeline.json -> {path_in_repo}")
            api.upload_file(path_or_fileobj=out, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="dataset")
        else:
            print("  警告: timeline.json 不存在，跳过")

        # kg_evaluation_queries.json — 替换 user_id，作为公开测试集上传
        kq_path = os.path.join(user_dir, "kg_evaluation_queries.json")
        if os.path.exists(kq_path):
            with open(kq_path) as f:
                kq_data = json.load(f)
            if isinstance(kq_data, dict) and user_id_raw and kq_data.get("user_id") == user_id_raw:
                kq_data["user_id"] = dir_name
            # 替换 query_id 前缀
            for q in kq_data.get("evaluation_queries", []):
                qid = q.get("query_id", "")
                parts = qid.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    q["query_id"] = f"{dir_name}_{parts[1]}"
            out = os.path.join(tmpdir, "kg_evaluation_queries.json")
            with open(out, "w") as f:
                json.dump(kq_data, f, ensure_ascii=False, indent=2)
            path_in_repo = f"data/{batch}/{dir_name}/kg_evaluation_queries.json"
            print(f"  上传: kg_evaluation_queries.json -> {path_in_repo}")
            api.upload_file(path_or_fileobj=out, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="dataset")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("  -> 上传完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上传用户数据到 HF 数据集")
    parser.add_argument("--repo", required=True, help="HF repo ID")
    parser.add_argument("--batch", required=True, help="批次 (如 202604)")
    parser.add_argument("--user-dir", required=True, help="用户数据目录")
    parser.add_argument("--dir-name", required=True, help="目标目录名 (如 user5022_AT_demo)")
    args = parser.parse_args()

    upload_user(args.repo, args.batch, args.user_dir, args.dir_name)
