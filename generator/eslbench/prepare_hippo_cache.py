"""
HippoRAG 预构建缓存下载脚本

从 HuggingFace ($ESLBENCH_HF_REPO, 默认 healthmemoryarena/ESL-Bench) 的 hippo_cache/ 目录
下载预构建的 HippoRAG 缓存到 benchmark/data/{benchmark}/.hippo_work_dirs/。

缓存包含: chunk_embeddings.npz, chunks.json, entity_embeddings.npz,
entity_list.json, graph.pkl, index_meta.json 等文件。

跳过逻辑: 如果本地用户目录已有完整缓存（index_meta.json 存在），则跳过该用户。

用法:
    python -m generator.eslbench.prepare_hippo_cache
    python -m generator.eslbench.prepare_hippo_cache --force   # 强制重新下载
"""

import argparse
import os
import shutil
from pathlib import Path

HF_REPO = os.getenv("ESLBENCH_HF_REPO", os.getenv("THETAGEN_HF_REPO", "healthmemoryarena/ESL-Bench"))

# 默认路径（ESLBench），ThetaGen 通过 download_hippo_cache(benchmark_dir=...) 覆盖
BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data" / "eslbench"


def _check_user_cache_complete(user_dir: Path) -> bool:
    """检查单个用户的 hippo 缓存是否完整"""
    required = ["index_meta.json", "chunks.json", "chunk_embeddings.npz",
                 "entity_list.json", "entity_embeddings.npz", "graph.pkl"]
    return all((user_dir / f).exists() for f in required)


def download_hippo_cache(*, benchmark_dir: Path | None = None, force: bool = False) -> bool:
    """从 HuggingFace 下载 HippoRAG 预构建缓存。返回是否有新下载。"""
    from huggingface_hub import snapshot_download

    if benchmark_dir is None:
        benchmark_dir = BENCHMARK_DIR
    hippo_work_dir = benchmark_dir / ".hippo_work_dirs"

    # 快速检查: 本地已有完整缓存则跳过
    if not force and hippo_work_dir.is_dir():
        user_dirs = sorted(d for d in hippo_work_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        if user_dirs and all(_check_user_cache_complete(d) for d in user_dirs):
            print(f"[hippo_cache] 本地缓存完整 ({len(user_dirs)} 个用户), 跳过下载")
            return False

    # 下载 hippo_cache/ 目录
    print("[hippo_cache] 从 HuggingFace 下载预构建缓存...")
    try:
        snapshot_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=["hippo_cache/**"],
        )
    except Exception as e:
        print(f"[hippo_cache] 下载失败: {e}")
        return False

    src_root = Path(snapshot_dir) / "hippo_cache"
    if not src_root.exists():
        print("[hippo_cache] 远端无 hippo_cache/ 目录, 跳过")
        return False

    # 复制到 .hippo_work_dirs/
    hippo_work_dir.mkdir(parents=True, exist_ok=True)
    copied_users = 0
    for src_dir in sorted(src_root.iterdir()):
        if not src_dir.is_dir() or src_dir.name.startswith("."):
            continue

        dst_dir = hippo_work_dir / src_dir.name

        # 非强制模式下，已有完整缓存则跳过
        if not force and _check_user_cache_complete(dst_dir):
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.iterdir():
            if src_file.is_file():
                shutil.copy2(str(src_file), str(dst_dir / src_file.name))

        copied_users += 1

    if copied_users:
        print(f"[hippo_cache] 下载完成: {copied_users} 个用户缓存已更新")
    else:
        print("[hippo_cache] 所有用户缓存已是最新")
    return copied_users > 0


def main():
    parser = argparse.ArgumentParser(description="下载 HippoRAG 预构建缓存")
    parser.add_argument("--force", action="store_true", help="强制重新下载（覆盖已有缓存）")
    args = parser.parse_args()

    print("=" * 60)
    print("HippoRAG 缓存下载")
    print("=" * 60)

    download_hippo_cache(force=args.force)

    # 汇总
    hippo_work_dir = BENCHMARK_DIR / ".hippo_work_dirs"
    if hippo_work_dir.is_dir():
        user_dirs = sorted(d for d in hippo_work_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        complete = sum(1 for d in user_dirs if _check_user_cache_complete(d))
        print(f"\n  缓存目录: {hippo_work_dir}/")
        print(f"  用户缓存: {complete}/{len(user_dirs)} 个完整")
    print("=" * 60)


if __name__ == "__main__":
    main()
