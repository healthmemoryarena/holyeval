#!/usr/bin/env bash
# gen_eslbench_jsonl.sh — 从已有 JSONL 生成 eslbench full/sample + 上传 + 评测
#
# 用法:
#   # 从指定 JSONL 生成 full/sample
#   bash gen_eslbench_jsonl.sh --jsonl /path/to/full_sample.jsonl
#
#   # 指定每维度抽样数（逗号分隔，生成多个 sample 文件）
#   bash gen_eslbench_jsonl.sh --jsonl /path/to/full_sample.jsonl --sample-n-per-dim 10,100
#
#   # 生成并上传到 ESL-Bench
#   bash gen_eslbench_jsonl.sh --jsonl /path/to/full_sample.jsonl --upload-eslbench
#
#   # 生成并运行评测
#   bash gen_eslbench_jsonl.sh --jsonl /path/to/full_sample.jsonl --benchmark thetagen_test_3 --dataset full_sample --model gpt-5.4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLYEVAL_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HOLYEVAL_VENV="$HOLYEVAL_DIR/.venv/bin/python"

# 参数默认值
JSONL_FILE=""
SAMPLE_N_PER_DIM="10,100"
SEED=42
UPLOAD_ESLBENCH=false
ESLBENCH_REPO="${ESLBENCH_HF_REPO:-healthmemoryarena/ESL-Bench}"
ESLBENCH_BATCH="${ESLBENCH_HF_BATCH:-202604}"
NO_EVAL=true
MODEL="gpt-5.4"
BENCHMARK="thetagen_test_3"
DATASET="full_sample"
PARALLELISM=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --jsonl) JSONL_FILE="$2"; shift 2 ;;
        --sample-n-per-dim) SAMPLE_N_PER_DIM="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --upload-eslbench) UPLOAD_ESLBENCH=true; shift ;;
        --eslbench-repo) ESLBENCH_REPO="$2"; shift 2 ;;
        --eslbench-batch) ESLBENCH_BATCH="$2"; shift 2 ;;
        --eval) NO_EVAL=false; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        -p) PARALLELISM="$2"; shift 2 ;;
        -*)
            echo "未知参数: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$JSONL_FILE" ]]; then
    echo "用法: $0 --jsonl <path> [--sample-n-per-dim N] [--upload-eslbench] [--eval]"
    exit 1
fi

if [[ ! -f "$JSONL_FILE" ]]; then
    echo "错误: JSONL 文件不存在: $JSONL_FILE" >&2
    exit 1
fi

# ============================================================
# 生成 sample JSONL（按难度分层抽样，支持多个 sample 大小）
# ============================================================

ESLBENCH_DIR="$HOLYEVAL_DIR/benchmark/data/eslbench"
DATE_TAG="$(date +%Y%m%d)"
INPUT_COUNT=$(wc -l < "$JSONL_FILE")

echo ""
echo "[生成 eslbench sample JSONL] (源: $JSONL_FILE, $INPUT_COUNT 条)"

IFS=',' read -ra SAMPLE_SIZES <<< "$SAMPLE_N_PER_DIM"
for N_PER_DIM in "${SAMPLE_SIZES[@]}"; do
    SAMPLE_TOTAL=$((N_PER_DIM * 5))
    SAMPLE_FILE="$ESLBENCH_DIR/sample${SAMPLE_TOTAL}-${DATE_TAG}.jsonl"

    "$HOLYEVAL_VENV" -c "
import json, random
from collections import defaultdict

with open('$JSONL_FILE') as f:
    items = [json.loads(line) for line in f]

groups = defaultdict(list)
for item in items:
    diff = item.get('eval', {}).get('difficulty', 'unknown')
    groups[diff].append(item)

n_per_dim = $N_PER_DIM
rng = random.Random($SEED)
sampled = []
for diff in sorted(groups):
    pool = groups[diff]
    n = min(n_per_dim, len(pool))
    sampled.extend(rng.sample(pool, n))

rng.shuffle(sampled)

with open('$SAMPLE_FILE', 'w') as f:
    for item in sampled:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'  → sample{len(sampled)}: $SAMPLE_FILE ({len(sampled)} 条, {n_per_dim}/dim)')
diff_counts = {}
for item in sampled:
    d = item.get('eval', {}).get('difficulty', 'unknown')
    diff_counts[d] = diff_counts.get(d, 0) + 1
for d in sorted(diff_counts):
    print(f'    {d}: {diff_counts[d]}')
"
done

# ============================================================
# 上传 eslbench JSONL 到 ESL-Bench HF repo（可选）
# ============================================================

# 加载 HF_TOKEN（如未设置则从 .env 读取）
if [[ -z "${HF_TOKEN:-}" && -f "$HOLYEVAL_DIR/.env" ]]; then
    HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' "$HOLYEVAL_DIR/.env" 2>/dev/null || true)"
    export HF_TOKEN
fi

if [[ "$UPLOAD_ESLBENCH" == true ]]; then
    ESLBENCH_JSONL_FILES=( $(find "$ESLBENCH_DIR" -name "*-${DATE_TAG}.jsonl" -type f 2>/dev/null) )

    if [[ ${#ESLBENCH_JSONL_FILES[@]} -gt 0 ]]; then
        echo ""
        echo "[上传 eslbench JSONL 到 ESL-Bench ($ESLBENCH_REPO batch=$ESLBENCH_BATCH)]"
        echo "  (自动去除内部字段: eval.evaluator/expected_value/key_points/source_data, target_overrides)"
        cd "$HOLYEVAL_DIR"
        for jf in "${ESLBENCH_JSONL_FILES[@]}"; do
            BASENAME="$(basename "$jf")"
            PATH_IN_REPO="data/${ESLBENCH_BATCH}/${BASENAME}"
            echo "  上传: $BASENAME -> $PATH_IN_REPO"
            STRIPPED_JSONL="$(mktemp)"
            "$HOLYEVAL_VENV" -c "
import json, sys
with open('$jf') as f_in, open('$STRIPPED_JSONL', 'w') as f_out:
    for line in f_in:
        obj = json.loads(line)
        if 'user' in obj:
            obj['user'].pop('target_overrides', None)
        if 'eval' in obj:
            for k in ('evaluator', 'expected_value', 'key_points', 'source_data'):
                obj['eval'].pop(k, None)
        f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
print(f'  stripped -> $STRIPPED_JSONL')
"
            "$HOLYEVAL_VENV" -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$STRIPPED_JSONL',
    path_in_repo='$PATH_IN_REPO',
    repo_id='$ESLBENCH_REPO',
    repo_type='dataset',
)
print('  -> 完成')
"
            rm -f "$STRIPPED_JSONL"
        done
    else
        echo ""
        echo "[未找到 eslbench/*-${DATE_TAG}.jsonl 文件，跳过上传]"
    fi
else
    echo ""
    echo "[跳过 eslbench JSONL 上传 (需要 --upload-eslbench)]"
fi

# ============================================================
# 运行评测（可选）
# ============================================================

if [[ "$NO_EVAL" == true ]]; then
    echo ""
    echo "[跳过评测 (默认，使用 --eval 开启)]"
    echo ""
    echo "手动运行评测:"
    echo "  cd $HOLYEVAL_DIR"
    echo "  .venv/bin/python -m benchmark.basic_runner $BENCHMARK $DATASET --target-type llm_api --target-model $MODEL -p $PARALLELISM"
else
    echo ""
    echo "[运行评测 ($MODEL + retrieve-tool)]"
    cd "$HOLYEVAL_DIR"
    "$HOLYEVAL_VENV" -m benchmark.basic_runner "$BENCHMARK" "$DATASET" \
        --target-type llm_api \
        --target-model "$MODEL" \
        -p "$PARALLELISM"
fi

echo ""
echo "============================================================"
echo "  完成!"
echo "============================================================"
