#!/usr/bin/env bash
# conv_eval.sh — 全流程: 生成题目 → 转 JSONL → 建 DuckDB → 评测
#
# 用法:
#   # 传 email（自动从 .data/ 查找用户目录）
#   bash conv_eval.sh --email user5022@demo
#
#   # 传 email + 指定用户数据源目录
#   bash conv_eval.sh --email user5022@demo --user-dir /path/to/user_dir
#
#   # 指定模型
#   bash conv_eval.sh --email user5022@demo --model gpt-5.4
#
#   # 跳过题目生成（已有 kg_evaluation_queries.json）
#   bash conv_eval.sh --email user5022@demo --skip-query-gen
#
#   # 只做数据准备不跑评测
#   bash conv_eval.sh --email user5022@demo --no-eval
#
#   # 批量处理多个用户（逗号分隔 email）
#   bash conv_eval.sh --email user5022@demo,user5023@demo,user5024@demo --no-eval
#
#   # 指定数字 user_id 目录（自动从 DB 查 email）
#   bash conv_eval.sh --user-dir /path/to/3941
#
#   # 逗号分隔多个目录
#   bash conv_eval.sh --user-dir /path/to/3941,/path/to/3942 --no-eval
#
#   # 父目录（自动扫描所有数字子目录）
#   bash conv_eval.sh --user-dir /path/to/data/ --no-eval
#
# 前置条件:
#   - thetagendata repo 有 .venv（用于题目生成）
#   - holyeval repo 有 .venv（用于转换 + 评测）
#   - 用户目录下有: device_data.json, events.json, exam_data.json, profile.json, *_timeline.json

set -euo pipefail

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLYEVAL_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HOLYEVAL_VENV="$HOLYEVAL_DIR/.venv/bin/python"

# thetagendata repo（题目生成器所在位置）
THETAGENDATA_DIR="${THETAGENDATA_DIR:-$HOME/caill/thetagendata}"
THETAGENDATA_VENV="$THETAGENDATA_DIR/.venv/bin/python"

# eslbench benchmark 数据目录（工具读取 .data/ 的位置）
THETAGEN_DATA_DIR="$HOLYEVAL_DIR/benchmark/data/eslbench/.data"

# ============================================================
# 参数解析
# ============================================================

USER_EMAIL=""
USER_DIR=""
MODEL="gpt-5.4"
BENCHMARK="thetagen_test_3"
DATASET="full_sample"
SKIP_QUERY_GEN=false
NO_EVAL=false
UPLOAD_HF=false
HF_REPO="${THETAGEN_HF_REPO:-cailiang/thetagen}"
HF_BATCH="${THETAGEN_HF_BATCH:-202604}"
UPLOAD_ESLBENCH=false
ESLBENCH_REPO="${ESLBENCH_HF_REPO:-healthmemoryarena/ESL-Bench}"
ESLBENCH_BATCH="${ESLBENCH_HF_BATCH:-202604}"
QUERIES_PER_USER=100
SEED=42
PARALLELISM=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --email) USER_EMAIL="$2"; shift 2 ;;
        --user-dir) USER_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --skip-query-gen) SKIP_QUERY_GEN=true; shift ;;
        --no-eval) NO_EVAL=true; shift ;;
        --upload-hf) UPLOAD_HF=true; shift ;;
        --hf-repo) HF_REPO="$2"; shift 2 ;;
        --hf-batch) HF_BATCH="$2"; shift 2 ;;
        --upload-eslbench) UPLOAD_ESLBENCH=true; shift ;;
        --eslbench-repo) ESLBENCH_REPO="$2"; shift 2 ;;
        --eslbench-batch) ESLBENCH_BATCH="$2"; shift 2 ;;
        --queries-per-user) QUERIES_PER_USER="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -p) PARALLELISM="$2"; shift 2 ;;
        -*)
            echo "未知参数: $1" >&2
            exit 1
            ;;
        *)
            # 向后兼容: 位置参数 → email
            if [[ -z "$USER_EMAIL" ]]; then
                USER_EMAIL="$1"
            fi
            shift
            ;;
    esac
done

# ============================================================
# 辅助函数
# ============================================================

email_to_dirname() {
    echo "$1" | sed 's/@/_AT_/'
}

# 从 DB 批量查询 user_id → email（输入: 逗号分隔的 id，输出: id,email 逐行）
lookup_emails_from_db() {
    local id_list="$1"
    local env_file="$THETAGENDATA_DIR/.env"
    if [[ ! -f "$env_file" ]]; then
        echo "错误: 找不到 thetagendata .env: $env_file" >&2
        return 1
    fi
    "$THETAGENDATA_VENV" -c "
from dotenv import load_dotenv
import os, psycopg2
load_dotenv('$env_file')
conn = psycopg2.connect(
    host=os.getenv('pg_host'), port=os.getenv('pg_port'),
    user=os.getenv('pg_user'), password=os.getenv('pg_password'),
    dbname=os.getenv('pg_dbname'))
cur = conn.cursor()
ids = [x.strip() for x in '$id_list'.split(',')]
cur.execute('SELECT id::varchar, email FROM theta_ai.health_app_user WHERE id::varchar = ANY(%s)', (ids,))
for row in cur.fetchall():
    print(f'{row[0]},{row[1]}')
conn.close()
"
}

# ============================================================
# 展开 --user-dir 为目录列表（支持: 单目录, 逗号分隔, 父目录含数字子目录）
# ============================================================

declare -a USER_DIR_LIST=()

if [[ -n "$USER_DIR" ]]; then
    IFS=',' read -ra _RAW_DIRS <<< "$USER_DIR"
    for _d in "${_RAW_DIRS[@]}"; do
        _d="$(echo "$_d" | xargs)"  # trim
        if [[ -d "$_d" ]]; then
            _base="$(basename "$_d")"
            if [[ "$_base" =~ ^[0-9]+$ ]]; then
                # 单个数字目录
                USER_DIR_LIST+=("$_d")
            else
                # 可能是父目录，扫描数字子目录
                _found=false
                for _sub in "$_d"/*/; do
                    [[ -d "$_sub" ]] || continue
                    if [[ "$(basename "$_sub")" =~ ^[0-9]+$ ]]; then
                        USER_DIR_LIST+=("${_sub%/}")
                        _found=true
                    fi
                done
                if [[ "$_found" == false ]]; then
                    # 非数字子目录，当作普通单目录（如 user5022_AT_demo）
                    USER_DIR_LIST+=("$_d")
                fi
            fi
        else
            echo "警告: 目录不存在，跳过: $_d" >&2
        fi
    done
fi

# ============================================================
# 构建 EMAIL_LIST + DIR_MAP（email → user_dir 的关联）
# ============================================================

declare -a EMAIL_LIST=()
declare -A DIR_MAP=()  # email → user_dir

if [[ ${#USER_DIR_LIST[@]} -gt 0 ]]; then
    # 从 --user-dir 构建列表
    # 收集需要 DB 查询的数字 ID
    declare -a NUMERIC_IDS=()
    declare -A ID_TO_DIR=()
    for _d in "${USER_DIR_LIST[@]}"; do
        _base="$(basename "$_d")"
        if [[ "$_base" =~ ^[0-9]+$ ]]; then
            NUMERIC_IDS+=("$_base")
            ID_TO_DIR["$_base"]="$_d"
        else
            # 非数字目录名，尝试转为 email（user5022_AT_demo → user5022@demo）
            _email="${_base//_AT_/@}"
            EMAIL_LIST+=("$_email")
            DIR_MAP["$_email"]="$_d"
        fi
    done

    # 批量 DB 查询数字 ID → email
    if [[ ${#NUMERIC_IDS[@]} -gt 0 ]]; then
        echo "从数据库查询 ${#NUMERIC_IDS[@]} 个 user_id 的 email..."
        _id_csv=$(IFS=','; echo "${NUMERIC_IDS[*]}")
        while IFS=',' read -r _id _email; do
            [[ -z "$_id" ]] && continue
            echo "  → user_id=$_id → email=$_email"
            EMAIL_LIST+=("$_email")
            DIR_MAP["$_email"]="${ID_TO_DIR[$_id]}"
        done < <(lookup_emails_from_db "$_id_csv")

        # 检查是否有未找到的 ID
        for _id in "${NUMERIC_IDS[@]}"; do
            _found=false
            for _e in "${EMAIL_LIST[@]}"; do
                if [[ "${DIR_MAP[$_e]}" == "${ID_TO_DIR[$_id]}" ]]; then
                    _found=true; break
                fi
            done
            if [[ "$_found" == false ]]; then
                echo "错误: 数据库中找不到 user_id=$_id 对应的 email" >&2
                exit 1
            fi
        done
    fi
elif [[ -n "$USER_EMAIL" ]]; then
    # 从 --email 构建列表
    IFS=',' read -ra EMAIL_LIST <<< "$USER_EMAIL"
else
    echo "用法: $0 --email <emails> | --user-dir <dirs> [--model MODEL] [--skip-query-gen] [--no-eval]"
    echo ""
    echo "示例:"
    echo "  $0 --email user5022@demo --model gpt-5.4"
    echo "  $0 --user-dir /path/to/3941                    # 单个数字目录，自动查 email"
    echo "  $0 --user-dir /path/to/3941,/path/to/3942      # 逗号分隔多个目录"
    echo "  $0 --user-dir /path/to/data/                   # 父目录，自动扫描数字子目录"
    echo "  $0 --email user5022@demo,user5023@demo         # 逗号分隔多个 email"
    exit 1
fi

echo "共 ${#EMAIL_LIST[@]} 个用户待处理"

# ============================================================
# 主循环: 逐用户处理
# ============================================================

for CURRENT_EMAIL in "${EMAIL_LIST[@]}"; do
    CURRENT_EMAIL="$(echo "$CURRENT_EMAIL" | xargs)"  # trim 空格
    USER_DIR_NAME="$(email_to_dirname "$CURRENT_EMAIL")"

    # 确定用户数据目录: DIR_MAP > THETAGEN_DATA_DIR > 报错
    if [[ -n "${DIR_MAP[$CURRENT_EMAIL]:-}" ]]; then
        CURRENT_USER_DIR="$(realpath "${DIR_MAP[$CURRENT_EMAIL]}")"
    elif [[ -d "$THETAGEN_DATA_DIR/$USER_DIR_NAME" ]]; then
        CURRENT_USER_DIR="$THETAGEN_DATA_DIR/$USER_DIR_NAME"
    else
        echo "错误: 找不到用户 $CURRENT_EMAIL 的数据目录" >&2
        echo "  尝试过: $THETAGEN_DATA_DIR/$USER_DIR_NAME" >&2
        echo "  请通过 --user-dir 指定" >&2
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "  ThetaGen Pipeline — $CURRENT_EMAIL"
    echo "============================================================"
    echo "  用户 email:  $CURRENT_EMAIL"
    echo "  目录名:      $USER_DIR_NAME"
    echo "  数据目录:    $CURRENT_USER_DIR"
    echo "  模型:        $MODEL"
    echo "  Benchmark:   $BENCHMARK"
    echo "  Dataset:     $DATASET"
    echo "============================================================"

# ============================================================
# Step 1: 生成评测题目 (kg_evaluation_queries.json)
# ============================================================

QUERIES_FILE="$CURRENT_USER_DIR/kg_evaluation_queries.json"

if [[ "$SKIP_QUERY_GEN" == true ]]; then
    echo ""
    echo "[Step 1] 跳过题目生成 (--skip-query-gen)"
    if [[ ! -f "$QUERIES_FILE" ]]; then
        echo "错误: $QUERIES_FILE 不存在" >&2
        exit 1
    fi
else
    echo ""
    echo "[Step 1] 生成评测题目..."
    if [[ ! -f "$THETAGENDATA_VENV" ]]; then
        echo "错误: thetagendata venv 不存在: $THETAGENDATA_VENV" >&2
        echo "请设置 THETAGENDATA_DIR 环境变量指向 thetagendata 仓库" >&2
        exit 1
    fi
    cd "$THETAGENDATA_DIR"
    "$THETAGENDATA_VENV" tools/kg_evaluation_query_generator.py \
        -d "$CURRENT_USER_DIR" \
        --language en \
        --queries-per-user "$QUERIES_PER_USER" \
        --seed "$SEED" \
        --force
    echo "  → 题目已生成: $QUERIES_FILE"
fi

# ============================================================
# Step 2: 转换为 JSONL
# ============================================================

echo ""
echo "[Step 2] 转换为 HolyEval JSONL..."

BENCH_DIR="$HOLYEVAL_DIR/benchmark/data/$BENCHMARK"
mkdir -p "$BENCH_DIR"

JSONL_FILE="$BENCH_DIR/${DATASET}.jsonl"

# 批量模式: 第一个用户覆盖写，后续追加
TEMP_JSONL="$(mktemp)"
cd "$HOLYEVAL_DIR"
"$HOLYEVAL_VENV" -m generator.eslbench.converter \
    --input "$QUERIES_FILE" \
    --output "$TEMP_JSONL" \
    --user-email "$CURRENT_EMAIL"

if [[ "$CURRENT_EMAIL" == "${EMAIL_LIST[0]}" ]]; then
    mv "$TEMP_JSONL" "$JSONL_FILE"
else
    cat "$TEMP_JSONL" >> "$JSONL_FILE"
    rm -f "$TEMP_JSONL"
fi

ITEM_COUNT=$(wc -l < "$JSONL_FILE")
echo "  → JSONL: $JSONL_FILE ($ITEM_COUNT 条，累计）"

# ============================================================
# Step 3: 创建 metadata.json（如不存在）
# ============================================================

METADATA_FILE="$BENCH_DIR/metadata.json"
if [[ ! -f "$METADATA_FILE" ]]; then
    echo ""
    echo "[Step 3] 复制 metadata.json..."
    cp "$HOLYEVAL_DIR/benchmark/data/eslbench/metadata.json" "$METADATA_FILE"
    echo "  → $METADATA_FILE"
else
    echo ""
    echo "[Step 3] metadata.json 已存在，跳过"
fi

# ============================================================
# Step 4: 创建数据目录 + DuckDB
# ============================================================

echo ""
echo "[Step 4] 准备数据目录 + DuckDB..."

TARGET_DATA_DIR="$THETAGEN_DATA_DIR/$USER_DIR_NAME"
mkdir -p "$TARGET_DATA_DIR"

# 软链接数据文件（注意: 不链接 device_data.json，该文件过大会导致 OOM/segfault）
# retrieve 工具应通过 timeline.json + DuckDB 访问设备数据
for f in events.json exam_data.json profile.json; do
    if [[ -f "$CURRENT_USER_DIR/$f" ]]; then
        ln -sf "$CURRENT_USER_DIR/$f" "$TARGET_DATA_DIR/$f"
    else
        echo "  警告: 缺少 $f" >&2
    fi
done

# 软链接 timeline（兼容 *_timeline.json 命名）
# 重要: retrieve.py 硬编码读取 "timeline.json"，必须创建标准名称的链接
TIMELINE=$(find "$CURRENT_USER_DIR" -name "*_timeline.json" -type f 2>/dev/null | head -1)
if [[ -n "$TIMELINE" ]]; then
    ln -sf "$TIMELINE" "$TARGET_DATA_DIR/$(basename "$TIMELINE")"
    # 如果原文件不叫 timeline.json，额外创建 timeline.json 链接
    if [[ "$(basename "$TIMELINE")" != "timeline.json" ]]; then
        ln -sf "$TIMELINE" "$TARGET_DATA_DIR/timeline.json"
        echo "  → timeline.json → $(basename "$TIMELINE")"
    fi
else
    echo "  警告: 未找到 *_timeline.json" >&2
fi

# 软链接 queries
if [[ -f "$QUERIES_FILE" ]]; then
    ln -sf "$QUERIES_FILE" "$TARGET_DATA_DIR/kg_evaluation_queries.json"
fi

# 构建 DuckDB
cd "$HOLYEVAL_DIR"
"$HOLYEVAL_VENV" -c "
from generator.eslbench.prepare_data import create_user_duckdb
from pathlib import Path
create_user_duckdb(Path('$TARGET_DATA_DIR'), force=True)
print('  → DuckDB 创建完成')
import duckdb
con = duckdb.connect('$TARGET_DATA_DIR/user.duckdb')
for t in ['device_indicators', 'exam_indicators', 'events']:
    r = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()
    print(f'    {t}: {r[0]} rows')
con.close()
"

echo "  → 数据目录: $TARGET_DATA_DIR"

# ============================================================
# Step 5: 上传到 HuggingFace（可选）
# ============================================================

HF_UPLOADER="$SCRIPT_DIR/hf_uploader.py"

if [[ "$UPLOAD_HF" == true ]]; then
    echo ""
    echo "[Step 5] 上传数据到 HuggingFace ($HF_REPO batch=$HF_BATCH)..."
    cd "$HOLYEVAL_DIR"
    "$HOLYEVAL_VENV" "$HF_UPLOADER" \
        --repo "$HF_REPO" --batch "$HF_BATCH" \
        --user-dir "$CURRENT_USER_DIR" --dir-name "$USER_DIR_NAME"
else
    echo ""
    echo "[Step 5] 跳过 HF 上传 (需要 --upload-hf)"
fi

if [[ "$UPLOAD_ESLBENCH" == true ]]; then
    echo ""
    echo "[Step 5b] 上传到 ESL-Bench ($ESLBENCH_REPO batch=$ESLBENCH_BATCH)..."
    cd "$HOLYEVAL_DIR"
    "$HOLYEVAL_VENV" "$HF_UPLOADER" \
        --repo "$ESLBENCH_REPO" --batch "$ESLBENCH_BATCH" \
        --user-dir "$CURRENT_USER_DIR" --dir-name "$USER_DIR_NAME"
else
    echo ""
    echo "[Step 5b] 跳过 ESL-Bench 上传 (需要 --upload-eslbench)"
fi

# ============================================================
# Step 6: 运行评测
# ============================================================

echo ""
echo "  ✓ 用户 $CURRENT_EMAIL 数据准备完成"

done  # END batch loop over EMAIL_LIST

# ============================================================
# Step 6.5: 生成 eslbench full + sample50（按难度分层抽样）
# ============================================================

ESLBENCH_DIR="$HOLYEVAL_DIR/benchmark/data/eslbench"
DATE_TAG="$(date +%Y%m%d)"
FULL_ESLBENCH="$ESLBENCH_DIR/full-${DATE_TAG}.jsonl"
SAMPLE50_ESLBENCH="$ESLBENCH_DIR/sample50-${DATE_TAG}.jsonl"

if [[ -f "$JSONL_FILE" ]]; then
    mkdir -p "$ESLBENCH_DIR"
    # 复制合并后的 full JSONL 作为 eslbench full
    cp "$JSONL_FILE" "$FULL_ESLBENCH"
    FULL_COUNT=$(wc -l < "$FULL_ESLBENCH")
    echo ""
    echo "[Step 6.5] 生成 eslbench JSONL..."
    echo "  → full: $FULL_ESLBENCH ($FULL_COUNT 条)"

    # 按 difficulty 分层抽样: 每种难度 10 条，共 50 条
    "$HOLYEVAL_VENV" -c "
import json, random
from collections import defaultdict

with open('$FULL_ESLBENCH') as f:
    items = [json.loads(line) for line in f]

# 按 difficulty 分组
groups = defaultdict(list)
for item in items:
    diff = item.get('eval', {}).get('difficulty', 'unknown')
    groups[diff].append(item)

rng = random.Random($SEED)
sampled = []
for diff in sorted(groups):
    pool = groups[diff]
    n = min(10, len(pool))
    sampled.extend(rng.sample(pool, n))

rng.shuffle(sampled)

with open('$SAMPLE50_ESLBENCH', 'w') as f:
    for item in sampled:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'  → sample50: $SAMPLE50_ESLBENCH ({len(sampled)} 条)')
diff_counts = {}
for item in sampled:
    d = item.get('eval', {}).get('difficulty', 'unknown')
    diff_counts[d] = diff_counts.get(d, 0) + 1
for d in sorted(diff_counts):
    print(f'    {d}: {diff_counts[d]}')
"
fi

# ============================================================
# Step 5c: 上传 eslbench JSONL 到 ESL-Bench HF repo（可选）
# ============================================================

if [[ "$UPLOAD_ESLBENCH" == true ]]; then
    ESLBENCH_JSONL_DIR="$HOLYEVAL_DIR/benchmark/data/eslbench"
    ESLBENCH_JSONL_FILES=( $(find "$ESLBENCH_JSONL_DIR" -name "*-${DATE_TAG}.jsonl" -type f 2>/dev/null) )

    if [[ ${#ESLBENCH_JSONL_FILES[@]} -gt 0 ]]; then
        echo ""
        echo "[Step 5c] 上传 eslbench JSONL 到 ESL-Bench ($ESLBENCH_REPO batch=$ESLBENCH_BATCH)..."
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
        # 删除 user.target_overrides
        if 'user' in obj:
            obj['user'].pop('target_overrides', None)
        # 删除 eval 内部字段
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
        echo "[Step 5c] 未找到 eslbench/*-20260324.jsonl 文件，跳过"
    fi
else
    echo ""
    echo "[Step 5c] 跳过 eslbench JSONL 上传 (需要 --upload-eslbench)"
fi

# ============================================================
# Step 6: 运行评测（在所有用户数据准备完成后）
# ============================================================

if [[ "$NO_EVAL" == true ]]; then
    echo ""
    echo "[Step 6] 跳过评测 (--no-eval)"
    echo ""
    echo "手动运行评测:"
    echo "  cd $HOLYEVAL_DIR"
    echo "  .venv/bin/python -m benchmark.basic_runner $BENCHMARK $DATASET --target-type llm_api --target-model $MODEL -p $PARALLELISM"
else
    echo ""
    echo "[Step 6] 运行评测 ($MODEL + retrieve-tool)..."
    echo "  并发: $PARALLELISM（大数据用户建议 -p 1 避免 OOM/segfault）"
    echo "  如遇 segfault 可用 --resume 恢复:"
    echo "    cd $HOLYEVAL_DIR"
    echo "    .venv/bin/python -m benchmark.basic_runner $BENCHMARK $DATASET --target-type llm_api --target-model $MODEL -p $PARALLELISM --resume"
    cd "$HOLYEVAL_DIR"
    "$HOLYEVAL_VENV" -m benchmark.basic_runner "$BENCHMARK" "$DATASET" \
        --target-type llm_api \
        --target-model "$MODEL" \
        -p "$PARALLELISM"
fi

echo ""
echo "============================================================"
echo "  Pipeline 完成!"
echo "============================================================"
