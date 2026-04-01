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
#   # 仅处理 kg_evaluation_queries（跳过其他 json 链接/上传/DuckDB）
#   bash conv_eval.sh --email user5022@demo --kg-query-only
#
#   # 指定外部 kg_evaluation_queries.json 文件，仅处理该文件
#   bash conv_eval.sh --email user5022@demo --kg-query-only /path/to/kg_evaluation_queries.json
#
#   # 指定每个维度抽样数量（默认 10，5维共50条）
#   bash conv_eval.sh --email user5022@demo --sample-n-per-dim 20
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
BENCHMARK="eslbench"
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
KG_QUERY_ONLY=false
KG_QUERY_ONLY_FILE=""
SAMPLE_N_PER_DIM=10

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
        --kg-query-only)
            KG_QUERY_ONLY=true
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                KG_QUERY_ONLY_FILE="$2"; shift
            fi
            shift ;;
        --sample-n-per-dim) SAMPLE_N_PER_DIM="$2"; shift 2 ;;
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
    echo "  $0 --email user5022@demo --kg-query-only                # 仅处理已有 kg_query"
    echo "  $0 --email user5022@demo --kg-query-only /path/to/file  # 指定外部 kg_query"
    echo "  $0 --email user5022@demo --sample-n-per-dim 20  # 每维度20条"
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
    if [[ "$KG_QUERY_ONLY" == true ]]; then
    echo "  KG Only:     true${KG_QUERY_ONLY_FILE:+ ($KG_QUERY_ONLY_FILE)}"
    fi
    echo "  Sample/dim:  $SAMPLE_N_PER_DIM"
    echo "============================================================"

# ============================================================
# Step 1: 生成评测题目 (kg_evaluation_queries.json)
# ============================================================

QUERIES_FILE="$CURRENT_USER_DIR/kg_evaluation_queries.json"

if [[ "$KG_QUERY_ONLY" == true ]]; then
    echo ""
    if [[ -n "$KG_QUERY_ONLY_FILE" ]]; then
        echo "[Step 1] 使用指定的 kg_evaluation_queries 文件: $KG_QUERY_ONLY_FILE"
        if [[ ! -f "$KG_QUERY_ONLY_FILE" ]]; then
            echo "错误: 指定的文件不存在: $KG_QUERY_ONLY_FILE" >&2
            exit 1
        fi
        cp "$KG_QUERY_ONLY_FILE" "$QUERIES_FILE"
        echo "  → 已复制到: $QUERIES_FILE"
    else
        echo "[Step 1] 使用已有的 kg_evaluation_queries 文件 (--kg-query-only)"
        if [[ ! -f "$QUERIES_FILE" ]]; then
            echo "错误: $QUERIES_FILE 不存在，请通过 --kg-query-only <file> 指定" >&2
            exit 1
        fi
    fi
elif [[ "$SKIP_QUERY_GEN" == true ]]; then
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

TARGET_DATA_DIR="$THETAGEN_DATA_DIR/$USER_DIR_NAME"
mkdir -p "$TARGET_DATA_DIR"

if [[ "$KG_QUERY_ONLY" == true ]]; then
    echo ""
    echo "[Step 4] 仅软链接 kg_evaluation_queries.json (--kg-query-only)..."
    if [[ -f "$QUERIES_FILE" ]]; then
        ln -sf "$QUERIES_FILE" "$TARGET_DATA_DIR/kg_evaluation_queries.json"
        echo "  → $TARGET_DATA_DIR/kg_evaluation_queries.json"
    else
        echo "  警告: $QUERIES_FILE 不存在" >&2
    fi
else
    echo ""
    echo "[Step 4] 准备数据目录 + DuckDB..."

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
fi

echo "  → 数据目录: $TARGET_DATA_DIR"

# ============================================================
# Step 5: 上传到 HuggingFace（可选）
# ============================================================

HF_UPLOADER="$SCRIPT_DIR/hf_uploader.py"

# 加载 HF_TOKEN（如未设置则从 .env 读取）
if [[ -z "${HF_TOKEN:-}" && -f "$HOLYEVAL_DIR/.env" ]]; then
    HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' "$HOLYEVAL_DIR/.env" 2>/dev/null || true)"
    export HF_TOKEN
fi

HF_EXTRA_ARGS=()
if [[ "$KG_QUERY_ONLY" == true ]]; then
    HF_EXTRA_ARGS+=(--kg-query-only)
fi

if [[ "$UPLOAD_HF" == true ]]; then
    echo ""
    echo "[Step 5] 上传数据到 HuggingFace ($HF_REPO batch=$HF_BATCH)..."
    cd "$HOLYEVAL_DIR"
    "$HOLYEVAL_VENV" "$HF_UPLOADER" \
        --repo "$HF_REPO" --batch "$HF_BATCH" \
        --user-dir "$CURRENT_USER_DIR" --dir-name "$USER_DIR_NAME" \
        "${HF_EXTRA_ARGS[@]}"
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
        --user-dir "$CURRENT_USER_DIR" --dir-name "$USER_DIR_NAME" \
        "${HF_EXTRA_ARGS[@]}"
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
# Step 6.5: 生成 full_mid / full-YYYYMMDD / sample JSONL
# ============================================================

ESLBENCH_DIR="$HOLYEVAL_DIR/benchmark/data/eslbench"
DATE_TAG="$(date +%Y%m%d)"
FULL_MID="$ESLBENCH_DIR/full_mid.jsonl"
FULL_ESLBENCH="$ESLBENCH_DIR/full-${DATE_TAG}.jsonl"

if [[ -f "$JSONL_FILE" ]]; then
    # full_mid.jsonl: 当前批次的合并 JSONL（每次覆盖）
    cp "$JSONL_FILE" "$FULL_MID"
    echo ""
    echo "[Step 6.5] full_mid.jsonl: $(wc -l < "$FULL_MID") 条"

    # full-YYYYMMDD.jsonl: 仅不存在时生成
    if [[ ! -f "$FULL_ESLBENCH" ]]; then
        cp "$FULL_MID" "$FULL_ESLBENCH"
        echo "  → 生成 $FULL_ESLBENCH ($(wc -l < "$FULL_ESLBENCH") 条)"
    else
        echo "  → $FULL_ESLBENCH 已存在，跳过"
    fi

    # 从 full_mid.jsonl 生成 sample50 + sample500
    GEN_ESLBENCH_ARGS=(--jsonl "$FULL_MID" --seed "$SEED")
    if [[ "$UPLOAD_ESLBENCH" == true ]]; then
        GEN_ESLBENCH_ARGS+=(--upload-eslbench --eslbench-repo "$ESLBENCH_REPO" --eslbench-batch "$ESLBENCH_BATCH")
    fi
    if [[ "$NO_EVAL" != true ]]; then
        GEN_ESLBENCH_ARGS+=(--eval --model "$MODEL" --benchmark "$BENCHMARK" --dataset "$DATASET" -p "$PARALLELISM")
    fi
    bash "$SCRIPT_DIR/gen_eslbench_jsonl.sh" "${GEN_ESLBENCH_ARGS[@]}"
fi

echo ""
echo "============================================================"
echo "  Pipeline 完成!"
echo "============================================================"
