# Dialogue Quality Evaluation (RECORD + RETRIEVAL)

## Background

theta-smart 是一个健康管理 AI 助手，核心场景：
- **RECORD**：用户记录健康数据（血压、饮食、用药、运动等），AI 一句话确认
- **RETRIEVAL**：用户查询健康数据，AI 取数据并附带分析

本评测验证这两个场景的对话质量：RECORD 回复是否简短不越界，RETRIEVAL 是否返回了正确数据。

## Branch

`dialogue-quality-eval`（本分支）

## Evaluator

`record_retrieval` — 纯规则，零 LLM，逐轮 checkpoint 检查。

代码位置：`evaluator/plugin/eval_agent/record_retrieval_eval_agent.py`

### Checkpoint 类型

| type | 检查内容 |
|------|---------|
| `record_ack` | 回复 <= 80 字 + 无建议关键词 + 回显用户数据 |
| `retrieval_data` | 回复包含预期数据（substring match） |
| `skip` | 不检查 |

## Datasets

| Dataset | Cases | 用途 |
|---------|-------|------|
| `smoke` | 5 | 快速验证 |
| `core` | 15 | 完整覆盖所有 record 类型 |
| `l2_core` | 12 | Layer 2 LLM-as-judge（3 场景 × 4 虚拟患者组合） |
| `l2_hard` | 9 | Layer 2 对抗性场景（矛盾信息 + 危险行为 + 长对话记忆） |

每个 case 使用独立测试用户（`user200@demo` ~ `user224@demo`，验证码 `000000`）。

## Running

支持两个 target 类型：
- **theta_api** — Holywell 后端（需设置 `THETA_API_BASE_URL`，默认远程测试环境）
- **theta_smart_api** — theta-smart 后端（需设置 `THETA_SMART_API_BASE_URL`，默认 localhost:8199）

### theta_api

需要覆盖 API 路径（本地后端无 `/holywell` 前缀）：

```bash
# smoke（约 5 分钟）
THETA_API_CHAT_PATH=/api/v1/chat/create_message \
THETA_API_LIST_MESSAGE_PATH=/api/v1/chat/list_message_chunks \
uv run python -m benchmark.basic_runner dialogue_quality smoke --target-type theta_api -p 1 -v

# core（约 15 分钟）
THETA_API_CHAT_PATH=/api/v1/chat/create_message \
THETA_API_LIST_MESSAGE_PATH=/api/v1/chat/list_message_chunks \
uv run python -m benchmark.basic_runner dialogue_quality core --target-type theta_api -p 1 -v
```

两个环境变量是因为本地后端 API 路径与远程不同（本地无 `/holywell` 前缀）。

### theta_smart_api

theta-smart 后端使用独立的 API 路径（`/api/v1/chat/create_message` + `/api/v1/chat/list_message_chunks`），无需覆盖环境变量：

```bash
# smoke
uv run python -m benchmark.basic_runner dialogue_quality smoke --target-type theta_smart_api -p 1 -v

# core
uv run python -m benchmark.basic_runner dialogue_quality core --target-type theta_smart_api -p 1 -v
```

### Running Layer 2

```bash
# theta_api
THETA_API_CHAT_PATH=/api/v1/chat/create_message \
THETA_API_LIST_MESSAGE_PATH=/api/v1/chat/list_message_chunks \
uv run python -m benchmark.basic_runner dialogue_quality l2_core --target-type theta_api -p 1 -v

# theta_smart_api
uv run python -m benchmark.basic_runner dialogue_quality l2_core --target-type theta_smart_api -p 1 -v
```

### Running Layer 2 Hard

对抗性场景，轮次更长（10-15 轮），预计 60-90 分钟：

```bash
# theta_api
THETA_API_CHAT_PATH=/api/v1/chat/create_message \
THETA_API_LIST_MESSAGE_PATH=/api/v1/chat/list_message_chunks \
uv run python -m benchmark.basic_runner dialogue_quality l2_hard --target-type theta_api -p 1 -v

# theta_smart_api
uv run python -m benchmark.basic_runner dialogue_quality l2_hard --target-type theta_smart_api -p 1 -v
```

l2_hard 包含 3 类对抗性场景：
- **矛盾信息管理**（max_turns=12）：用户给出前后矛盾的健康信息，AI 是否发现并澄清
- **危险行为识别**（max_turns=10）：用户提到擅自停药/换保健品等危险行为，AI 是否质疑
- **长对话记忆压力**（max_turns=15）：10+ 轮对话穿插话题切换，AI 是否遗忘早期关键信息

judge 会根据虚拟用户的 persona 类型动态注入评分标准（persona-aware judge），对 AI 是否正确应对困难用户进行更严格的评估。

## Baseline

smoke 数据集基线：4/5 通过，avg_score=0.90。core 数据集尚未跑基线。

## Layer 2: Dialogue Quality (LLM-as-Judge)

Layer 2 已实现。使用 LLM-as-judge 对对话质量进行多维度打分，覆盖健康管理场景中无法用规则判断的回复质量。

**Evaluator：** `dialogue_quality` — LLM-as-judge，6 维度加权评分

代码位置：`evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py`

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| accuracy（数据准确性） | 25% | 数据正确、引用可靠来源 |
| personalization（个性化） | 20% | 结合用户实际健康数据，而非通用回答 |
| comprehensiveness（全面性） | 20% | 覆盖多维度（体重、代谢、睡眠、运动等） |
| readability（可读性） | 15% | 排版、结构清晰、好理解 |
| actionability（可操作性） | 10% | 给出具体可执行的建议 |
| context_memory（上下文记忆） | 10% | 长对话不丢失前文信息 |

### Dataset: l2_core

- **12 个用例**，覆盖 3 个场景 × 4 个虚拟患者 persona 组合
- **场景**：健康数据解读、运动/营养方案制定、睡眠质量分析
- **Persona 维度**：
  - Expression（表达方式）：normal / vague / incoherent
  - Logic（逻辑一致性）：consistent / occasional_contradiction / fabricating

### Design Notes

以下是基于用户调研的背景设计思路，供参考。

#### 用户真实关注的质量维度

来自用户访谈，按优先级排列：

1. **信息准确性** — 数据正确、引用可靠来源
2. **个性化程度** — 结合用户实际健康数据，而非通用回答
3. **回答全面性** — 覆盖多维度（体重、代谢、睡眠、运动等）
4. **逻辑清晰度** — 结构化、分层次、好理解
5. **可操作性** — 给出具体可执行的建议
6. **语言自然度** — 不像机器人、不啰嗦

#### 用户关注的功能场景

- 解读健康数据（血检、体检报告等）
- 制定运动 / 营养 / 减重方案
- 睡眠质量分析
- 可穿戴设备数据分析（Apple Watch、Garmin 等）
- 日常健康问答
- 上传文件让 AI 分析（PDF、图片等）
- 上下文记忆能力（长对话不丢失前文信息）
- 回复的阅读体验（排版、图表、条理性）

#### Layer 2 评测方案草案

**评测方式：** LLM-as-judge 多维度打分（非规则）

| 维度 | 权重 | 评测方式 | 对应用户诉求 |
|------|------|---------|------------|
| 数据准确性 | 25% | LLM judge | 信息准确性 |
| 个性化 | 20% | LLM judge | 结合用户实际健康数据 |
| 全面性 | 20% | LLM judge | 覆盖多维度 |
| 可读性 | 15% | LLM judge | 排版、结构、图表 |
| 可操作性 | 10% | LLM judge | 具体可执行建议 |
| 上下文记忆 | 10% | LLM judge | 长对话不丢前文 |

> Layer 2 全部使用 LLM judge，不混合规则。Layer 1 能用规则是因为 RECORD/RETRIEVAL 有明确 ground truth，Layer 2 的场景（健康解读、方案制定）没有确定性答案，硬写规则会造成假阳性。

#### 虚拟患者 APAF

可复用 med-dialogue-bench 的 APAF 6 维度模型，适配为健康管理用户行为：
- Expression → 用户记录数据的方式（精确 vs 模糊口语化）
- Disclosure → 用户一次记录多少信息
- Attitude → 用户对 AI 回复的耐心
- Cognition → 用户对健康数据的理解程度
- Logic → 用户记录数据的一致性
- Purpose → 使用目的（纯记录 / 查数据 / 要分析）

## Known Issues

- 血压值 AI 可能重格式化（"135/88" → "收缩压 135 / 舒张压 88"），checkpoint 已改为拆分匹配 ["135", "88"]
- 本地后端 verify 接口不返回 `success` 字段，已修复 target agent 兼容
- 评测必须在同一 session 内完成 RECORD → RETRIEVAL（跨 session 查不到刚记录的数据）
