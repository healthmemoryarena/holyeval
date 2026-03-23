# 生成 Benchmark 数据

> **在 Claude Code 中使用**: 输入 `/add-benchmark`，后跟你的 benchmark 来源（论文链接、GitHub 仓库），Claude 会自动完成研究 → 转换 → 验证全流程。

## 概述

Benchmark 数据集是评测的输入，存放在 `benchmark/data/<benchmark>/<dataset>.jsonl`。你可以从外部评测基准（如 HealthBench、MedCalc-Bench）转换数据，也可以手动编写测试用例。

## 数据格式

每行是一个 JSON 对象，对应一个 `BenchItem`（运行时自动转为 `TestCase`）：

```json
{
  "id": "case_001",
  "name": "示例用例",
  "tags": ["topic:cardiology"],
  "user": {
    "type": "manual",
    "strict_inputs": ["我最近心跳很快，怎么回事？"]
  },
  "eval": {
    "evaluator": "semantic",
    "threshold": 0.7
  }
}
```

> **注意**: `BenchItem` 通常不包含 `target` 配置 — 被测系统在运行时通过 CLI 参数或 Web UI 指定，使同一数据集可复用于不同模型。如果 benchmark 需要固定 target 配置，在 `metadata.json` 中指定。

### 关键字段

| 字段 | 说明 |
|------|------|
| `user.type` | `manual`（脚本驱动）或 `auto`（LLM 驱动） |
| `user.strict_inputs` | 预设输入列表（`List[str]`），manual 模式逐条按序发送给被测系统。单条 = 单轮问答；多条 = 逐轮预注入上下文后再提问（如先发病历、再问诊断），中间回复不影响评估 |
| `eval.evaluator` | 评估器类型（`semantic` / `keyword` / `healthbench` / `medcalc` / `preset_answer`） |
| `history` | 可选，评测前的历史对话 `[{role, content}]`。与 `strict_inputs` 不同：history 作为预加载上下文直接注入，不走对话循环 |

## 从外部数据集转换

### 已内置的转换器

| 转换器 | 命令 | 说明 |
|--------|------|------|
| HealthBench | `python -m generator.healthbench.converter input.jsonl output.jsonl --target-model gpt-4.1` | prompt → history + strict_inputs, rubrics → eval.rubrics |
| MedCalc-Bench | `python -m generator.medcalc.converter` | Patient Note + Question → strict_inputs, Answer → eval.ground_truth |
| AgentClinic | `python -m generator.agentclinic.converter input.jsonl output.jsonl` | OSCE/MCQ 两种格式 → strict_inputs，Correct_Diagnosis → eval.standard_answer（keyword 匹配） |
| MedHall (生成) | `python -m generator.medhall.data_gen --count 15 --output raw_data.jsonl` | GPT-4o 批量生成 factual/contextual/citation 三类幻觉场景 |
| MedHall (转换) | `python -m generator.medhall.converter raw_data.jsonl benchmark/data/medhall/theta.jsonl` | 幻觉场景原始 JSONL → BenchItem JSONL，使用 hallucination 评估器 |
| MemoryArena | `python -m generator.memoryarena.converter` | questions → strict_inputs, answers → eval.ground_truths, domain → tags |

### 自定义转换器

在 `generator/<benchmark>/` 目录下创建转换脚本，核心步骤：

1. 读取源数据（JSONL / CSV / 其他格式）
2. 映射为 `BenchItem` 结构（user + eval，不含 target）
3. 写入 JSONL 文件到 `benchmark/data/<benchmark>/`
4. 创建 `metadata.json` 描述数据集

> **推荐**: 使用 `/add-benchmark` skill 自动化完成上述全流程。

## 目录结构

```
benchmark/
├── data/
│   ├── healthbench/          # 评测套件名
│   │   ├── full.jsonl        # 数据集
│   │   ├── sample.jsonl
│   │   └── metadata.json     # 套件元数据（Web UI 展示）
│   ├── medcalc/
│   │   ├── full.jsonl
│   │   ├── sample.jsonl
│   │   └── metadata.json
│   ├── extraction/
│   │   ├── simple.jsonl
│   │   └── metadata.json
│   └── memoryarena/
│       ├── full.jsonl
│       ├── sample.jsonl
│       └── metadata.json
└── report/                   # 报告输出（自动生成）
```

### metadata.json 格式

每个评测套件目录下的 `metadata.json` 提供元数据，其中 `description` 字段（Markdown 格式）会展示在 Web UI 的数据集详情页：

```json
{
  "description": "# 评测套件名\n\n简要描述...\n\n## 子集\n\n| 子集 | 数量 | 说明 |\n|------|------|------|\n| sample | 100 | 快速验证 |\n\n**评估器**: `evaluator_name`",
  "target": {
    "type": "llm_api",
    "model": "gpt-4.1"
  },
  "target_configurable": true
}
```

| 字段 | 说明 |
|------|------|
| `description` | Markdown 格式的套件说明，展示在 Web UI |
| `target` | 默认 target 配置，Web UI / CLI 创建任务时的初始值 |
| `target_configurable` | `true` 允许用户修改 target 参数，`false` 锁定（如 extraction 固定使用 theta_api） |

## 验证数据

```bash
# 快速跑几条验证
python -m benchmark.basic_runner <benchmark> <dataset> --target-type llm_api --target-model gpt-4.1 --limit 5 -v
```

或通过 Web UI 执行：启动 `python -m web`，在 http://localhost:8000/tasks 页面创建任务，可实时查看进度。
