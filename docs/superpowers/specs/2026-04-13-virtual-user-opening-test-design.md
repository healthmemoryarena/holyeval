# 虚拟用户开场白测试 — 设计文档

> 日期: 2026-04-13
> 分支: virtual-user
> 状态: 设计中

## 背景与目标

### 问题

Theta Smart 是一款慢病管理产品，用户打开 APP 后产品主动发送一句开场白。开场白的质量直接决定用户是否愿意"开口" — 即录入健康数据或提出问题。

当前没有真实用户，无法验证哪些开场白有效、哪些会把用户吓跑。

### 目标

围绕**肥胖**病种，用 LLM 生成的虚拟用户批量测试不同开场白，通过统计"主动录入率"排除不合理的开场白。

这是三层指标体系中**第一层（信任建立）**的验证。第二层（洞察质量）后续扩展。

### 成功标准

- 能区分不同开场白的录入率差异
- 能看到不同对抗画像维度对录入率的影响
- 迭代效率：改开场白 → 重新生成 → 跑 → 看报告，单轮 < 30 分钟

## 整体架构

```
开场白列表 (openings.json) × 用户画像池 (profiles.jsonl)
        ↓
   case_gen.py（笛卡尔积）
        ↓
  benchmark/data/virtual_user/<dataset>.jsonl
        ↓
   basic_runner.py（BatchSession, theta_smart_api）
        ↓
  对每条用例:
    history = [{"role": "assistant", "content": 开场白}]
    AutoTestAgent（五维对抗画像）↔ ThetaSmartApi（1~3 轮短对话）
        ↓
   EngagementEvalAgent 判定: engaged / not_engaged
        ↓
  BenchReport → analyzer.py 按 opening_id 分组统计录入率
```

## 组件设计

### 1. 画像生成器 (`generator/virtual_user/profile_gen.py`)

用 LLM 批量生成肥胖患者基本画像，输出 `profiles.jsonl`。

每个画像包含:

```json
{
  "profile_id": "obesity_001",
  "age": 35,
  "gender": "female",
  "bmi": 32.5,
  "occupation": "小学教师",
  "comorbidities": ["高血压前期"],
  "motivation": "体检报告异常，医生建议减重",
  "background": "尝试过节食和跑步，坚持不到两周就放弃。家里人都偏胖，觉得是遗传。"
}
```

- 初始生成 20 个画像，覆盖年龄/性别/BMI/职业/合并症/减肥动机的多样性
- 后续可扩展到 100 个

### 2. 开场白列表 (`generator/virtual_user/openings.json`)

```json
[
  {
    "opening_id": "01",
    "content": "你好！我是你的健康管理助手，很高兴认识你。我可以帮你记录和分析健康数据，发现身体变化的规律。有什么我能帮到你的吗？"
  },
  {
    "opening_id": "02",
    "content": "看起来你最近在关注体重管理方面的问题。很多和你年龄相仿的用户都有类似的困扰，我可以根据你的具体情况给出个性化的建议。方便告诉我你目前的情况吗？"
  }
]
```

迭代时只需修改这个文件。

### 3. 用例组装器 (`generator/virtual_user/case_gen.py`)

读取 `profiles.jsonl` + `openings.json`，做笛卡尔积，输出 benchmark JSONL。

**对抗维度分配策略:**

- 档 1 (baseline) 占比 50%
- 档 2 (中间) 占比 30%
- 档 3 (极端) 占比 20%

每个画像随机分配五维对抗档位。随机种子固定，保证可复现。

**生成的每条用例:**

```json
{
  "id": "vu_o01_p003",
  "title": "开场白01 × 35岁女教师BMI32 [reluctant/compliant]",
  "tags": ["opening:01", "profile:003", "disclosure:reluctant", "attitude:compliant", "cognition:accurate", "logic:consistent", "expression:normal"],
  "user": {
    "type": "auto",
    "goal": "根据自身情况决定是否愿意与这个健康助手交流。如果不想交流，你可以选择沉默（回复\"[沉默]\"）、敷衍、或直接拒绝。如果愿意就主动描述自己的健康困扰。",
    "context": "你是一位35岁的小学教师，BMI 32.5，有高血压前期。尝试过节食和跑步，坚持不到两周就放弃。家里人都偏胖，觉得是遗传。体检报告异常，医生建议减重。",
    "persona": {
      "disclosure": "reluctant",
      "attitude": "compliant",
      "cognition": "accurate",
      "logic": "consistent",
      "expression": "normal"
    },
    "max_turns": 3
  },
  "history": [
    {"role": "assistant", "content": "你好！我是你的健康管理助手..."}
  ],
  "target": {"type": "theta_smart_api"},
  "eval": {"evaluator": "engagement", "threshold": 0.5}
}
```

**CLI:**

```bash
python -m generator.virtual_user.case_gen \
  --profiles generator/virtual_user/profiles.jsonl \
  --openings generator/virtual_user/openings.json \
  --output benchmark/data/virtual_user/round1.jsonl \
  --seed 42
```

### 4. Engagement 评估器 (`evaluator/plugin/eval_agent/engagement_eval_agent.py`)

新的 EvalAgent 插件，LLM-as-Judge 二元判定。

**判定标准:**

| 结果 | score | 定义 | 典型表现 |
|------|-------|------|----------|
| engaged | 1.0 | 主动披露健康信息或提出具体问题 | "我最近胖了10斤..."、"我想问问血糖的事" |
| not_engaged | 0.0 | 敷衍、回避、质疑、拒绝、或沉默 | "嗯"、"你是谁"、"我不需要"、"[沉默]" |

**Judge Prompt 核心:**

```
你是一个用户行为分析师。以下是一款健康管理APP的开场对话。
产品先发了一句开场白，用户做出了回应。

请判断用户是否产生了"主动参与"行为：
- engaged: 用户主动描述了自己的健康状况、提出了具体问题、或表达了使用意愿
- not_engaged: 用户敷衍回复、表示拒绝、质疑产品、未提供任何有效信息、或选择沉默

输出 JSON: {"result": "engaged" | "not_engaged", "reason": "..."}
```

**EvalInfo 配置:**

```json
{
  "evaluator": "engagement",
  "threshold": 0.5
}
```

score = 1.0 (engaged) 或 0.0 (not_engaged)，threshold 0.5 → pass_rate 即"主动录入率"。

### 5. 对抗画像扩展 (`evaluator/core/schema.py`)

扩展 PersonaConfig 从 2 维到 5 维:

```python
class PersonaConfig(BaseModel):
    disclosure: Literal["responsive", "reluctant", "withholding"] = "responsive"
    attitude: Literal["compliant", "impatient", "dominant"] = "compliant"
    cognition: Literal["accurate", "partial_understanding", "complete_denial"] = "accurate"
    logic: Literal["consistent", "occasional_contradiction", "fabricating"] = "consistent"
    expression: Literal["normal", "vague", "incoherent"] = "normal"
```

所有新维度带默认值 = 档 1 (baseline)，向后兼容现有用例。

### 6. AutoTestAgent prompt 扩展 (`evaluator/plugin/test_agent/auto_test_agent.py`)

五维画像 → 自然语言行为描述的映射表，注入 system prompt。

示例映射 (`disclosure: reluctant` + `attitude: impatient`):

> 你不太愿意主动透露自己的详细健康信息，除非对方追问才会说一点。你比较急躁，如果对方啰嗦或没有快速给出有用信息，你会表现出不耐烦。

### 7. 对比分析器 (`generator/virtual_user/analyzer.py`)

读取 BenchReport JSON，按 tags 分组统计。

**输出示例:**

```
═══════════════════════════════════════════════════
  开场白 A/B 对比报告 — 肥胖虚拟用户（N=20）
═══════════════════════════════════════════════════

开场白                              录入率    样本数
──────────────────────────────────────────────────
opening_01: "你好，我是你的..."       75%     15/20
opening_02: "看起来你最近..."         40%      8/20
opening_03: "很多和你情况类似..."      85%     17/20
──────────────────────────────────────────────────

按对抗维度交叉分析（opening_03）:
  disclosure=responsive        100%  (6/6)
  disclosure=reluctant          83%  (5/6)
  disclosure=withholding        75%  (6/8)

  attitude=compliant            90%  (9/10)
  attitude=impatient            80%  (4/5)
  attitude=dominant             80%  (4/5)
```

**CLI:**

```bash
python -m generator.virtual_user.analyzer benchmark/report/virtual_user/round1_xxx.json
```

### 8. Benchmark 元数据 (`benchmark/data/virtual_user/metadata.json`)

```json
{
  "description": "# 虚拟用户开场白测试\n\n用 LLM 生成的虚拟肥胖患者测试不同开场白的用户参与率。",
  "target": [
    {
      "type": "theta_smart_api",
      "fields": {
        "email": {"default": "user1@demo", "editable": true},
        "agent": {"default": "expert", "editable": true}
      }
    }
  ]
}
```

## 文件清单

### 新增

| 文件 | 说明 |
|------|------|
| `generator/virtual_user/profile_gen.py` | LLM 批量生成肥胖患者画像 → profiles.jsonl |
| `generator/virtual_user/case_gen.py` | 画像 × 开场白 笛卡尔积 → benchmark JSONL |
| `generator/virtual_user/openings.json` | 开场白列表（迭代入口） |
| `generator/virtual_user/analyzer.py` | 报告分组统计 + 交叉分析 |
| `evaluator/plugin/eval_agent/engagement_eval_agent.py` | Engagement 评估器插件 |
| `benchmark/data/virtual_user/metadata.json` | Benchmark 元数据 |

### 修改

| 文件 | 改动 |
|------|------|
| `evaluator/core/schema.py` | PersonaConfig 加 3 个字段（disclosure, attitude, cognition），带默认值 |
| `evaluator/plugin/test_agent/auto_test_agent.py` | prompt 构造支持五维画像 → 自然语言描述 |

### 不改动

orchestrator.py, basic_runner.py, BatchSession, 现有 benchmark 数据, Web UI — 全部原样复用。

## 使用流程

```bash
# 1. 生成画像池（一次性）
python -m generator.virtual_user.profile_gen --count 20 --output generator/virtual_user/profiles.jsonl

# 2. 编辑开场白
# 直接修改 generator/virtual_user/openings.json

# 3. 生成用例（开场白 × 画像 笛卡尔积）
python -m generator.virtual_user.case_gen \
  --profiles generator/virtual_user/profiles.jsonl \
  --openings generator/virtual_user/openings.json \
  --output benchmark/data/virtual_user/round1.jsonl \
  --seed 42

# 4. 跑测试
python -m benchmark.basic_runner virtual_user round1 --target-type theta_smart_api -p 5

# 5. 查看对比报告
python -m generator.virtual_user.analyzer benchmark/report/virtual_user/round1_xxx.json

# 6. 迭代：修改 openings.json → 重复 3-5
```

## 第二层扩展预留

当前设计聚焦第一层（开场话术 → 录入率）。第二层（洞察质量）扩展时:

- `max_turns` 从 3 调大到 10-15，模拟多天数据录入
- 新增洞察质量相关的 EvalAgent（评估产品回复的个性化程度、信息价值）
- 画像池和五维对抗体系直接复用，不需重建
