# Dialogue Quality Layer 2 — LLM-as-Judge Multi-Dimension Evaluation

## Overview

Layer 2 evaluates the overall dialogue quality of Theta Health AI across complex health management scenarios (health data interpretation, health plan creation, daily health Q&A) using LLM-as-judge multi-dimension scoring. Unlike Layer 1 (rule-based RECORD/RETRIEVAL correctness), Layer 2 has no deterministic ground truth — quality is assessed holistically across 6 user-research-driven dimensions.

**Key decisions:**
- Separate evaluator plugin (`dialogue_quality`), independent of Layer 1's `record_retrieval`
- Auto TestAgent with persona-driven virtual users (Expression + Logic dimensions from APAF)
- Whole-conversation evaluation with per-turn evidence citations
- All 6 dimensions scored by LLM judge — no rule-based components

## 1. Evaluator Plugin

### Registration

New file: `evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py`
Registered name: `dialogue_quality`

### Config Model

```python
class DialogueQualityEvalInfo(BaseModel):
    evaluator: Literal["dialogue_quality"] = "dialogue_quality"
    dimensions: dict[str, float] = {
        "accuracy": 0.25,
        "personalization": 0.20,
        "comprehensiveness": 0.20,
        "readability": 0.15,
        "actionability": 0.10,
        "context_memory": 0.10,
    }
    threshold: float = 0.6
    judge_model: str = "gpt-4.1"
```

- `dimensions`: dimension name -> weight, must sum to 1.0
- `threshold`: weighted score >= threshold -> pass
- `judge_model`: configurable per case, default gpt-4.1

### Evaluation Flow

1. Extract full conversation from `memory_list` + `history` (reuse pattern from hallucination evaluator)
2. Build judge prompt with: user context, user goal, formatted dialogue, dimension definitions, scoring rubric
3. Call `do_execute()` with judge model
4. Parse JSON output -> per-dimension scores (1-5) + evidence
5. Weighted sum -> normalize to 0-1: `(weighted_sum - 1) / 4`
6. score >= threshold -> pass

### EvalResult Output

- `score`: weighted total (0-1)
- `feedback`: `overall_comment` from judge
- `trace.eval_detail`: full dimensions dict with scores + evidence

### Judge Prompt

```
你是一个健康管理 AI 助手的质量评估专家。请评估以下多轮对话中 AI 助手的表现。

## 用户背景
{user_context}

## 对话目标
{user_goal}

## 完整对话
{formatted_dialogue}

## 评估维度

请对 AI 助手在以下 6 个维度上打分（1-5 分）：

1. **数据准确性 (accuracy)**: AI 提供的健康数据、医学知识是否正确？有无编造数据或错误解读？
2. **个性化 (personalization)**: AI 是否结合了用户的具体健康数据和背景给出针对性回答？还是给通用模板回答？
3. **全面性 (comprehensiveness)**: AI 是否覆盖了用户问题涉及的多个维度？有无明显遗漏？
4. **可读性 (readability)**: AI 回复的结构是否清晰、分层合理、易于理解？
5. **可操作性 (actionability)**: AI 是否给出了具体可执行的建议？还是泛泛而谈？
6. **上下文记忆 (context_memory)**: 在多轮对话中，AI 是否记住了前文信息？有无重复提问或遗忘已知信息？

## 评分标准
- 5分：优秀，该维度表现出色
- 4分：良好，基本满足但有小瑕疵
- 3分：及格，存在明显不足
- 2分：较差，该维度严重欠缺
- 1分：极差，完全不满足

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：
{
  "dimensions": {
    "accuracy": {"score": <1-5>, "evidence": [{"turn": <轮次号>, "quote": "<引用原文>", "comment": "<评价>"}]},
    "personalization": {"score": <1-5>, "evidence": [...]},
    "comprehensiveness": {"score": <1-5>, "evidence": [...]},
    "readability": {"score": <1-5>, "evidence": [...]},
    "actionability": {"score": <1-5>, "evidence": [...]},
    "context_memory": {"score": <1-5>, "evidence": [...]}
  },
  "overall_comment": "<一句话总评>"
}
```

## 2. Virtual User (TestAgent Persona)

### Approach

Reuse existing `auto` TestAgent. Add `persona` optional field to `AutoUserInfo` in `schema.py` (not `UserInfo` — that's a discriminated union). `BenchAutoUserInfo` inherits from `AutoUserInfo`, so benchmark cases get `persona` for free. Auto TestAgent reads `persona` and appends behavioral instructions to its system prompt.

### Persona Dimensions

| Dimension | Values | Description |
|-----------|--------|-------------|
| Expression | `normal`, `vague`, `incoherent` | How clearly the user describes health issues |
| Logic | `consistent`, `occasional_contradiction`, `fabricating` | Consistency of user's descriptions across turns |

### Behavioral Prompt Mapping

| Dimension | Value | Injected prompt fragment |
|-----------|-------|--------------------------|
| Expression | `normal` | (none, default behavior) |
| Expression | `vague` | 你表达能力有限，描述健康问题时用模糊口语："有一阵子了"、"就是不太舒服"、"大概这一块"。即使 AI 追问也无法给出更精确的描述。 |
| Expression | `incoherent` | 你很难把话说清楚。经常答非所问，说到一半思路断了，用"那个""就是""怎么说呢"代替关键词。AI 很难从你的回答中提取有用信息。 |
| Logic | `consistent` | (none, default behavior) |
| Logic | `occasional_contradiction` | 你偶尔会前后矛盾。比如先说"大概两天了"，后来又说"可能有一周了"。如果 AI 指出矛盾你会修正。 |
| Logic | `fabricating` | 你会编造健康信息。比如问到没有的症状你说"有"并编造细节，夸大不适感，编造用药经历。 |

### Implementation

Modify `auto_test_agent.py:_build_system_prompt()`: after the context section (line ~101), if `self.user_info.persona` exists, look up each dimension/value pair in the mapping table and append the corresponding Chinese prompt fragment as a new `# Your Behavioral Traits` section.

Schema change: add `persona: Optional[Dict[str, str]] = None` to `AutoUserInfo` in `schema.py`. `BenchAutoUserInfo` inherits it automatically since it extends `AutoUserInfo`.

## 3. Dataset

### File Layout

```
benchmark/data/dialogue_quality/
├── smoke.jsonl           # Layer 1 smoke (existing)
├── core.jsonl            # Layer 1 core (existing)
├── l2_core.jsonl         # Layer 2 core (NEW, ~12 cases)
├── metadata.json         # Updated description
└── README.md             # Updated docs
```

### Case Design

3 scenes x 4 persona combinations = 12 cases:

| Scene | Description | Case IDs |
|-------|-------------|----------|
| Health data interpretation | Blood test / physical exam report analysis | dq_l2_interpret_001 ~ 004 |
| Health plan creation | Exercise / nutrition / weight management plans | dq_l2_plan_001 ~ 004 |
| Daily health Q&A | General health questions and advice | dq_l2_qa_001 ~ 004 |

Persona combinations per scene:
1. `normal` + `consistent` (baseline)
2. `vague` + `consistent` (expression challenge)
3. `normal` + `occasional_contradiction` (logic challenge)
4. `incoherent` + `fabricating` (worst case)

Each case uses a unique test user (`user225@demo` ~ `user236@demo`).

### Case Structure

```json
{
  "id": "dq_l2_interpret_001",
  "title": "血检报告解读 - 正常表达/一致逻辑",
  "user": {
    "type": "auto",
    "goal": "你想了解最近血检报告中几个异常指标的含义和注意事项",
    "context": "你是35岁的上班族，最近体检发现总胆固醇6.2mmol/L（偏高）、低密度脂蛋白4.1mmol/L（偏高）、空腹血糖6.0mmol/L（临界）。你没有明显不适症状，平时饮食不太注意，运动较少。",
    "max_turns": 8,
    "persona": {
      "expression": "normal",
      "logic": "consistent"
    },
    "target_overrides": {
      "theta_api": {"email": "user225@demo"}
    }
  },
  "eval": {
    "evaluator": "dialogue_quality",
    "dimensions": {
      "accuracy": 0.25,
      "personalization": 0.20,
      "comprehensiveness": 0.20,
      "readability": 0.15,
      "actionability": 0.10,
      "context_memory": 0.10
    },
    "threshold": 0.6
  },
  "tags": ["scene:health_interpret", "expression:normal", "logic:consistent", "l2"]
}
```

## 4. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py` | NEW | Evaluator plugin: judge prompt, scoring, JSON parsing |
| `evaluator/core/schema.py` | MODIFY | Add optional `persona: Dict[str, str]` to `AutoUserInfo` |
| `evaluator/plugin/test_agent/auto_test_agent.py` | MODIFY | Read `persona`, append behavioral descriptions to prompt |
| `benchmark/data/dialogue_quality/l2_core.jsonl` | NEW | 12 cases (3 scenes x 4 combinations) |
| `benchmark/data/dialogue_quality/metadata.json` | MODIFY | Update description for Layer 2 |
| `benchmark/data/dialogue_quality/README.md` | MODIFY | Update documentation |

## 5. What's NOT In Scope

- **Layer 1** (`record_retrieval`): untouched, runs independently
- **Orchestrator / BatchSession**: no changes needed, auto-adapts via plugin system
- **Web UI**: auto-discovers new evaluator via plugin registry
- **Dynamic state transitions**: not implementing APAF dynamic_state for virtual users
- **New TestAgent plugin**: reusing existing `auto` TestAgent with persona extension
