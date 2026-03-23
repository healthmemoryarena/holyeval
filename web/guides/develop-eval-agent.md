# Developing Evaluation Logic (EvalAgent)

> **Using in Claude Code**: Type `/add-eval-agent <agent-name>`, followed by your evaluation logic description (e.g., scoring rules, input/output requirements). Claude will automatically generate the configuration, implementation, and registration code.

## Overview

EvalAgent is a core component of the evaluation framework, responsible for assessing the system under test after a conversation ends. The framework supports multiple evaluation strategies through a plugin mechanism, allowing you to develop custom evaluation logic based on your needs.

## Existing Evaluators

| Name | Type | Description |
|------|------|------|
| `semantic` | LLM | Semantic understanding evaluation, uses LLM to judge response quality |
| `healthbench` | LLM | HealthBench rubric scoring, based on multi-dimensional grading criteria |
| `medcalc` | LLM + Rules | MedCalc-Bench medical calculation evaluation (LLM answer extraction + typed numerical matching) |
| `hallucination` | LLM | Medical hallucination detection (LLM-as-Judge), detects factual/contextual/citation hallucinations, scoring 0~1 |
| `indicator` | LLM + API | Health data indicator evaluation |
| `keyword` | Rules | Keyword matching evaluation |
| `preset_answer` | Rules | Preset answer matching (exact/keyword modes) |
| `redteam_compliance` | LLM | Red-team compliance evaluation (LLM-as-Judge medical compliance scoring) |
| `memoryarena` | LLM | MemoryArena multi-subtask evaluation (LLM-as-Judge + Progress Score) |

## Development Steps

### 1. Design the Evaluation Logic

Determine what your evaluator needs:
- **LLM-based** or **deterministic rules**? LLM is suitable for semantic judgment; rules are suitable for exact matching
- What **configuration parameters** are needed? (e.g., threshold, keyword list, model name, etc.)
- Is the scoring method **binary** (pass/fail) or **continuous** (0~1)?

### 2. Define the Configuration Model

Add a Pydantic configuration in `evaluator/core/schema.py`:

```python
class MyEvalInfo(BaseModel):
    """My custom evaluation configuration"""
    model_config = ConfigDict(extra="forbid")

    evaluator: Literal["my_eval"] = "my_eval"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Custom fields...
```

Then add it to the `EvalInfo` union type.

### 3. Implement the Evaluator

Create the implementation file in `evaluator/plugin/eval_agent/`:

```python
class MyEvalAgent(AbstractEvalAgent, name="my_eval"):
    async def run(self, memory_list, session_info=None) -> EvalResult:
        # memory_list: List[TestAgentMemory] — full conversation (user actions + target system responses)
        # session_info: SessionInfo | None — target system session info (authentication data, etc.)
        # self.history / self.user_info / self.case_id — static context (injected at init)
        # Return EvalResult(result="pass"|"fail", score=0.8, feedback="...")
```

### 4. Register the Plugin

Import the new class in `evaluator/plugin/eval_agent/__init__.py`.

### 5. Verify

```bash
# Check registration
python -c "from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; print(AbstractEvalAgent.get_all())"

# Lint check
ruff check evaluator/core/schema.py evaluator/plugin/eval_agent/
```

## Key Files

| File | Description |
|------|------|
| `evaluator/core/schema.py` | Configuration model definitions |
| `evaluator/core/interfaces/abstract_eval_agent.py` | Abstract interface |
| `evaluator/plugin/eval_agent/` | Plugin implementation directory |
| `evaluator/plugin/eval_agent/semantic_eval_agent.py` | LLM evaluation reference implementation |
| `evaluator/plugin/eval_agent/medcalc_eval_agent.py` | LLM + rules hybrid reference implementation |
| `evaluator/plugin/eval_agent/hallucination_eval_agent.py` | LLM-as-Judge hallucination detection reference implementation |
| `evaluator/plugin/eval_agent/preset_answer_eval_agent.py` | Rule-based evaluation reference implementation |
| `evaluator/plugin/eval_agent/redteam_compliance_eval_agent.py` | LLM-as-Judge compliance evaluation reference implementation |
| `evaluator/plugin/eval_agent/memoryarena_eval_agent.py` | LLM-as-Judge multi-subtask evaluation reference implementation |
