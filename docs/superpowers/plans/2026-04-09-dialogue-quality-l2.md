# Dialogue Quality Layer 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-as-judge multi-dimension evaluator for health management dialogue quality, with persona-driven virtual users and a 12-case benchmark dataset.

**Architecture:** New `dialogue_quality` EvalAgent plugin scored by a single LLM judge call across 6 dimensions. Virtual users use existing `auto` TestAgent with a new `persona` field injecting behavioral traits (Expression + Logic). Dataset lives alongside Layer 1 in `benchmark/data/dialogue_quality/`.

**Tech Stack:** Python 3.11+, Pydantic v2, langchain, `do_execute()` LLM wrapper, pytest

**Spec:** `docs/superpowers/specs/2026-04-09-dialogue-quality-l2-design.md`

---

### Task 1: Add `persona` field to `AutoUserInfo`

**Files:**
- Modify: `evaluator/core/schema.py:106-169` (AutoUserInfo class)

- [ ] **Step 1: Write the failing test**

Create `evaluator/tests/test_dialogue_quality.py` with a test that parses an AutoUserInfo with persona:

```python
"""
DialogueQuality Layer 2 单元测试

运行：uv run pytest evaluator/tests/test_dialogue_quality.py -v -s
"""

import pytest
from evaluator.core.schema import AutoUserInfo


class TestAutoUserInfoPersona:
    """AutoUserInfo.persona 字段测试"""

    def test_persona_accepted(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"expression": "vague", "logic": "consistent"},
        )
        assert info.persona == {"expression": "vague", "logic": "consistent"}

    def test_persona_default_none(self):
        info = AutoUserInfo(type="auto", goal="test goal")
        assert info.persona is None

    def test_persona_empty_dict(self):
        info = AutoUserInfo(type="auto", goal="test goal", persona={})
        assert info.persona == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestAutoUserInfoPersona -v`
Expected: FAIL — `AutoUserInfo` has `extra="forbid"`, so `persona` field is rejected.

- [ ] **Step 3: Add persona field to AutoUserInfo**

In `evaluator/core/schema.py`, add the `persona` field to `AutoUserInfo` after `finish_condition` (line ~169):

```python
    persona: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "虚拟用户行为特征 — 键值对形式的行为维度配置。"
            "支持的维度：expression (normal/vague/incoherent), logic (consistent/occasional_contradiction/fabricating)。"
            "auto TestAgent 会据此在 system prompt 中注入对应的行为描述。"
        ),
    )
```

Note: `Dict` is already imported at the top of schema.py. `Optional` is also available.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestAutoUserInfoPersona -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluator/core/schema.py evaluator/tests/test_dialogue_quality.py
git commit -m "feat(schema): add persona field to AutoUserInfo for Layer 2 virtual user behavioral traits"
```

---

### Task 2: Inject persona into auto TestAgent prompt

**Files:**
- Modify: `evaluator/plugin/test_agent/auto_test_agent.py:85-114` (_build_system_prompt method)
- Test: `evaluator/tests/test_dialogue_quality.py`

- [ ] **Step 1: Write the failing test**

Append to `evaluator/tests/test_dialogue_quality.py`:

```python
from evaluator.core.schema import AutoUserInfo
from evaluator.plugin.test_agent.auto_test_agent import AutoTestAgent


class TestAutoTestAgentPersona:
    """AutoTestAgent persona prompt injection 测试"""

    def test_no_persona_no_behavioral_section(self):
        info = AutoUserInfo(type="auto", goal="test goal", context="35岁上班族")
        agent = AutoTestAgent(user_info=info)
        assert "行为特征" not in agent.system_prompt

    def test_vague_expression_injected(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"expression": "vague"},
        )
        agent = AutoTestAgent(user_info=info)
        assert "模糊口语" in agent.system_prompt

    def test_incoherent_expression_injected(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"expression": "incoherent"},
        )
        agent = AutoTestAgent(user_info=info)
        assert "答非所问" in agent.system_prompt

    def test_normal_expression_no_injection(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"expression": "normal"},
        )
        agent = AutoTestAgent(user_info=info)
        # normal = default, no behavioral traits section needed
        assert "模糊口语" not in agent.system_prompt
        assert "答非所问" not in agent.system_prompt

    def test_occasional_contradiction_injected(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"logic": "occasional_contradiction"},
        )
        agent = AutoTestAgent(user_info=info)
        assert "前后矛盾" in agent.system_prompt

    def test_fabricating_injected(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"logic": "fabricating"},
        )
        agent = AutoTestAgent(user_info=info)
        assert "编造健康信息" in agent.system_prompt

    def test_consistent_logic_no_injection(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"logic": "consistent"},
        )
        agent = AutoTestAgent(user_info=info)
        assert "前后矛盾" not in agent.system_prompt
        assert "编造健康信息" not in agent.system_prompt

    def test_combined_persona(self):
        info = AutoUserInfo(
            type="auto",
            goal="test goal",
            persona={"expression": "vague", "logic": "fabricating"},
        )
        agent = AutoTestAgent(user_info=info)
        assert "模糊口语" in agent.system_prompt
        assert "编造健康信息" in agent.system_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestAutoTestAgentPersona -v`
Expected: FAIL — persona is not yet injected into the prompt.

- [ ] **Step 3: Add persona prompt mapping and injection to _build_system_prompt**

In `evaluator/plugin/test_agent/auto_test_agent.py`, add the mapping table as a module-level constant before the `AutoTestAgent` class:

```python
# Persona behavioral prompt fragments (Layer 2 dialogue quality)
_PERSONA_PROMPTS: dict[str, dict[str, str]] = {
    "expression": {
        "vague": (
            "你表达能力有限，描述健康问题时用模糊口语："有一阵子了"、"就是不太舒服"、"大概这一块"。"
            "即使 AI 追问也无法给出更精确的描述。"
        ),
        "incoherent": (
            "你很难把话说清楚。经常答非所问，说到一半思路断了，"
            "用"那个""就是""怎么说呢"代替关键词。AI 很难从你的回答中提取有用信息。"
        ),
    },
    "logic": {
        "occasional_contradiction": (
            "你偶尔会前后矛盾。比如先说"大概两天了"，后来又说"可能有一周了"。"
            "如果 AI 指出矛盾你会修正。"
        ),
        "fabricating": (
            "你会编造健康信息。比如问到没有的症状你说"有"并编造细节，"
            "夸大不适感，编造用药经历。"
        ),
    },
}
```

Then modify `_build_system_prompt` in `AutoTestAgent`. After the context section (after line 101) and before the finish condition, add:

```python
        # Persona behavioral traits injection (Layer 2)
        if hasattr(self.user_info, "persona") and self.user_info.persona:
            trait_lines = []
            for dim, value in self.user_info.persona.items():
                prompts = _PERSONA_PROMPTS.get(dim, {})
                fragment = prompts.get(value)
                if fragment:
                    trait_lines.append(f"- {fragment}")
            if trait_lines:
                lines.append("\n# Your Behavioral Traits (行为特征)")
                lines.extend(trait_lines)
```

This goes after line 101 (`lines.append(f"\n# Your Background\n{self.user_info.context}")`), before the finish_condition block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py -v`
Expected: All tests in both `TestAutoUserInfoPersona` and `TestAutoTestAgentPersona` PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluator/plugin/test_agent/auto_test_agent.py evaluator/tests/test_dialogue_quality.py
git commit -m "feat(test-agent): inject persona behavioral traits into auto TestAgent prompt"
```

---

### Task 3: Create `dialogue_quality` evaluator plugin

**Files:**
- Create: `evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py`
- Test: `evaluator/tests/test_dialogue_quality.py`

- [ ] **Step 1: Write the failing test for config model and registration**

Append to `evaluator/tests/test_dialogue_quality.py`:

```python
from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent


class TestDialogueQualityRegistration:
    """dialogue_quality evaluator registration 测试"""

    def test_registered(self):
        assert AbstractEvalAgent.has("dialogue_quality")

    def test_lookup(self):
        cls = AbstractEvalAgent.get("dialogue_quality")
        assert cls.__name__ == "DialogueQualityEvalAgent"


class TestDialogueQualityEvalInfo:
    """DialogueQualityEvalInfo config model 测试"""

    def test_default_dimensions(self):
        from evaluator.plugin.eval_agent.dialogue_quality_eval_agent import DialogueQualityEvalInfo

        info = DialogueQualityEvalInfo()
        assert abs(sum(info.dimensions.values()) - 1.0) < 1e-6
        assert info.threshold == 0.6
        assert info.judge_model == "gpt-4.1"

    def test_custom_dimensions(self):
        from evaluator.plugin.eval_agent.dialogue_quality_eval_agent import DialogueQualityEvalInfo

        info = DialogueQualityEvalInfo(
            dimensions={"accuracy": 0.5, "personalization": 0.5},
            threshold=0.7,
        )
        assert info.dimensions == {"accuracy": 0.5, "personalization": 0.5}
        assert info.threshold == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestDialogueQualityRegistration -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the evaluator plugin file**

Create `evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py`:

```python
"""
DialogueQualityEvalAgent — LLM-as-Judge multi-dimension dialogue quality evaluator

Registered name: "dialogue_quality"

Evaluates multi-turn health management dialogue quality across 6 dimensions:
  - accuracy:          健康数据、医学知识是否正确
  - personalization:   是否结合用户具体健康数据给出针对性回答
  - comprehensiveness: 是否覆盖用户问题涉及的多个维度
  - readability:       回复结构是否清晰、分层合理
  - actionability:     是否给出具体可执行的建议
  - context_memory:    多轮对话中是否记住前文信息

Evaluation flow:
1. Extract full conversation from memory_list + history
2. Build judge prompt with user context, goal, dialogue, dimensions
3. Single LLM call to judge model
4. Parse JSON → per-dimension 1-5 scores + evidence
5. Weighted sum → normalize to 0-1 → pass/fail

Config example:
{
    "evaluator": "dialogue_quality",
    "dimensions": {"accuracy": 0.25, "personalization": 0.20, ...},
    "threshold": 0.6,
    "judge_model": "gpt-4.1"
}
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, EvalTrace, SessionInfo, TestAgentMemory
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)

_DEFAULT_DIMENSIONS = {
    "accuracy": 0.25,
    "personalization": 0.20,
    "comprehensiveness": 0.20,
    "readability": 0.15,
    "actionability": 0.10,
    "context_memory": 0.10,
}

_JUDGE_PROMPT = """你是一个健康管理 AI 助手的质量评估专家。请评估以下多轮对话中 AI 助手的表现。

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
{{
  "dimensions": {{
    "accuracy": {{"score": <1-5>, "evidence": [{{"turn": <轮次号>, "quote": "<引用原文>", "comment": "<评价>"}}]}},
    "personalization": {{"score": <1-5>, "evidence": [...]}},
    "comprehensiveness": {{"score": <1-5>, "evidence": [...]}},
    "readability": {{"score": <1-5>, "evidence": [...]}},
    "actionability": {{"score": <1-5>, "evidence": [...]}},
    "context_memory": {{"score": <1-5>, "evidence": [...]}}
  }},
  "overall_comment": "<一句话总评>"
}}"""


class DialogueQualityEvalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator: Literal["dialogue_quality"] = Field(default="dialogue_quality", description="Evaluator type")
    dimensions: Dict[str, float] = Field(default_factory=lambda: dict(_DEFAULT_DIMENSIONS), description="维度名 → 权重")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Pass threshold (0-1)")
    judge_model: str = Field(default="gpt-4.1", description="Judge LLM model")


class DialogueQualityEvalAgent(
    AbstractEvalAgent,
    name="dialogue_quality",
    params_model=DialogueQualityEvalInfo,
):
    _display_meta = {
        "icon": "M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z",
        "color": "#6366f1",
        "features": ["LLM-as-Judge", "6 Dimensions", "Multi-turn Quality"],
    }
    _cost_meta = {"est_cost_per_case": 0.03}

    def __init__(self, eval_config: DialogueQualityEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)

    async def run(self, memory_list: List[TestAgentMemory], session_info: SessionInfo | None = None) -> EvalResult:
        cfg: DialogueQualityEvalInfo = self.eval_config

        # 1. Extract conversation
        dialogue = self._format_dialogue(memory_list, self.history)
        if not dialogue:
            return EvalResult(result="fail", score=0.0, feedback="No dialogue content to evaluate")

        # 2. Build judge prompt
        user_context = self.user_info.context if self.user_info else "（未提供）"
        user_goal = self.user_info.goal if self.user_info and hasattr(self.user_info, "goal") else "（未提供）"
        prompt = _JUDGE_PROMPT.format(
            user_context=user_context,
            user_goal=user_goal,
            formatted_dialogue=dialogue,
        )

        # 3. Call judge LLM
        try:
            result = await do_execute(
                model=cfg.judge_model,
                system_prompt="You are a dialogue quality evaluation expert. Always respond with valid JSON.",
                input=prompt,
            )
        except Exception as e:
            logger.error("[DialogueQualityEval] Judge LLM call failed: %s", e)
            return EvalResult(result="error", score=0.0, feedback=f"Judge LLM error: {e}")

        # 4. Parse JSON response
        raw = result.content or ""
        parsed = self._parse_judge_response(raw, cfg.dimensions)
        if parsed is None:
            return EvalResult(
                result="error",
                score=0.0,
                feedback=f"Failed to parse judge response: {raw[:200]}",
                trace=EvalTrace(eval_detail={"raw_response": raw}),
            )

        # 5. Compute weighted score
        dimensions_result, overall_comment = parsed
        score = self._compute_weighted_score(dimensions_result, cfg.dimensions)
        passed = score >= cfg.threshold

        feedback = overall_comment or f"Weighted score: {score:.2f}"

        return EvalResult(
            result="pass" if passed else "fail",
            score=score,
            feedback=feedback,
            trace=EvalTrace(eval_detail={"dimensions": dimensions_result, "overall_comment": overall_comment}),
        )

    @staticmethod
    def _format_dialogue(memory_list: List[TestAgentMemory], history: List[BaseMessage] | None = None) -> str:
        """Format conversation history + memory_list into numbered dialogue text."""
        lines = []
        turn = 1

        # Pre-evaluation history
        if history:
            for msg in history:
                if isinstance(msg, HumanMessage):
                    lines.append(f"[Turn {turn}] 用户: {msg.content}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"[Turn {turn}] AI: {msg.content}")
                    turn += 1

        # Evaluation conversation
        for mem in memory_list:
            user_text = mem.test_reaction.action.semantic_content if mem.test_reaction else ""
            ai_text = mem.target_response.extract_text() if mem.target_response else ""

            if user_text:
                lines.append(f"[Turn {turn}] 用户: {user_text}")
            if ai_text:
                lines.append(f"[Turn {turn}] AI: {ai_text}")
                turn += 1

        return "\n".join(lines)

    @staticmethod
    def _parse_judge_response(raw: str, dimensions: Dict[str, float]) -> tuple[Dict[str, Any], str] | None:
        """Parse judge LLM JSON response. Returns (dimensions_dict, overall_comment) or None."""
        # Try to extract JSON from markdown code block or raw text
        json_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        json_str = json_match.group(1).strip() if json_match else raw.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("[DialogueQualityEval] JSON parse failed: %s", json_str[:200])
            return None

        if "dimensions" not in data:
            logger.warning("[DialogueQualityEval] Missing 'dimensions' key in response")
            return None

        dims = data["dimensions"]
        overall = data.get("overall_comment", "")

        # Validate: each configured dimension should have a score
        result = {}
        for dim_name in dimensions:
            if dim_name in dims and isinstance(dims[dim_name], dict) and "score" in dims[dim_name]:
                score = dims[dim_name]["score"]
                if isinstance(score, (int, float)) and 1 <= score <= 5:
                    result[dim_name] = dims[dim_name]
                    continue
            # Fallback: missing or invalid → score 3 (neutral)
            logger.warning("[DialogueQualityEval] Dimension %r missing or invalid, defaulting to 3", dim_name)
            result[dim_name] = {"score": 3, "evidence": [], "comment": "dimension missing from judge response"}

        return result, overall

    @staticmethod
    def _compute_weighted_score(dimensions: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Compute weighted score normalized to 0-1.

        Each dimension is 1-5. Weighted sum maps to 0-1 via: (sum - 1) / 4
        """
        total_weight = sum(weights.get(d, 0) for d in dimensions)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(dimensions[d]["score"] * weights.get(d, 0) / total_weight for d in dimensions)
        return (weighted_sum - 1) / 4
```

- [ ] **Step 4: Run registration and config tests**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestDialogueQualityRegistration evaluator/tests/test_dialogue_quality.py::TestDialogueQualityEvalInfo -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py evaluator/tests/test_dialogue_quality.py
git commit -m "feat(eval): add dialogue_quality evaluator plugin with LLM-as-judge multi-dimension scoring"
```

---

### Task 4: Unit test scoring and parsing logic

**Files:**
- Test: `evaluator/tests/test_dialogue_quality.py`
- Existing: `evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py`

- [ ] **Step 1: Write tests for _compute_weighted_score and _parse_judge_response**

Append to `evaluator/tests/test_dialogue_quality.py`:

```python
from evaluator.plugin.eval_agent.dialogue_quality_eval_agent import DialogueQualityEvalAgent


class TestComputeWeightedScore:
    """_compute_weighted_score 计算测试"""

    def test_all_fives(self):
        dims = {d: {"score": 5} for d in ["accuracy", "personalization", "comprehensiveness", "readability", "actionability", "context_memory"]}
        weights = {"accuracy": 0.25, "personalization": 0.20, "comprehensiveness": 0.20, "readability": 0.15, "actionability": 0.10, "context_memory": 0.10}
        score = DialogueQualityEvalAgent._compute_weighted_score(dims, weights)
        assert abs(score - 1.0) < 1e-6

    def test_all_ones(self):
        dims = {d: {"score": 1} for d in ["accuracy", "personalization", "comprehensiveness", "readability", "actionability", "context_memory"]}
        weights = {"accuracy": 0.25, "personalization": 0.20, "comprehensiveness": 0.20, "readability": 0.15, "actionability": 0.10, "context_memory": 0.10}
        score = DialogueQualityEvalAgent._compute_weighted_score(dims, weights)
        assert abs(score - 0.0) < 1e-6

    def test_all_threes(self):
        dims = {d: {"score": 3} for d in ["accuracy", "personalization"]}
        weights = {"accuracy": 0.5, "personalization": 0.5}
        score = DialogueQualityEvalAgent._compute_weighted_score(dims, weights)
        assert abs(score - 0.5) < 1e-6

    def test_mixed_scores(self):
        dims = {"accuracy": {"score": 5}, "personalization": {"score": 1}}
        weights = {"accuracy": 0.5, "personalization": 0.5}
        score = DialogueQualityEvalAgent._compute_weighted_score(dims, weights)
        # weighted_sum = 5*0.5 + 1*0.5 = 3.0, normalized = (3-1)/4 = 0.5
        assert abs(score - 0.5) < 1e-6


class TestParseJudgeResponse:
    """_parse_judge_response 解析测试"""

    def test_valid_json(self):
        raw = json.dumps({
            "dimensions": {
                "accuracy": {"score": 4, "evidence": [{"turn": 1, "quote": "test", "comment": "good"}]},
                "personalization": {"score": 3, "evidence": []},
            },
            "overall_comment": "Decent performance",
        })
        weights = {"accuracy": 0.5, "personalization": 0.5}
        result = DialogueQualityEvalAgent._parse_judge_response(raw, weights)
        assert result is not None
        dims, comment = result
        assert dims["accuracy"]["score"] == 4
        assert dims["personalization"]["score"] == 3
        assert comment == "Decent performance"

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"dimensions": {"accuracy": {"score": 5, "evidence": []}}, "overall_comment": "Great"}\n```'
        weights = {"accuracy": 1.0}
        result = DialogueQualityEvalAgent._parse_judge_response(raw, weights)
        assert result is not None
        dims, comment = result
        assert dims["accuracy"]["score"] == 5

    def test_missing_dimension_defaults_to_3(self):
        raw = json.dumps({
            "dimensions": {"accuracy": {"score": 4, "evidence": []}},
            "overall_comment": "ok",
        })
        weights = {"accuracy": 0.5, "personalization": 0.5}
        result = DialogueQualityEvalAgent._parse_judge_response(raw, weights)
        assert result is not None
        dims, _ = result
        assert dims["personalization"]["score"] == 3

    def test_invalid_json_returns_none(self):
        result = DialogueQualityEvalAgent._parse_judge_response("not json at all", {"accuracy": 1.0})
        assert result is None

    def test_missing_dimensions_key_returns_none(self):
        raw = json.dumps({"overall_comment": "no dims"})
        result = DialogueQualityEvalAgent._parse_judge_response(raw, {"accuracy": 1.0})
        assert result is None

    def test_score_out_of_range_defaults_to_3(self):
        raw = json.dumps({
            "dimensions": {"accuracy": {"score": 10, "evidence": []}},
            "overall_comment": "ok",
        })
        result = DialogueQualityEvalAgent._parse_judge_response(raw, {"accuracy": 1.0})
        assert result is not None
        dims, _ = result
        assert dims["accuracy"]["score"] == 3
```

Note: add `import json` at the top of the test file if not already present.

- [ ] **Step 2: Run tests**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestComputeWeightedScore evaluator/tests/test_dialogue_quality.py::TestParseJudgeResponse -v`
Expected: All PASS (these test static methods on the already-created class).

- [ ] **Step 3: Commit**

```bash
git add evaluator/tests/test_dialogue_quality.py
git commit -m "test(eval): add scoring and parsing unit tests for dialogue_quality evaluator"
```

---

### Task 5: Unit test _format_dialogue

**Files:**
- Test: `evaluator/tests/test_dialogue_quality.py`

- [ ] **Step 1: Write the test**

Append to `evaluator/tests/test_dialogue_quality.py`:

```python
from evaluator.core.schema import TestAgentAction, TestAgentReaction, TargetAgentReaction, TestAgentMemory
from langchain_core.messages import HumanMessage, AIMessage


class TestFormatDialogue:
    """_format_dialogue 对话格式化测试"""

    def _make_memory(self, user_text: str, ai_text: str) -> TestAgentMemory:
        return TestAgentMemory(
            test_reaction=TestAgentReaction(
                action=TestAgentAction(type="semantic", semantic_content=user_text),
                reason="test",
                is_finished=False,
            ),
            target_response=TargetAgentReaction(raw_output=ai_text),
        )

    def test_basic_two_turns(self):
        mems = [
            self._make_memory("你好", "你好，有什么可以帮你的？"),
            self._make_memory("我血压高", "建议你监测血压"),
        ]
        result = DialogueQualityEvalAgent._format_dialogue(mems)
        assert "[Turn 1] 用户: 你好" in result
        assert "[Turn 1] AI: 你好，有什么可以帮你的？" in result
        assert "[Turn 2] 用户: 我血压高" in result
        assert "[Turn 2] AI: 建议你监测血压" in result

    def test_with_history(self):
        history = [HumanMessage(content="之前聊过"), AIMessage(content="记得的")]
        mems = [self._make_memory("继续聊", "好的")]
        result = DialogueQualityEvalAgent._format_dialogue(mems, history)
        assert "[Turn 1] 用户: 之前聊过" in result
        assert "[Turn 1] AI: 记得的" in result
        assert "[Turn 2] 用户: 继续聊" in result

    def test_empty_memory_list(self):
        result = DialogueQualityEvalAgent._format_dialogue([])
        assert result == ""
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py::TestFormatDialogue -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add evaluator/tests/test_dialogue_quality.py
git commit -m "test(eval): add dialogue formatting tests for dialogue_quality evaluator"
```

---

### Task 6: Create benchmark dataset `l2_core.jsonl`

**Files:**
- Create: `benchmark/data/dialogue_quality/l2_core.jsonl`
- Modify: `benchmark/data/dialogue_quality/metadata.json`

- [ ] **Step 1: Create l2_core.jsonl with 12 cases**

Create `benchmark/data/dialogue_quality/l2_core.jsonl`. Each line is one JSON object. 3 scenes x 4 persona combos = 12 cases:

```jsonl
{"id": "dq_l2_interpret_001", "title": "血检报告解读 - 正常表达/一致逻辑", "user": {"type": "auto", "goal": "你想了解最近血检报告中几个异常指标的含义和注意事项", "context": "你是35岁的上班族，最近体检发现总胆固醇6.2mmol/L（偏高）、低密度脂蛋白4.1mmol/L（偏高）、空腹血糖6.0mmol/L（临界）。你没有明显不适症状，平时饮食不太注意，运动较少。", "max_turns": 8, "persona": {"expression": "normal", "logic": "consistent"}, "target_overrides": {"theta_api": {"email": "user225@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_interpret", "expression:normal", "logic:consistent", "l2"]}
{"id": "dq_l2_interpret_002", "title": "血检报告解读 - 模糊表达/一致逻辑", "user": {"type": "auto", "goal": "你想了解最近血检报告中几个异常指标的含义，但你表达不太清楚", "context": "你是48岁的工厂工人，最近体检说血脂高、血糖也有点高，但你记不太清具体数值。你只知道医生说了几个指标不太好。平时吃得比较油腻，很少运动。", "max_turns": 8, "persona": {"expression": "vague", "logic": "consistent"}, "target_overrides": {"theta_api": {"email": "user226@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_interpret", "expression:vague", "logic:consistent", "l2"]}
{"id": "dq_l2_interpret_003", "title": "体检报告解读 - 正常表达/偶尔矛盾", "user": {"type": "auto", "goal": "你想让AI帮你看看体检报告里的肝功能指标", "context": "你是42岁的销售经理，体检发现谷丙转氨酶ALT 65U/L（偏高）、谷草转氨酶AST 48U/L（偏高）。你平时应酬多，喝酒比较频繁，但有时候会说自己不怎么喝酒。", "max_turns": 8, "persona": {"expression": "normal", "logic": "occasional_contradiction"}, "target_overrides": {"theta_api": {"email": "user227@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_interpret", "expression:normal", "logic:occasional_contradiction", "l2"]}
{"id": "dq_l2_interpret_004", "title": "体检报告解读 - 表达混乱/编造信息", "user": {"type": "auto", "goal": "你想问问最近体检的几个指标是什么意思", "context": "你是55岁的退休老人，体检发现尿酸偏高、肌酐临界。你说话经常说不清楚，而且有时候会编造一些症状来引起AI的重视。", "max_turns": 8, "persona": {"expression": "incoherent", "logic": "fabricating"}, "target_overrides": {"theta_api": {"email": "user228@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_interpret", "expression:incoherent", "logic:fabricating", "l2"]}
{"id": "dq_l2_plan_001", "title": "减重方案制定 - 正常表达/一致逻辑", "user": {"type": "auto", "goal": "你想让AI帮你制定一个减重计划，目标是3个月减5公斤", "context": "你是30岁的程序员，身高175cm体重82kg，BMI 26.8（超重）。久坐工作，每天外卖为主，几乎不运动。没有慢性病，但最近觉得容易疲劳。", "max_turns": 8, "persona": {"expression": "normal", "logic": "consistent"}, "target_overrides": {"theta_api": {"email": "user229@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_plan", "expression:normal", "logic:consistent", "l2"]}
{"id": "dq_l2_plan_002", "title": "运动方案制定 - 模糊表达/一致逻辑", "user": {"type": "auto", "goal": "你想开始运动但不知道从哪里开始，想让AI给你一些建议", "context": "你是40岁的全职妈妈，之前很少运动，最近觉得体力下降想动一动。你对运动知识了解很少，说不清自己想要什么样的运动方式。", "max_turns": 8, "persona": {"expression": "vague", "logic": "consistent"}, "target_overrides": {"theta_api": {"email": "user230@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_plan", "expression:vague", "logic:consistent", "l2"]}
{"id": "dq_l2_plan_003", "title": "营养方案制定 - 正常表达/偶尔矛盾", "user": {"type": "auto", "goal": "你想改善饮食习惯，让AI帮你做一个营养调整方案", "context": "你是38岁的教师，轻度脂肪肝，医生建议控制饮食。你对营养有一些了解，但描述饮食习惯时前后会有出入，比如有时说自己吃得很清淡，有时又提到经常吃火锅烧烤。", "max_turns": 8, "persona": {"expression": "normal", "logic": "occasional_contradiction"}, "target_overrides": {"theta_api": {"email": "user231@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_plan", "expression:normal", "logic:occasional_contradiction", "l2"]}
{"id": "dq_l2_plan_004", "title": "减重方案制定 - 表达混乱/编造信息", "user": {"type": "auto", "goal": "你想减肥但总是说不清楚自己的情况", "context": "你是50岁的出租车司机，体重偏重，有高血压在吃药。你说话啰嗦且混乱，而且为了让AI觉得你很重视健康，会编造一些运动经历和饮食习惯。", "max_turns": 8, "persona": {"expression": "incoherent", "logic": "fabricating"}, "target_overrides": {"theta_api": {"email": "user232@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_plan", "expression:incoherent", "logic:fabricating", "l2"]}
{"id": "dq_l2_qa_001", "title": "日常健康问答 - 正常表达/一致逻辑", "user": {"type": "auto", "goal": "你最近睡眠不太好，想问问AI有什么改善睡眠的方法", "context": "你是28岁的设计师，最近项目压力大，连续两周入睡困难，每天只能睡5-6小时。没有其他健康问题，不想吃安眠药。", "max_turns": 8, "persona": {"expression": "normal", "logic": "consistent"}, "target_overrides": {"theta_api": {"email": "user233@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_qa", "expression:normal", "logic:consistent", "l2"]}
{"id": "dq_l2_qa_002", "title": "日常健康问答 - 模糊表达/一致逻辑", "user": {"type": "auto", "goal": "你经常头疼，想问问AI这是怎么回事", "context": "你是45岁的快递员，最近经常头疼但说不清是哪种疼、什么时候疼。你只知道"就是疼"，问什么都说"不太清楚"。没去过医院检查。", "max_turns": 8, "persona": {"expression": "vague", "logic": "consistent"}, "target_overrides": {"theta_api": {"email": "user234@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_qa", "expression:vague", "logic:consistent", "l2"]}
{"id": "dq_l2_qa_003", "title": "日常健康问答 - 正常表达/偶尔矛盾", "user": {"type": "auto", "goal": "你想咨询一下关于维生素补充的问题", "context": "你是33岁的白领，最近觉得容易疲劳、口腔溃疡频繁，朋友说可能缺维生素。你自己也查了一些资料，但有时候会把不同维生素的功能搞混。", "max_turns": 8, "persona": {"expression": "normal", "logic": "occasional_contradiction"}, "target_overrides": {"theta_api": {"email": "user235@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_qa", "expression:normal", "logic:occasional_contradiction", "l2"]}
{"id": "dq_l2_qa_004", "title": "日常健康问答 - 表达混乱/编造信息", "user": {"type": "auto", "goal": "你想问AI关于胃不舒服的问题", "context": "你是60岁的退休工人，最近胃不太舒服，但你说话总是东拉西扯说不到重点。而且你会编造一些症状（比如说自己吐血了其实没有）来让AI更重视你。", "max_turns": 8, "persona": {"expression": "incoherent", "logic": "fabricating"}, "target_overrides": {"theta_api": {"email": "user236@demo"}}}, "eval": {"evaluator": "dialogue_quality", "threshold": 0.6}, "tags": ["scene:health_qa", "expression:incoherent", "logic:fabricating", "l2"]}
```

- [ ] **Step 2: Update metadata.json**

Replace the `description` field in `benchmark/data/dialogue_quality/metadata.json` to include Layer 2:

```json
{
  "description": "# Dialogue Quality Evaluation\n\nMulti-turn dialogue quality evaluation for Theta Health AI.\n\n## Layer 1 (RECORD + RETRIEVAL)\nTests RECORD acknowledgment quality and RETRIEVAL data accuracy.\n**Evaluator**: `record_retrieval` (zero LLM, per-turn checkpoints)\n\n| Dataset | Cases | Purpose |\n|---------|-------|---------|\n| smoke | 5 | CI gate — fast smoke test |\n| core | 15+ | Full evaluation — all record types |\n\n## Layer 2 (Dialogue Quality)\nLLM-as-judge multi-dimension quality scoring for complex health management scenarios.\n**Evaluator**: `dialogue_quality` (6 dimensions, LLM judge)\n\n| Dataset | Cases | Purpose |\n|---------|-------|---------|\n| l2_core | 12 | Health interpret + plan + Q&A, with persona combos |\n\n**Target**: `theta_api` (Theta Health API)",
  "target": [
    {
      "type": "theta_api",
      "fields": {
        "agent": {"default": "expert", "editable": true}
      }
    }
  ]
}
```

- [ ] **Step 3: Verify dataset loads**

Run: `uv run python -m evaluator list -b | grep dialogue_quality`
Expected: Shows `dialogue_quality` with `smoke`, `core`, and `l2_core` datasets.

- [ ] **Step 4: Commit**

```bash
git add benchmark/data/dialogue_quality/l2_core.jsonl benchmark/data/dialogue_quality/metadata.json
git commit -m "feat(benchmark): add Layer 2 dialogue quality dataset (12 cases, 3 scenes x 4 persona combos)"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `benchmark/data/dialogue_quality/README.md`

- [ ] **Step 1: Update README to document Layer 2**

Add a Layer 2 section after the existing Layer 1 content (before the "Layer 2 设计笔记" section). Replace the "Layer 2 设计笔记（待实现）" header with "Layer 2: Dialogue Quality (LLM-as-Judge)" and mark it as implemented. Key content to add:

- Layer 2 evaluator: `dialogue_quality`, LLM-as-judge, 6 dimensions
- Dataset: `l2_core` (12 cases)
- Persona dimensions: Expression (normal/vague/incoherent) + Logic (consistent/occasional_contradiction/fabricating)
- Running command:

```bash
THETA_API_CHAT_PATH=/api/v1/chat/create_message \
THETA_API_LIST_MESSAGE_PATH=/api/v1/chat/list_message_chunks \
uv run python -m benchmark.basic_runner dialogue_quality l2_core --target-type theta_api -p 1 -v
```

Remove the "待实现" marker from the Layer 2 section header. Keep the Layer 2 design notes subsections (用户真实关注的质量维度, 用户关注的功能场景, etc.) as reference context.

- [ ] **Step 2: Commit**

```bash
git add benchmark/data/dialogue_quality/README.md
git commit -m "docs: update dialogue quality README with Layer 2 evaluator and dataset documentation"
```

---

### Task 8: Run full test suite and lint

**Files:** None (verification only)

- [ ] **Step 1: Run all dialogue quality tests**

Run: `uv run pytest evaluator/tests/test_dialogue_quality.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run full test suite to check no regressions**

Run: `uv run pytest evaluator/tests/ -v`
Expected: All existing tests PASS.

- [ ] **Step 3: Run lint**

Run: `uv run ruff check evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py evaluator/plugin/test_agent/auto_test_agent.py evaluator/core/schema.py evaluator/tests/test_dialogue_quality.py`
Expected: No lint errors.

Run: `uv run ruff format --check evaluator/plugin/eval_agent/dialogue_quality_eval_agent.py evaluator/plugin/test_agent/auto_test_agent.py evaluator/core/schema.py evaluator/tests/test_dialogue_quality.py`
Expected: No formatting issues.

- [ ] **Step 4: Fix any issues and commit if needed**

If lint or tests failed, fix and commit:

```bash
git add -u
git commit -m "fix: address lint and test issues"
```
