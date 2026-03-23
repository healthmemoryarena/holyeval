"""
JSONL → TestCase 数据流测试

测试覆盖：
1. JSONL 解析 → TestCase 对象（4 行，手动模式 + 自动模式 + 携带历史对话 + llm_api 历史对话）
2. Discriminated Union 配置 — eval 自动路由为 SemanticEvalInfo 实例
3. 自定义 criteria 从 JSONL 正确传递到 SemanticEvalInfo
4. Agent 构造正确性 — 配置正确传递到 SemanticEvalAgent
5. history 字段 — JSON dict 自动转换为 langchain BaseMessage

运行：uv run pytest evaluator/tests/test_data_flow.py -v -s
"""

import json
from pathlib import Path
from typing import List

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from evaluator.core.schema import TestCase
from evaluator.plugin.eval_agent.semantic_eval_agent import SemanticEvalInfo
from evaluator.plugin.eval_agent.semantic_eval_agent import SemanticEvalAgent

JSONL_PATH = Path(__file__).parent / "fixtures" / "test_cases.jsonl"


# ============================================================
# JSONL 加载
# ============================================================


def load_cases_from_jsonl(path: Path) -> List[TestCase]:
    """从 JSONL 文件加载测试用例列表（每行一个 JSON → TestCase）"""
    cases: List[TestCase] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(TestCase(**data))
    return cases


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def cases() -> List[TestCase]:
    return load_cases_from_jsonl(JSONL_PATH)


# ============================================================
# 1. JSONL 解析正确性
# ============================================================


def test_jsonl_parse_count(cases: List[TestCase]):
    """JSONL 应该解析出 4 条用例"""
    assert len(cases) == 4


def test_jsonl_parse_basic_fields(cases: List[TestCase]):
    """验证基础字段解析"""
    case = cases[0]
    assert case.id == "manual_headache_001"
    assert case.title == "偏头痛咨询 — 手动模式"
    assert case.user.type == "manual"
    assert not hasattr(case.user, "goal")  # ManualUserInfo 不含 goal 字段
    assert not hasattr(case.user, "max_turns")  # ManualUserInfo 不含 max_turns 字段
    assert len(case.user.strict_inputs) == 2
    assert case.target.type == "theta_api"
    assert case.eval.evaluator == "semantic"
    assert "symptom:头痛" in case.tags


def test_jsonl_parse_custom_criteria(cases: List[TestCase]):
    """验证自定义 criteria 正确解析到 SemanticEvalInfo"""
    case = cases[1]  # auto_cough_child_002 — 有自定义三维度
    assert isinstance(case.eval, SemanticEvalInfo)
    assert len(case.eval.criteria) == 3
    assert case.eval.criteria[0].name == "triage_accuracy"
    assert case.eval.threshold == 0.75


def test_jsonl_parse_default_eval(cases: List[TestCase]):
    """验证仅指定 evaluator + threshold → 使用默认 criteria"""
    case = cases[0]
    assert isinstance(case.eval, SemanticEvalInfo)
    assert case.eval.threshold == 0.7
    assert len(case.eval.criteria) == 3  # 默认三维度


def test_jsonl_parse_strict_inputs_multi(cases: List[TestCase]):
    """验证多条 strict_inputs（手动模式用例）"""
    case = cases[0]
    assert len(case.user.strict_inputs) == 2
    assert "太阳穴" in case.user.strict_inputs[0]
    assert "发作" in case.user.strict_inputs[1]


def test_jsonl_parse_auto_mode(cases: List[TestCase]):
    """验证自动模式用例的字段"""
    case = cases[1]
    assert case.user.type == "auto"
    assert case.user.context is not None
    assert case.user.finish_condition is not None
    assert len(case.user.strict_inputs) == 1  # 开场白前缀


# ============================================================
# 2. Discriminated Union 配置正确性
# ============================================================


def test_eval_config_default(cases: List[TestCase]):
    """默认配置 → SemanticEvalInfo 使用默认值"""
    case = cases[0]  # threshold=0.7, 无 criteria
    assert isinstance(case.eval, SemanticEvalInfo)
    assert case.eval.threshold == 0.7
    assert len(case.eval.criteria) == 3  # 默认三维度
    assert case.eval.criteria[0].name == "goal_completion"
    print(f"\n[default] eval type: {type(case.eval).__name__}, threshold={case.eval.threshold}")


def test_eval_config_custom_criteria(cases: List[TestCase]):
    """自定义 criteria → SemanticEvalInfo 正确解析"""
    case = cases[1]  # 自定义三维度 + threshold=0.75
    assert isinstance(case.eval, SemanticEvalInfo)
    assert case.eval.threshold == 0.75
    assert len(case.eval.criteria) == 3
    assert case.eval.criteria[0].name == "triage_accuracy"
    assert case.eval.criteria[0].weight == 50
    assert case.eval.criteria[1].display_name == "共情能力"
    print(f"\n[custom] criteria count={len(case.eval.criteria)}, threshold={case.eval.threshold}")


def test_target_config(cases: List[TestCase]):
    """验证 target 配置正确解析"""
    case = cases[0]
    assert case.target.type == "theta_api"
    assert case.target.email == "demo1@symptom_entry_evaluation.com"


# ============================================================
# 3. Agent 构造正确性
# ============================================================


def test_agent_from_default_config(cases: List[TestCase]):
    """默认配置 → SemanticEvalAgent 构造正确"""
    case = cases[0]
    agent = SemanticEvalAgent(case.eval)
    assert agent.eval_config.threshold == 0.7
    assert len(agent.eval_config.criteria) == 3
    print(f"\n[agent_default] threshold={agent.eval_config.threshold}")


def test_agent_from_custom_config(cases: List[TestCase]):
    """自定义配置 → SemanticEvalAgent 构造正确"""
    case = cases[1]
    agent = SemanticEvalAgent(case.eval)
    assert agent.eval_config.threshold == 0.75
    assert len(agent.eval_config.criteria) == 3
    assert agent.eval_config.criteria[0].name == "triage_accuracy"
    print(f"\n[agent_custom] criteria count={len(agent.eval_config.criteria)}")


# ============================================================
# 4. history 字段解析正确性
# ============================================================


def test_jsonl_parse_history(cases: List[TestCase]):
    """验证 history 从 JSON dict 格式正确转换为 langchain BaseMessage"""
    case = cases[2]  # auto_history_sleep_003
    assert case.id == "auto_history_sleep_003"
    assert len(case.history) == 4

    # 交替 user / assistant
    assert isinstance(case.history[0], HumanMessage)
    assert isinstance(case.history[1], AIMessage)
    assert isinstance(case.history[2], HumanMessage)
    assert isinstance(case.history[3], AIMessage)

    # 内容正确
    assert "失眠" in case.history[0].content
    assert "作息" in case.history[3].content


def test_jsonl_parse_llm_api_history(cases: List[TestCase]):
    """验证 llm_api + history 用例正确解析"""
    case = cases[3]  # auto_history_llm_004
    assert case.id == "auto_history_llm_004"
    assert case.target.type == "llm_api"
    assert case.target.model == "gpt-5.2"
    assert case.target.system_prompt is not None
    assert len(case.history) == 4
    assert isinstance(case.history[0], HumanMessage)
    assert isinstance(case.history[1], AIMessage)
    assert "胃疼" in case.history[0].content


def test_no_history_defaults_empty(cases: List[TestCase]):
    """无 history 字段的用例 → 默认空列表"""
    case = cases[0]  # manual_headache_001
    assert case.history == []
