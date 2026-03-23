"""
PresetAnswerEvalAgent 单元测试

测试覆盖：
1. 数字匹配 — 精确命中
2. 数字匹配 — 容差范围内命中
3. 数字匹配 — 超出容差未命中
4. 数字匹配 — 多数字部分命中
5. 关键词匹配 — 命中
6. 关键词匹配 — 未命中
7. 精确匹配 — 命中
8. 精确匹配 — 未命中
9. 空回复 → fail
10. 标准答案无数字时 fallback 到关键词
11. 注册表查找
12. 通过 run() 接口的集成测试

运行：uv run pytest evaluator/tests/test_preset_answer_eval.py -v -s
"""

import pytest

from evaluator.plugin.eval_agent.preset_answer_eval_agent import PresetAnswerEvalInfo
from evaluator.plugin.eval_agent.preset_answer_eval_agent import (
    PresetAnswerEvalAgent,
    _numbers_close,
    _numbers_close_with_precision,
    _decimal_places,
)


# ============================================================
# 工具函数测试
# ============================================================


class TestNumbersClose:
    """_numbers_close 工具函数测试"""

    def test_exact_match(self):
        assert _numbers_close(59.06, 59.06, 0.01) is True

    def test_within_tolerance(self):
        # 59.06 * 1% = 0.5906 → 59.06 ± 0.59
        assert _numbers_close(59.06, 59.5, 0.01) is True

    def test_beyond_tolerance(self):
        assert _numbers_close(59.06, 61.0, 0.01) is False

    def test_both_zero(self):
        assert _numbers_close(0.0, 0.0, 0.01) is True

    def test_one_zero(self):
        assert _numbers_close(0.0, 0.001, 0.01) is False
        assert _numbers_close(0.0, 0.0000001, 0.01) is True

    def test_negative_numbers(self):
        assert _numbers_close(-100.0, -100.5, 0.01) is True
        assert _numbers_close(-100.0, -110.0, 0.01) is False

    def test_large_numbers(self):
        assert _numbers_close(8102.23, 8102.0, 0.01) is True
        assert _numbers_close(8102.23, 9000.0, 0.01) is False


# ============================================================
# 静态匹配方法测试
# ============================================================


class TestMatchNumber:
    """数字匹配测试"""

    def test_exact_number_match(self):
        ok, detail = PresetAnswerEvalAgent._match_number(
            "您的心率变异性是 59.06", "59.06"
        )
        assert ok is True
        assert "1/1 命中" in detail

    def test_number_within_tolerance(self):
        ok, detail = PresetAnswerEvalAgent._match_number(
            "您的心率变异性约为 59.1", "59.06"
        )
        assert ok is True

    def test_number_beyond_tolerance(self):
        ok, detail = PresetAnswerEvalAgent._match_number(
            "您的心率变异性是 70.0", "59.06"
        )
        assert ok is False

    def test_multiple_numbers_partial_match(self):
        """标准答案有 2 个数字，生成答案命中 1 个 → 50% ≥ 50% → pass"""
        ok, detail = PresetAnswerEvalAgent._match_number(
            "步数 8102，心率 0", "8102.23 和 59.06",
            threshold=0.5,
        )
        assert ok is True
        assert "1/2 命中" in detail

    def test_multiple_numbers_all_miss(self):
        ok, detail = PresetAnswerEvalAgent._match_number(
            "数据不可用", "8102.23 和 59.06",
        )
        assert ok is False
        assert "0/2 命中" in detail

    def test_comma_formatted_numbers(self):
        """支持千分位格式 1,234"""
        ok, _ = PresetAnswerEvalAgent._match_number(
            "总共 1,234 步", "1,234"
        )
        assert ok is True

    def test_no_numbers_in_standard_fallback_keyword(self):
        """标准答案无数字 → fallback 到关键词匹配"""
        ok, detail = PresetAnswerEvalAgent._match_number(
            "建议您尽快就医检查", "建议 就医"
        )
        assert ok is True
        assert "关键词" in detail


class TestMatchKeyword:
    """关键词匹配测试"""

    def test_keyword_hit(self):
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "建议您尽快到医院就医", "建议就医"
        )
        assert ok is True

    def test_keyword_miss(self):
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "天气不错", "建议就医",
            threshold=0.5,
        )
        assert ok is False

    def test_short_standard_substring(self):
        """标准答案拆词后无长度≥2的词 → 子串匹配"""
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "是的", "是",
        )
        assert "子串" in detail


class TestMatchExact:
    """精确匹配测试"""

    def test_exact_hit(self):
        ok, _ = PresetAnswerEvalAgent._match_exact(
            "您的心率变异性是 59.06 ms", "59.06"
        )
        assert ok is True

    def test_exact_miss(self):
        ok, _ = PresetAnswerEvalAgent._match_exact(
            "数据暂时不可用", "59.06"
        )
        assert ok is False


# ============================================================
# run() 集成测试（mock memory）
# ============================================================


def _make_memory_with_reply(reply_text: str) -> list:
    """构造包含 target 回复的 memory_list"""
    from evaluator.core.schema import (
        TargetAgentReaction,
        TestAgentAction,
        TestAgentMemory,
        TestAgentReaction,
    )
    from datetime import datetime

    reaction = TargetAgentReaction(
        type="message",
        message_list=[{"content": reply_text}],
    )
    test_reaction = TestAgentReaction(
        action=TestAgentAction(type="semantic", semantic_content="test"),
        is_finished=False,
    )
    return [TestAgentMemory(
        test_reaction=test_reaction,
        test_reaction_time=datetime.now(),
        target_response=reaction,
        target_response_time=datetime.now(),
    )]


@pytest.mark.asyncio
async def test_run_number_correct():
    """run() — 数字匹配正确 → pass"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="59.06",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply("您的心率变异性是 59.06 ms"),
    )
    assert result.result == "pass"
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_run_number_wrong():
    """run() — 数字匹配错误 → fail"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="59.06",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply("抱歉，我无法查询到该数据"),
    )
    assert result.result == "fail"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_run_empty_reply():
    """run() — 空回复 → fail"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="59.06",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply(""),
    )
    assert result.result == "fail"
    assert "为空" in result.feedback


@pytest.mark.asyncio
async def test_run_keyword_mode():
    """run() — 关键词模式"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="建议就医",
        match_mode="keyword",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply("根据您的情况，建议您尽快到医院就医检查"),
    )
    assert result.result == "pass"


@pytest.mark.asyncio
async def test_run_exact_mode():
    """run() — 精确模式"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="59.06",
        match_mode="exact",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply("您的心率变异性是 59.06"),
    )
    assert result.result == "pass"


# ============================================================
# 注册表测试
# ============================================================


def test_registry_lookup():
    """应能通过 AbstractEvalAgent.get('preset_answer') 获取"""
    from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
    import evaluator.plugin.eval_agent  # noqa: F401

    cls = AbstractEvalAgent.get("preset_answer")
    assert cls is PresetAnswerEvalAgent


def test_registry_has():
    """注册表应包含 'preset_answer'"""
    from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
    import evaluator.plugin.eval_agent  # noqa: F401

    assert AbstractEvalAgent.has("preset_answer")


# ============================================================
# schema 测试
# ============================================================


def test_preset_answer_eval_info_parse():
    """PresetAnswerEvalInfo 应能正确解析"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="59.06",
    )
    assert config.match_mode == "number"
    assert config.number_tolerance == 0.01
    assert config.threshold == 0.5


def test_eval_info_discriminated_union():
    """EvalInfo Discriminated Union 应路由到 PresetAnswerEvalInfo"""
    from pydantic import TypeAdapter
    from evaluator.core.schema import EvalInfo

    adapter = TypeAdapter(EvalInfo)
    config = adapter.validate_python({
        "evaluator": "preset_answer",
        "standard_answer": "42",
    })
    assert isinstance(config, PresetAnswerEvalInfo)
    assert config.standard_answer == "42"


# ============================================================
# CamelCase 归一化测试
# ============================================================


class TestKeywordCamelCase:
    """关键词匹配 — CamelCase 归一化"""

    def test_camelcase_indicator_name(self):
        """DailyStepCount 应匹配 Daily Step Count"""
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "The affected indicator is Daily Step Count.",
            "DailyStepCount",
        )
        assert ok is True

    def test_camelcase_with_abbreviation(self):
        """BodyFatPercentage-BFP 应匹配 Body Fat Percentage BFP"""
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "Body Fat Percentage (BFP) is one of the indicators.",
            "BodyFatPercentage-BFP",
        )
        assert ok is True

    def test_camelcase_semicolon_list(self):
        """分号分隔的 CamelCase 列表应匹配英文全称列表"""
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "The indicators are: Daily Step Count, Sleep Efficiency, Stress Score.",
            "DailyStepCount; SleepEfficiency; StressScore",
        )
        assert ok is True

    def test_camelcase_no_false_positive(self):
        """不相关内容不应匹配"""
        ok, detail = PresetAnswerEvalAgent._match_keyword(
            "The weather is nice today.",
            "DailyStepCount; SleepEfficiency",
            threshold=0.5,
        )
        assert ok is False


# ============================================================
# 502 系统错误检测测试
# ============================================================


@pytest.mark.asyncio
async def test_run_502_error():
    """run() — 502 Bad Gateway → error（非 fail）"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="true",
        match_mode="keyword",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply(
            'API 调用失败: list_message returned 502: <html>\r\n'
            '<head><title>502 Bad Gateway</title></head>\r\n'
            '<body>\r\n<center><h1>502 Bad Gateway</h1></center>\r\n'
            '</body>\r\n</html>\r\n'
        ),
    )
    assert result.result == "error"
    assert "system_error" in result.feedback
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_run_api_failure():
    """run() — API 调用失败 → error"""
    config = PresetAnswerEvalInfo(
        evaluator="preset_answer",
        standard_answer="42",
    )
    agent = PresetAnswerEvalAgent(config)

    result = await agent.run(
        _make_memory_with_reply("API 调用失败: connection timeout"),
    )
    assert result.result == "error"
    assert "system_error" in result.feedback


# ============================================================
# 精度对齐测试
# ============================================================


class TestDecimalPlaces:
    """_decimal_places 工具函数测试"""

    def test_integer(self):
        assert _decimal_places("25") == 0

    def test_two_decimals(self):
        assert _decimal_places("463.74") == 2

    def test_four_decimals(self):
        assert _decimal_places("463.7415") == 4

    def test_comma_format(self):
        assert _decimal_places("1,234.56") == 2


class TestPrecisionAlignment:
    """精度对齐数字比对测试"""

    def test_rounding_4dp_vs_2dp(self):
        """463.7415 vs 463.74 — 精度对齐后应通过"""
        assert _numbers_close_with_precision(
            463.7415, 463.74, "463.7415", "463.74", 0.01
        ) is True

    def test_negative_rounding(self):
        """-8.0519 vs -8.05 — 精度对齐后应通过"""
        assert _numbers_close_with_precision(
            -8.0519, -8.05, "-8.0519", "-8.05", 0.01
        ) is True

    def test_small_value_rounding(self):
        """0.3761 vs 0.38 — 精度对齐后应通过"""
        assert _numbers_close_with_precision(
            0.3761, 0.38, "0.3761", "0.38", 0.01
        ) is True

    def test_genuinely_wrong_still_fails(self):
        """25 vs 26 — 整数不涉及精度对齐，真错就是错"""
        assert _numbers_close_with_precision(
            25.0, 26.0, "25", "26", 0.01
        ) is False

    def test_large_diff_still_fails(self):
        """463.7415 vs 500 — 差距过大，精度对齐也救不了"""
        assert _numbers_close_with_precision(
            463.7415, 500.0, "463.7415", "500", 0.01
        ) is False

    def test_same_precision_original_tolerance(self):
        """同精度数字走原始容差路径"""
        assert _numbers_close_with_precision(
            59.06, 59.5, "59.06", "59.50", 0.01
        ) is True

    def test_match_number_integration(self):
        """通过 _match_number 端到端测试精度对齐"""
        ok, detail = PresetAnswerEvalAgent._match_number(
            "The averages are 463.74, 507.64, and 481.35 minutes.",
            "463.7415; 507.6379; 481.3482",
            tolerance=0.01,
        )
        assert ok is True
        assert "3/3 命中" in detail
