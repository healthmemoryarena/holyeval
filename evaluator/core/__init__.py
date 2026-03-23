"""
core — 框架核心

- schema.py:        所有核心数据结构（TestCase / TestResult / 交互协议 / 记忆）
- orchestrator.py:  编排器 do_single_test — 框架唯一入口
- interfaces/:      抽象基类（BaseUserAgent / BaseTargetAgent / BaseEvalAgent）

依赖：evaluator.utils
被依赖：evaluator.plugin, benchmark, generator
"""
