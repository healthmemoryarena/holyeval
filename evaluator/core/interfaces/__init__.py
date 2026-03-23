"""
interfaces — 核心抽象接口

定义三类 Agent 的抽象基类，供 evaluator.plugin 实现：
- AbstractTestAgent:    测试代理 / 虚拟用户（do_generate 生成用户动作）
- AbstractTargetAgent:  被测目标代理（do_generate 转发动作并接收响应）
- AbstractEvalAgent:    评估代理（run 执行评估并返回结果）
"""
