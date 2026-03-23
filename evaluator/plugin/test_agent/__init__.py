"""test_agent — 测试代理（虚拟用户）插件，实现 AbstractTestAgent

导入本模块会触发子类注册（__init_subclass__），使测试代理可通过名称查找:
    AbstractTestAgent.get("auto")    -> AutoTestAgent   (LLM 驱动)
    AbstractTestAgent.get("manual")  -> ManualTestAgent  (脚本驱动)
"""

from evaluator.plugin.test_agent.auto_test_agent import AutoTestAgent
from evaluator.plugin.test_agent.manual_test_agent import ManualTestAgent

__all__ = ["AutoTestAgent", "ManualTestAgent"]
