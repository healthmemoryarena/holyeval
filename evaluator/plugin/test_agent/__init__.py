"""test_agent — Test agent (virtual user) plugins implementing AbstractTestAgent

Importing this module triggers subclass registration (__init_subclass__), enabling test agents to be looked up by name:
    AbstractTestAgent.get("auto")    -> AutoTestAgent   (LLM-driven)
    AbstractTestAgent.get("manual")  -> ManualTestAgent  (script-driven)
"""

from evaluator.plugin.test_agent.auto_test_agent import AutoTestAgent
from evaluator.plugin.test_agent.manual_test_agent import ManualTestAgent

__all__ = ["AutoTestAgent", "ManualTestAgent"]
