"""target_agent — 被测目标代理插件，实现 AbstractTargetAgent

导入本模块会触发子类注册（__init_subclass__），使目标代理可通过名称查找:
    AbstractTargetAgent.get("theta_api")       -> ThetaApiTargetAgent
    AbstractTargetAgent.get("theta_smart_api") -> ThetaSmartApiTargetAgent
    AbstractTargetAgent.get("llm_api")         -> LlmApiTargetAgent
    AbstractTargetAgent.get("theta_miroflow")  -> ThetaMiroflowTargetAgent
    AbstractTargetAgent.get("hippo_rag_api")   -> HippoRagApiTargetAgent
    AbstractTargetAgent.get("dyg_rag_api")     -> DygRagApiTargetAgent
"""

from evaluator.plugin.target_agent.dyg_rag_api_target_agent import DygRagApiTargetAgent
from evaluator.plugin.target_agent.hippo_rag_api_target_agent import HippoRagApiTargetAgent
from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetAgent
from evaluator.plugin.target_agent.theta_api_target_agent import ThetaApiTargetAgent
from evaluator.plugin.target_agent.theta_miroflow_target_agent import ThetaMiroflowTargetAgent
from evaluator.plugin.target_agent.theta_smart_api_target_agent import ThetaSmartApiTargetAgent

__all__ = [
    "DygRagApiTargetAgent",
    "HippoRagApiTargetAgent",
    "LlmApiTargetAgent",
    "ThetaApiTargetAgent",
    "ThetaMiroflowTargetAgent",
    "ThetaSmartApiTargetAgent",
]
