"""target_agent — 被测目标代理插件，实现 AbstractTargetAgent

导入本模块会触发子类注册（__init_subclass__），使目标代理可通过名称查找。
内部插件（theta_api 等）按需加载，缺失时静默跳过。
"""

from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetAgent

__all__ = ["LlmApiTargetAgent"]

# —— 内部插件（发布时排除，import 失败不影响框架运行）——

try:
    from evaluator.plugin.target_agent.theta_api_target_agent import ThetaApiTargetAgent  # noqa: F401

    __all__.append("ThetaApiTargetAgent")
except ImportError:
    pass

try:
    from evaluator.plugin.target_agent.theta_smart_api_target_agent import ThetaSmartApiTargetAgent  # noqa: F401

    __all__.append("ThetaSmartApiTargetAgent")
except ImportError:
    pass

try:
    from evaluator.plugin.target_agent.theta_miroflow_target_agent import ThetaMiroflowTargetAgent  # noqa: F401

    __all__.append("ThetaMiroflowTargetAgent")
except ImportError:
    pass

try:
    from evaluator.plugin.target_agent.hippo_rag_api_target_agent import HippoRagApiTargetAgent  # noqa: F401

    __all__.append("HippoRagApiTargetAgent")
except ImportError:
    pass

try:
    from evaluator.plugin.target_agent.dyg_rag_api_target_agent import DygRagApiTargetAgent  # noqa: F401

    __all__.append("DygRagApiTargetAgent")
except ImportError:
    pass
