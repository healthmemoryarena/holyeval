"""target_agent — Target agent plugins implementing AbstractTargetAgent

Importing this module triggers subclass registration (__init_subclass__), enabling target agents to be looked up by name.
Internal plugins (theta_api, etc.) are loaded on demand; missing ones are silently skipped.
"""

from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetAgent

__all__ = ["LlmApiTargetAgent"]

# -- Internal plugins (excluded in open-source release, import failures do not affect framework) --

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
