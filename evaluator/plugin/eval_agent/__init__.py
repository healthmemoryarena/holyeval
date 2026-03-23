"""eval_agent — 评估代理插件包

导入本模块会触发子类注册（__init_subclass__），使评估器可通过名称查找。
内部插件（indicator 等）按需加载，缺失时静默跳过。
"""

from evaluator.plugin.eval_agent.hallucination_eval_agent import HallucinationEvalAgent
from evaluator.plugin.eval_agent.healthbench_eval_agent import HealthBenchEvalAgent
from evaluator.plugin.eval_agent.keyword_eval_agent import KeywordEvalAgent
from evaluator.plugin.eval_agent.medcalc_eval_agent import MedCalcEvalAgent
from evaluator.plugin.eval_agent.memoryarena_eval_agent import MemoryArenaEvalAgent
from evaluator.plugin.eval_agent.preset_answer_eval_agent import PresetAnswerEvalAgent
from evaluator.plugin.eval_agent.redteam_compliance_eval_agent import RedteamComplianceEvalAgent
from evaluator.plugin.eval_agent.semantic_eval_agent import SemanticEvalAgent

__all__ = [
    "SemanticEvalAgent",
    "KeywordEvalAgent",
    "PresetAnswerEvalAgent",
    "HealthBenchEvalAgent",
    "MedCalcEvalAgent",
    "HallucinationEvalAgent",
    "RedteamComplianceEvalAgent",
    "MemoryArenaEvalAgent",
]

# —— 内部插件（发布时排除，import 失败不影响框架运行）——

try:
    from evaluator.plugin.eval_agent.indicator_eval_agent import IndicatorEvalAgent  # noqa: F401

    __all__.append("IndicatorEvalAgent")
except ImportError:
    pass

try:
    from evaluator.plugin.eval_agent.indicator_recall_eval_agent import IndicatorRecallEvalAgent  # noqa: F401

    __all__.append("IndicatorRecallEvalAgent")
except ImportError:
    pass

try:
    from evaluator.plugin.eval_agent.kg_qa_eval_agent import KgQaEvalAgent  # noqa: F401

    __all__.append("KgQaEvalAgent")
except ImportError:
    pass
