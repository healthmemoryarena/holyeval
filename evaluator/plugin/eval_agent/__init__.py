"""eval_agent — 评估代理插件包

导入本模块会触发子类注册（__init_subclass__），使评估器可通过名称查找:
    AbstractEvalAgent.get("semantic")          -> SemanticEvalAgent
    AbstractEvalAgent.get("indicator")         -> IndicatorEvalAgent
    AbstractEvalAgent.get("keyword")           -> KeywordEvalAgent
    AbstractEvalAgent.get("preset_answer")     -> PresetAnswerEvalAgent
    AbstractEvalAgent.get("healthbench")       -> HealthBenchEvalAgent
    AbstractEvalAgent.get("medcalc")           -> MedCalcEvalAgent
    AbstractEvalAgent.get("hallucination")     -> HallucinationEvalAgent
    AbstractEvalAgent.get("indicator_recall")  -> IndicatorRecallEvalAgent
    AbstractEvalAgent.get("redteam_compliance") -> RedteamComplianceEvalAgent
AbstractEvalAgent.get("memoryarena")       -> MemoryArenaEvalAgent
    AbstractEvalAgent.get("kg_qa")             -> KgQaEvalAgent
"""

from evaluator.plugin.eval_agent.hallucination_eval_agent import HallucinationEvalAgent
from evaluator.plugin.eval_agent.healthbench_eval_agent import HealthBenchEvalAgent
from evaluator.plugin.eval_agent.indicator_eval_agent import IndicatorEvalAgent
from evaluator.plugin.eval_agent.indicator_recall_eval_agent import IndicatorRecallEvalAgent
from evaluator.plugin.eval_agent.kg_qa_eval_agent import KgQaEvalAgent
from evaluator.plugin.eval_agent.keyword_eval_agent import KeywordEvalAgent
from evaluator.plugin.eval_agent.medcalc_eval_agent import MedCalcEvalAgent
from evaluator.plugin.eval_agent.memoryarena_eval_agent import MemoryArenaEvalAgent
from evaluator.plugin.eval_agent.preset_answer_eval_agent import PresetAnswerEvalAgent
from evaluator.plugin.eval_agent.redteam_compliance_eval_agent import RedteamComplianceEvalAgent
from evaluator.plugin.eval_agent.semantic_eval_agent import SemanticEvalAgent

__all__ = [
    "SemanticEvalAgent",
    "IndicatorEvalAgent",
    "KeywordEvalAgent",
    "PresetAnswerEvalAgent",
    "HealthBenchEvalAgent",
    "MedCalcEvalAgent",
    "HallucinationEvalAgent",
    "IndicatorRecallEvalAgent",
    "RedteamComplianceEvalAgent",
    "MemoryArenaEvalAgent",
    "KgQaEvalAgent",
]
