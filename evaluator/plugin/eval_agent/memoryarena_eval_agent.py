"""
MemoryArenaEvalAgent — 基于 LLM-as-Judge 的多子任务评估器

注册名称: "memoryarena"

工作原理:
1. 从 TestAgent / TargetAgent 的 memory_list 中提取多轮对话
2. 按轮次将每个 assistant 回复与对应的 ground_truth 配对
3. 并发调用 LLM 逐子任务判断回答是否正确
4. 计算 Progress Score = 通过子任务数 / 总子任务数
5. 结果为 scored，score = Progress Score

支持领域: bundled_shopping, progressive_search, group_travel_planner,
formal_reasoning_math, formal_reasoning_phys。

参考: https://arxiv.org/abs/2602.16313
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    SessionInfo,
    TestAgentMemory,
)
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1"

# ============================================================
# LLM Judge Prompt
# ============================================================

JUDGE_TEMPLATE = """You are an evaluator for the MemoryArena benchmark, which tests an AI agent's ability to solve interdependent multi-step tasks.

Your task: determine if the assistant's response correctly answers the given subtask.

# Subtask Question
<<question>>

# Expected Answer
<<expected_answer>>

# Assistant's Response
<<response>>

# Domain Context
This subtask belongs to the "<<domain>>" domain.

# Evaluation Guidelines
- Focus on whether the assistant's response contains the correct answer, not on formatting or verbosity.
- For product selection (bundled_shopping): check if the correct product was identified (matching ASIN or key attributes).
- For search queries (progressive_search): check if the answer matches the expected information semantically.
- For travel planning (group_travel_planner): check if the plan satisfies the key constraints mentioned in the expected answer.
- For formal reasoning (math/physics): check if the answer is logically and numerically correct.
- Be lenient on formatting differences but strict on factual correctness.

# Output Format
Return a JSON object:
```json
{
  "explanation": "Brief explanation of why the answer is correct or incorrect",
  "correct": true or false
}
```

Return ONLY the JSON object in markdown format. Do not include any other text."""


# ============================================================
# MemoryArenaEvalInfo — 配置模型
# ============================================================


class MemoryArenaEvalInfo(BaseModel):
    """MemoryArena 多子任务评估配置 — LLM-as-Judge + Progress Score

    按轮次将 assistant 回复与 ground_truth 配对，并发调用 LLM 逐子任务判断正确性，
    计算 Progress Score = 通过子任务数 / 总子任务数，结果为 scored。
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "memoryarena",
                    "domain": "bundled_shopping",
                    "ground_truths": ["Product A with ASIN B00XXX", "Total price is $45.99"],
                },
                {
                    "evaluator": "memoryarena",
                    "model": "gpt-4.1",
                    "domain": "formal_reasoning_math",
                    "ground_truths": ["42", "The answer is 3.14"],
                },
            ]
        },
    )

    evaluator: Literal["memoryarena"] = Field(default="memoryarena", description="评估器类型")
    model: Optional[str] = Field(default="gpt-4.1", description="Judge LLM 模型（默认 gpt-4.1）")
    domain: str = Field(
        description=(
            "评测领域 — bundled_shopping, progressive_search, group_travel_planner, "
            "formal_reasoning_math, formal_reasoning_phys"
        ),
    )
    ground_truths: List[str] = Field(description="各子任务的标准答案列表，按轮次与对话对齐")


class MemoryArenaEvalAgent(AbstractEvalAgent, name="memoryarena", params_model=MemoryArenaEvalInfo):
    """MemoryArena 多子任务评估器 — LLM-as-Judge + Progress Score"""

    _display_meta = {
        "icon": (
            "M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0"
            " 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987"
            " 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
        ),
        "color": "#8b5cf6",
        "features": ["MemoryArena", "多子任务", "Progress Score"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.06,  # ~6 subtasks × gpt-4.1 judging, USD/case
    }

    def __init__(self, eval_config: MemoryArenaEvalInfo, model: str | None = None, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.eval_config: MemoryArenaEvalInfo = eval_config
        self.model = model or eval_config.model or DEFAULT_MODEL
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # 框架接口
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        try:
            # 提取多轮对话中的 (question, response) 对
            qa_pairs = self._extract_qa_pairs(memory_list, self.history)
            if not qa_pairs:
                return EvalResult(result="fail", score=0.0, feedback="无对话记录，无法评估")

            ground_truths = self.eval_config.ground_truths
            domain = self.eval_config.domain
            num_subtasks = len(ground_truths)

            if not ground_truths:
                return EvalResult(result="fail", score=0.0, feedback="未配置 ground_truths，无法评估")

            logger.info(
                "[MemoryArenaEval] 开始评估 — %d 个子任务, domain=%s, model=%s",
                num_subtasks,
                domain,
                self.model,
            )

            # 对齐子任务数量（取 min 避免越界）
            n = min(len(qa_pairs), num_subtasks)
            if len(qa_pairs) < num_subtasks:
                logger.warning(
                    "[MemoryArenaEval] 对话轮次 (%d) 少于子任务数 (%d)，缺失部分计为未通过",
                    len(qa_pairs),
                    num_subtasks,
                )

            # 并发评估所有已有的子任务
            judge_results = await asyncio.gather(
                *[
                    self._judge_subtask(qa_pairs[i]["question"], qa_pairs[i]["response"], ground_truths[i], domain)
                    for i in range(n)
                ]
            )

            # 缺失的子任务默认不通过
            for _ in range(num_subtasks - n):
                judge_results.append({"correct": False, "explanation": "No response (conversation ended early)"})

            # 计算 Progress Score
            correct_count = sum(1 for r in judge_results if r.get("correct"))
            progress_score = correct_count / num_subtasks if num_subtasks > 0 else 0.0
            success = correct_count == num_subtasks

            # 构建详情
            subtask_details = []
            for i, result in enumerate(judge_results):
                subtask_details.append(
                    {
                        "subtask_index": i,
                        "question": qa_pairs[i]["question"][:200] if i < len(qa_pairs) else "(missing)",
                        "ground_truth": ground_truths[i][:200] if i < num_subtasks else "",
                        "correct": result.get("correct", False),
                        "explanation": result.get("explanation", ""),
                    }
                )

            feedback = (
                f"MemoryArena [{domain}] 评估: {correct_count}/{num_subtasks} 子任务通过, "
                f"PS={progress_score:.3f}, SR={'1.0' if success else '0.0'}"
            )

            logger.info(
                "[MemoryArenaEval] 评估完成 — PS=%.3f, SR=%s (%d/%d)",
                progress_score,
                "1.0" if success else "0.0",
                correct_count,
                num_subtasks,
            )

            return EvalResult(
                result="scored",
                score=progress_score,
                feedback=feedback,
                trace=EvalTrace(
                    eval_detail={
                        "domain": domain,
                        "subtask_results": subtask_details,
                        "progress_score": progress_score,
                        "success_rate": 1.0 if success else 0.0,
                        "correct_count": correct_count,
                        "total_subtasks": num_subtasks,
                    }
                ),
            )

        except Exception as e:
            logger.error("[MemoryArenaEval] 评估过程出错: %s", e, exc_info=True)
            return EvalResult(result="fail", score=0.0, feedback=f"评估过程出错: {e}")

    # ----------------------------------------------------------
    # 核心方法
    # ----------------------------------------------------------

    async def _judge_subtask(self, question: str, response: str, ground_truth: str, domain: str) -> dict:
        """调用 LLM 判断单个子任务回答是否正确"""
        prompt = (
            JUDGE_TEMPLATE.replace("<<question>>", question)
            .replace("<<expected_answer>>", ground_truth)
            .replace("<<response>>", response)
            .replace("<<domain>>", domain)
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a precise evaluator for multi-step agent tasks.",
                    input=prompt,
                    max_tokens=500,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json_response(result.content)
                if parsed is not None and "correct" in parsed:
                    if isinstance(parsed["correct"], bool):
                        return parsed

                logger.warning(
                    "[MemoryArenaEval] judge 输出格式异常 (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    result.content[:200],
                )
            except Exception as e:
                logger.warning(
                    "[MemoryArenaEval] judge 调用失败 (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

        logger.error("[MemoryArenaEval] judge 重试耗尽，默认 correct=False")
        return {"explanation": "Judging failed after retries", "correct": False}

    # ----------------------------------------------------------
    # 对话提取
    # ----------------------------------------------------------

    @staticmethod
    def _extract_qa_pairs(
        memory_list: List[TestAgentMemory],
        history: List[BaseMessage] | None = None,
    ) -> List[Dict[str, str]]:
        """从 memory_list 中提取 (question, response) 对

        每轮对话中 user 发送的内容为 question，assistant 回复为 response。
        history 中的消息不计入子任务评估（它们是预加载上下文）。
        """
        pairs: List[Dict[str, str]] = []

        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue

            user_text = mem.test_reaction.action.semantic_content
            target_text = mem.target_response.extract_text() if mem.target_response else ""

            if user_text and target_text:
                pairs.append({"question": user_text, "response": target_text})

        return pairs

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """从 LLM 响应中提取 JSON（兼容 markdown 包裹）"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
