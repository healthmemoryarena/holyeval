# 虚拟用户开场白测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建虚拟用户开场白 A/B 测试能力，通过批量虚拟用户统计不同开场白的"主动录入率"。

**Architecture:** 在 benchmark 框架上新增 virtual_user benchmark。generator 负责 LLM 画像生成 + 笛卡尔积组装。新增 engagement EvalAgent 做二元判定。analyzer 做分组统计。框架侧仅扩展 PersonaConfig（3 个新字段）和 AutoTestAgent prompt 映射。

**Tech Stack:** Python 3.11+, Pydantic v2, asyncio, LLM (gpt-4.1), HolyEval 框架 (orchestrator, BatchSession, basic_runner)

---

## File Structure

### 新增文件

| 文件 | 职责 |
|------|------|
| `evaluator/plugin/eval_agent/engagement_eval_agent.py` | Engagement 评估器插件（LLM-as-Judge 二元判定） |
| `generator/virtual_user/__init__.py` | 包初始化 |
| `generator/virtual_user/__main__.py` | CLI 入口分发 |
| `generator/virtual_user/profile_gen.py` | LLM 批量生成肥胖患者画像 |
| `generator/virtual_user/case_gen.py` | 画像 × 开场白笛卡尔积组装 BenchItem JSONL |
| `generator/virtual_user/openings.json` | 开场白列表（迭代入口） |
| `generator/virtual_user/analyzer.py` | 报告分组统计 + 对抗维度交叉分析 |
| `benchmark/data/virtual_user/metadata.json` | Benchmark 元数据 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `evaluator/core/schema.py:170` | PersonaConfig 从 Dict 改为结构化模型，新增 disclosure/attitude/cognition 三个字段 |
| `evaluator/plugin/test_agent/auto_test_agent.py:30-47` | `_PERSONA_PROMPTS` 增加三个新维度的行为描述映射 |

---

### Task 1: 扩展 PersonaConfig（schema.py）

**Files:**
- Modify: `evaluator/core/schema.py:170-177`

- [ ] **Step 1: 修改 PersonaConfig 字段定义**

当前 `persona` 是 `Optional[Dict[str, str]]`，改为结构化 Pydantic 模型以获得类型校验，同时保持向后兼容（dict 输入自动解析）。

在 `evaluator/core/schema.py` 的 `AutoUserInfo` 类前面（约第 100 行 `class AutoUserInfo` 之前），新增 PersonaConfig 类：

```python
class PersonaConfig(BaseModel):
    """虚拟用户行为特征配置 — 五维三档对抗画像"""

    model_config = ConfigDict(extra="forbid")

    disclosure: Literal["responsive", "reluctant", "withholding"] = "responsive"
    attitude: Literal["compliant", "impatient", "dominant"] = "compliant"
    cognition: Literal["accurate", "partial_understanding", "complete_denial"] = "accurate"
    logic: Literal["consistent", "occasional_contradiction", "fabricating"] = "consistent"
    expression: Literal["normal", "vague", "incoherent"] = "normal"
```

然后将 `AutoUserInfo.persona` 字段从：
```python
    persona: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "虚拟用户行为特征 — 键值对形式的行为维度配置。"
            "支持的维度：expression (normal/vague/incoherent), logic (consistent/occasional_contradiction/fabricating)。"
            "auto TestAgent 会据此在 system prompt 中注入对应的行为描述。"
        ),
    )
```

改为：
```python
    persona: Optional[PersonaConfig] = Field(
        None,
        description=(
            "虚拟用户行为特征 — 五维三档对抗画像配置。\n"
            "维度: disclosure (信息披露), attitude (态度), cognition (认知), logic (逻辑), expression (表达)。\n"
            "每个维度三档: 档1(baseline) → 档2(中间) → 档3(极端)。\n"
            "auto TestAgent 会据此在 system prompt 中注入对应的行为描述。"
        ),
    )
```

- [ ] **Step 2: 运行现有测试验证向后兼容**

Run: `pytest evaluator/tests/ -x -q 2>&1 | head -30`

现有用例中 persona 有两种形态：(1) None（大多数）(2) `{"expression": "vague", "logic": "fabricating"}` 之类的 dict。Pydantic v2 会把 dict 自动解析为 PersonaConfig 实例（多余字段 extra="forbid" 不影响，因为旧数据只用了 expression 和 logic）。验证测试全部通过。

- [ ] **Step 3: Commit**

```bash
git add evaluator/core/schema.py
git commit -m "feat(schema): expand PersonaConfig to 5-dimension adversarial profile

Add disclosure, attitude, cognition dimensions alongside existing
logic and expression. All new fields have baseline defaults for
backward compatibility."
```

---

### Task 2: 扩展 AutoTestAgent 的 persona prompt 映射

**Files:**
- Modify: `evaluator/plugin/test_agent/auto_test_agent.py:30-47`

- [ ] **Step 1: 扩展 `_PERSONA_PROMPTS` 映射表**

在 `auto_test_agent.py` 的 `_PERSONA_PROMPTS` dict 中，新增 disclosure、attitude、cognition 三个维度的映射（只映射非 baseline 档位，baseline 不需要注入任何行为描述）：

```python
_PERSONA_PROMPTS: dict[str, dict[str, str]] = {
    "disclosure": {
        "reluctant": (
            "你不太愿意主动透露自己的详细健康信息，除非对方追问才会说一点。"
            "如果被问到敏感话题（体重、饮食习惯），你会回避或模糊带过。"
        ),
        "withholding": (
            "你完全不愿意透露个人健康信息。无论对方怎么问，你都拒绝提供具体数据。"
            "你会用"这个不方便说"、"跟你没关系"之类的话回绝。"
        ),
    },
    "attitude": {
        "impatient": (
            "你比较急躁，没有耐心听长篇大论。如果对方啰嗦或没有快速给出有用信息，"
            "你会表现出不耐烦，催促对方"说重点"。"
        ),
        "dominant": (
            "你很强势，喜欢主导对话。你会打断对方、质疑对方的专业性、"
            "坚持自己的看法（即使不一定正确）。你认为自己对自己的身体最了解。"
        ),
    },
    "cognition": {
        "partial_understanding": (
            "你对自己的健康状况只有模糊的理解。你听说过一些健康知识但经常混淆，"
            "比如分不清BMI和体脂率，或者把别人的经验当成医学常识。"
        ),
        "complete_denial": (
            "你完全否认自己有健康问题。即使数据显示异常，你也认为"没什么大不了"、"
            ""我身体好得很"。你会拒绝接受任何负面的健康评估。"
        ),
    },
    "expression": {
        "vague": (
            '你表达能力有限，描述健康问题时用模糊口语："有一阵子了"、"就是不太舒服"、"大概这一块"。'
            "即使 AI 追问也无法给出更精确的描述。"
        ),
        "incoherent": (
            "你很难把话说清楚。经常答非所问，说到一半思路断了，"
            '用"那个""就是""怎么说呢"代替关键词。AI 很难从你的回答中提取有用信息。'
        ),
    },
    "logic": {
        "occasional_contradiction": (
            '你偶尔会前后矛盾。比如先说"大概两天了"，后来又说"可能有一周了"。如果 AI 指出矛盾你会修正。'
        ),
        "fabricating": (
            '你会编造健康信息。比如问到没有的症状你说"有"并编造细节，夸大不适感，编造用药经历。'
        ),
    },
}
```

- [ ] **Step 2: 更新 `_build_system_prompt` 中的 persona 处理逻辑**

当前代码（第 123-132 行）已经遍历 `self.user_info.persona.items()`。PersonaConfig 是 Pydantic BaseModel，需要改用 `model_dump()` 获取 dict：

将：
```python
        if hasattr(self.user_info, "persona") and self.user_info.persona:
            trait_lines = []
            for dim, value in self.user_info.persona.items():
```

改为：
```python
        if self.user_info.persona:
            trait_lines = []
            persona_dict = (
                self.user_info.persona.model_dump()
                if hasattr(self.user_info.persona, "model_dump")
                else self.user_info.persona
            )
            for dim, value in persona_dict.items():
```

这样既支持新的 PersonaConfig 对象，也兼容任何遗留 dict。

- [ ] **Step 3: 运行测试验证**

Run: `pytest evaluator/tests/ -x -q 2>&1 | head -30`
Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add evaluator/plugin/test_agent/auto_test_agent.py
git commit -m "feat(test-agent): add persona prompts for disclosure/attitude/cognition

Expand _PERSONA_PROMPTS with 3 new behavioral dimensions.
Update _build_system_prompt to handle PersonaConfig model."
```

---

### Task 3: Engagement 评估器插件

**Files:**
- Create: `evaluator/plugin/eval_agent/engagement_eval_agent.py`

- [ ] **Step 1: 创建 EngagementEvalAgent**

```python
"""
EngagementEvalAgent — 用户参与度评估器 (LLM-as-Judge)

Registered name: "engagement"

判定虚拟用户在看到产品开场白后是否产生"主动参与"行为：
- engaged (1.0):     主动披露健康信息、提出具体问题、或表达使用意愿
- not_engaged (0.0): 敷衍回复、拒绝、质疑产品、未提供有效信息、或沉默([沉默])

用于开场白 A/B 测试，通过 pass_rate 统计"主动录入率"。
"""

import json
import logging
from typing import Any, List, Literal, Optional

from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    SessionInfo,
    TargetAgentReaction,
    TestAgentMemory,
)
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1"

# ============================================================
# Judge Prompt
# ============================================================

_JUDGE_PROMPT = """你是一个用户行为分析师。以下是一款健康管理 APP 的开场对话。
产品先发了一句开场白，用户做出了回应。

# 开场白（产品发出）
{opening}

# 用户回应
{user_responses}

# 任务
请判断用户是否产生了"主动参与"行为：

- **engaged**: 用户主动描述了自己的健康状况、提出了具体问题、表达了使用意愿、或提供了个人健康相关信息
- **not_engaged**: 用户敷衍回复（如"嗯"、"好的"）、表示拒绝（如"不需要"、"不用了"）、质疑产品（如"你是谁"、"靠谱吗"）、未提供任何有效健康信息、或选择沉默（如"[沉默]"）

请严格按以下 JSON 格式输出，不要输出其他内容：
```json
{{"result": "engaged" | "not_engaged", "reason": "一句话说明判断依据"}}
```"""


# ============================================================
# Config model
# ============================================================


class EngagementEvalInfo(BaseModel):
    """Engagement 评估配置 — 用户参与度二元判定"""

    model_config = ConfigDict(extra="forbid")

    evaluator: Literal["engagement"] = Field(description="评估器类型")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="通过阈值（0.5 → engaged=pass, not_engaged=fail）",
    )
    model: Optional[str] = Field(None, description="Judge LLM 模型（默认 gpt-4.1）")


# ============================================================
# EvalAgent
# ============================================================


class EngagementEvalAgent(AbstractEvalAgent, name="engagement", params_model=EngagementEvalInfo):
    """用户参与度评估器 — LLM-as-Judge 二元判定"""

    _display_meta = {
        "icon": (
            "M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933"
            " 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
        ),
        "color": "#10b981",
        "features": ["LLM-as-Judge", "Binary", "Engagement"],
    }
    _cost_meta = {"est_cost_per_case": 0.005}

    def __init__(self, eval_config: EngagementEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.threshold = eval_config.threshold
        self.model = eval_config.model or DEFAULT_MODEL

    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """执行参与度评估"""
        try:
            # 1. 提取开场白（来自 history）
            opening = ""
            if self.history:
                # history 中最后一条 AI 消息即为开场白
                for msg in reversed(self.history):
                    if hasattr(msg, "type") and msg.type == "ai":
                        opening = msg.content
                        break
                    if hasattr(msg, "content") and not opening:
                        opening = msg.content

            if not opening:
                return EvalResult(
                    result="error",
                    score=0.0,
                    feedback="无法提取开场白（history 为空）",
                )

            # 2. 提取用户回应
            user_responses = []
            for mem in memory_list:
                if mem.test_reaction and mem.test_reaction.action:
                    content = mem.test_reaction.action.semantic_content
                    if content:
                        user_responses.append(content)

            if not user_responses:
                # 无回应 = not_engaged
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback="用户无任何回应 → not_engaged",
                )

            user_text = "\n".join(f"- {r}" for r in user_responses)

            # 3. 调用 Judge LLM
            prompt = _JUDGE_PROMPT.format(opening=opening, user_responses=user_text)

            result = await do_execute(
                model=self.model,
                system_prompt="You are a user behavior analyst. Output valid JSON only.",
                input=prompt,
                max_tokens=200,
            )

            # 4. 解析结果
            content = result.content.strip()
            # 去掉 markdown 包裹
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            is_engaged = parsed.get("result") == "engaged"
            reason = parsed.get("reason", "")

            score = 1.0 if is_engaged else 0.0
            passed = score >= self.threshold

            return EvalResult(
                result="pass" if passed else "fail",
                score=score,
                feedback=f"{'engaged' if is_engaged else 'not_engaged'}: {reason}",
                trace=EvalTrace(
                    eval_memory=[],
                    raw={"judge_output": parsed, "opening": opening, "user_responses": user_responses},
                ),
            )

        except json.JSONDecodeError as e:
            logger.error("[EngagementEval] JSON 解析失败: %s, raw: %s", e, content if "content" in dir() else "N/A")
            return EvalResult(
                result="error",
                score=0.0,
                feedback=f"Judge 输出解析失败: {e}",
            )
        except Exception as e:
            logger.error("[EngagementEval] 评估异常: %s", e, exc_info=True)
            return EvalResult(
                result="error",
                score=0.0,
                feedback=f"评估异常: {e}",
            )
```

- [ ] **Step 2: 验证插件自动注册**

Run: `python -c "from evaluator.plugin.eval_agent import *; from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; print('engagement' in AbstractEvalAgent._registry)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add evaluator/plugin/eval_agent/engagement_eval_agent.py
git commit -m "feat(eval): add engagement eval agent for opening A/B testing

LLM-as-Judge binary classification: engaged (1.0) vs not_engaged (0.0).
Extracts opening from history, user responses from memory_list."
```

---

### Task 4: 画像生成器 (profile_gen.py)

**Files:**
- Create: `generator/virtual_user/__init__.py`
- Create: `generator/virtual_user/profile_gen.py`

- [ ] **Step 1: 创建包初始化文件**

`generator/virtual_user/__init__.py`:
```python
"""virtual_user — 虚拟用户开场白测试数据生成器"""
```

- [ ] **Step 2: 创建 profile_gen.py**

```python
"""
虚拟用户画像生成器 — 用 LLM 批量生成肥胖患者画像

输出 profiles.jsonl，每行一个 JSON 对象:
{
    "profile_id": "obesity_001",
    "age": 35,
    "gender": "female",
    "bmi": 32.5,
    "occupation": "小学教师",
    "comorbidities": ["高血压前期"],
    "motivation": "体检报告异常，医生建议减重",
    "background": "尝试过节食和跑步..."
}

用法:
    python -m generator.virtual_user.profile_gen --count 20 --output generator/virtual_user/profiles.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are a medical data specialist creating realistic virtual patient profiles for product testing. Output valid JSON only."

_GENERATION_PROMPT = """请生成 {count} 个**肥胖相关**的虚拟患者画像，用于测试一款慢病管理 APP 的开场白效果。

要求：
1. 覆盖多样性：年龄（20~65）、性别、BMI 范围（28~45）、职业、合并症、减肥动机
2. 画像要真实可信，包含具体的生活细节
3. 每个画像的 background 应包含 2~3 句话，描述该患者的减肥经历或健康困扰

输出 JSON 数组，每个对象包含：
- "profile_id": 字符串，格式 "obesity_XXX"（三位数字，从 {start_index} 开始）
- "age": 整数
- "gender": "male" 或 "female"
- "bmi": 浮点数（保留一位小数）
- "occupation": 字符串（中文）
- "comorbidities": 字符串列表（如 ["高血压", "2型糖尿病"]，可为空列表）
- "motivation": 字符串（中文，一句话描述减肥动机）
- "background": 字符串（中文，2~3 句话描述该患者的具体情况）

Return ONLY the JSON array, no other text."""


async def _generate_batch(count: int, start_index: int) -> list[dict]:
    """调用 LLM 生成一批画像"""
    from evaluator.utils.llm import do_execute

    prompt = _GENERATION_PROMPT.format(count=count, start_index=start_index)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await do_execute(
                model="gpt-4.1",
                system_prompt=_SYSTEM_PROMPT,
                input=prompt,
                max_tokens=4000,
            )
            content = result.content.strip()

            # 去掉 markdown 包裹
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            items = json.loads(content)
            if isinstance(items, list):
                logger.info("[ProfileGen] 生成 %d 个画像", len(items))
                return items

            logger.warning("[ProfileGen] 格式异常 (attempt %d/%d)", attempt + 1, max_retries)
        except json.JSONDecodeError as e:
            logger.warning("[ProfileGen] JSON 解析失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)
        except Exception as e:
            logger.warning("[ProfileGen] 生成失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)

    logger.error("[ProfileGen] 生成失败，重试耗尽")
    return []


async def generate(output_path: str | Path, count: int = 20) -> int:
    """批量生成虚拟患者画像

    Args:
        output_path: 输出 JSONL 路径
        count: 生成数量

    Returns:
        成功生成的画像数
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 分批生成（每批最多 15 个，避免超出 token 限制）
    batch_size = 15
    tasks = []
    start = 1
    remaining = count
    while remaining > 0:
        this_batch = min(remaining, batch_size)
        tasks.append(_generate_batch(this_batch, start))
        start += this_batch
        remaining -= this_batch

    logger.info("[ProfileGen] 启动 %d 个生成任务...", len(tasks))
    results = await asyncio.gather(*tasks)

    total = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for items in results:
            for item in items:
                if isinstance(item, dict):
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total += 1

    logger.info("[ProfileGen] 生成完成: %d 个画像 → %s", total, output_path)
    return total


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="用 LLM 批量生成肥胖患者虚拟画像")
    parser.add_argument("--count", type=int, default=20, help="生成数量（默认 20）")
    parser.add_argument(
        "--output",
        default="generator/virtual_user/profiles.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    total = asyncio.run(generate(args.output, args.count))
    print(f"生成完成: {total} 个画像 → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证 CLI 帮助**

Run: `python -m generator.virtual_user.profile_gen --help`
Expected: 显示帮助信息，无 import 错误

- [ ] **Step 4: Commit**

```bash
git add generator/virtual_user/__init__.py generator/virtual_user/profile_gen.py
git commit -m "feat(generator): add virtual user profile generator

LLM-powered batch generation of obesity patient profiles.
Outputs profiles.jsonl with age, BMI, comorbidities, motivation."
```

---

### Task 5: 开场白列表 + 用例组装器

**Files:**
- Create: `generator/virtual_user/openings.json`
- Create: `generator/virtual_user/case_gen.py`

- [ ] **Step 1: 创建示例开场白列表**

`generator/virtual_user/openings.json`:
```json
[
  {
    "opening_id": "01",
    "content": "你好！我是你的健康管理助手，很高兴认识你。我可以帮你记录和分析健康数据，发现身体变化的规律。有什么我能帮到你的吗？"
  },
  {
    "opening_id": "02",
    "content": "看起来你最近在关注体重管理方面的问题。很多和你年龄相仿的用户都有类似的困扰，我可以根据你的具体情况给出个性化的建议。方便告诉我你目前的情况吗？"
  },
  {
    "opening_id": "03",
    "content": "欢迎来到 Theta Health！我注意到你对健康管理有兴趣。作为你的私人健康助手，我能帮你追踪体重变化趋势、分析饮食和运动对身体的影响。想先从哪里开始聊聊？"
  }
]
```

- [ ] **Step 2: 创建 case_gen.py**

```python
"""
虚拟用户用例组装器 — 画像 × 开场白笛卡尔积 → BenchItem JSONL

读取 profiles.jsonl + openings.json，对每个画像随机分配五维对抗档位，
然后与开场白做笛卡尔积，输出 benchmark JSONL。

用法:
    python -m generator.virtual_user.case_gen \
        --profiles generator/virtual_user/profiles.jsonl \
        --openings generator/virtual_user/openings.json \
        --output benchmark/data/virtual_user/round1.jsonl \
        --seed 42
"""

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# 五维三档定义及分配权重
_DIMENSIONS: dict[str, list[tuple[str, float]]] = {
    "disclosure": [("responsive", 0.5), ("reluctant", 0.3), ("withholding", 0.2)],
    "attitude": [("compliant", 0.5), ("impatient", 0.3), ("dominant", 0.2)],
    "cognition": [("accurate", 0.5), ("partial_understanding", 0.3), ("complete_denial", 0.2)],
    "logic": [("consistent", 0.5), ("occasional_contradiction", 0.3), ("fabricating", 0.2)],
    "expression": [("normal", 0.5), ("vague", 0.3), ("incoherent", 0.2)],
}

_GOAL = (
    "根据自身情况决定是否愿意与这个健康助手交流。"
    '如果不想交流，你可以选择沉默（回复"[沉默]"）、敷衍、或直接拒绝。'
    "如果愿意就主动描述自己的健康困扰。"
)


def _assign_persona(rng: random.Random) -> dict[str, str]:
    """按权重随机分配五维对抗档位"""
    persona = {}
    for dim, levels in _DIMENSIONS.items():
        values, weights = zip(*levels)
        persona[dim] = rng.choices(values, weights=weights, k=1)[0]
    return persona


def _build_context(profile: dict) -> str:
    """从画像构建虚拟用户 context 描述"""
    parts = []
    age = profile.get("age", "")
    gender_map = {"male": "男性", "female": "女性"}
    gender = gender_map.get(profile.get("gender", ""), "")
    occupation = profile.get("occupation", "")
    bmi = profile.get("bmi", "")

    if age and gender and occupation:
        parts.append(f"你是一位{age}岁的{gender}，职业是{occupation}，BMI {bmi}。")

    comorbidities = profile.get("comorbidities", [])
    if comorbidities:
        parts.append(f"合并症：{'、'.join(comorbidities)}。")

    motivation = profile.get("motivation", "")
    if motivation:
        parts.append(f"减肥动机：{motivation}。")

    background = profile.get("background", "")
    if background:
        parts.append(background)

    return "".join(parts)


def generate_cases(
    profiles_path: str | Path,
    openings_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> int:
    """生成用例 JSONL

    Args:
        profiles_path: 画像 JSONL 路径
        openings_path: 开场白 JSON 路径
        output_path: 输出 BenchItem JSONL 路径
        seed: 随机种子

    Returns:
        生成的用例数
    """
    profiles_path = Path(profiles_path)
    openings_path = Path(openings_path)
    output_path = Path(output_path)

    # 读取画像
    profiles = []
    with open(profiles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))
    logger.info("读取 %d 个画像", len(profiles))

    # 读取开场白
    with open(openings_path, "r", encoding="utf-8") as f:
        openings = json.load(f)
    logger.info("读取 %d 句开场白", len(openings))

    # 为每个画像分配对抗维度（固定种子）
    rng = random.Random(seed)
    profile_personas: list[dict[str, str]] = []
    for _ in profiles:
        profile_personas.append(_assign_persona(rng))

    # 笛卡尔积
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for opening in openings:
            oid = opening["opening_id"]
            opening_content = opening["content"]

            for i, profile in enumerate(profiles):
                pid = profile.get("profile_id", f"p{i:03d}")
                persona = profile_personas[i]
                context = _build_context(profile)

                # 构建 tags
                tags = [f"opening:{oid}", f"profile:{pid}"]
                for dim, val in persona.items():
                    tags.append(f"{dim}:{val}")

                # 构建 title（简短）
                age = profile.get("age", "?")
                gender_short = {"male": "男", "female": "女"}.get(profile.get("gender", ""), "")
                bmi = profile.get("bmi", "?")
                persona_summary = "/".join(v for d, v in persona.items() if v != list(dict(_DIMENSIONS[d]).keys())[0])
                title = f"开场白{oid} × {age}岁{gender_short}BMI{bmi}"
                if persona_summary:
                    title += f" [{persona_summary}]"

                bench_item = {
                    "id": f"vu_o{oid}_p{pid.replace('obesity_', '')}",
                    "title": title,
                    "user": {
                        "type": "auto",
                        "goal": _GOAL,
                        "context": context,
                        "persona": persona,
                        "max_turns": 3,
                        "finish_condition": "对话进行了 1~3 轮后自然结束，或用户明确选择沉默/离开。",
                    },
                    "history": [{"role": "assistant", "content": opening_content}],
                    "eval": {"evaluator": "engagement", "threshold": 0.5},
                    "tags": tags,
                }

                fout.write(json.dumps(bench_item, ensure_ascii=False) + "\n")
                total += 1

    logger.info("生成完成: %d 条用例 → %s", total, output_path)
    return total


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="画像 × 开场白笛卡尔积 → BenchItem JSONL")
    parser.add_argument(
        "--profiles",
        default="generator/virtual_user/profiles.jsonl",
        help="画像 JSONL 路径",
    )
    parser.add_argument(
        "--openings",
        default="generator/virtual_user/openings.json",
        help="开场白 JSON 路径",
    )
    parser.add_argument(
        "--output",
        default="benchmark/data/virtual_user/round1.jsonl",
        help="输出 BenchItem JSONL 路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    total = generate_cases(args.profiles, args.openings, args.output, args.seed)
    print(f"生成完成: {total} 条用例 → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证 CLI 帮助**

Run: `python -m generator.virtual_user.case_gen --help`
Expected: 显示帮助信息，无 import 错误

- [ ] **Step 4: Commit**

```bash
git add generator/virtual_user/openings.json generator/virtual_user/case_gen.py
git commit -m "feat(generator): add case assembler for opening A/B test

Cartesian product of profiles × openings with weighted random
persona assignment. Outputs BenchItem JSONL for basic_runner."
```

---

### Task 6: Benchmark 元数据

**Files:**
- Create: `benchmark/data/virtual_user/metadata.json`

- [ ] **Step 1: 创建 metadata.json**

```json
{
  "description": "# 虚拟用户开场白测试\n\n用 LLM 生成的虚拟肥胖患者测试不同开场白的用户参与率。\n\n## 评估指标\n\n- **主动录入率** (pass_rate): engaged / total\n- **按开场白分组**: 通过 tags `opening:XX` 聚合\n- **按对抗维度交叉分析**: 通过 tags `disclosure:XX` 等聚合\n\n## 使用流程\n\n```bash\n# 1. 生成画像\npython -m generator.virtual_user.profile_gen --count 20\n\n# 2. 生成用例\npython -m generator.virtual_user.case_gen\n\n# 3. 跑测试\npython -m benchmark.basic_runner virtual_user round1 --target-type theta_smart_api -p 5\n\n# 4. 分析报告\npython -m generator.virtual_user.analyzer benchmark/report/virtual_user/round1_xxx.json\n```",
  "target": [
    {
      "type": "theta_smart_api",
      "fields": {
        "email": {"default": "user1@demo", "editable": true},
        "agent": {"default": "expert", "editable": true}
      }
    },
    {
      "type": "llm_api",
      "fields": {
        "model": {"default": "gpt-4.1", "editable": true, "required": true}
      }
    }
  ]
}
```

- [ ] **Step 2: 验证 benchmark 能被框架发现**

Run: `python -m evaluator list -b 2>&1 | grep virtual_user`
Expected: 输出包含 `virtual_user`

- [ ] **Step 3: Commit**

```bash
git add benchmark/data/virtual_user/metadata.json
git commit -m "feat(benchmark): add virtual_user benchmark metadata

Supports theta_smart_api and llm_api targets."
```

---

### Task 7: 对比分析器 (analyzer.py)

**Files:**
- Create: `generator/virtual_user/analyzer.py`

- [ ] **Step 1: 创建 analyzer.py**

```python
"""
虚拟用户开场白对比分析器

读取 BenchReport JSON，按 opening_id 分组统计录入率（pass_rate），
并按对抗维度做交叉分析。

用法:
    python -m generator.virtual_user.analyzer benchmark/report/virtual_user/round1_xxx.json
    python -m generator.virtual_user.analyzer report.json --opening 03
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# 对抗维度列表（用于交叉分析）
_ADVERSARIAL_DIMS = ["disclosure", "attitude", "cognition", "logic", "expression"]


def _extract_tag(tags: list[str], prefix: str) -> str | None:
    """从 tags 列表中提取指定前缀的值"""
    for tag in tags:
        if tag.startswith(f"{prefix}:"):
            return tag.split(":", 1)[1]
    return None


def _compute_rate(cases: list[dict]) -> tuple[int, int, float]:
    """计算 pass 数、总数、pass_rate"""
    total = len(cases)
    if total == 0:
        return 0, 0, 0.0
    passed = sum(1 for c in cases if c.get("eval", {}).get("result") == "pass")
    return passed, total, passed / total


def analyze(report_path: str | Path, focus_opening: str | None = None) -> dict:
    """分析报告

    Args:
        report_path: BenchReport JSON 路径
        focus_opening: 仅分析指定 opening_id（None = 全部）

    Returns:
        分析结果 dict
    """
    report_path = Path(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("cases", [])
    if not cases:
        logger.warning("报告中无用例结果")
        return {}

    # 按 opening 分组
    by_opening: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        tags = case.get("tags", [])
        oid = _extract_tag(tags, "opening")
        if oid:
            by_opening[oid].append(case)

    # 总体统计
    result = {"total_cases": len(cases), "openings": {}}

    for oid in sorted(by_opening.keys()):
        if focus_opening and oid != focus_opening:
            continue

        opening_cases = by_opening[oid]
        passed, total, rate = _compute_rate(opening_cases)

        opening_result = {
            "pass_count": passed,
            "total": total,
            "pass_rate": round(rate, 4),
            "cross_analysis": {},
        }

        # 交叉分析：每个对抗维度
        for dim in _ADVERSARIAL_DIMS:
            by_dim: dict[str, list[dict]] = defaultdict(list)
            for case in opening_cases:
                tags = case.get("tags", [])
                val = _extract_tag(tags, dim)
                if val:
                    by_dim[val].append(case)

            dim_stats = {}
            for val in sorted(by_dim.keys()):
                p, t, r = _compute_rate(by_dim[val])
                dim_stats[val] = {"pass_count": p, "total": t, "pass_rate": round(r, 4)}
            opening_result["cross_analysis"][dim] = dim_stats

        result["openings"][oid] = opening_result

    return result


def print_report(result: dict) -> None:
    """将分析结果打印为终端表格"""
    print()
    print("=" * 60)
    print(f"  开场白 A/B 对比报告 — 虚拟用户（N={result.get('total_cases', 0)}）")
    print("=" * 60)
    print()

    openings = result.get("openings", {})
    if not openings:
        print("  (无数据)")
        return

    # 总览表
    print(f"{'开场白':<15} {'录入率':>8} {'样本数':>10}")
    print("-" * 40)
    for oid, data in openings.items():
        rate_pct = f"{data['pass_rate'] * 100:.0f}%"
        sample = f"{data['pass_count']}/{data['total']}"
        print(f"opening_{oid:<8} {rate_pct:>8} {sample:>10}")
    print()

    # 交叉分析（逐 opening）
    for oid, data in openings.items():
        cross = data.get("cross_analysis", {})
        if not cross:
            continue
        print(f"按对抗维度交叉分析（opening_{oid}）:")
        for dim, dim_stats in cross.items():
            for val, stats in dim_stats.items():
                rate_pct = f"{stats['pass_rate'] * 100:.0f}%"
                sample = f"({stats['pass_count']}/{stats['total']})"
                print(f"  {dim}={val:<30} {rate_pct:>6}  {sample}")
            print()


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="虚拟用户开场白对比分析器")
    parser.add_argument("report", help="BenchReport JSON 路径")
    parser.add_argument("--opening", default=None, help="仅分析指定 opening_id")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = analyze(args.report, focus_opening=args.opening)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 CLI 帮助**

Run: `python -m generator.virtual_user.analyzer --help`
Expected: 显示帮助信息

- [ ] **Step 3: Commit**

```bash
git add generator/virtual_user/analyzer.py
git commit -m "feat(generator): add opening A/B test report analyzer

Groups results by opening_id, computes engagement rate,
cross-analyzes by adversarial persona dimensions."
```

---

### Task 8: CLI 入口 + 端到端验证

**Files:**
- Create: `generator/virtual_user/__main__.py`

- [ ] **Step 1: 创建 __main__.py**

```python
"""
generator.virtual_user CLI 入口

子命令:
    profile_gen  — 生成画像
    case_gen     — 组装用例
    analyzer     — 分析报告

用法:
    python -m generator.virtual_user profile_gen --count 20
    python -m generator.virtual_user case_gen --seed 42
    python -m generator.virtual_user analyzer report.json
"""

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m generator.virtual_user <subcommand> [args...]")
        print("子命令: profile_gen, case_gen, analyzer")
        sys.exit(1)

    subcommand = sys.argv[1]
    # 移除子命令名，让子模块的 argparse 正常工作
    sys.argv = [f"generator.virtual_user.{subcommand}"] + sys.argv[2:]

    if subcommand == "profile_gen":
        from generator.virtual_user.profile_gen import main as sub_main
    elif subcommand == "case_gen":
        from generator.virtual_user.case_gen import main as sub_main
    elif subcommand == "analyzer":
        from generator.virtual_user.analyzer import main as sub_main
    else:
        print(f"未知子命令: {subcommand}")
        print("可用: profile_gen, case_gen, analyzer")
        sys.exit(1)

    sub_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 端到端冒烟测试（无需真实 LLM 调用）**

用 case_gen 验证笛卡尔积逻辑（手动创建一个小的 profiles.jsonl）：

```bash
# 创建测试画像（2 个）
echo '{"profile_id":"obesity_001","age":35,"gender":"female","bmi":32.5,"occupation":"教师","comorbidities":["高血压"],"motivation":"体检异常","background":"尝试过节食"}' > /tmp/test_profiles.jsonl
echo '{"profile_id":"obesity_002","age":45,"gender":"male","bmi":30.0,"occupation":"工程师","comorbidities":[],"motivation":"医嘱","background":"久坐不动"}' >> /tmp/test_profiles.jsonl

# 运行 case_gen（2 画像 × 3 开场白 = 6 条）
python -m generator.virtual_user.case_gen \
  --profiles /tmp/test_profiles.jsonl \
  --openings generator/virtual_user/openings.json \
  --output /tmp/test_cases.jsonl \
  --seed 42

# 验证行数
wc -l /tmp/test_cases.jsonl
# Expected: 6

# 验证 JSON 格式
python -c "
import json
with open('/tmp/test_cases.jsonl') as f:
    for line in f:
        item = json.loads(line)
        assert 'id' in item
        assert 'history' in item and len(item['history']) == 1
        assert item['history'][0]['role'] == 'assistant'
        assert item['eval']['evaluator'] == 'engagement'
        assert any(t.startswith('opening:') for t in item['tags'])
        print(f\"  {item['id']}: {item['title'][:50]}\")
print('All 6 cases validated!')
"
```

Expected: 6 条用例，每条包含 history（开场白）、engagement eval、opening/profile tags。

- [ ] **Step 3: Commit**

```bash
git add generator/virtual_user/__main__.py
git commit -m "feat(generator): add virtual_user CLI entry point

Dispatches to profile_gen, case_gen, analyzer subcommands."
```

---

### Task 9: 最终验证 — ruff + 测试

**Files:** (无新文件)

- [ ] **Step 1: Ruff lint + format**

```bash
ruff check evaluator/core/schema.py evaluator/plugin/test_agent/auto_test_agent.py evaluator/plugin/eval_agent/engagement_eval_agent.py generator/virtual_user/
ruff format evaluator/core/schema.py evaluator/plugin/test_agent/auto_test_agent.py evaluator/plugin/eval_agent/engagement_eval_agent.py generator/virtual_user/
```

Fix any issues that come up.

- [ ] **Step 2: 运行全量测试**

```bash
pytest evaluator/tests/ -x -q
```

Expected: 全部通过，无回归。

- [ ] **Step 3: 修复任何问题并提交**

如果有 lint/test 问题，修复后：
```bash
git add -u
git commit -m "fix: lint and test fixes for virtual user feature"
```
