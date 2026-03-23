# 开发评测逻辑（EvalAgent）

> **在 Claude Code 中使用**: 输入 `/add-eval-agent <agent-name>`，后跟你的评测逻辑描述（如评分规则、输入输出要求），Claude 会自动生成配置、实现和注册代码。

## 概述

EvalAgent 是评测框架的核心组件，负责在对话结束后评估被测系统的表现。框架通过插件机制支持多种评估策略，你可以根据业务需求开发自定义评测逻辑。

## 现有评估器

| 名称 | 类型 | 说明 |
|------|------|------|
| `semantic` | LLM | 语义理解评估，通过 LLM 判断回答质量 |
| `healthbench` | LLM | HealthBench rubric 评分，基于多维度打分标准 |
| `medcalc` | LLM + 规则 | MedCalc-Bench 医疗计算评估（LLM 提取答案 + 类型化数值匹配） |
| `hallucination` | LLM | 医疗幻觉检测（LLM-as-Judge），检测事实/上下文/引用三类幻觉，评分 0~1 |
| `indicator` | LLM + API | 健康数据指标评估 |
| `keyword` | 规则 | 关键词匹配评估 |
| `preset_answer` | 规则 | 预设答案匹配（exact/keyword 模式） |
| `redteam_compliance` | LLM | 红队合规评估（LLM-as-Judge 医疗合规性评分） |
| `memoryarena` | LLM | MemoryArena 多子任务评估（LLM-as-Judge + Progress Score） |

## 开发步骤

### 1. 设计评估逻辑

确定你的评估器需要：
- **LLM-based** 还是 **确定性规则**？LLM 适合语义判断，规则适合精确匹配
- 需要哪些 **配置参数**？（如阈值、关键词列表、模型名称等）
- 评分方式是 **二元判定**（pass/fail）还是 **连续评分**（0~1）？

### 2. 定义配置模型

在 `evaluator/core/schema.py` 中新增 Pydantic 配置：

```python
class MyEvalInfo(BaseModel):
    """我的自定义评估配置"""
    model_config = ConfigDict(extra="forbid")

    evaluator: Literal["my_eval"] = "my_eval"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # 自定义字段...
```

然后将其加入 `EvalInfo` 联合类型。

### 3. 实现评估器

在 `evaluator/plugin/eval_agent/` 创建实现文件：

```python
class MyEvalAgent(AbstractEvalAgent, name="my_eval"):
    async def run(self, memory_list, session_info=None) -> EvalResult:
        # memory_list: List[TestAgentMemory] — 完整对话（用户行为 + 目标系统响应）
        # session_info: SessionInfo | None — 目标系统会话信息（认证数据等）
        # self.history / self.user_info / self.case_id — 静态上下文（init 时注入）
        # 返回 EvalResult(result="pass"|"fail", score=0.8, feedback="...")
```

### 4. 注册插件

在 `evaluator/plugin/eval_agent/__init__.py` 中导入新类。

### 5. 验证

```bash
# 检查注册
python -c "from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; print(AbstractEvalAgent.get_all())"

# 代码检查
ruff check evaluator/core/schema.py evaluator/plugin/eval_agent/
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `evaluator/core/schema.py` | 配置模型定义 |
| `evaluator/core/interfaces/abstract_eval_agent.py` | 抽象接口 |
| `evaluator/plugin/eval_agent/` | 插件实现目录 |
| `evaluator/plugin/eval_agent/semantic_eval_agent.py` | LLM 评估参考实现 |
| `evaluator/plugin/eval_agent/medcalc_eval_agent.py` | LLM + 规则混合参考实现 |
| `evaluator/plugin/eval_agent/hallucination_eval_agent.py` | LLM-as-Judge 幻觉检测参考实现 |
| `evaluator/plugin/eval_agent/preset_answer_eval_agent.py` | 规则评估参考实现 |
| `evaluator/plugin/eval_agent/redteam_compliance_eval_agent.py` | LLM-as-Judge 合规评估参考实现 |
| `evaluator/plugin/eval_agent/memoryarena_eval_agent.py` | LLM-as-Judge 多子任务评估参考实现 |
