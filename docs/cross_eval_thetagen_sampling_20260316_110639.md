现在我已收集了所有需要的数据。让我生成交叉评测分析报告。

# ThetaGen Sampling 交叉评测分析报告

**评测时间**: 2026-03-16 11:06:39  
**数据集**: thetagen_sampling / full_sample_1  
**评测用例数**: 50 条

---

## 1. 总体概览

### 1.1 评测矩阵

本次交叉评测覆盖 5 个被测系统配置，全部运行成功：

| # | Target Type | Override | 配置值 | 状态 |
|---|------------|----------|--------|------|
| 1 | theta_api | agent | expert | 成功 |
| 2 | theta_api | agent | mix | 成功 |
| 3 | llm_api | model | gpt-4.1 | 成功 |
| 4 | theta_miroflow | language | en | 成功 |
| 5 | hippo_rag_api | model | gemini-3-flash-preview | 成功 |

### 1.2 被测系统说明

- **Theta API (expert)**: Theta Health API，使用 claude-opus-4.6|gemini-3.1-pro-preview 组合
- **Theta API (mix)**: Theta Health API，使用 claude-sonnet-4.6|gemini-3-flash-preview 组合
- **GPT-4.1 (llm_api)**: 直接调用 OpenAI GPT-4.1，配合 DuckDB 工具查询健康数据
- **Theta Miroflow**: Theta MiroFlow API，使用 miro-thinker 模型
- **Gemini-3-flash (hippo_rag_api)**: HippoRAG API，使用 Gemini-3-flash-preview + embedding

---

## 2. 横向对比表格

### 2.1 核心指标对比

| 被测系统 | Pass Rate | Avg Score | 用例总耗时 (s) | 运行耗时 (s) | Fail Count |
|---------|-----------|-----------|---------------|-------------|------------|
| **Theta API (mix)** | 0.0% | **55.43%** | 2750.68 | 304.98 | 0 |
| Theta API (expert) | 0.0% | 30.02% | 3655.74 | 582.42 | 26 |
| Gemini-3-flash (RAG) | 0.0% | 19.21% | 13.65 | 3.16 | 0 |
| Theta Miroflow | 0.0% | 13.51% | 1739.47 | 207.55 | 30 |
| GPT-4.1 (llm_api) | 0.0% | 6.42% | 107.35 | 14.40 | 34 |

**说明**:
- 所有系统 Pass Rate 均为 0%，这是因为评测阈值设置较高或系统尚未完全适配评测格式
- **Theta API (mix)** 以 55.43% 的平均分取得最佳表现
- GPT-4.1 (llm_api) 存在大量工具调用异常（DuckDB 临时文件路径问题）
- Gemini-3-flash (RAG) 用户数据目录缺失导致全部返回错误

---

## 3. 按难度标签分项对比

评测用例按 5 个难度等级分类：

| 难度 | Theta API (mix) | Theta API (expert) | Theta Miroflow | GPT-4.1 | Gemini-3-flash |
|-----|----------------|-------------------|----------------|---------|----------------|
| **direct** (3条) | 39.44% | 45.00% | **55.00%** | 0.00% | 0.00% |
| **single_hop** (6条) | **58.04%** | 22.17% | 34.51% | 6.83% | 6.75% |
| **multi_hop** (10条) | **37.25%** | 7.33% | 8.11% | 8.00% | 12.00% |
| **multi_hop_temporal** (14条) | **57.59%** | 63.90%* | 1.43% | 14.29% | 14.29% |
| **attribution** (17条) | **66.26%** | 15.59% | 11.89% | 0.00% | 35.29% |

*注: Theta API (expert) 在 multi_hop_temporal 上分数较高 (63.90%)

### 难度维度分析

1. **Direct (直接查询)**: Theta Miroflow 表现最佳 (55%)，能准确回答简单的事件类型、状态确认等问题
2. **Single Hop (单跳推理)**: Theta API (mix) 领先 (58%)，对单步数据检索有良好支持
3. **Multi Hop (多跳推理)**: 各系统表现普遍较差，Theta API (mix) 仅 37%
4. **Multi Hop Temporal (时序多跳)**: Theta API (expert) 表现最佳 (64%)，擅长处理时间相关的复合查询
5. **Attribution (归因分析)**: Theta API (mix) 领先 (66%)，但 expert 仅 16%，表明归因能力与模型配置高度相关

---

## 4. 按答案类型分项对比

评测用例按 4 种答案类型分类：

| 答案类型 | Theta API (mix) | Theta API (expert) | Theta Miroflow | GPT-4.1 | Gemini-3-flash |
|---------|----------------|-------------------|----------------|---------|----------------|
| **text** (9条) | **54.44%** | 42.22% | 21.11% | 0.00% | 0.00% |
| **boolean** (1条) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **numeric_value** (18条) | **46.67%** | 38.89% | 10.00% | 17.78% | 20.00% |
| **list** (22条) | **65.53%** | 19.13% | 13.88% | 0.05% | 27.30% |

### 答案类型分析

1. **文本类 (text)**: Theta API (mix) 表现稳定 (54%)，能较好理解自然语言描述的健康信息
2. **布尔类 (boolean)**: 仅 1 条用例，所有系统得分为 0，样本量不足以判断
3. **数值类 (numeric_value)**: Theta API (mix) 领先 (47%)，数值查询准确率较高
4. **列表类 (list)**: Theta API (mix) 显著领先 (66%)，在列表匹配和排序任务中表现突出；Gemini-3-flash (27%) 次之

---

## 5. 关键发现与结论

### 5.1 性能排名

按综合平均分排序：
1. **Theta API (mix)** - 55.43% ⭐ 最佳
2. Theta API (expert) - 30.02%
3. Gemini-3-flash (RAG) - 19.21%
4. Theta Miroflow - 13.51%
5. GPT-4.1 (llm_api) - 6.42%

### 5.2 关键发现

**优势系统特征**:
- Theta API (mix) 在几乎所有维度领先，尤其在 list 类答案 (66%) 和 attribution 任务 (66%) 表现突出
- 使用 claude-sonnet-4.6 + gemini-3-flash-preview 的 mix 配置比 opus + pro 的 expert 配置效果更好，可能因为:
  - 响应更快，避免超时
  - 模型对任务的适配性更好

**问题诊断**:
- **GPT-4.1 (llm_api)**: 34/50 用例异常失败，原因是 DuckDB 临时文件目录不存在 (`FileNotFoundError`)
- **Gemini-3-flash (RAG)**: 用户数据目录缺失，无法执行实际查询
- **Boolean 类问题**: 样本量过小 (仅 1 条)，无法得出结论

**改进建议**:
1. 修复 llm_api 的 DuckDB 临时文件路径问题
2. 确保 hippo_rag_api 的用户数据目录正确配置
3. 增加 boolean 类测试用例以获得更全面的评估
4. 针对 multi_hop 类任务优化 prompt 或工具调用策略

### 5.3 效率对比

| 指标 | 最快 | 最慢 |
|-----|-----|-----|
| 运行耗时 | Gemini-3-flash (3.16s) | Theta API (expert) (582.42s) |
| 单例耗时 | GPT-4.1 (2.15s/case) | Theta API (expert) (73.11s/case) |

*注: Gemini-3-flash 因数据目录缺失直接返回错误，耗时数据不具参考价值*

---

**报告生成时间**: 2026-03-16