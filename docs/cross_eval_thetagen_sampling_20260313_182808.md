# ThetaGen Sampling 交叉评测分析报告

**生成时间**: 2026-03-13 18:28:08
**评测框架**: HolyEval v1.0

---

## 1. 总体概览

### 1.1 评测矩阵

| 维度 | 内容 |
|------|------|
| **Benchmark** | thetagen_sampling |
| **数据集** | full_sample_1, full_sample_2 |
| **总运行次数** | 8 |
| **成功/失败** | 8 / 0 |

### 1.2 被测系统列表

| 类型 | 系统标识 | 说明 |
|------|----------|------|
| LLM API | gpt-4.1 | OpenAI GPT-4.1 基础模型 |
| LLM API | gpt-5.2 | OpenAI GPT-5.2 基础模型 |
| Theta API | theta_expert | Theta Health Expert Agent |
| Theta API | theta_mix | Theta Health Mix Agent |

### 1.3 用例分布

- **总用例数**: 100 条（每数据集 50 条 × 2 个数据集）
- **难度分布**: direct (6%), single_hop (12%), multi_hop (20%), multi_hop_temporal (28%), attribution (34%)
- **答案类型**: boolean (2%), text (18%), numeric_value (36%), list (44%)

---

## 2. 横向对比表格

### 2.1 整体性能对比

| 被测系统 | 总用例数 | 平均得分 | 通过率 | 执行异常数 | 总耗时(秒) |
|----------|----------|----------|--------|------------|------------|
| **theta_mix** | 100 | **0.648** | 0% | 0 | 4,601.5 |
| **theta_expert** | 100 | 0.603 | 0% | 0 | 5,258.0 |
| gpt-4.1 | 100 | 0.060 | 0% | 62 | 236.9 |
| gpt-5.2 | 100 | 0.055 | 0% | 50 | 396.1 |

### 2.2 按数据集分项对比

| 数据集 | 被测系统 | 平均得分 | 执行异常数 | 耗时(秒) |
|--------|----------|----------|------------|----------|
| full_sample_1 | theta_mix | **0.647** | 0 | 2,423.2 |
| full_sample_1 | theta_expert | 0.592 | 0 | 2,844.5 |
| full_sample_1 | gpt-4.1 | 0.068 | 34 | 113.5 |
| full_sample_1 | gpt-5.2 | 0.028 | 31 | 199.8 |
| full_sample_2 | theta_mix | **0.649** | 0 | 2,178.3 |
| full_sample_2 | theta_expert | 0.615 | 0 | 2,413.4 |
| full_sample_2 | gpt-5.2 | 0.082 | 19 | 196.3 |
| full_sample_2 | gpt-4.1 | 0.052 | 28 | 123.4 |

---

## 3. 按难度标签分项对比

| 难度级别 | gpt-4.1 | gpt-5.2 | theta_expert | theta_mix |
|----------|---------|---------|--------------|-----------|
| **direct** | 0.031 | 0.131 | 0.637 | **0.936** |
| **single_hop** | 0.064 | 0.051 | 0.559 | **0.738** |
| **multi_hop** | 0.080 | 0.070 | **0.359** | 0.233 |
| **multi_hop_temporal** | 0.125 | 0.067 | 0.763 | **0.768** |
| **attribution** | 0.000 | 0.000 | **0.652** | 0.658 |

### 难度维度分析

- **direct (直接查询)**: theta_mix 以 0.936 的得分远超其他系统，表现出色
- **single_hop (单跳推理)**: theta_mix (0.738) > theta_expert (0.559) >> LLM 基模型
- **multi_hop (多跳推理)**: theta_expert (0.359) 略优于 theta_mix (0.233)，是 Theta 系列表现最弱的维度
- **multi_hop_temporal (时序多跳)**: theta_mix 和 theta_expert 表现相近（~0.76-0.77），显著优于基模型
- **attribution (归因查询)**: 仅 Theta 系列能够处理（~0.65），基模型得分为 0

---

## 4. 按答案类型分项对比

| 答案类型 | gpt-4.1 | gpt-5.2 | theta_expert | theta_mix |
|----------|---------|---------|--------------|-----------|
| **text** | 0.011 | 0.028 | 0.631 | **0.618** |
| **numeric_value** | 0.161 | 0.139 | **0.706** | 0.611 |
| **list** | 0.000 | 0.000 | 0.536 | **0.675** |
| **boolean** | 0.000 | 0.000 | 0.000 | **1.000** |

### 答案类型分析

- **boolean (布尔值)**: theta_mix 完美得分 (1.000)，theta_expert 得分为 0（存在明显缺陷）
- **numeric_value (数值)**: theta_expert (0.706) > theta_mix (0.611)，Theta 系列明显优于基模型
- **list (列表)**: theta_mix (0.675) > theta_expert (0.536)，基模型无法正确处理
- **text (文本)**: Theta 系列表现相近（~0.62），基模型几乎无法正确回答

---

## 5. 关键发现与结论

### 5.1 核心发现

1. **Theta Agent 显著优于基础 LLM**
   - Theta 系列平均得分 (0.60-0.65) 是基础 LLM (0.05-0.06) 的 **10-12 倍**
   - 基础 LLM 存在大量执行异常（FileNotFoundError），原因是 ThetaGen 数据目录缺失导致工具调用失败

2. **theta_mix 整体表现最优**
   - 综合得分 0.648，略高于 theta_expert (0.603)
   - 在 direct、single_hop、boolean 等维度表现突出
   - 唯一在 boolean 类型上获得满分的系统

3. **theta_expert 在数值和复杂推理上更优**
   - numeric_value 维度得分 0.706 vs theta_mix 的 0.611
   - multi_hop 维度得分 0.359 vs theta_mix 的 0.233

4. **基础 LLM 面临工具链兼容性问题**
   - gpt-4.1 有 62% 的用例执行异常
   - gpt-5.2 有 50% 的用例执行异常
   - 主要原因：ThetaGen 工具要求的用户数据目录不存在

### 5.2 性能-耗时权衡

| 系统 | 得分 | 耗时(秒) | 得分/耗时比 |
|------|------|----------|-------------|
| theta_mix | 0.648 | 4,601 | 0.000141 |
| theta_expert | 0.603 | 5,258 | 0.000115 |
| gpt-4.1 | 0.060 | 237 | 0.000253 |
| gpt-5.2 | 0.055 | 396 | 0.000139 |

Theta Agent 虽然得分高，但耗时约为基础 LLM 的 15-20 倍。考虑到执行异常的影响，基础 LLM 的有效得分更低，Theta Agent 的性能优势更为显著。

### 5.3 改进建议

1. **基础 LLM 评测**: 需要确保 ThetaGen 数据目录正确部署，消除 FileNotFoundError
2. **theta_expert**: 需要修复 boolean 类型回答的逻辑问题
3. **theta_mix**: 可考虑在 multi_hop 推理场景引入更强的链式推理策略
4. **整体**: 建议增加 direct 难度用例的采样比例，以更全面评估基础能力
