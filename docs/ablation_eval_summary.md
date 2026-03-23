# ThetaGen Ablation 评测报告

> 共 **35** 轮评测（排除 theta_smart_api / hippo_rag），覆盖 **5** 个数据集 × **7** 个 Target
>
> 每个数据集 50 条用例，含 5 种难度各 10 条 | Scored: 1734 | Failed: 16

## 1. 总体得分 (avg_score)

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.519 | 0.447 | 0.501 | 0.629 | 0.624 | **0.544** |
| anthropic/claude-sonnet-4.6 | 0.555 | 0.550 | 0.611 | 0.593 | 0.598 | **0.582** |
| gemini-3.1-flash-lite-preview | 0.526 | 0.391 | 0.493 | 0.608 | 0.593 | **0.522** |
| z-ai/glm-5 | 0.575 | 0.512 | 0.558 | 0.539 | 0.537 | **0.544** |
| minimax/minimax-m2.5 | 0.501 | 0.371 | 0.530 | 0.513 | 0.590 | **0.501** |
| theta_api/expert | 0.602 | 0.583 | 0.671 | 0.638 | 0.605 | **0.620** |
| theta_api/general | 0.640 | 0.583 | 0.628 | 0.685 | 0.640 | **0.635** |

## 2. 按难度维度得分

### Direct

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.615 | 0.788 | 0.772 | 0.609 | 0.902 | **0.737** |
| anthropic/claude-sonnet-4.6 | 0.560 | 0.690 | 0.835 | 0.668 | 0.687 | **0.688** |
| gemini-3.1-flash-lite-preview | 0.505 | 0.640 | 0.812 | 0.735 | 0.686 | **0.676** |
| z-ai/glm-5 | 0.718 | 0.740 | 0.777 | 0.551 | 0.747 | **0.707** |
| minimax/minimax-m2.5 | 0.690 | 0.740 | 0.665 | 0.532 | 0.770 | **0.679** |
| theta_api/expert | 0.497 | 0.757 | 0.809 | 0.751 | 0.672 | **0.697** |
| theta_api/general | 0.607 | 0.635 | 0.848 | 0.807 | 0.745 | **0.728** |

### Single-hop

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.597 | 0.307 | 0.385 | 0.660 | 0.785 | **0.547** |
| anthropic/claude-sonnet-4.6 | 0.619 | 0.600 | 0.540 | 0.740 | 0.795 | **0.659** |
| gemini-3.1-flash-lite-preview | 0.597 | 0.455 | 0.460 | 0.560 | 0.800 | **0.574** |
| z-ai/glm-5 | 0.723 | 0.647 | 0.585 | 0.720 | 0.805 | **0.696** |
| minimax/minimax-m2.5 | 0.404 | 0.417 | 0.455 | 0.420 | 0.730 | **0.485** |
| theta_api/expert | 0.796 | 0.563 | 0.627 | 0.609 | 0.885 | **0.696** |
| theta_api/general | 0.674 | 0.655 | 0.622 | 0.760 | 0.820 | **0.706** |

### Multi-hop

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.550 | 0.545 | 0.490 | 0.790 | 0.507 | **0.576** |
| anthropic/claude-sonnet-4.6 | 0.655 | 0.605 | 0.538 | 0.510 | 0.575 | **0.576** |
| gemini-3.1-flash-lite-preview | 0.494 | 0.335 | 0.420 | 0.690 | 0.590 | **0.506** |
| z-ai/glm-5 | 0.534 | 0.485 | 0.523 | 0.554 | 0.588 | **0.537** |
| minimax/minimax-m2.5 | 0.474 | 0.275 | 0.515 | 0.550 | 0.544 | **0.471** |
| theta_api/expert | 0.575 | 0.725 | 0.636 | 0.705 | 0.510 | **0.630** |
| theta_api/general | 0.761 | 0.825 | 0.643 | 0.710 | 0.490 | **0.686** |

### Temporal (Multi-hop)

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.534 | 0.395 | 0.525 | 0.605 | 0.450 | **0.502** |
| anthropic/claude-sonnet-4.6 | 0.535 | 0.539 | 0.580 | 0.520 | 0.535 | **0.542** |
| gemini-3.1-flash-lite-preview | 0.608 | 0.385 | 0.495 | 0.607 | 0.450 | **0.509** |
| z-ai/glm-5 | 0.687 | 0.535 | 0.565 | 0.750 | 0.372 | **0.582** |
| minimax/minimax-m2.5 | 0.672 | 0.317 | 0.595 | 0.807 | 0.579 | **0.594** |
| theta_api/expert | 0.683 | 0.465 | 0.705 | 0.693 | 0.455 | **0.600** |
| theta_api/general | 0.857 | 0.524 | 0.580 | 0.687 | 0.590 | **0.647** |

### Attribution

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.299 | 0.200 | 0.333 | 0.484 | 0.478 | **0.359** |
| anthropic/claude-sonnet-4.6 | 0.405 | 0.318 | 0.563 | 0.529 | 0.376 | **0.438** |
| gemini-3.1-flash-lite-preview | 0.425 | 0.142 | 0.278 | 0.449 | 0.438 | **0.346** |
| z-ai/glm-5 | 0.214 | 0.116 | 0.340 | 0.143 | 0.182 | **0.199** |
| minimax/minimax-m2.5 | 0.266 | 0.101 | 0.392 | 0.258 | 0.304 | **0.264** |
| theta_api/expert | 0.459 | 0.406 | 0.578 | 0.433 | 0.500 | **0.475** |
| theta_api/general | 0.302 | 0.275 | 0.449 | 0.463 | 0.557 | **0.409** |

## 3. Target 汇总统计

| Target | 数据集数 | 总用例 | Scored | Failed | 平均得分 |
|:---|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 5 | 250 | 250 | 0 | 0.544 |
| anthropic/claude-sonnet-4.6 | 5 | 250 | 249 | 1 | 0.582 |
| gemini-3.1-flash-lite-preview | 5 | 250 | 249 | 1 | 0.522 |
| z-ai/glm-5 | 5 | 250 | 246 | 4 | 0.544 |
| minimax/minimax-m2.5 | 5 | 250 | 240 | 10 | 0.501 |
| theta_api/expert | 5 | 250 | 250 | 0 | 0.620 |
| theta_api/general | 5 | 250 | 250 | 0 | 0.635 |

## 4. 数据集汇总

| 数据集 | Target 数 | 平均得分 | 最高 | 最高 Target | 最低 | 最低 Target |
|:---|:---:|:---:|:---:|:---|:---:|:---|
| Del 10% | 7 | 0.560 | 0.640 | theta_api/general | 0.501 | minimax/minimax-m2.5 |
| Del 20% | 7 | 0.491 | 0.583 | theta_api/expert | 0.371 | minimax/minimax-m2.5 |
| Del 30% | 7 | 0.570 | 0.671 | theta_api/expert | 0.493 | gemini-3.1-flash-lite-preview |
| Noise 10% | 7 | 0.601 | 0.685 | theta_api/general | 0.513 | minimax/minimax-m2.5 |
| Noise 20% | 7 | 0.598 | 0.640 | theta_api/general | 0.537 | z-ai/glm-5 |

## 5. 难度维度总览（跨数据集平均）

| Target | Direct | Single-hop | Multi-hop | Temporal | Attribution | 总平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.737 | 0.547 | 0.576 | 0.502 | 0.359 | **0.544** |
| anthropic/claude-sonnet-4.6 | 0.688 | 0.659 | 0.576 | 0.542 | 0.438 | **0.581** |
| gemini-3.1-flash-lite-preview | 0.676 | 0.574 | 0.506 | 0.509 | 0.346 | **0.522** |
| z-ai/glm-5 | 0.707 | 0.696 | 0.537 | 0.582 | 0.199 | **0.544** |
| minimax/minimax-m2.5 | 0.679 | 0.485 | 0.471 | 0.594 | 0.264 | **0.499** |
| theta_api/expert | 0.697 | 0.696 | 0.630 | 0.600 | 0.475 | **0.620** |
| theta_api/general | 0.728 | 0.706 | 0.686 | 0.647 | 0.409 | **0.635** |

## 6. 剩余 Fail 用例（共 16 条，重试后仍失败）

| 数据集 | Target | Fail 数 | Fail IDs |
|:---|:---|:---:|:---|
| Del 10% | gemini-3.1-flash-lite-preview | 1 | user303_AT_demo_Q073 |
| Del 20% | minimax/minimax-m2.5 | 1 | user328_AT_demo_Q082 |
| Del 20% | z-ai/glm-5 | 1 | user324_AT_demo_Q094 |
| Del 30% | minimax/minimax-m2.5 | 2 | user343_AT_demo_Q099, user343_AT_demo_Q100 |
| Noise 10% | z-ai/glm-5 | 1 | user368_AT_demo_Q085 |
| Noise 20% | anthropic/claude-sonnet-4.6 | 1 | user389_AT_demo_Q097 |
| Noise 20% | minimax/minimax-m2.5 | 7 | user382_AT_demo_Q009, user390_AT_demo_Q053, user382_AT_demo_Q047, user389_AT_demo_Q078, user390_AT_demo_Q076, user384_AT_demo_Q078, user381_AT_demo_Q096 |
| Noise 20% | z-ai/glm-5 | 2 | user386_AT_demo_Q047, user382_AT_demo_Q047 |
