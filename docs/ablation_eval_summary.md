# ThetaGen Ablation 评测报告

> 共 **50** 轮评测，覆盖 **5** 个数据集 × **10** 个 Target
>
> 每个数据集 50 条用例，含 5 种难度各 10 条 | Scored: 2431 | Failed: 69

> ⚠️ **gemini-3-flash-preview 数据可信度警告**：该模型 250 条用例中有 61 条 fail（24.4%），仅 189 条有效评分。
> fail 集中在 **Attribution** 难度（50 条仅 12 条 scored，缺失率 76%），该维度得分（0.278）极不可信。
> 其余难度（Direct/Single-hop/Multi-hop/Temporal）缺失率约 10-20%，得分仅供参考，不宜与其他 target 直接比较。

## 1. 总体得分 (avg_score)

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.519 | 0.447 | 0.501 | 0.629 | 0.624 | **0.544** |
| anthropic/claude-sonnet-4.6 | 0.555 | 0.550 | 0.611 | 0.593 | 0.598 | **0.581** |
| gemini-3.1-flash-lite-preview | 0.526 | 0.391 | 0.493 | 0.608 | 0.593 | **0.522** |
| gemini-3-flash-preview ⚠️ | 0.750 | 0.713 | 0.761 | 0.767 | 0.774 | **0.753** |
| z-ai/glm-5 | 0.575 | 0.512 | 0.558 | 0.533 | 0.539 | **0.544** |
| minimax/minimax-m2.5 | 0.501 | 0.371 | 0.530 | 0.513 | 0.580 | **0.499** |
| theta_api/expert | 0.602 | 0.583 | 0.671 | 0.638 | 0.605 | **0.620** |
| theta_api/general | 0.640 | 0.583 | 0.628 | 0.685 | 0.640 | **0.635** |
| theta_smart_api/expert | 0.557 | 0.582 | 0.621 | 0.589 | 0.610 | **0.592** |
| hippo_rag_api/gpt-5.4 | 0.363 | 0.362 | 0.433 | 0.426 | 0.381 | **0.393** |

## 2. 按难度维度得分

### Direct

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.615 | 0.788 | 0.772 | 0.609 | 0.902 | **0.737** |
| anthropic/claude-sonnet-4.6 | 0.560 | 0.690 | 0.835 | 0.668 | 0.687 | **0.688** |
| gemini-3.1-flash-lite-preview | 0.505 | 0.640 | 0.812 | 0.735 | 0.686 | **0.676** |
| gemini-3-flash-preview | 0.711 | 0.880 | 0.861 | 0.797 | 1.000 | **0.850** |
| z-ai/glm-5 | 0.718 | 0.740 | 0.777 | 0.551 | 0.747 | **0.707** |
| minimax/minimax-m2.5 | 0.690 | 0.740 | 0.665 | 0.532 | 0.784 | **0.682** |
| theta_api/expert | 0.497 | 0.757 | 0.809 | 0.751 | 0.672 | **0.697** |
| theta_api/general | 0.607 | 0.635 | 0.848 | 0.807 | 0.745 | **0.728** |
| theta_smart_api/expert | 0.526 | 0.870 | 0.952 | 0.808 | 0.985 | **0.828** |
| hippo_rag_api/gpt-5.4 | 0.307 | 0.640 | 0.561 | 0.510 | 0.599 | **0.523** |

### Single-hop

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.597 | 0.307 | 0.385 | 0.660 | 0.785 | **0.547** |
| anthropic/claude-sonnet-4.6 | 0.619 | 0.600 | 0.540 | 0.740 | 0.795 | **0.659** |
| gemini-3.1-flash-lite-preview | 0.597 | 0.455 | 0.460 | 0.560 | 0.800 | **0.574** |
| gemini-3-flash-preview | 0.864 | 0.656 | 0.857 | 0.840 | 0.900 | **0.823** |
| z-ai/glm-5 | 0.723 | 0.647 | 0.585 | 0.720 | 0.805 | **0.696** |
| minimax/minimax-m2.5 | 0.404 | 0.417 | 0.455 | 0.420 | 0.730 | **0.485** |
| theta_api/expert | 0.796 | 0.563 | 0.627 | 0.609 | 0.885 | **0.696** |
| theta_api/general | 0.674 | 0.655 | 0.622 | 0.760 | 0.820 | **0.706** |
| theta_smart_api/expert | 0.773 | 0.616 | 0.581 | 0.740 | 0.885 | **0.719** |
| hippo_rag_api/gpt-5.4 | 0.418 | 0.361 | 0.480 | 0.385 | 0.295 | **0.388** |

### Multi-hop

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.550 | 0.545 | 0.490 | 0.790 | 0.507 | **0.576** |
| anthropic/claude-sonnet-4.6 | 0.655 | 0.605 | 0.538 | 0.510 | 0.575 | **0.576** |
| gemini-3.1-flash-lite-preview | 0.494 | 0.335 | 0.420 | 0.690 | 0.590 | **0.506** |
| gemini-3-flash-preview | 0.694 | 0.760 | 0.667 | 0.678 | 0.620 | **0.684** |
| z-ai/glm-5 | 0.534 | 0.485 | 0.523 | 0.554 | 0.590 | **0.537** |
| minimax/minimax-m2.5 | 0.474 | 0.275 | 0.515 | 0.550 | 0.494 | **0.462** |
| theta_api/expert | 0.575 | 0.725 | 0.636 | 0.705 | 0.510 | **0.630** |
| theta_api/general | 0.761 | 0.825 | 0.643 | 0.710 | 0.490 | **0.686** |
| theta_smart_api/expert | 0.530 | 0.495 | 0.418 | 0.497 | 0.370 | **0.462** |
| hippo_rag_api/gpt-5.4 | 0.144 | 0.155 | 0.140 | 0.325 | 0.206 | **0.194** |

### Temporal (Multi-hop)

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.534 | 0.395 | 0.525 | 0.605 | 0.450 | **0.502** |
| anthropic/claude-sonnet-4.6 | 0.535 | 0.539 | 0.580 | 0.520 | 0.535 | **0.542** |
| gemini-3.1-flash-lite-preview | 0.608 | 0.385 | 0.495 | 0.607 | 0.450 | **0.509** |
| gemini-3-flash-preview | 0.880 | 0.656 | 0.814 | 0.822 | 0.631 | **0.761** |
| z-ai/glm-5 | 0.687 | 0.535 | 0.565 | 0.695 | 0.372 | **0.571** |
| minimax/minimax-m2.5 | 0.672 | 0.317 | 0.595 | 0.807 | 0.613 | **0.601** |
| theta_api/expert | 0.683 | 0.465 | 0.705 | 0.693 | 0.455 | **0.600** |
| theta_api/general | 0.857 | 0.524 | 0.580 | 0.687 | 0.590 | **0.647** |
| theta_smart_api/expert | 0.512 | 0.428 | 0.600 | 0.445 | 0.440 | **0.485** |
| hippo_rag_api/gpt-5.4 | 0.593 | 0.429 | 0.620 | 0.480 | 0.315 | **0.487** |

### Attribution

| Target | Del 10% | Del 20% | Del 30% | Noise 10% | Noise 20% | **平均** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.299 | 0.200 | 0.333 | 0.484 | 0.478 | **0.359** |
| anthropic/claude-sonnet-4.6 | 0.405 | 0.318 | 0.563 | 0.529 | 0.396 | **0.442** |
| gemini-3.1-flash-lite-preview | 0.425 | 0.142 | 0.278 | 0.449 | 0.438 | **0.346** |
| gemini-3-flash-preview | 0.100 | 0.300 | 0.463 | 0.100 | 0.425 | **0.278** |
| z-ai/glm-5 | 0.214 | 0.154 | 0.340 | 0.143 | 0.182 | **0.207** |
| minimax/minimax-m2.5 | 0.266 | 0.101 | 0.392 | 0.258 | 0.279 | **0.259** |
| theta_api/expert | 0.459 | 0.406 | 0.578 | 0.433 | 0.500 | **0.475** |
| theta_api/general | 0.302 | 0.275 | 0.449 | 0.463 | 0.557 | **0.409** |
| theta_smart_api/expert | 0.443 | 0.486 | 0.554 | 0.455 | 0.370 | **0.462** |
| hippo_rag_api/gpt-5.4 | 0.354 | 0.224 | 0.363 | 0.429 | 0.488 | **0.372** |

## 3. Target 汇总统计

| Target | 数据集数 | 总用例 | Scored | Failed | 平均得分 |
|:---|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 5 | 250 | 250 | 0 | 0.544 |
| anthropic/claude-sonnet-4.6 | 5 | 250 | 250 | 0 | 0.581 |
| gemini-3.1-flash-lite-preview | 5 | 250 | 249 | 1 | 0.522 |
| gemini-3-flash-preview | 5 | 250 | 189 | 61 | 0.753 |
| z-ai/glm-5 | 5 | 250 | 250 | 0 | 0.544 |
| minimax/minimax-m2.5 | 5 | 250 | 244 | 6 | 0.499 |
| theta_api/expert | 5 | 250 | 250 | 0 | 0.620 |
| theta_api/general | 5 | 250 | 250 | 0 | 0.635 |
| theta_smart_api/expert | 5 | 250 | 249 | 1 | 0.592 |
| hippo_rag_api/gpt-5.4 | 5 | 250 | 250 | 0 | 0.393 |

## 4. 数据集汇总

| 数据集 | Target 数 | 平均得分 | 最高 | 最高 Target | 最低 | 最低 Target |
|:---|:---:|:---:|:---:|:---|:---:|:---|
| Del 10% | 10 | 0.559 | 0.750 | gemini-3-flash-preview | 0.363 | hippo_rag_api/gpt-5.4 |
| Del 20% | 10 | 0.509 | 0.713 | gemini-3-flash-preview | 0.362 | hippo_rag_api/gpt-5.4 |
| Del 30% | 10 | 0.581 | 0.761 | gemini-3-flash-preview | 0.433 | hippo_rag_api/gpt-5.4 |
| Noise 10% | 10 | 0.598 | 0.767 | gemini-3-flash-preview | 0.426 | hippo_rag_api/gpt-5.4 |
| Noise 20% | 10 | 0.594 | 0.774 | gemini-3-flash-preview | 0.381 | hippo_rag_api/gpt-5.4 |

## 5. 难度维度总览（跨数据集平均）

| Target | Direct | Single-hop | Multi-hop | Temporal | Attribution | 总平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-5.4 | 0.737 | 0.547 | 0.576 | 0.502 | 0.359 | **0.544** |
| anthropic/claude-sonnet-4.6 | 0.688 | 0.659 | 0.576 | 0.542 | 0.442 | **0.581** |
| gemini-3.1-flash-lite-preview | 0.676 | 0.574 | 0.506 | 0.509 | 0.346 | **0.522** |
| gemini-3-flash-preview | 0.850 | 0.823 | 0.684 | 0.761 | 0.278 | **0.679** |
| z-ai/glm-5 | 0.707 | 0.696 | 0.537 | 0.571 | 0.207 | **0.544** |
| minimax/minimax-m2.5 | 0.682 | 0.485 | 0.462 | 0.601 | 0.259 | **0.498** |
| theta_api/expert | 0.697 | 0.696 | 0.630 | 0.600 | 0.475 | **0.620** |
| theta_api/general | 0.728 | 0.706 | 0.686 | 0.647 | 0.409 | **0.635** |
| theta_smart_api/expert | 0.828 | 0.719 | 0.462 | 0.485 | 0.462 | **0.591** |
| hippo_rag_api/gpt-5.4 | 0.523 | 0.388 | 0.194 | 0.487 | 0.372 | **0.393** |

## 6. 剩余 Fail 用例（共 69 条，重试后仍失败）

| 数据集 | Target | Fail 数 | Fail IDs |
|:---|:---|:---:|:---|
| Del 10% | gemini-3-flash-preview | 14 | user302_AT_demo_Q006, user304_AT_demo_Q024, user305_AT_demo_Q044, user307_AT_demo_Q052, user304_AT_demo_Q063, user310_AT_demo_Q077, user303_AT_demo_Q100, user301_AT_demo_Q095, user305_AT_demo_Q091, user306_AT_demo_Q092, user301_AT_demo_Q099, user303_AT_demo_Q098, user308_AT_demo_Q093, user309_AT_demo_Q094 |
| Del 10% | gemini-3.1-flash-lite-preview | 1 | user303_AT_demo_Q073 |
| Del 20% | gemini-3-flash-preview | 11 | user327_AT_demo_Q021, user322_AT_demo_Q032, user321_AT_demo_Q078, user323_AT_demo_Q086, user321_AT_demo_Q091, user330_AT_demo_Q093, user324_AT_demo_Q094, user330_AT_demo_Q098, user323_AT_demo_Q093, user329_AT_demo_Q091, user324_AT_demo_Q099 |
| Del 20% | minimax/minimax-m2.5 | 1 | user328_AT_demo_Q082 |
| Del 20% | theta_smart_api/expert | 1 | user328_AT_demo_Q082 |
| Del 30% | gemini-3-flash-preview | 13 | user347_AT_demo_Q021, user345_AT_demo_Q018, user342_AT_demo_Q026, user345_AT_demo_Q061, user347_AT_demo_Q081, user348_AT_demo_Q083, user348_AT_demo_Q086, user344_AT_demo_Q092, user343_AT_demo_Q099, user341_AT_demo_Q099, user341_AT_demo_Q093, user348_AT_demo_Q096, user348_AT_demo_Q091 |
| Del 30% | minimax/minimax-m2.5 | 2 | user343_AT_demo_Q099, user343_AT_demo_Q100 |
| Noise 10% | gemini-3-flash-preview | 12 | user362_AT_demo_Q065, user362_AT_demo_Q088, user368_AT_demo_Q085, user364_AT_demo_Q096, user366_AT_demo_Q100, user364_AT_demo_Q092, user361_AT_demo_Q100, user368_AT_demo_Q091, user362_AT_demo_Q093, user361_AT_demo_Q097, user369_AT_demo_Q094, user367_AT_demo_Q100 |
| Noise 20% | gemini-3-flash-preview | 11 | user381_AT_demo_Q017, user389_AT_demo_Q086, user382_AT_demo_Q078, user389_AT_demo_Q097, user384_AT_demo_Q091, user386_AT_demo_Q092, user382_AT_demo_Q096, user388_AT_demo_Q093, user384_AT_demo_Q092, user388_AT_demo_Q095, user381_AT_demo_Q096 |
| Noise 20% | minimax/minimax-m2.5 | 3 | user382_AT_demo_Q047, user390_AT_demo_Q076, user384_AT_demo_Q078 |
