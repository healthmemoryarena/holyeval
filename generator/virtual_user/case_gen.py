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

# 五维三档定义及分配权重（档1=50%, 档2=30%, 档3=20%）
_DIMENSIONS: dict[str, list[tuple[str, float]]] = {
    "disclosure": [("responsive", 0.5), ("reluctant", 0.3), ("withholding", 0.2)],
    "attitude": [("compliant", 0.5), ("impatient", 0.3), ("dominant", 0.2)],
    "cognition": [("accurate", 0.5), ("partial_understanding", 0.3), ("complete_denial", 0.2)],
    "logic": [("consistent", 0.5), ("occasional_contradiction", 0.3), ("fabricating", 0.2)],
    "expression": [("normal", 0.5), ("vague", 0.3), ("incoherent", 0.2)],
}

# 每个维度的 baseline 值（档1）
_BASELINES: dict[str, str] = {dim: levels[0][0] for dim, levels in _DIMENSIONS.items()}

_GOAL = (
    "你刚打开了一款健康管理 APP，屏幕上弹出了一条消息。"
    "根据你的性格和当前状态，自然地做出反应。"
    "你有以下选择，每种都是完全合理的：\n"
    "  - 主动描述自己的健康困扰\n"
    "  - 简单敷衍几句\n"
    "  - 质疑这个 APP 是什么\n"
    '  - 直接无视这条消息（回复"[沉默]"表示你关掉了弹窗或划走了）\n'
    "  - 表示不感兴趣然后离开\n"
    "真实用户打开一个新 APP 时，大部分人不会立刻跟弹窗消息深入交流。"
    "不要因为你在'测试'就刻意配合，按你的性格自然反应。"
)


def _assign_persona(rng: random.Random) -> dict[str, str]:
    """按权重随机分配五维对抗档位"""
    persona = {}
    for dim, levels in _DIMENSIONS.items():
        values, weights = zip(*levels)
        persona[dim] = rng.choices(values, weights=weights, k=1)[0]
    return persona


# 说话风格模板 — 按年龄段和职业特征分配，对抗人格同质化
_SPEECH_STYLES: list[dict[str, str]] = [
    {"tag": "年轻直接型", "desc": "说话简短直接，爱用网络用语，不耐烦时会发'...'或'？'。口头禅类似'emmm'、'就挺无语的'、'好吧'。"},
    {"tag": "中年务实型", "desc": "说话讲重点，关心实际效果，不喜欢花哨的东西。口头禅类似'有用吗'、'说具体点'、'我时间有限'。"},
    {"tag": "老年谨慎型", "desc": "说话慢，会反复确认，对新事物持怀疑态度。口头禅类似'这个靠谱不'、'我不太会用这种'、'年轻人才搞这些'。"},
    {"tag": "知识分子型", "desc": "表达清晰有条理，会用专业词汇，但有时过度分析。口头禅类似'从科学角度来说'、'我了解过一些'、'但是数据支持吗'。"},
    {"tag": "体力劳动者型", "desc": "说话朴实，用大白话，不喜欢绕弯子。口头禅类似'别整那些虚的'、'直说就行'、'我也不懂那些'。"},
    {"tag": "焦虑型", "desc": "话多但主题跳跃，容易联想到最坏情况。口头禅类似'不会吧'、'我看网上说可能是xxx'、'会不会很严重'。"},
    {"tag": "佛系型", "desc": "无所谓的态度，回复极简，不主动展开。口头禅类似'随便吧'、'都行'、'嗯'、'看情况'。"},
    {"tag": "社交型", "desc": "喜欢聊天，会扯到不相关的事（工作、家庭、朋友），信息埋在闲聊里。口头禅类似'对了我跟你说'、'我朋友也是这样'。"},
]


def _assign_speech_style(profile: dict, rng: random.Random) -> str:
    """根据画像特征分配说话风格，增加个体差异"""
    age = profile.get("age", 35)
    occupation = profile.get("occupation", "")

    # 按年龄和职业倾向性加权选择（不是硬规则，保留随机性）
    weights = [1.0] * len(_SPEECH_STYLES)

    if age <= 28:
        weights[0] *= 3  # 年轻直接型
        weights[5] *= 2  # 焦虑型（年轻人更容易焦虑）
        weights[6] *= 2  # 佛系型
    elif age <= 45:
        weights[1] *= 3  # 中年务实型
        weights[3] *= 2  # 知识分子型
    else:
        weights[2] *= 3  # 老年谨慎型
        weights[7] *= 2  # 社交型

    if any(k in occupation for k in ["工程师", "研究", "医", "护士", "教师"]):
        weights[3] *= 2  # 知识分子型
    if any(k in occupation for k in ["司机", "厨师", "餐饮", "超市", "货"]):
        weights[4] *= 2  # 体力劳动者型
    if any(k in occupation for k in ["学生", "研究生"]):
        weights[0] *= 2  # 年轻直接型
        weights[6] *= 2  # 佛系型

    style = rng.choices(_SPEECH_STYLES, weights=weights, k=1)[0]
    return style["desc"]


def _build_context(profile: dict, speech_style: str) -> str:
    """从画像 + 说话风格构建虚拟用户 context 描述"""
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

    # 注入个性化说话风格
    parts.append(f"\n\n你的说话风格：{speech_style}")

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
    with open(profiles_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))
    logger.info("读取 %d 个画像", len(profiles))

    # 读取开场白
    with open(openings_path, encoding="utf-8") as f:
        openings = json.load(f)
    logger.info("读取 %d 句开场白", len(openings))

    # 为每个画像分配对抗维度和说话风格（固定种子）
    rng = random.Random(seed)
    profile_personas: list[dict[str, str]] = [_assign_persona(rng) for _ in profiles]
    profile_styles: list[str] = [_assign_speech_style(p, rng) for p in profiles]

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
                context = _build_context(profile, profile_styles[i])

                # 构建 tags
                tags = [f"opening:{oid}", f"profile:{pid}"]
                for dim, val in persona.items():
                    tags.append(f"{dim}:{val}")

                # 构建 title（简短）
                age = profile.get("age", "?")
                gender_short = {"male": "男", "female": "女"}.get(profile.get("gender", ""), "")
                bmi = profile.get("bmi", "?")
                non_baseline = [v for d, v in persona.items() if v != _BASELINES[d]]
                persona_summary = "/".join(non_baseline)
                title = f"开场白{oid} × {age}岁{gender_short}BMI{bmi}"
                if persona_summary:
                    title += f" [{persona_summary}]"

                bench_item = {
                    "id": f"vu_o{oid}_{pid.replace('obesity_', 'p')}",
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
