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
