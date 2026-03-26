"""
TestResult → 可读 Markdown 渲染器

用法:
    from evaluator.utils.result_renderer import render_result_markdown

    md_text = render_result_markdown(result)
"""

from evaluator.core.schema import TestResult


def render_result_markdown(result: TestResult) -> str:
    """将 TestResult 渲染为人类可读的 Markdown 文本"""
    duration = (result.end - result.start).total_seconds()
    ev = result.eval
    trace = ev.trace
    parts: list[str] = []

    # ---- 标题 & 概览 ----
    status_emoji = "PASS" if ev.result == "pass" else "FAIL"
    parts.append(f"# {result.id} — {status_emoji} ({ev.score:.2f})\n")
    parts.append("| 项目 | 值 |")
    parts.append("|------|------|")
    parts.append(f"| 结果 | **{ev.result.upper()}** |")
    parts.append(f"| 得分 | {ev.score:.2f} |")
    parts.append(f"| 耗时 | {duration:.1f}s |")
    parts.append(f"| 开始 | {result.start.strftime('%Y-%m-%d %H:%M:%S')} |")
    parts.append(f"| 结束 | {result.end.strftime('%Y-%m-%d %H:%M:%S')} |")
    parts.append("")

    # ---- 评估维度（如有）----
    if trace and trace.eval_detail:
        dimensions = trace.eval_detail.get("dimensions", {})
        if dimensions:
            parts.append("## 评估维度\n")
            parts.append("| 维度 | 分数 | 权重 | 说明 |")
            parts.append("|------|------|------|------|")
            for dim_name, dim_data in dimensions.items():
                display = dim_data.get("display_name", dim_name)
                score = dim_data.get("score", "-")
                weight = dim_data.get("weight", "-")
                reason = dim_data.get("reason", "")
                if len(reason) > 120:
                    reason = reason[:120] + "..."
                parts.append(f"| {display} | {score} | {weight}% | {reason} |")
            parts.append("")

    # ---- 对话记录 ----
    if trace and trace.test_memory:
        # 跳过 is_finished 幽灵轮次（未发送给 target），不计入轮次
        effective_memory = [
            mem for mem in trace.test_memory
            if not (mem.test_reaction.is_finished and mem.target_response is None)
        ]
        turns = len(effective_memory)
        parts.append(f"## 对话记录（共 {turns} 轮）\n")

        for i, mem in enumerate(effective_memory, 1):
            reaction = mem.test_reaction

            # 虚拟用户发言
            user_text = reaction.action.semantic_content or ""
            source = ""
            if reaction.reason == "强制输入":
                source = " `[strict_input]`"

            parts.append(f"### Round {i}{source}\n")
            parts.append(f"> **虚拟用户**: {user_text}\n")

            if reaction.reason and reaction.reason != "强制输入":
                parts.append(f"*reason: {reaction.reason}*\n")

            # AI 助手回复
            if mem.target_response:
                target_text = mem.target_response.extract_text()
                if target_text:
                    # 清理多余空行
                    cleaned = "\n".join(line for line in target_text.split("\n") if line.strip())
                    parts.append(f"**AI 助手**: {cleaned}\n")

            parts.append("---\n")

    # ---- 评估反馈 ----
    if ev.feedback:
        parts.append("## 评估反馈\n")
        # 按 " | " 分段展示
        segments = ev.feedback.split(" | ")
        for seg in segments:
            seg = seg.strip()
            if seg.startswith("优点:"):
                parts.append("**优点**\n")
                for item in seg[len("优点:") :].split(";"):
                    item = item.strip()
                    if item:
                        parts.append(f"- {item}")
            elif seg.startswith("问题:"):
                parts.append("\n**问题**\n")
                for item in seg[len("问题:") :].split(";"):
                    item = item.strip()
                    if item:
                        parts.append(f"- {item}")
            else:
                parts.append(f"{seg}\n")
        parts.append("")

    # ---- 成本 ----
    cost = result.cost
    has_cost = bool(cost.test or cost.eval or cost.target)
    if has_cost:
        parts.append("## Token 消耗\n")
        parts.append("| Agent | 模型 | Input | Output | Total |")
        parts.append("|-------|------|-------|--------|-------|")
        for label, cost_dict in [("TestAgent", cost.test), ("EvalAgent", cost.eval), ("TargetAgent", cost.target)]:
            if not cost_dict:
                continue
            for model_name, usage in cost_dict.items():
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                total = usage.get("total_tokens", 0)
                parts.append(f"| {label} | {model_name} | {inp:,} | {out:,} | {total:,} |")
        parts.append("")

    return "\n".join(parts)
