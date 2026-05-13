"""Built-in signals for the rubric evaluator.

Each signal is a pure function `(response_text, reaction) -> value`. Keep
detection conservative: prefer false negatives over false positives, since
signals back hard assertions (`signal_check`).
"""

from __future__ import annotations

import re
from typing import Any

from evaluator.utils.signals import register_signal

# Markdown image: ![alt](url)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# A markdown table requires a separator row like `|---|---|` (three or more
# dashes per column, optional leading/trailing pipe, optional alignment colons)
# sandwiched between two pipe rows.
_MD_TABLE_SEP = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
_MD_PIPE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)

# Hint keywords mentioned inside reaction annotations or tool-call names.
_CHART_HINTS = ("chart", "sparkline", "plot", "graph", "图表", "趋势图", "柱状图", "折线图")


def _reaction_items(reaction: Any) -> list[dict]:
    if reaction is None:
        return []
    items = getattr(reaction, "message_list", None) or []
    return [i for i in items if isinstance(i, dict)]


def _reaction_tool_calls(reaction: Any) -> list[dict]:
    if reaction is None:
        return []
    tcs = getattr(reaction, "tool_calls", None) or []
    return [i for i in tcs if isinstance(i, dict)]


@register_signal("has_chart")
def _has_chart(response: str, reaction: Any) -> bool:
    """True if the turn rendered a chart — via markdown image, annotation chunk
    whose content hints at a chart, or tool call whose name mentions charting."""
    if _MD_IMAGE.search(response or ""):
        return True
    for item in _reaction_items(reaction):
        content = item.get("content")
        text = content if isinstance(content, str) else str(content or "")
        low = text.lower()
        if any(k in low for k in _CHART_HINTS):
            return True
    for tc in _reaction_tool_calls(reaction):
        name = (tc.get("name") or "").lower()
        if any(k in name for k in _CHART_HINTS):
            return True
    return False


@register_signal("has_table")
def _has_table(response: str, reaction: Any) -> bool:
    """True if the reply contains a markdown table (separator row + ≥2 pipe rows)."""
    if not response:
        return False
    if not _MD_TABLE_SEP.search(response):
        return False
    return len(_MD_PIPE_ROW.findall(response)) >= 2


@register_signal("char_count")
def _char_count(response: str, reaction: Any) -> int:
    return len(response or "")


@register_signal("md_image_count")
def _md_image_count(response: str, reaction: Any) -> int:
    return len(_MD_IMAGE.findall(response or ""))


@register_signal("tool_calls_count")
def _tool_calls_count(response: str, reaction: Any) -> int:
    return len(_reaction_tool_calls(reaction))


@register_signal("has_annotation")
def _has_annotation(response: str, reaction: Any) -> bool:
    for item in _reaction_items(reaction):
        if item.get("type") == "annotation":
            return True
    return False
