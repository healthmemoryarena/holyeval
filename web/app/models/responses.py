"""API 响应模型"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

# 共享数据模型（从 evaluator 核心层 re-export）
from evaluator.core.bench_schema import ApiCallResult  # noqa: F401
from evaluator.core.schema import (  # noqa: F401
    AgentInfo,
    BenchmarkSummary,
    CaseSummary,
    DatasetDetail,
    DatasetInfo,
    ReportEntry,
    TargetInfo,
)


# ==================== Task（Web 专用）====================


class TaskCreateRequest(BaseModel):
    benchmark: str
    dataset: str
    target_type: str | None = None  # 目标系统类型（多 target 时指定，如 "llm_api"）
    target: dict[str, Any] | None = None  # 前端传入 target 配置（editable 字段覆盖）
    ids: str | None = None
    limit: int | None = None
    max_concurrency: int = 3


class TaskSummary(BaseModel):
    task_id: str
    benchmark: str
    dataset: str
    target_type: str | None = None  # 被测系统类型（eval-only 模式为 None）
    target_model: str | None = None  # 被测系统模型（可选）
    status: str  # pending | running | completed | cancelled | error
    total: int
    completed: int
    created_at: datetime


class TaskDetail(BaseModel):
    task_id: str
    benchmark: str
    dataset: str
    runtime_target: dict[str, Any] | None = None  # runtime_target 配置（eval-only 模式为 None）
    status: str
    total: int
    completed: int
    cancelled: bool
    created_at: datetime
    cases: dict[str, Any]
    stats_by_tag: dict[str, Any] | None = None
    report_path: str | None = None
    report_summary: dict[str, Any] | None = None  # eval-only 完成后的报告摘要


# ==================== Eval-Only（Web 专用）====================


class EvalCheckRequest(BaseModel):
    benchmark: str
    dataset: str
    results: list[ApiCallResult]


class EvalCheckResponse(BaseModel):
    benchmark: str
    dataset: str
    benchmark_description: str = ""
    dataset_total: int  # 数据集总条数
    submitted: int  # 提交的结果条数
    matched: int  # ID 匹配成功的条数
    missed_ids: list[str] = []  # 数据集中有但提交中缺失的 ID（最多 50 条）
    extra_ids: list[str] = []  # 提交中有但数据集中不存在的 ID
    tag_distribution: dict[str, int] = {}  # 匹配用例的 tag 分布


class EvalOnlyRequest(BaseModel):
    benchmark: str
    dataset: str
    results: list[ApiCallResult]
    max_concurrency: int = 5


class EvalOnlyResponse(BaseModel):
    task_id: str
    status: str
    total: int  # 匹配成功的评测条数
    dataset_total: int  # 数据集总条数
    submitted: int  # 提交的结果条数
    matched: int  # ID 匹配成功的条数
    missed_ids: list[str] = []  # 数据集中有但提交中缺失的 ID（最多 50 条）
    extra_ids: list[str] = []  # 提交中有但数据集中不存在的 ID
    created_at: datetime


# ==================== Report（Web 专用）====================


class ReportListResponse(BaseModel):
    reports: list[ReportEntry]


# ==================== Checkpoint（Web 专用）====================


class CheckpointSummary(BaseModel):
    session_id: str
    benchmark: str
    dataset: str
    target_type: str
    case_count: int  # 总用例数
    completed_count: int  # 已完成数
    started_at: str
