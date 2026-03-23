"""检查点管理器 — 断点续跑的文件持久化工具

每个运行中的评测会话对应两个文件:
- {session_id}.meta.json  — 会话元数据（开始时写入一次）
- {session_id}.results.jsonl — 已完成用例的 TestResult（逐条追加，crash-safe）

使用模式:
    # 创建检查点
    mgr = CheckpointManager(session_id)
    mgr.save_meta(meta)           # 开始时保存元数据
    mgr.append_result(result)      # 每完成一个用例追加结果
    mgr.cleanup()                  # 全部完成后删除检查点

    # 恢复
    meta, results = CheckpointManager.load(session_id)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.schema import TestResult

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "report" / ".checkpoints"


class CheckpointMeta(BaseModel):
    """检查点元数据 — 记录评测会话的配置信息，用于断点续跑"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="会话 ID")
    benchmark: str = Field(description="评测类型（如 healthbench）")
    dataset: str = Field(description="数据集名称（如 sample）")
    target_type: str = Field(description="目标系统类型（如 llm_api）")
    cli_overrides: dict[str, Any] | None = Field(None, description="CLI/UI 传入的 target 覆盖参数")
    runtime_target: dict[str, Any] = Field(description="运行时 target 配置（JSON 序列化的 TargetInfo）")
    case_ids: list[str] = Field(description="所有待执行用例 ID 列表（有序）")
    max_concurrency: int = Field(default=0, description="最大并发数")
    started_at: str = Field(description="开始时间（ISO 格式）")
    data_file_hash: str = Field(default="", description="JSONL 数据文件的 SHA-256 前 16 位")


class CheckpointManager:
    """检查点管理器 — 提供 save/append/load/cleanup 操作"""

    def __init__(self, session_id: str, checkpoint_dir: Path | None = None):
        self.session_id = session_id
        self.checkpoint_dir = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        self.meta_path = self.checkpoint_dir / f"{session_id}.meta.json"
        self.results_path = self.checkpoint_dir / f"{session_id}.results.jsonl"
        self._lock = threading.Lock()

    def save_meta(self, meta: CheckpointMeta) -> None:
        """保存会话元数据（原子写入）"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = meta.model_dump(mode="json")
        # 先写临时文件，再原子替换
        fd, tmp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.meta_path)
        except Exception:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info("检查点元数据已保存: %s", self.meta_path)

    def append_result(self, result: TestResult) -> None:
        """追加一条完成的 TestResult 到 results.jsonl（线程安全）"""
        line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with open(self.results_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def cleanup(self) -> None:
        """删除检查点文件（评测成功完成后调用）"""
        for path in (self.meta_path, self.results_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        logger.info("检查点已清理: %s", self.session_id)

    @staticmethod
    def load(session_id: str, checkpoint_dir: Path | None = None) -> tuple[CheckpointMeta, list[TestResult]]:
        """加载检查点：返回 (元数据, 已完成结果列表)

        Raises:
            FileNotFoundError: meta.json 不存在
        """
        d = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        meta_path = d / f"{session_id}.meta.json"
        results_path = d / f"{session_id}.results.jsonl"

        # 读取 meta
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = CheckpointMeta.model_validate_json(f.read())

        # 读取已完成结果（容忍末行截断）
        results: list[TestResult] = []
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(TestResult.model_validate_json(line))
                    except Exception as e:
                        logger.warning("检查点第 %d 行解析失败（可能为截断数据），已跳过: %s", i, e)

        logger.info("检查点已加载: %s（%d 条已完成结果）", session_id, len(results))
        return meta, results

    @staticmethod
    def find_checkpoints(
        benchmark: str | None = None,
        dataset: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> list[CheckpointMeta]:
        """查找活跃的检查点，按开始时间倒序"""
        d = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        if not d.is_dir():
            return []

        result: list[CheckpointMeta] = []
        for meta_file in d.glob("*.meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = CheckpointMeta.model_validate_json(f.read())
                if benchmark and meta.benchmark != benchmark:
                    continue
                if dataset and meta.dataset != dataset:
                    continue
                result.append(meta)
            except Exception as e:
                logger.warning("无法解析检查点 %s: %s", meta_file.name, e)

        result.sort(key=lambda m: m.started_at, reverse=True)
        return result

    @staticmethod
    def completed_count(session_id: str, checkpoint_dir: Path | None = None) -> int:
        """快速获取已完成用例数（不解析 TestResult，仅计行数）"""
        d = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        results_path = d / f"{session_id}.results.jsonl"
        if not results_path.exists():
            return 0
        count = 0
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    @staticmethod
    def compute_data_hash(jsonl_path: Path) -> str:
        """计算 JSONL 数据文件的 SHA-256 前 16 位"""
        h = hashlib.sha256()
        with open(jsonl_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
