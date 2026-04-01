# Mem0 RAG TargetAgent 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 thetagen 评测中新增 mem0 作为被测系统（TargetAgent），与 HippoRAG 横向对比记忆/检索能力。

**Architecture:** 复用 HippoRAG 的文档分块逻辑（profile + timeline，排除 events.json），将文本块通过 `mem0.add(text, user_id, infer=True)` 写入 mem0（LLM 自动提取记忆）。查询时 `mem0.search(query, user_id)` 检索相关记忆，拼接进 RAG prompt 送 LLM 生成答案。与 HippoRAG 共享同一套 system prompt、few-shot 示例和生成模型。

**Tech Stack:** mem0ai (pip), text-embedding-3-large (OpenAI embedding), Qdrant (本地向量库, mem0 内置), gemini-3-flash-preview (生成/记忆提取)

---

### Task 1: 安装 mem0ai 依赖

**Files:**
- Modify: `pyproject.toml`

**Step 1: 添加依赖**

在 `pyproject.toml` 的 `[project] dependencies` 中添加 `mem0ai`：

```toml
"mem0ai>=0.1.0",
```

**Step 2: 安装**

Run: `cd /mnt/elements/fat/xiaotong/holyeval && uv pip install mem0ai`
Expected: 安装成功

**Step 3: 验证**

Run: `.venv/bin/python -c "from mem0 import Memory; print('mem0 OK')"`
Expected: `mem0 OK`

---

### Task 2: 提取 HippoRAG 的文档分块为共享模块

**Files:**
- Create: `evaluator/utils/thetagen_chunker.py`
- Modify: `evaluator/plugin/target_agent/hippo_rag_api_target_agent.py`（改为 import 共享模块）

**目的:** `_load_user_documents()` 和 `_email_to_dir()` 逻辑需要在 HippoRAG 和 mem0 之间共享，避免代码重复。

**Step 1: 创建共享分块模块**

将 `hippo_rag_api_target_agent.py` 中的以下函数提取到 `evaluator/utils/thetagen_chunker.py`：

```python
"""thetagen 用户数据分块工具 — HippoRAG / mem0 等 RAG agent 共享"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# benchmark/data/ 目录
BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data"


def email_to_dir(email: str) -> str:
    """邮箱 → 目录名: user110@demo → user110_AT_demo"""
    return email.replace("@", "_AT_")


def resolve_user_dir(data_group: str, user_email: str) -> Path:
    """解析用户数据目录路径"""
    return BENCHMARK_DATA_DIR / data_group / ".data" / email_to_dir(user_email)


def load_user_documents(user_dir: Path, max_chunk_chars: int = 15000) -> tuple[list[str], set[int]]:
    """加载用户数据文件，按类型智能切分

    返回 (chunks, openie_indices):
    - chunks: 所有文本块
    - openie_indices: 需要做 OpenIE 的 chunk 索引集合

    注意: 只加载 profile.json + timeline.json，不加载 events.json（含答案）
    """
    # ... 从 hippo_rag_api_target_agent.py 的 _load_user_documents 完整复制逻辑 ...
```

**Step 2: 修改 HippoRAG 改为 import 共享模块**

在 `hippo_rag_api_target_agent.py` 中：

```python
# 替换原有的 _email_to_dir 和 _load_user_documents
from evaluator.utils.thetagen_chunker import email_to_dir as _email_to_dir, load_user_documents as _load_user_documents, BENCHMARK_DATA_DIR as _BENCHMARK_DATA_DIR, resolve_user_dir
```

删除原文件中的 `_email_to_dir`、`_load_user_documents`、`_chunk_by_chars`、`_BENCHMARK_DATA_DIR` 定义。

**Step 3: 运行现有测试验证不破坏 HippoRAG**

Run: `cd /mnt/elements/fat/xiaotong/holyeval && .venv/bin/python -c "from evaluator.plugin.target_agent import HippoRagApiTargetAgent; print('import OK')"`
Expected: `import OK`

**Step 4: Commit**

```bash
git add evaluator/utils/thetagen_chunker.py evaluator/plugin/target_agent/hippo_rag_api_target_agent.py
git commit -m "refactor: extract thetagen chunker as shared module for HippoRAG/mem0"
```

---

### Task 3: 实现 Mem0RagApiTargetAgent

**Files:**
- Create: `evaluator/plugin/target_agent/mem0_rag_api_target_agent.py`

**Step 1: 创建配置模型 + Agent 类**

```python
"""
Mem0RagApiTargetAgent — mem0 记忆增强问答

注册名称: "mem0_rag_api"

实现流程:
1. 加载用户数据文件 → 复用 thetagen_chunker 智能切分（排除 events.json）
2. 通过 mem0.add(chunk, user_id, infer=True) 批量写入记忆（LLM 自动提取关键事实）
3. 查询时 mem0.search(query, user_id) 检索相关记忆
4. 拼接 RAG prompt + few-shot 示例 → LLM 生成答案

配置与 HippoRAG 对齐:
    model: gemini-3-flash-preview（生成模型）
    embedding_model: text-embedding-3-large（向量编码，通过 mem0 config 传入）
    top_k: 10（检索记忆条数）
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    SessionInfo,
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.utils.llm import BasicMessage, do_execute
from evaluator.utils.thetagen_chunker import BENCHMARK_DATA_DIR, load_user_documents, resolve_user_dir

logger = logging.getLogger(__name__)

# 全局已加载用户集合（避免同一用户重复灌入记忆）
_LOADED_USERS: set[str] = set()
# 全局 Memory 实例缓存
_MEM0_INSTANCE: Any = None


def _get_mem0(embedding_model: str = "text-embedding-3-large") -> Any:
    """获取或创建 mem0 Memory 单例（配置 text-embedding-3-large）"""
    global _MEM0_INSTANCE
    if _MEM0_INSTANCE is not None:
        return _MEM0_INSTANCE

    from mem0 import Memory

    config = {
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
            },
        },
    }
    _MEM0_INSTANCE = Memory.from_config(config)
    return _MEM0_INSTANCE


async def _ensure_user_loaded(
    mem0_instance: Any,
    user_id: str,
    user_dir: Path,
    max_chunk_chars: int,
    marker_dir: Path,
) -> int:
    """确保用户数据已灌入 mem0（幂等，有磁盘标记文件）

    返回写入的记忆条数。
    """
    # 检查内存缓存
    if user_id in _LOADED_USERS:
        return 0

    # 检查磁盘标记
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_file = marker_dir / ".mem0_loaded"

    # 计算数据哈希（检测数据变更）
    chunks, _ = load_user_documents(user_dir, max_chunk_chars)
    data_hash = hashlib.md5("".join(chunks).encode()).hexdigest()[:12]

    if marker_file.exists():
        try:
            meta = json.loads(marker_file.read_text())
            if meta.get("data_hash") == data_hash and meta.get("chunk_count") == len(chunks):
                _LOADED_USERS.add(user_id)
                logger.info("[mem0] 用户 %s 数据已加载（缓存命中），跳过", user_id)
                return 0
        except (json.JSONDecodeError, OSError):
            pass

    # 清除旧记忆
    logger.info("[mem0] 清除用户 %s 旧记忆并重新加载 %d 块数据", user_id, len(chunks))
    try:
        mem0_instance.delete_all(user_id=user_id)
    except Exception:
        pass  # 首次加载时可能无旧记忆

    # 逐条写入（mem0 无 batch API）
    # infer=True: mem0 用 LLM 提取关键事实
    count = 0
    for chunk in chunks:
        try:
            mem0_instance.add(chunk, user_id=user_id, infer=True)
            count += 1
        except Exception as e:
            logger.warning("[mem0] 写入记忆失败: %s", e)

    # 写标记文件
    marker_file.write_text(json.dumps({
        "data_hash": data_hash,
        "chunk_count": len(chunks),
        "memory_count": count,
    }))
    _LOADED_USERS.add(user_id)
    logger.info("[mem0] 用户 %s 加载完成，写入 %d 条记忆", user_id, count)
    return count


# ============================================================
# RAG prompt — 复用 HippoRAG 的 system prompt 和 few-shot
# ============================================================

# 直接从 hippo_rag 导入共享的 prompt 常量
from evaluator.plugin.target_agent.hippo_rag_api_target_agent import (
    _RAG_SYSTEM_PROMPT,
    _SHOT_LIST_INPUT,
    _SHOT_LIST_OUTPUT,
    _SHOT_NUMERIC_INPUT,
    _SHOT_NUMERIC_OUTPUT,
)


def _build_rag_input(query: str, memories: list[dict]) -> str:
    """将 mem0 search 结果拼接为 RAG 输入"""
    if not memories:
        context = "(No relevant memories found)"
    else:
        context = "\n\n".join(m.get("memory", "") for m in memories)
    return f"{context}\n\nQuestion: {query}\nThought: "


# ============================================================
# 配置模型
# ============================================================


class Mem0RagApiTargetInfo(BaseModel):
    """mem0 记忆增强被测目标配置"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "默认配置（embedding 与 HippoRAG 一致）",
                    "type": "mem0_rag_api",
                    "model": "gemini-3-flash-preview",
                    "data_group": "thetagen",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["mem0_rag_api"] = Field(description="目标类型")
    model: Literal[
        "gpt-5.4",
        "gpt-5.2",
        "gpt-4.1",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "minimax/minimax-m2.5",
        "z-ai/glm-5",
    ] = Field(description="生成模型名称")
    embedding_model: str = Field(
        "text-embedding-3-large",
        description="嵌入模型（与 HippoRAG 一致）",
    )
    data_group: str = Field(description="数据目录（benchmark 名称，如 'thetagen'）")
    user_email: Optional[str] = Field(None, description="用户邮箱（映射到 .data/{user_dir}/）")
    top_k: int = Field(10, description="检索记忆条数", ge=1, le=100)
    max_chunk_chars: int = Field(15000, description="文档切分字符数", ge=1000, le=50000)
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词（默认使用 HippoRAG 同款）")


# ============================================================
# Agent 实现
# ============================================================


class Mem0RagApiTargetAgent(AbstractTargetAgent, name="mem0_rag_api", params_model=Mem0RagApiTargetInfo):
    """mem0 记忆增强问答"""

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125"
            " 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0"
            " 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
        ),
        "color": "#f59e0b",
        "features": ["RAG", "记忆管理", "mem0"],
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(self, target_config: Mem0RagApiTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: Mem0RagApiTargetInfo = target_config

        # 对话历史
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # 用户数据目录
        self._user_dir: Path | None = None
        if target_config.user_email:
            self._user_dir = resolve_user_dir(target_config.data_group, target_config.user_email)
            if not self._user_dir.is_dir():
                logger.warning("[mem0] 用户数据目录不存在: %s", self._user_dir)

        # mem0 缓存目录（磁盘标记文件）
        self._marker_dir: Path | None = None
        if self._user_dir:
            self._marker_dir = (
                BENCHMARK_DATA_DIR / target_config.data_group / ".mem0_work_dirs" / self._user_dir.name
            )

        # 成本统计
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage
        self._cost = accumulate_usage(self._cost, usage)

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        if test_action is None:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "你好，有什么可以帮你的？"}],
            )

        user_text = self._extract_user_input(test_action)

        if not self._user_dir or not self._user_dir.is_dir():
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": f"错误: 用户数据目录不存在 ({self._user_dir})"}],
            )

        # 1. 获取 mem0 实例 & 确保用户数据已加载
        m = _get_mem0(self.config.embedding_model)
        user_id = self._user_dir.name  # e.g. "user110_AT_demo"

        await _ensure_user_loaded(
            mem0_instance=m,
            user_id=user_id,
            user_dir=self._user_dir,
            max_chunk_chars=self.config.max_chunk_chars,
            marker_dir=self._marker_dir,
        )

        # 2. 检索相关记忆
        results = m.search(query=user_text, user_id=user_id, limit=self.config.top_k)
        # mem0 search 返回 {"results": [...]} 或直接 list
        memories = results.get("results", results) if isinstance(results, dict) else results

        logger.debug(
            "[mem0] Query: %s, Retrieved %d memories, calling %s",
            user_text[:80],
            len(memories) if memories else 0,
            self.config.model,
        )

        # 3. 构建 RAG prompt
        rag_input = _build_rag_input(user_text, memories)
        system_prompt = self.config.system_prompt or _RAG_SYSTEM_PROMPT

        # 4. 调用 LLM（带 few-shot 示例，与 HippoRAG 一致）
        history_with_oneshot = [
            BasicMessage(role="user", content=_SHOT_NUMERIC_INPUT),
            BasicMessage(role="assistant", content=_SHOT_NUMERIC_OUTPUT),
            BasicMessage(role="user", content=_SHOT_LIST_INPUT),
            BasicMessage(role="assistant", content=_SHOT_LIST_OUTPUT),
        ]
        if self._conversation:
            history_with_oneshot.extend(self._conversation)

        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=rag_input,
            history_messages=history_with_oneshot,
        )

        self._accumulate_cost(result.usage)

        # 更新对话历史
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=result.content))

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": result.content}],
            usage=result.usage,
        )
```

**Step 2: 验证 import**

Run: `.venv/bin/python -c "from evaluator.plugin.target_agent.mem0_rag_api_target_agent import Mem0RagApiTargetAgent; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add evaluator/plugin/target_agent/mem0_rag_api_target_agent.py
git commit -m "feat: add mem0 RAG target agent for thetagen benchmark"
```

---

### Task 4: 注册插件

**Files:**
- Modify: `evaluator/plugin/target_agent/__init__.py`

**Step 1: 添加 import 和 __all__**

```python
from evaluator.plugin.target_agent.mem0_rag_api_target_agent import Mem0RagApiTargetAgent

__all__ = [
    ...
    "Mem0RagApiTargetAgent",
]
```

同时更新文件顶部 docstring 注册表。

**Step 2: 验证注册**

Run: `.venv/bin/python -c "from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; from evaluator.plugin import target_agent; print(AbstractTargetAgent.get('mem0_rag_api'))"`
Expected: `<class '...Mem0RagApiTargetAgent'>`

**Step 3: Commit**

```bash
git add evaluator/plugin/target_agent/__init__.py
git commit -m "feat: register mem0_rag_api target agent plugin"
```

---

### Task 5: 在 thetagen_test_1 metadata.json 中添加 mem0 target

**Files:**
- Modify: `benchmark/data/thetagen_test_1/metadata.json`

**Step 1: 在 target 数组中添加 mem0_rag_api 条目**

在 `hippo_rag_api` 条目后面添加:

```json
{
  "type": "mem0_rag_api",
  "fields": {
    "model": {
      "default": "gemini-3-flash-preview",
      "editable": true,
      "required": true
    },
    "data_group": {
      "default": "thetagen",
      "editable": false
    },
    "user_email": {
      "default": null,
      "editable": false
    },
    "embedding_model": {
      "default": "text-embedding-3-large",
      "editable": true,
      "required": false
    },
    "top_k": {
      "default": 10,
      "editable": true,
      "required": false
    }
  }
}
```

**Step 2: 在 full_sample.jsonl 中为每条用例添加 mem0 target_overrides**

每条 JSONL 的 `target_overrides` 中已有 `hippo_rag_api`，需要同样加上 `mem0_rag_api`：

```json
"target_overrides": {
  "hippo_rag_api": {"user_email": "userXXX@demo"},
  "dyg_rag_api": {"user_email": "userXXX@demo"},
  "mem0_rag_api": {"user_email": "userXXX@demo"}
}
```

用脚本批量添加（读取每行 JSON，在 target_overrides 中复制 hippo_rag_api 的 user_email 给 mem0_rag_api）。

**Step 3: Commit**

```bash
git add benchmark/data/thetagen_test_1/metadata.json benchmark/data/thetagen_test_1/full_sample.jsonl
git commit -m "feat: add mem0_rag_api target to thetagen_test_1 metadata and test cases"
```

---

### Task 6: 端到端冒烟测试

**Step 1: 用 1 条用例测试**

Run:
```bash
cd /mnt/elements/fat/xiaotong/holyeval
.venv/bin/python -m benchmark.basic_runner thetagen_test_1 full_sample \
  --target-type mem0_rag_api \
  --target-model gemini-3-flash-preview \
  --limit 1 -v
```

Expected: 成功运行 1 条，输出包含 score/pass 信息。

**验证点:**
- mem0 数据加载日志: `[mem0] 用户 xxx 加载完成，写入 N 条记忆`
- 检索日志: `[mem0] Query: ..., Retrieved N memories`
- 生成结果包含 `Thought:` 和 `Answer:` 格式
- 无 events.json 相关数据泄露

**Step 2: 第二次运行同用户，验证缓存命中**

Run: 再次运行同一条用例
Expected: 日志显示 `[mem0] 用户 xxx 数据已加载（缓存命中），跳过`

---

### Task 7: 运行完整 50 条评测

**Step 1: 跑全量 full_sample**

Run:
```bash
.venv/bin/python -m benchmark.basic_runner thetagen_test_1 full_sample \
  --target-type mem0_rag_api \
  --target-model gemini-3-flash-preview \
  -p 3 -v
```

Expected: 50 条全部跑完，生成报告到 `benchmark/report/thetagen_test_1/`

**Step 2: 检查报告**

Run: `ls -la benchmark/report/thetagen_test_1/`
Expected: 看到新生成的 `full_sample_mem0_rag_api_gemini-3-flash-preview_*.json` 报告文件

---

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 数据加载 | 只读 profile.json + timeline.json | events.json 含 affected_indicators 答案，会泄露 |
| 分块策略 | 复用 HippoRAG 的 thetagen_chunker | 保持对比公平性 |
| mem0 infer | `infer=True` | 让 mem0 自动提取关键事实，体现其记忆能力 |
| embedding | text-embedding-3-large | 与 HippoRAG 一致 |
| 生成模型 | gemini-3-flash-preview | 与 HippoRAG 默认一致 |
| system prompt | 共享 _RAG_SYSTEM_PROMPT | 公平对比 |
| 缓存策略 | 磁盘标记文件 + 内存集合 | 避免重复灌入（首次加载慢，但只需一次） |
| mem0 存储 | 默认 Qdrant 本地 | 无需额外部署 |

## 注意事项

- 首次加载用户数据较慢（每个 chunk 都需 LLM 提取，thetagen 单用户 ~1000+ chunks），但有磁盘缓存
- mem0 没有原生 batch API，逐条 add 有速率限制风险
- 并发度 `-p` 不宜过高（mem0 单例共享，Qdrant 写入串行），建议 3-5
