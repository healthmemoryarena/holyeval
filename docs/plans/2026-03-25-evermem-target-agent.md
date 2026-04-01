# EverMem Target Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 添加 `evermem` target agent，通过 EverMemOS 远程 API 实现"预灌入 + 检索问答"的 RAG 评测流程。

**Architecture:** 分两部分：(1) 预灌入脚本，复用 `thetagen_chunker` 切分用户数据，通过 HTTP API 批量写入 EverMemOS；(2) target agent 插件，评测时仅调 search API 检索 + LLM 生成答案。user_id 格式为 `holyeval_eslbench_{dir_name}` 防止冲突。EverMemOS HTTP 客户端内联实现，不依赖外部 mirobody 包。

**Tech Stack:** aiohttp (HTTP client), thetagen_chunker (数据切分), langchain + do_execute (LLM 调用), tqdm (进度显示)

---

## Task 1: 实现 EverMemOS HTTP 客户端

**Files:**
- Create: `evaluator/utils/evermemos_client.py`

**Step 1: 创建客户端文件**

```python
"""EverMemOS HTTP 客户端 — 内联实现，不依赖 mirobody

仅实现 add / search 两个 API，供预灌入脚本和 target agent 使用。
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class EverMemOSClient:
    """EverMemOS 远程记忆服务客户端"""

    def __init__(self, api_key: str, base_url: str = "https://api.evermind.ai"):
        self._api_key = api_key.strip() if api_key else ""
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=30)
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout, headers=self._headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "EverMemOSClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def add(
        self,
        user_id: str,
        content: str,
        *,
        flush: bool = True,
    ) -> tuple[str | None, str | None]:
        """写入一条记忆。返回 (request_id, error)"""
        if not self._api_key:
            return None, "Missing EverMemOS API key"

        now = datetime.now(timezone.utc)
        payload = {
            "message_id": f"{user_id}_{int(now.timestamp() * 1e9):x}",
            "create_time": now.isoformat(timespec="seconds"),
            "sender": user_id,
            "content": content,
            "flush": flush,
        }

        try:
            session = self._get_session()
            async with session.post(f"{self._base_url}/api/v0/memories", json=payload) as resp:
                text = await resp.text()
                if not resp.ok:
                    return None, f"HTTP {resp.status}: {text}"
                data = json.loads(text)
                if data.get("status") in ("ok", "queued"):
                    return data.get("request_id", ""), None
                return None, data.get("message", text)
        except Exception as e:
            return None, str(e)

    async def search(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 10,
    ) -> tuple[list[dict] | None, str | None]:
        """检索记忆。返回 (results, error)，results 为 profiles + memories 合并列表"""
        if not self._api_key:
            return None, "Missing EverMemOS API key"

        payload = {
            "user_id": user_id,
            "query": query,
            "retrieve_method": "rrf",
            "top_k": top_k,
        }

        try:
            session = self._get_session()
            async with session.get(f"{self._base_url}/api/v0/memories/search", json=payload) as resp:
                text = await resp.text()
                if not resp.ok:
                    return None, f"HTTP {resp.status}: {text}"
                data = json.loads(text)
                if data.get("status") == "ok":
                    result = data.get("result", {})
                    items = result.get("profiles", [])
                    items.extend(result.get("memories", []))
                    return items, None
                return None, data.get("message", text)
        except Exception as e:
            return None, str(e)
```

**Step 2: 验证导入**

Run: `.venv/bin/python -c "from evaluator.utils.evermemos_client import EverMemOSClient; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add evaluator/utils/evermemos_client.py
git commit -m "feat: add EverMemOS HTTP client (inline, no external deps)"
```

---

## Task 2: 实现预灌入脚本

**Files:**
- Create: `scripts/evermem_preload.py`

**Step 1: 创建预灌入脚本**

```python
"""EverMemOS 预灌入脚本 — 将 thetagen 用户数据批量写入 EverMemOS

用法:
    EVERMEMOS_API_KEY=xxx python scripts/evermem_preload.py [--benchmark eslbench] [--users user110@demo,user111@demo] [--dry-run]

user_id 格式: holyeval_eslbench_{dir_name}（如 holyeval_eslbench_user110_AT_demo）
数据切分: 复用 thetagen_chunker，只灌入语义 chunks（openie_indices 标记的）
幂等性: 磁盘标记文件 benchmark/data/thetagen/.evermem_work_dirs/{dir_name}/.loaded
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from evaluator.utils.evermemos_client import EverMemOSClient
from evaluator.utils.thetagen_chunker import BENCHMARK_DATA_DIR, load_user_documents

logger = logging.getLogger(__name__)

USER_ID_PREFIX = "holyeval_eslbench"
DATA_GROUP = "thetagen"
MARKER_BASE = BENCHMARK_DATA_DIR / DATA_GROUP / ".evermem_work_dirs"


def make_user_id(dir_name: str) -> str:
    return f"{USER_ID_PREFIX}_{dir_name}"


async def preload_one_user(
    client: EverMemOSClient,
    user_dir: Path,
    max_chunk_chars: int = 15000,
    dry_run: bool = False,
) -> tuple[str, int, str | None]:
    """灌入单个用户，返回 (dir_name, chunk_count, error)"""
    dir_name = user_dir.name
    evermem_user_id = make_user_id(dir_name)

    # 加载 & 切分
    all_chunks, openie_indices = load_user_documents(user_dir, max_chunk_chars)
    semantic_chunks = [all_chunks[i] for i in sorted(openie_indices) if i < len(all_chunks)]

    if not semantic_chunks:
        return dir_name, 0, None

    data_hash = hashlib.md5("".join(semantic_chunks).encode()).hexdigest()[:12]

    # 检查磁盘标记（幂等）
    marker_dir = MARKER_BASE / dir_name
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_file = marker_dir / ".loaded"

    if marker_file.exists():
        try:
            meta = json.loads(marker_file.read_text())
            if meta.get("data_hash") == data_hash and meta.get("chunk_count") == len(semantic_chunks):
                return dir_name, 0, None  # 已加载
        except (json.JSONDecodeError, OSError):
            pass

    if dry_run:
        return dir_name, len(semantic_chunks), None

    # 逐条灌入（最后一条 flush=True，其余 flush=False 提高吞吐）
    count = 0
    for i, chunk in enumerate(semantic_chunks):
        is_last = i == len(semantic_chunks) - 1
        req_id, err = await client.add(evermem_user_id, chunk, flush=is_last)
        if err:
            logger.warning("[evermem] %s chunk %d/%d 写入失败: %s", dir_name, i + 1, len(semantic_chunks), err)
        else:
            count += 1

    # 写标记
    marker_file.write_text(json.dumps({
        "data_hash": data_hash,
        "chunk_count": len(semantic_chunks),
        "written_count": count,
        "user_id": evermem_user_id,
    }))

    return dir_name, count, None


async def main() -> None:
    parser = argparse.ArgumentParser(description="EverMemOS 预灌入")
    parser.add_argument("--benchmark", default="eslbench", help="benchmark 名称")
    parser.add_argument("--users", default=None, help="指定用户（逗号分隔邮箱），默认全部")
    parser.add_argument("--max-chunk-chars", type=int, default=15000)
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不实际写入")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    api_key = os.environ.get("EVERMEMOS_API_KEY", "")
    if not api_key and not args.dry_run:
        print("错误: 请设置 EVERMEMOS_API_KEY 环境变量")
        return

    # 发现用户目录
    data_dir = BENCHMARK_DATA_DIR / DATA_GROUP / ".data"
    if not data_dir.is_dir():
        print(f"错误: 数据目录不存在 {data_dir}")
        return

    if args.users:
        from evaluator.utils.thetagen_chunker import email_to_dir
        dir_names = [email_to_dir(e.strip()) for e in args.users.split(",")]
        user_dirs = [data_dir / d for d in dir_names if (data_dir / d).is_dir()]
    else:
        user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    print(f"{'[DRY RUN] ' if args.dry_run else ''}准备灌入 {len(user_dirs)} 个用户到 EverMemOS")
    print(f"user_id 前缀: {USER_ID_PREFIX}_")

    async with EverMemOSClient(api_key) as client:
        tasks = [
            preload_one_user(client, ud, args.max_chunk_chars, args.dry_run)
            for ud in user_dirs
        ]
        results = []
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="灌入进度"):
            results.append(await coro)

    total_chunks = sum(c for _, c, _ in results)
    loaded = sum(1 for _, c, _ in results if c > 0)
    skipped = len(results) - loaded
    print(f"\n完成: {loaded} 用户写入 {total_chunks} chunks, {skipped} 用户跳过（已缓存）")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: dry-run 验证**

Run: `.venv/bin/python scripts/evermem_preload.py --dry-run`
Expected: 输出用户数和 chunk 统计，不实际写入

**Step 3: Commit**

```bash
git add scripts/evermem_preload.py
git commit -m "feat: add EverMemOS preload script for eslbench"
```

---

## Task 3: 实现 evermem target agent 插件

**Files:**
- Create: `evaluator/plugin/target_agent/evermem_target_agent.py`
- Modify: `evaluator/plugin/target_agent/__init__.py`

**Step 1: 创建 target agent**

核心逻辑: 纯检索模式，不做数据灌入。
- 从 `target_overrides` 拿到 `user_email` → 转为 `holyeval_eslbench_{dir_name}`
- 调 `EverMemOSClient.search()` 检索
- 拼 RAG prompt + few-shot → 调 LLM 生成答案
- 复用 `hippo_rag_api` 的 `_RAG_SYSTEM_PROMPT` 和 few-shot 示例

```python
"""
EvermemTargetAgent — EverMemOS 记忆检索问答（预灌入模式）

注册名称: "evermem"

流程（评测时）:
1. user_email → holyeval_eslbench_{dir_name} 映射
2. EverMemOSClient.search(user_id, query, top_k) 检索相关记忆
3. 拼接 RAG prompt + few-shot → LLM 生成 Thought + Answer

预灌入: 通过 scripts/evermem_preload.py 提前完成，本 agent 不负责数据写入。

配置:
    model: gemini-3-flash-preview（生成模型）
    top_k: 10（检索条数）
    evermemos_api_key: 从环境变量 EVERMEMOS_API_KEY 读取
"""

import logging
import os
from typing import Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import TargetAgentReaction, TestAgentAction
from evaluator.utils.evermemos_client import EverMemOSClient
from evaluator.utils.llm import BasicMessage, do_execute
from evaluator.utils.thetagen_chunker import email_to_dir

logger = logging.getLogger(__name__)

# 复用 HippoRAG 的 RAG prompt 和 few-shot
from evaluator.plugin.target_agent.hippo_rag_api_target_agent import (
    _RAG_SYSTEM_PROMPT,
    _SHOT_LIST_INPUT,
    _SHOT_LIST_OUTPUT,
    _SHOT_NUMERIC_INPUT,
    _SHOT_NUMERIC_OUTPUT,
)

USER_ID_PREFIX = "holyeval_eslbench"

# 全局客户端单例
_CLIENT: EverMemOSClient | None = None


def _get_client() -> EverMemOSClient:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("EVERMEMOS_API_KEY", "")
        if not api_key:
            raise RuntimeError("EVERMEMOS_API_KEY 环境变量未设置")
        _CLIENT = EverMemOSClient(api_key)
    return _CLIENT


def _build_rag_input(query: str, memories: list[dict]) -> str:
    if not memories:
        context = "(No relevant memories found)"
    else:
        parts = []
        for m in memories:
            # search 返回的每条记忆可能有 memory / content / text 等字段
            text = m.get("memory") or m.get("content") or m.get("text") or str(m)
            parts.append(text)
        context = "\n\n".join(parts)
    return f"{context}\n\nQuestion: {query}\nThought: "


# ============================================================
# 配置模型
# ============================================================

class EvermemTargetInfo(BaseModel):
    """EverMemOS 记忆检索被测目标配置"""

    model_config = ConfigDict(extra="forbid")
    type: Literal["evermem"] = Field(description="目标类型")
    model: Literal[
        "gpt-5.4", "gpt-5.2", "gpt-4.1",
        "gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview", "gemini-3-flash-preview",
        "anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6",
        "minimax/minimax-m2.5", "z-ai/glm-5",
    ] = Field(description="生成模型名称")
    user_email: Optional[str] = Field(None, description="用户邮箱（映射为 evermem user_id）")
    top_k: int = Field(10, description="检索记忆条数", ge=1, le=100)
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词")


# ============================================================
# Agent 实现
# ============================================================

class EvermemTargetAgent(AbstractTargetAgent, name="evermem", params_model=EvermemTargetInfo):
    """EverMemOS 记忆检索问答（预灌入模式）"""

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125"
            " 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0"
            " 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
        ),
        "color": "#8b5cf6",
        "features": ["RAG", "EverMemOS", "预灌入"],
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(self, target_config: EvermemTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: EvermemTargetInfo = target_config

        # 对话历史
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # 映射 user_id
        self._evermem_user_id: str | None = None
        if target_config.user_email:
            dir_name = email_to_dir(target_config.user_email)
            self._evermem_user_id = f"{USER_ID_PREFIX}_{dir_name}"

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
            return TargetAgentReaction(type="message", message_list=[{"content": "你好，有什么可以帮你的？"}])

        user_text = self._extract_user_input(test_action)

        if not self._evermem_user_id:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "错误: 未配置 user_email，无法映射 EverMemOS user_id"}],
            )

        # 1. 检索
        client = _get_client()
        memories, err = await client.search(self._evermem_user_id, user_text, top_k=self.config.top_k)
        if err:
            logger.warning("[evermem] search 失败: %s", err)
            memories = []

        logger.debug(
            "[evermem] Query: %s, Retrieved %d memories, calling %s",
            user_text[:80], len(memories) if memories else 0, self.config.model,
        )

        # 2. RAG prompt + few-shot
        rag_input = _build_rag_input(user_text, memories or [])
        system_prompt = self.config.system_prompt or _RAG_SYSTEM_PROMPT

        history_with_oneshot = [
            BasicMessage(role="user", content=_SHOT_NUMERIC_INPUT),
            BasicMessage(role="assistant", content=_SHOT_NUMERIC_OUTPUT),
            BasicMessage(role="user", content=_SHOT_LIST_INPUT),
            BasicMessage(role="assistant", content=_SHOT_LIST_OUTPUT),
        ]
        if self._conversation:
            history_with_oneshot.extend(self._conversation)

        # 3. LLM 生成
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

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        return str(test_action.custom_content) if test_action.custom_content else ""
```

**Step 2: 注册插件到 `__init__.py`**

在 `evaluator/plugin/target_agent/__init__.py` 末尾添加:

```python
try:
    from evaluator.plugin.target_agent.evermem_target_agent import EvermemTargetAgent  # noqa: F401

    __all__.append("EvermemTargetAgent")
except ImportError:
    pass
```

**Step 3: 验证导入注册**

Run: `.venv/bin/python -c "from evaluator.plugin.target_agent import *; from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; print(AbstractTargetAgent.has('evermem'))"`
Expected: `True`

**Step 4: Commit**

```bash
git add evaluator/plugin/target_agent/evermem_target_agent.py evaluator/plugin/target_agent/__init__.py
git commit -m "feat: add evermem target agent plugin (search-only, preload mode)"
```

---

## Task 4: 注册到 ESLBench metadata.json

**Files:**
- Modify: `benchmark/data/eslbench/metadata.json`

**Step 1: 在 `target` 数组中添加 evermem 条目**

```json
{
    "type": "evermem",
    "fields": {
        "model": {
            "default": "gemini-3-flash-preview",
            "editable": true,
            "required": true
        },
        "user_email": {
            "default": null,
            "editable": false
        },
        "top_k": {
            "default": 10,
            "editable": true,
            "required": false
        }
    }
}
```

**Step 2: 在 eslbench.jsonl 的 target_overrides 中添加 evermem 映射**

每条 case 的 `user.target_overrides` 需要包含 `evermem` key（与其他 RAG agent 对齐）:

```json
"evermem": {
    "user_email": "user113@demo"
}
```

如果所有 case 已经有 `hippo_rag_api.user_email` 或 `mem0_rag_api.user_email`，则用脚本批量添加（值相同）。

**Step 3: Commit**

```bash
git add benchmark/data/eslbench/metadata.json benchmark/data/eslbench/eslbench.jsonl
git commit -m "feat: register evermem target in eslbench metadata + target_overrides"
```

---

## Task 5: 添加 EVERMEMOS_API_KEY 到环境变量配置

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md` (Environment Variables 表格)

**Step 1: 在 `.env.example` 中添加**

```
EVERMEMOS_API_KEY=xxx
```

**Step 2: 在 CLAUDE.md 环境变量表格中添加**

| `EVERMEMOS_API_KEY` | evermem 评测必填 | EverMemOS API 密钥（预灌入 + target agent 检索） |

**Step 3: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: add EVERMEMOS_API_KEY to env config"
```

---

## Task 6: 端到端验证

**Step 1: 预灌入 dry-run**

Run: `.venv/bin/python scripts/evermem_preload.py --dry-run`
Expected: 输出用户数和 chunk 统计

**Step 2: 实际预灌入（需要 API key）**

Run: `EVERMEMOS_API_KEY=xxx .venv/bin/python scripts/evermem_preload.py --users user110@demo -v`
Expected: 1 个用户灌入成功

**Step 3: 单条评测**

Run: `.venv/bin/python -m benchmark.basic_runner eslbench eslbench --target-type evermem --target-model gemini-3-flash-preview --limit 1 -v`
Expected: 成功完成 1 条评测，输出包含 evermem search 日志

**Step 4: Commit（如有修复）**

```bash
git commit -m "fix: evermem end-to-end adjustments"
```
