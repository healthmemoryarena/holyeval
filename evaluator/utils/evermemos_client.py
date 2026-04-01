"""EverMemOS memory API client.

Lightweight async HTTP client for the local EverMemOS memory service (v1 API).
Only depends on aiohttp (no additional external packages).

仅支持本地部署 (localhost:1995)，无需 API key。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

# Timeout constants
_ADD_TIMEOUT = aiohttp.ClientTimeout(total=30)  # add 可能需要处理时间
_SEARCH_TIMEOUT = aiohttp.ClientTimeout(total=10)
_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10)


class EverMemOSClient:
    """Async client for the EverMemOS memory API (v1)."""

    def __init__(self, base_url: str = "http://localhost:1995"):
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        self._session: aiohttp.ClientSession | None = None

    # -- context manager --

    async def __aenter__(self) -> EverMemOSClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- session management --

    def _get_session(self, timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientSession:
        """Lazy session creation. Recreates if closed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=timeout or _DEFAULT_TIMEOUT,
                headers=self._headers,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- public API --

    async def add(
        self,
        user_id: str,
        content: str,
    ) -> tuple[str | None, str | None]:
        """Add memory content for a user (v1: POST /api/v1/memories).

        Returns:
            (request_id, None) on success, (None, error_message) on failure.
        """
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return None, "Invalid user ID."
        if not content or not isinstance(content, str) or not content.strip():
            return None, "Invalid content."

        user_id = user_id.strip()
        content = content.strip()

        url = f"{self._base_url}/api/v1/memories"
        now = datetime.now(timezone.utc)

        payload = {
            "message_id": f"{user_id}_{int(now.timestamp() * 1e9):x}",
            "create_time": now.isoformat(timespec="seconds"),
            "sender": user_id,
            "content": content,
        }

        try:
            session = self._get_session(_ADD_TIMEOUT)
            async with session.post(url, json=payload) as response:
                text = await response.text()
                if not response.ok:
                    return None, f"HTTP {response.status}: {text}"

                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    return None, text

                # v1 本地有两种成功响应:
                # 1. {"status": "ok", "message": "...", "result": {...}}
                # 2. {"message": "Request accepted, ...", "request_id": "..."}（后台处理）
                status = data.get("status", "")
                if status in ("ok", "queued"):
                    return data.get("request_id", ""), None
                if data.get("request_id"):
                    return data["request_id"], None
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
        """Search memories using a query string (v1: GET /api/v1/memories/search).

        Returns:
            (results_list, None) on success, (None, error_message) on failure.
            results_list contains memory objects from the API response.
        """
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            return None, "Invalid user ID."
        if not query or not isinstance(query, str) or not query.strip():
            return None, "Invalid query."

        user_id = user_id.strip()
        query = query.strip()

        url = f"{self._base_url}/api/v1/memories/search"

        payload: dict = {
            "user_id": user_id,
            "query": query,
            "retrieve_method": "rrf",
        }

        if top_k is not None and 1 <= top_k <= 100:
            payload["top_k"] = top_k

        try:
            session = self._get_session(_SEARCH_TIMEOUT)
            async with session.get(url, json=payload) as response:
                text = await response.text()
                if not response.ok:
                    return None, f"HTTP {response.status}: {text}"

                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    return None, text

                if data.get("status") == "ok":
                    result = data.get("result", {})
                    # v1 memories 按 group_id 分组: [{"gid": [mem1, mem2]}, ...]
                    # 展平为 [mem1, mem2, ...] 供下游使用
                    raw_memories = result.get("memories", [])
                    flat: list[dict] = []
                    for item in raw_memories:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, list):
                                    flat.extend(v)
                                else:
                                    flat.append(item)
                                    break
                        else:
                            flat.append(item)
                    # 兼容 v0 的 profiles
                    profiles = result.get("profiles", [])
                    if profiles:
                        profiles.extend(flat)
                        return profiles, None
                    return flat, None
                else:
                    return None, data.get("message", text)

        except Exception as e:
            return None, str(e)
