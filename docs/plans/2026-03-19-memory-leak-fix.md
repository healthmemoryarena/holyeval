# Memory Leak Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix web service memory leaks — completed tasks retain heavy objects indefinitely, JSON cache unbounded by size.

**Architecture:** Two targeted fixes: (1) TaskManager 完成任务后释放重对象（session/对话历史/eval_results），只保留轻量元数据 + report_path 供 API 回退读文件；(2) retrieve.py JSON 缓存增加总大小上限。

**Tech Stack:** Python, asyncio, dataclass

---

### Task 1: TaskManager — 完成任务后自动释放重对象

**Files:**
- Modify: `web/app/services/task_manager.py`

**分析:**
- `_tasks` dict 永不清理，completed 的 `TaskEntry` 保留 `session`（含所有 CaseContext + Agent + memory_list）、`eval_results`、`resumed_results`
- API 端已有 fallback：`get_task_results` 在 `entry.report_path` 存在时从文件读取；`get_snapshot` 在 `entry.report_path` 存在时从文件读 stats_by_tag
- 因此：任务完成后释放 session / eval_results / resumed_results，保留元数据即可

**Step 1: 在 TaskEntry 上添加 `_release_heavy_data()` 方法**

在 `TaskEntry` dataclass 后面添加方法：

```python
def _release_heavy_data(self) -> None:
    """释放重对象，仅保留轻量元数据供 API 查询"""
    self.session = None
    self.eval_results = []
    self.resumed_results = []
    self.asyncio_task = None
```

**Step 2: 在 `_run_session` 完成后调用释放**

在 `_run_session` 中，`entry.status = "completed"` 和 `entry.status = "cancelled"` 之后，以及 `except` 块中 `entry.status = "error"` 之后，都调用 `entry._release_heavy_data()`。

具体位置（3 处）：

1. completed 分支（`mgr.cleanup()` 之后）:
```python
                entry.status = "completed"
                mgr.cleanup()
            entry._release_heavy_data()  # 新增：释放重对象
```
注意：放在 if/else 之后（即无论 cancelled 还是 completed 都释放）。

2. except 块：
```python
            entry.status = "error"
            entry.error = str(e)
            entry._release_heavy_data()  # 新增
```

**Step 3: 在 `_run_eval_session` 完成后调用释放**

在 `_run_eval_session` 中：

1. completed（`entry.status = "completed"` 之后）:
```python
            entry.status = "completed"
            entry._release_heavy_data()  # 新增
```

2. except 块：
```python
            entry.status = "error"
            entry.error = str(e)
            entry._release_heavy_data()  # 新增
```

**Step 4: 修复 `get_task_results` 中对已释放数据的处理**

`get_task_results` 当前逻辑已有 `entry.report_path` fallback，但中间分支 `entry.session and entry.status == "completed"` 在 session 被释放后不会命中，会自然 fall through 到 report_path 分支。无需修改。

`get_snapshot` 同理 — `entry.session is None` 时走 eval-only 分支，但 eval-only 分支依赖 `entry.eval_results`（已清空）和 `entry.report_path`（保留）。需确认 eval-only 完成后 `report_path` 已设置。检查代码：`_run_eval_session` 中 `entry.report_path = str(report_path)` 在 `entry.status = "completed"` 之前，✅ OK。

**Step 5: 修复 `list_tasks` 中的安全访问**

`list_tasks` API 中 `e.session.total` / `e.session.completed` 在 session 被释放后会 NPE。当前已有 `if e.session` 守卫，✅ OK。

**Step 6: 修复 `get_case_result` 中的安全访问**

`get_case_result` 在 `entry.session.contexts.get(case_id)` 处，session 被释放后会 NPE。需加守卫：

```python
    if not entry.session:
        raise HTTPException(status_code=410, detail="任务已完成，请从报告中查看结果")
```

**Step 7: 验证**

Run: `ruff check web/app/services/task_manager.py web/app/api/tasks.py`

---

### Task 2: retrieve.py JSON 缓存 — 增加总大小上限

**Files:**
- Modify: `benchmark/data/thetagen/tools/retrieve.py`

**分析:**
- `_JSON_CACHE` 是 OrderedDict，LRU 上限 6 个文件
- timeline.json 单文件 15-50MB，6 个 = 最高 300MB
- 改进：减少条目数 + 增加总大小估算上限

**Step 1: 添加大小限制常量和追踪变量**

```python
_MAX_JSON_LRU = 4          # 从 6 降到 4
_MAX_JSON_CACHE_BYTES = 64 * 1024 * 1024  # 64MB 总上限

_JSON_CACHE: OrderedDict[str, tuple[int, int, Any]] = OrderedDict()
_json_cache_bytes: int = 0  # 当前缓存的总文件大小（近似）
```

**Step 2: 修改 `_load_json_cached` 函数**

```python
def _load_json_cached(fp: Path) -> Any:
    """加载并缓存 JSON（按文件 mtime/size 自动失效，含总大小上限）。"""
    global _json_cache_bytes
    stat = fp.stat()
    cache_key = str(fp)
    cached = _JSON_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        _JSON_CACHE.move_to_end(cache_key)
        return cached[2]

    # 缓存未命中 — 先驱逐旧条目（如果 key 已存在则先减去旧 size）
    if cache_key in _JSON_CACHE:
        _json_cache_bytes -= _JSON_CACHE[cache_key][1]
        del _JSON_CACHE[cache_key]

    raw = fp.read_bytes()
    data = _loads_json(raw)
    _JSON_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, data)
    _json_cache_bytes += stat.st_size
    _JSON_CACHE.move_to_end(cache_key)

    # 驱逐：条目数超限 或 总大小超限
    while len(_JSON_CACHE) > _MAX_JSON_LRU or _json_cache_bytes > _MAX_JSON_CACHE_BYTES:
        if not _JSON_CACHE:
            break
        _, (_, old_size, _) = _JSON_CACHE.popitem(last=False)
        _json_cache_bytes -= old_size

    return data
```

**Step 3: 验证**

Run: `ruff check benchmark/data/thetagen/tools/retrieve.py`

---

### Task 3: 验证 + 提交

**Step 1:** `ruff check web/app/services/task_manager.py web/app/api/tasks.py benchmark/data/thetagen/tools/retrieve.py`
**Step 2:** `pytest evaluator/tests/ -x -q`（确保不破坏现有测试）
**Step 3:** 重启 web 服务验证
