# 量化交易 Claude Code Harness 4级约束框架（教学演示）

> 日期：2026-04-07
> 用途：教学演示，展示如何在量化交易场景下分层配置 Claude Code harness

---

## 架构总览

```
优先级（高→低）：Managed > Project > User > Local

┌─────────────────────────────────────────────────────────────┐
│  L1 Managed（组织强制）                                       │
│  位置：/etc/claude/settings.json 或 MDM 部署                  │
│  谁管：合规/IT 团队        不可覆盖：✓                         │
│  职责：安全红线、合规底线、审计要求                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  L2 Project（项目团队）                                   │ │
│  │  位置：.claude/settings.json                             │ │
│  │  谁管：Tech Lead        提交 git：✓                      │ │
│  │  职责：项目工具链、Hook自动化、团队权限                      │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  L3 User（个人全局）                                  │ │ │
│  │  │  位置：~/.claude/settings.json                       │ │ │
│  │  │  谁管：开发者本人      跨项目：✓                       │ │ │
│  │  │  职责：个人偏好、全局快捷方式、通用工具权限               │ │ │
│  │  │  ┌─────────────────────────────────────────────────┐ │ │ │
│  │  │  │  L4 Local（个人项目私有）                          │ │ │ │
│  │  │  │  位置：.claude/settings.local.json               │ │ │ │
│  │  │  │  谁管：开发者本人    gitignore：✓                  │ │ │ │
│  │  │  │  职责：本地API密钥、调试开关、实验性配置              │ │ │ │
│  │  │  └─────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 冲突解决规则

- **deny 优先于 allow**：任何层级的 deny 都不可被低层级 allow 覆盖
- **高层级 deny 不可撤销**：Managed deny 的规则，Project/User/Local 无法 allow
- **allow 可累加**：低层级可以 allow 高层级未提及的权限
- **env 变量**：低层级覆盖高层级同名变量

---

## L1 Managed — 组织强制层

> 文件：`/etc/claude/settings.json`（由 IT/合规团队通过 MDM 部署，开发者无法修改）

```json
{
  "permissions": {
    "deny": [
      "Bash(curl *://*/api/v*/orders*)",
      "Bash(curl *://*/api/v*/withdraw*)",
      "Bash(*scp*)",
      "Bash(*rsync*external*)",
      "Bash(rm -rf /)",
      "Bash(sudo *)",
      "Bash(chmod 777 *)",
      "Edit(*.pem)",
      "Edit(*.key)",
      "Edit(*credentials*)",
      "Edit(*secrets*)",
      "Write(*.pem)",
      "Write(*.key)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "/opt/company/claude-hooks/compliance-check.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "/opt/company/claude-hooks/audit-logger.sh"
          }
        ]
      }
    ]
  }
}
```

**设计意图：**
- **deny 规则不可覆盖**：禁止直接调用交易/提款 API、传输文件到外部、编辑密钥文件
- **合规检查 Hook**：每次工具调用前检查是否触碰敏感操作
- **审计日志 Hook**：每次工具调用后记录到合规审计系统
- 这些规则即使开发者在 Project/User/Local 层 allow 也无法绕过

---

## L2 Project — 项目团队层

> 文件：`.claude/settings.json`（提交到 git，团队共享）

```json
{
  "permissions": {
    "allow": [
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(git *)",
      "Bash(ruff *)",
      "Bash(mypy *)",
      "Bash(docker compose *)",
      "Bash(redis-cli *)",
      "Read",
      "Edit",
      "Glob",
      "Grep",
      "WebFetch"
    ],
    "deny": [
      "Bash(git push * main)",
      "Bash(git push * master)",
      "Bash(git push --force *)",
      "Bash(docker push *)",
      "Bash(pip install *)",
      "Edit(.env*)",
      "Edit(config/prod_*)",
      "Edit(strategies/*_live.py)"
    ]
  },
  "env": {
    "PYTHONPATH": "${CLAUDE_PROJECT_DIR}",
    "TRADING_ENV": "backtest",
    "LOG_LEVEL": "INFO"
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/block-secrets.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/auto-format.sh"
          }
        ]
      },
      {
        "matcher": "Bash|Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/audit-log.sh"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "echo '=== QuantX Trading Platform ==='; echo 'ENV: backtest (safe mode)'; echo 'Live trading files are READ-ONLY'; git log --oneline -3"
          }
        ]
      }
    ]
  }
}
```

**设计意图：**
- **allow 白名单**：只允许项目必需的工具（python/pytest/git/docker compose 等）
- **deny 保护**：禁止直推 main、force push、编辑 .env/生产配置/实盘策略文件
- **环境锁定**：`TRADING_ENV=backtest` 确保 Claude 操作始终在回测环境
- **Hook 自动化**：密钥泄露阻断 + 代码自动格式化 + 操作审计
- **会话提示**：启动时明确告知当前环境和限制

---

## L3 User — 个人全局层

> 文件：`~/.claude/settings.json`（个人所有项目通用）

```json
{
  "permissions": {
    "allow": [
      "Bash(top:*)",
      "Bash(htop)",
      "Bash(df *)",
      "Bash(free *)",
      "Bash(nvidia-smi)",
      "Bash(tmux *)",
      "WebSearch"
    ]
  },
  "env": {
    "EDITOR": "vim",
    "LANG": "zh_CN.UTF-8"
  },
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' '需要你的注意' 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

**设计意图：**
- **通用工具权限**：系统监控命令（top/htop/df/nvidia-smi）对所有项目都安全
- **个人偏好**：编辑器、语言等不因项目而异的配置
- **通知 Hook**：Claude 等待输入时弹桌面通知，跨项目生效
- 注意：这里 allow 的权限如果被 L1/L2 deny，仍然会被阻止

---

## L4 Local — 个人项目私有层

> 文件：`.claude/settings.local.json`（gitignore，不提交）

```json
{
  "env": {
    "OPENAI_API_KEY": "sk-proj-xxxxx",
    "BINANCE_API_KEY": "abc123_testnet_only",
    "BINANCE_SECRET_KEY": "def456_testnet_only",
    "REDIS_URL": "redis://localhost:6379/0",
    "TRADING_ENV": "backtest",
    "LOG_LEVEL": "DEBUG",
    "BACKTEST_DATA_DIR": "/data/ssd/market-data"
  }
}
```

**设计意图：**
- **API 密钥**：个人的 API Key 绝不提交到 git
- **本地覆盖**：`LOG_LEVEL=DEBUG` 覆盖 L2 的 `INFO`（方便调试）
- **本地路径**：指向本地 SSD 上的行情数据目录
- **最低优先级**：如果与 L1/L2 冲突，高层级规则胜出

---

## CLAUDE.md 指令文件

### 项目根 CLAUDE.md

```markdown
# QuantX Trading Platform

量化交易策略研发平台。

## 项目结构
- `strategies/` — 策略代码（*_live.py 为实盘，*_bt.py 为回测）
- `data/` — 行情数据管道
- `backtest/` — 回测引擎
- `risk/` — 风控模块
- `infra/` — 基础设施（Docker, K8s）

## 关键命令
python -m pytest tests/ -v                    # 全量测试
python -m pytest tests/test_strategy.py -k "test_macd"  # 单策略测试
python -m backtest.run --strategy macd --period 2024    # 回测
ruff check . && mypy strategies/              # 代码质量

## 安全红线
- 永远不要在代码中硬编码 API Key
- 永远不要修改 *_live.py 文件（实盘策略需走 PR 审批）
- 永远不要直接调用交易所 API（使用 backtest mock）
- 所有策略变更必须附带回测报告

## 导入
@.claude/rules/strategy-code.md
@.claude/rules/data-pipeline.md
@.claude/rules/testing.md
```

### .claude/CLAUDE.md（补充指令）

```markdown
# 项目补充指南

## 工具链
- Python 3.11+, uv 包管理
- Redis 缓存行情数据
- ClickHouse 存储回测结果
- Docker Compose 本地开发环境

## 代码规范
- Ruff 格式化，行长 120
- mypy strict 模式
- 策略类必须继承 BaseStrategy
- 所有金额用 Decimal，禁止 float

## Git 工作流
- 功能分支：`feat/策略名-描述`
- 回测分支：`bt/策略名-参数`
- PR 必须包含回测报告截图
```

---

## .claude/rules/ 按路径生效的规则

### strategy-code.md

```markdown
---
paths:
  - "strategies/**/*.py"
---

# 策略代码规范

## 必须遵守
- 所有策略继承 `BaseStrategy`，实现 `on_bar()` 和 `on_tick()` 方法
- 金额和价格使用 `Decimal` 类型，禁止 `float`（浮点精度问题会导致真金白银损失）
- 每个策略必须定义 `MAX_POSITION_SIZE` 和 `STOP_LOSS_PCT` 常量
- 日志使用 `self.logger`，不要用 `print()`

## 禁止操作
- 禁止在策略代码中直接 `import requests`（所有外部调用走 DataProvider 接口）
- 禁止硬编码交易对（使用 `self.config.symbols`）
- 禁止在 `on_bar()` 中做 IO 操作（会阻塞事件循环）

## 命名约定
- 策略类：`PascalCase` + `Strategy` 后缀（如 `MACDStrategy`）
- 信号方法：`signal_` 前缀（如 `signal_entry()`、`signal_exit()`）
- 指标方法：`calc_` 前缀（如 `calc_rsi()`、`calc_macd()`）
```

### data-pipeline.md

```markdown
---
paths:
  - "data/**/*.py"
  - "data/**/*.sql"
---

# 数据管道规范

## 数据源
- 行情数据：Binance/OKX WebSocket → Redis → ClickHouse
- 基本面数据：定时任务拉取 → PostgreSQL

## 必须遵守
- 所有时间戳统一使用 UTC（`datetime.utcnow()`）
- DataFrame 索引必须是 `DatetimeIndex`，timezone-aware
- 大数据集使用 `polars` 而非 `pandas`（10x 性能差距）
- SQL 查询必须参数化，禁止字符串拼接

## 数据质量
- 入库前检查：空值率 < 1%，时间连续性，价格合理范围
- 异常数据标记 `is_anomaly=True`，不删除原始数据
```

### testing.md

```markdown
---
paths:
  - "tests/**/*.py"
---

# 测试规范

## 分层测试
- `tests/unit/` — 纯逻辑测试（无 IO），< 1s
- `tests/integration/` — 含 Redis/DB 交互，< 10s
- `tests/backtest/` — 完整回测，可能 > 1min

## 策略测试必须包含
1. **信号正确性**：给定历史数据，验证买卖信号位置
2. **边界条件**：空数据、单条数据、极端价格
3. **风控触发**：止损、最大持仓、每日亏损限额
4. **回测指标**：Sharpe > 1.0, MaxDrawdown < 20%, WinRate > 45%

## 禁止
- 禁止 mock 交易所 API 返回值（使用 `fixtures/` 下的真实历史数据）
- 禁止 `time.sleep()` 在测试中
- 禁止跳过失败的测试（`@pytest.mark.skip` 需注明原因和 issue 编号）
```

---

## Hook 脚本

### .claude/hooks/block-secrets.sh

```bash
#!/bin/bash
# L2 Hook: 阻止在代码中写入密钥/敏感信息
# 触发：PreToolUse (Bash|Edit|Write)
# 退出码：0=放行, 2=阻止

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

# 检查 Edit/Write 的文件内容是否包含密钥模式
if [[ "$TOOL" == "Edit" || "$TOOL" == "Write" ]]; then
  CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

  SECRET_PATTERNS=(
    'sk-[a-zA-Z0-9]{20,}'           # OpenAI Key
    'AKIA[0-9A-Z]{16}'              # AWS Access Key
    '[a-f0-9]{64}'                   # 64位 hex（可能是 Secret Key）
    'ghp_[a-zA-Z0-9]{36}'           # GitHub Token
    'password\s*=\s*["\x27][^"\x27]+["\x27]'  # 硬编码密码
  )

  for pattern in "${SECRET_PATTERNS[@]}"; do
    if echo "$CONTENT" | grep -qP "$pattern"; then
      echo "BLOCKED: 检测到疑似密钥/敏感信息，禁止写入代码" >&2
      echo "匹配模式: $pattern" >&2
      echo "请使用环境变量或 .env 文件管理密钥" >&2
      exit 2
    fi
  done
fi

# 检查 Bash 命令是否在打印密钥
if [[ "$TOOL" == "Bash" ]]; then
  CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
  if echo "$CMD" | grep -qiP '(echo|cat|print).*(_KEY|_SECRET|_TOKEN|PASSWORD)'; then
    echo "BLOCKED: 禁止在终端输出密钥/敏感信息" >&2
    exit 2
  fi
fi

exit 0
```

### .claude/hooks/audit-log.sh

```bash
#!/bin/bash
# L2 Hook: 记录所有工具调用到审计日志
# 触发：PostToolUse (Bash|Edit|Write)

INPUT=$(cat)
LOG_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/audit"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOOL=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
SESSION=${CLAUDE_SESSION_ID:-"unknown"}

# 提取关键信息（不记录完整内容，避免日志膨胀）
case "$TOOL" in
  Bash)
    DETAIL=$(echo "$INPUT" | jq -r '.tool_input.command // empty' | head -c 200)
    ;;
  Edit)
    DETAIL=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    ;;
  Write)
    DETAIL=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    ;;
  *)
    DETAIL="N/A"
    ;;
esac

# 追加到日志（JSONL 格式，方便后续分析）
echo "{\"ts\":\"$TIMESTAMP\",\"session\":\"$SESSION\",\"tool\":\"$TOOL\",\"detail\":\"$DETAIL\"}" \
  >> "$LOG_DIR/$(date -u +%Y-%m-%d).jsonl"

exit 0
```

### .claude/hooks/auto-format.sh

```bash
#!/bin/bash
# L2 Hook: 编辑 Python 文件后自动格式化
# 触发：PostToolUse (Edit|Write)

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# 只处理 Python 文件
if [[ "$FILE_PATH" == *.py ]]; then
  if command -v ruff &> /dev/null; then
    ruff format "$FILE_PATH" 2>/dev/null || true
    ruff check --fix "$FILE_PATH" 2>/dev/null || true
  fi
fi

exit 0
```

---

## Skills

### .claude/skills/backtest/SKILL.md

```markdown
---
name: backtest
description: 运行策略回测并生成报告。当用户说"回测"、"backtest"、"跑一下策略"时使用。
---

## 回测流程

1. 确认参数：
   - 策略名称（必填）
   - 时间范围（默认最近 1 年）
   - 交易对（默认 BTC/USDT）
   - 初始资金（默认 100,000 USDT）

2. 运行回测：
   python -m backtest.run \
     --strategy {策略名} \
     --start {开始日期} \
     --end {结束日期} \
     --symbol {交易对} \
     --capital {初始资金}

3. 检查关键指标（不达标则警告）：
   - Sharpe Ratio > 1.0
   - Max Drawdown < 20%
   - Win Rate > 45%
   - Profit Factor > 1.5

4. 生成报告到 `reports/backtest/{策略名}_{日期}.html`

5. 如果指标不达标，分析原因并建议优化方向
```

### .claude/skills/deploy-strategy/SKILL.md

```markdown
---
name: deploy-strategy
description: 将回测通过的策略部署到模拟盘。当用户说"部署策略"、"上模拟盘"时使用。
---

## 部署检查清单（必须全部通过）

1. **回测报告存在**：`reports/backtest/{策略名}_*.html`
2. **指标达标**：Sharpe > 1.0, MaxDD < 20%
3. **代码审查**：无硬编码参数，有完整注释
4. **测试通过**：`pytest tests/ -k {策略名}` 全绿
5. **风控配置**：`MAX_POSITION_SIZE` 和 `STOP_LOSS_PCT` 已设置

## 部署步骤

1. 创建部署分支：`git checkout -b deploy/{策略名}`
2. 生成配置：`python -m infra.gen_config --strategy {策略名} --env paper`
3. 构建镜像：`docker compose build strategy-{策略名}`
4. 推送到模拟环境：需要用户手动确认后执行
5. 监控 5 分钟：检查日志无报错、持仓合理

## 禁止
- 禁止直接部署到实盘（`--env live` 会被 Managed 层 Hook 阻断）
- 禁止跳过回测报告检查
```

---

## 个人全局 CLAUDE.md

> 文件：`~/.claude/CLAUDE.md`

```markdown
# 个人偏好

## 响应风格
- 简洁直接，不要废话
- 代码注释用中文
- 错误信息给出修复建议，不只是报错

## 常用工作流
- 新策略：先写测试 → 再写策略 → 最后回测
- Debug：先看日志 → 再加断点 → 最后改代码
- 提交：ruff format → pytest → git commit

## 快捷操作
- "跑测试" = pytest tests/ -v --tb=short
- "查 GPU" = nvidia-smi
- "查内存" = free -h && df -h /data
```

---

## 4级约束冲突解决示例

```
场景：开发者想编辑 .env 文件添加新的 API Key

L4 Local:  （未设置相关规则）     → 无意见
L3 User:   （未设置相关规则）     → 无意见
L2 Project: deny["Edit(.env*)"]  → ❌ 阻止
L1 Managed: （未设置相关规则）     → 无意见

结果：被 L2 阻止。即使 L4 尝试 allow 也无法覆盖。
开发者需要手动编辑 .env 文件。

---

场景：开发者想 force push 到远程

L4 Local:  （未设置相关规则）              → 无意见
L3 User:   allow["Bash(git *)"]          → ✅ 允许 git 命令
L2 Project: deny["Bash(git push --force *)"] → ❌ 阻止 force push
L1 Managed: （未设置相关规则）              → 无意见

结果：被 L2 阻止。L3 允许了 git 命令，但 deny 优先于 allow。

---

场景：开发者想直接调用币安下单 API

L4 Local:  （未设置相关规则）                        → 无意见
L3 User:   （未设置相关规则）                        → 无意见
L2 Project: （未设置相关规则）                       → 无意见
L1 Managed: deny["Bash(curl *://*/api/v*/orders*)"] → ❌ 阻止

结果：被 L1 强制阻止。任何层级都无法覆盖 Managed 的 deny 规则。
这是合规团队设置的安全红线。
```

---

## 文件清单

| 文件 | 层级 | 提交 git | 说明 |
|------|------|---------|------|
| `managed/settings.json` | L1 | N/A (IT部署) | 组织安全红线 |
| `.claude/settings.json` | L2 | ✅ | 项目权限+Hook |
| `~/.claude/settings.json` | L3 | ❌ | 个人通用权限 |
| `.claude/settings.local.json` | L4 | ❌ (.gitignore) | 本地密钥+调试 |
| `CLAUDE.md` | — | ✅ | 项目主指令 |
| `.claude/CLAUDE.md` | — | ✅ | 项目补充指令 |
| `~/.claude/CLAUDE.md` | — | ❌ | 个人全局指令 |
| `.claude/rules/*.md` | — | ✅ | 按路径生效的规则 |
| `.claude/hooks/*.sh` | — | ✅ | Hook 脚本 |
| `.claude/skills/*/SKILL.md` | — | ✅ | 可复用技能 |
