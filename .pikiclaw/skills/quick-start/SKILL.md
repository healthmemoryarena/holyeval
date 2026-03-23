---
name: quick-start
description: 引导项目初始化 — 环境安装、配置、验证、启动 Web UI（含 hma-web 公开评测平台）。
---

# Quick Start

引导用户完成 HolyEval 项目的初始化设置。按顺序执行以下步骤，每步完成后继续下一步，不要等待用户确认。

## Workflow

### Step 1: 检查环境

确认以下工具已安装：

```bash
python3 --version   # 需要 Python >= 3.11
uv --version        # 需要 uv 包管理器
node --version      # 需要 Node.js >= 18（hma-web 使用）
```

如果缺少，给出安装指引并停止：
- Python: https://www.python.org/downloads/ 或 `brew install python@3.11`
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node.js: https://nodejs.org/ 或 `brew install node`

### Step 2: 安装依赖

```bash
# 框架层（evaluator / benchmark / generator / web）
uv sync

# hma-web（独立业务应用，不属于 uv workspace）
cd hma-web && npm install && cd ..
```

### Step 3: 配置环境变量

#### 3a. 框架层 `.env`

检查根目录 `.env` 文件是否存在。如果不存在则从 `.env.example` 复制：

```bash
cp .env.example .env
```

读取 `.env` 文件，检查是否已填写 API Key。如果所有 key 还是占位符（sk-xxx / xxx / hf_xxx），**暂停并提示用户**至少配一个 LLM 提供商：

| 变量 | 说明 | 获取方式 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 密钥（gpt-4.1 等） | https://platform.openai.com/api-keys |
| `GOOGLE_API_KEY` | Google AI API 密钥（gemini-3 等） | https://aistudio.google.com/apikey |
| `OPENROUTER_API_KEY` | OpenRouter 统一多提供商访问（可选） | https://openrouter.ai/keys |
| `HF_TOKEN` | HuggingFace Token（thetagen 私有数据集必填，未配置会导致启动时数据准备失败） | https://huggingface.co/settings/tokens |
| `THETA_API_BASE_URL` | Theta Health API 地址（可选） | 内部系统，联系团队获取 |

如果已有有效 key，直接继续。

#### 3b. hma-web `.env`

检查 `hma-web/.env` 是否存在。如果不存在则从 `hma-web/.env.example` 复制：

```bash
cp hma-web/.env.example hma-web/.env
```

hma-web 使用**远程配置中心**加载大部分配置，本地 `.env` 只需配置引导变量：

| 变量 | 说明 |
|------|------|
| `ENV` | 环境标识（test / production） |
| `CONFIG_SERVER` | 配置中心地址 |
| `CONFIG_TOKEN` | 配置中心认证 Token |
| `CONFIG_ENCRYPTION_KEY` | 配置加密密钥 |
| `HF_TOKEN` | HuggingFace Token（数据集同步） |

**注意**: DB 连接（`PG_*_EVAL`）、Auth（`AUTH_SECRET`、`EMAIL_SMTP_*`）等由远程配置中心下发，不需要本地配置。如果缺少远程配置凭据，**暂停并提示用户**联系团队获取。

### Step 4: 验证安装

```bash
# 验证框架层模块导入
python -c "import evaluator; import benchmark; import generator; print('OK')"
```

如果有报错，诊断并修复后再继续。

### Step 5: 启动框架层 Web UI

在后台启动 Web 服务器，然后用系统命令打开浏览器：

```bash
# 后台启动（使用 run_in_background）
python -m web

# 等服务就绪后打开浏览器
sleep 2 && open http://localhost:8000
```

**注意**: 使用 Bash 工具的 `run_in_background: true` 参数启动 `python -m web`，然后单独执行 `sleep 2 && open http://localhost:8000`。端口可通过 `HOLYEVAL_PORT` 环境变量自定义（默认 8000）。

### Step 6: 启动 hma-web（可选）

hma-web 是 HMA 公开评测平台，独立于框架层。启动前需要：
- PostgreSQL 数据库可达（由远程配置中心提供连接信息）
- 远程配置中心凭据已配好（Step 3b）

```bash
cd hma-web && npm run dev
```

启动时 `instrumentation.ts` 会自动执行：
1. 加载远程配置（CONFIG_SERVER）
2. 构造 DATABASE_URL（从 PG_*_EVAL 变量）
3. 执行 Prisma 数据库迁移（`prisma migrate deploy`）
4. 运行种子数据（`prisma/seed.mts`）
5. 从 HuggingFace manifest 同步数据集版本到 DB

启动成功后访问 http://localhost:3000。

**常见问题**:
- 如果报 `缺少远程配置引导变量`，检查 `hma-web/.env` 中 `CONFIG_SERVER`、`CONFIG_TOKEN`、`ENV` 是否配置
- 如果报 `缺少 PG_*_EVAL 配置`，说明远程配置中心未返回数据库连接信息，联系团队排查
- 如果数据集同步失败，检查 `HF_TOKEN` 是否有效（不影响启动）

### Step 7: 展示项目概览

读取 `web/guides/overview.md` 文件，在对话窗口输出项目概览内容。

**输出内容包括**：
- **项目简介**：一句话描述 HolyEval 的定位
- **核心架构**：
  - 三大目录：evaluator（评测框架 + 对接服务）、generator（数据转换）、benchmark（批量调度）
  - 三层 Agent 架构：TestAgent / TargetAgent / EvalAgent
  - 执行流程：对话循环 → 评估 → 生成报告
- **业务应用**：hma-web 公开评测平台（独立部署，单向依赖框架层）
- **核心功能**：插件化 Agent 系统、批量评测执行、多维度评估、数据管理
- **技术栈**：
  - 框架层：Python 3.11+, FastAPI, uv workspace, Pydantic v2, LangChain
  - hma-web：Next.js 16, React 19, TypeScript, Tailwind CSS v4, Prisma, NextAuth

**引导用户**：
- 框架层 Web UI 已在浏览器打开：**http://localhost:8000**
- 访问项目概览页查看完整架构图：**http://localhost:8000/guides/overview**
- 如果启动了 hma-web：**http://localhost:3000**
- 推荐后续操作（skills 列表）

**输出格式**：
- 使用 markdown 表格和列表
- 去掉图片引用部分（对话窗口无法显示，引导用户到 Web UI 查看）
- 保持简洁易读，突出重点信息

**推荐后续操作**：

| Skill | 说明 |
|-------|------|
| `/add-benchmark` | 集成外部 benchmark（研究 → 转换 → 验证全流程） |
| `/add-eval-agent` | 新增评测逻辑插件 |
| `/add-target-agent` | 新增被测系统插件 |
| `/run-e2e-test` | 运行端到端测试 |
| `/run-benchmark` | 运行 benchmark 跑分 |
| `/review-architecture` | 审查项目架构健康度 |
