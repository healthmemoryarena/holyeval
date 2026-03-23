"""FastAPI 应用工厂 + 路由注册"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 触发插件注册
import evaluator.plugin.eval_agent  # noqa: F401
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.test_agent  # noqa: F401

from web.app.api import agents, benchmarks, guides, reports, tasks
from web.app.services.prepare_manager import prepare_manager
from web.app.services.task_manager import task_manager

from fastapi.responses import JSONResponse

_WEB_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：扫描并运行 benchmark 准备脚本（非阻塞，后台执行）
    await prepare_manager.start_all()
    yield
    # 关闭时取消所有活跃任务和准备脚本
    task_manager.cancel_all()
    prepare_manager.cancel_all()


def create_app() -> FastAPI:
    app = FastAPI(title="HolyEval Web UI", lifespan=lifespan)

    # 静态文件
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

    # K8s 健康检查探针
    @app.get("/api/health")
    async def health():
        return JSONResponse({"status": "ok"})

    # API 路由
    app.include_router(benchmarks.router, prefix="/api")
    app.include_router(agents.router, prefix="/api")
    app.include_router(tasks.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")
    app.include_router(guides.router, prefix="/api")

    # 页面路由
    @app.get("/")
    async def index(request: Request):
        return templates.TemplateResponse("tasks/index.html", {"request": request, "nav": "tasks"})

    @app.get("/tasks")
    async def tasks_page(request: Request):
        return templates.TemplateResponse("tasks/index.html", {"request": request, "nav": "tasks"})

    @app.get("/tasks/{task_id}")
    async def task_detail_page(request: Request, task_id: str):
        return templates.TemplateResponse("tasks/detail.html", {"request": request, "nav": "tasks", "task_id": task_id})

    @app.get("/benchmarks")
    async def benchmarks_page(request: Request):
        return templates.TemplateResponse("benchmarks/index.html", {"request": request, "nav": "benchmarks"})

    @app.get("/benchmarks/{benchmark}/{dataset}")
    async def benchmark_detail_page(request: Request, benchmark: str, dataset: str):
        return templates.TemplateResponse(
            "benchmarks/detail.html",
            {"request": request, "nav": "benchmarks", "benchmark": benchmark, "dataset": dataset},
        )

    @app.get("/virtual-users")
    async def virtual_users_page(request: Request):
        return templates.TemplateResponse("virtual_users/index.html", {"request": request, "nav": "virtual_users"})

    @app.get("/agents/target")
    async def agents_target_page(request: Request):
        return templates.TemplateResponse("agents/target.html", {"request": request, "nav": "agents_target"})

    @app.get("/agents/eval")
    async def agents_eval_page(request: Request):
        return templates.TemplateResponse("agents/eval.html", {"request": request, "nav": "agents_eval"})

    @app.get("/agents/test")
    async def agents_test_page(request: Request):
        return templates.TemplateResponse("agents/test.html", {"request": request, "nav": "agents_test"})

    @app.get("/reports/{benchmark}/{filename}")
    async def report_detail_page(request: Request, benchmark: str, filename: str):
        return templates.TemplateResponse(
            "reports/detail.html",
            {"request": request, "nav": "tasks", "benchmark": benchmark, "filename": filename},
        )

    @app.get("/guides/{guide_name}")
    async def guide_page(request: Request, guide_name: str):
        title = next((g["title"] for g in guides.GUIDE_LIST if g["name"] == guide_name), guide_name)
        return templates.TemplateResponse(
            "guides/detail.html",
            {"request": request, "nav": f"guide_{guide_name}", "guide_name": guide_name, "guide_title": title},
        )

    return app
