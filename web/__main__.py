"""python -m web — 启动 HolyEval Web UI"""

import logging
import os
import sys

import uvicorn

LOG_FILE = "web.log"


def _setup_logging():
    """配置日志同时输出到终端和文件"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def main():
    _setup_logging()
    logging.getLogger(__name__).info("Web UI 启动，日志文件: %s", LOG_FILE)

    port = int(os.environ.get("HOLYEVAL_WEB_PORT", 8000))
    # 生产/容器环境禁用 reload（K8s 中文件不变，reload 无意义且影响信号处理）
    reload = os.environ.get("HOLYEVAL_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "web.app.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
