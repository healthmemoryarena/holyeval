"""python -m web — start HolyEval Web UI"""

import logging
import os
import sys

import uvicorn

LOG_FILE = "web.log"


def _setup_logging():
    """Configure logging to both terminal and file"""
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
    logging.getLogger(__name__).info("Web UI started, log file: %s", LOG_FILE)

    port = int(os.environ.get("HOLYEVAL_WEB_PORT", 8000))
    # Disable reload in production/container (files don't change in K8s, reload is unnecessary and affects signal handling)
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
