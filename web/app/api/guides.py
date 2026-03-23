"""快速开始指南 API"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["guides"])

_GUIDES_DIR = Path(__file__).resolve().parent.parent.parent / "guides"

# 指南元数据（顺序即侧栏顺序）
GUIDE_LIST = [
    {"name": "overview", "title": "项目概览"},
    {"name": "develop-eval-agent", "title": "开发 EvalAgent"},
    {"name": "develop-target-agent", "title": "开发 TargetAgent"},
    {"name": "generate-benchmark", "title": "生成 Benchmark"},
    {"name": "e2e-test", "title": "E2E 测试"},
    {"name": "run-benchmark", "title": "Benchmark 跑分"},
]

_VALID_NAMES = {g["name"] for g in GUIDE_LIST}


@router.get("/guides")
async def list_guides() -> list[dict]:
    return GUIDE_LIST


@router.get("/guides/{guide_name}")
async def get_guide(guide_name: str) -> dict:
    if guide_name not in _VALID_NAMES:
        raise HTTPException(status_code=404, detail=f"指南不存在: {guide_name}")
    md_path = _GUIDES_DIR / f"{guide_name}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"指南文件不存在: {guide_name}")
    content = md_path.read_text(encoding="utf-8")
    title = next((g["title"] for g in GUIDE_LIST if g["name"] == guide_name), guide_name)
    return {"name": guide_name, "title": title, "content": content}
