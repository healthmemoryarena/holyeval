"""报告 API"""

from typing import Any

from fastapi import APIRouter, HTTPException

from web.app.models.responses import ReportListResponse
from web.app.services.report_reader import delete_report, get_report_content, list_reports

router = APIRouter(tags=["reports"])


@router.get("/reports", response_model=ReportListResponse)
async def get_reports():
    return ReportListResponse(reports=list_reports())


@router.get("/reports/{benchmark}/{filename}")
async def get_report(benchmark: str, filename: str) -> dict[str, Any]:
    try:
        return get_report_content(benchmark, filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/reports/{benchmark}/{filename}")
async def remove_report(benchmark: str, filename: str):
    try:
        delete_report(benchmark, filename)
        return {"status": "deleted", "benchmark": benchmark, "filename": filename}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
