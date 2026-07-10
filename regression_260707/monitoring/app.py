"""FastAPI entry point for the standalone MFT campaign monitor."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .readers import TARGETS, ArtifactService, SchedulerReader


HERE = Path(__file__).resolve().parent
DEFAULT_REGRESSION_ROOT = HERE.parent


def create_app(
    regression_root: str | Path | None = None,
    service: ArtifactService | None = None,
) -> FastAPI:
    root = Path(
        regression_root or os.environ.get("MFT_MONITOR_ROOT") or DEFAULT_REGRESSION_ROOT
    ).resolve()
    if service is None:
        scheduler = SchedulerReader(
            base_url=os.environ.get("MFT_SCHEDULER_URL", "http://127.0.0.1:8000"),
            task_prefix=os.environ.get("MFT_MONITOR_TASK_PREFIX", "mft"),
            timeout=float(os.environ.get("MFT_SCHEDULER_TIMEOUT", "2")),
        )
        service = ArtifactService(
            root,
            scheduler=scheduler,
            record_runtime=os.environ.get("MFT_MONITOR_DISABLE_HISTORY", "0") != "1",
        )

    app = FastAPI(
        title="MFT 1MW Campaign Monitor",
        description="Read-only project dashboard for data, surrogate, NSGA-II, and FEA verification status.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.service = service
    app.state.regression_root = root
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def no_cache_api(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    async def invoke(call: Callable[[], Any], section: str) -> JSONResponse:
        try:
            payload = await run_in_threadpool(call)
            return JSONResponse(payload)
        except Exception as exc:  # A single bad live artifact must not kill the UI.
            return JSONResponse(
                {
                    "schema_version": 1,
                    "available": False,
                    "section": section,
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
                status_code=200,
            )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": "MFT 1MW 최적설계 모니터",
                "refresh_seconds": 20,
                "project_root": str(root),
            },
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {
            "status": "ok",
            "service": "mft-monitor",
            "regression_root": str(root),
        }

    @app.get("/api/dashboard")
    async def api_dashboard():
        return await invoke(service.dashboard, "dashboard")

    @app.get("/api/status")
    async def api_status():
        return await invoke(service.status, "status")

    @app.get("/api/data")
    async def api_data():
        return await invoke(service.data, "data")

    @app.get("/api/models")
    async def api_models():
        return await invoke(service.models, "models")

    @app.get("/api/models/{target}/history")
    async def api_model_history(target: str):
        if target not in {item["name"] for item in TARGETS}:
            raise HTTPException(status_code=404, detail="unknown model target")
        return await invoke(lambda: service.model_history(target), "model_history")

    @app.get("/api/nsga2")
    async def api_nsga2():
        return await invoke(service.nsga2, "nsga2")

    @app.get("/api/verification")
    async def api_verification():
        return await invoke(service.verification, "verification")

    @app.get("/api/history")
    async def api_history():
        return await invoke(service.history, "history")

    return app


app = create_app()
