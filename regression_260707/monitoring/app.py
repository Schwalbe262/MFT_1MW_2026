"""FastAPI entry point for the standalone MFT campaign monitor."""

from __future__ import annotations

import os
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from filelock import FileLock

from .readers import TARGETS, ArtifactService, SchedulerReader


HERE = Path(__file__).resolve().parent
DEFAULT_REGRESSION_ROOT = HERE.parent
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "").strip()
if not _LOCALAPPDATA:
    _LOCALAPPDATA = str(Path.home() / "AppData" / "Local")
CAMPAIGN_MUTATION_LOCK_PATH = (
    Path(_LOCALAPPDATA) / "MFT_1MW_2026" / "campaign-mutation.lock")


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
            project_name=os.environ.get("MFT_SCHEDULER_PROJECT", "MFT_1MW_2026v1"),
            timeout=float(os.environ.get("MFT_SCHEDULER_TIMEOUT", "2")),
            optional_timeout=float(
                os.environ.get("MFT_SCHEDULER_OPTIONAL_TIMEOUT", "1")
            ),
        )
        service = ArtifactService(
            root,
            scheduler=scheduler,
            record_runtime=os.environ.get("MFT_MONITOR_DISABLE_HISTORY", "0") != "1",
        )

    app = FastAPI(
        title="MFT 1MW Campaign Monitor",
        description="Project dashboard for MFT campaign status and bounded local operator control.",
        version="1.1.0",
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
        # The operator dashboard is a live local tool.  Do not let a browser
        # keep an older HTML/JS/CSS bundle while the service has already moved
        # to a newer response schema.
        if request.url.path == "/" or request.url.path.startswith(("/api/", "/static/")):
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

    def require_local_operator_request(request: Request) -> None:
        """Keep the one mutation endpoint loopback-only and CSRF-resistant."""
        client_host = request.client.host if request.client else ""
        try:
            if not ip_address(client_host).is_loopback:
                raise ValueError("not loopback")
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail="operator control is available only from localhost",
            ) from exc
        host_header = request.headers.get("host", "").strip()
        host_name = urlsplit(f"//{host_header}").hostname or ""
        if host_name.lower() != "localhost":
            try:
                if not ip_address(host_name).is_loopback:
                    raise ValueError("host is not loopback")
            except ValueError as exc:
                raise HTTPException(
                    status_code=403,
                    detail="operator control Host must be localhost",
                ) from exc
        if request.headers.get("x-mft-operator-control") != "parallel-target-v1":
            raise HTTPException(status_code=403, detail="operator control header is required")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="operator control requires application/json")
        if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
            raise HTTPException(status_code=403, detail="cross-site operator control is forbidden")
        origin = request.headers.get("origin", "").strip()
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != host_header.lower():
                raise HTTPException(status_code=403, detail="operator control origin mismatch")

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

    @app.get("/api/models/{target}/parity")
    async def api_model_parity(target: str):
        if target not in {item["name"] for item in TARGETS}:
            raise HTTPException(status_code=404, detail="unknown model target")
        return await invoke(lambda: service.model_parity(target), "model_parity")

    @app.get("/api/nsga2")
    async def api_nsga2():
        return await invoke(service.nsga2, "nsga2")

    @app.get("/api/verification")
    async def api_verification():
        return await invoke(service.verification, "verification")

    @app.get("/api/history")
    async def api_history():
        return await invoke(service.history, "history")

    @app.patch("/api/operator/parallel-target")
    async def api_set_parallel_target(request: Request):
        require_local_operator_request(request)
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        target = payload.get("target")
        if type(target) is not int or not 1 <= target <= 300:
            raise HTTPException(
                status_code=422,
                detail="target must be an integer between 1 and 300",
            )
        scheduler = getattr(service, "scheduler", None)
        setter = getattr(scheduler, "set_parallel_target", None)
        if not callable(setter):
            raise HTTPException(status_code=503, detail="scheduler operator control is unavailable")
        try:
            def set_under_campaign_lock():
                CAMPAIGN_MUTATION_LOCK_PATH.parent.mkdir(
                    parents=True, exist_ok=True)
                with FileLock(
                        str(CAMPAIGN_MUTATION_LOCK_PATH), timeout=15 * 60):
                    return setter(target)

            result = await run_in_threadpool(set_under_campaign_lock)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"scheduler target update failed: {type(exc).__name__}: {exc}",
            ) from exc
        return {
            "schema_version": 1,
            "updated": True,
            **result,
        }

    return app


app = create_app()
