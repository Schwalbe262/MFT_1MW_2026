"""FastAPI entry point for the standalone MFT campaign monitor."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from filelock import FileLock

from .readers import (
    TARGETS,
    ArtifactService,
    CampaignDemandConflict,
    SchedulerReader,
    SimulationPolicyConflict,
)


HERE = Path(__file__).resolve().parent
DEFAULT_REGRESSION_ROOT = HERE.parent
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "").strip()
if not _LOCALAPPDATA:
    _LOCALAPPDATA = str(Path.home() / "AppData" / "Local")
CAMPAIGN_MUTATION_LOCK_PATH = (
    Path(_LOCALAPPDATA) / "MFT_1MW_2026" / "campaign-mutation.lock")
TRUSTED_OPERATOR_LAN = ip_network("192.168.0.0/24")
DEFAULT_OPERATOR_HOSTS = ("localhost", "127.0.0.1", "::1")
LOGGER = logging.getLogger(__name__)


def _operator_host_allowlist() -> frozenset[str]:
    configured = os.environ.get("MFT_MONITOR_OPERATOR_HOSTS", "")
    values = configured.split(",") if configured.strip() else DEFAULT_OPERATOR_HOSTS
    return frozenset(
        value.strip().lower().removeprefix("[").removesuffix("]")
        for value in values
        if value.strip()
    )


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
        description="Project dashboard for MFT campaign status and bounded trusted-LAN operator control.",
        version="1.2.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.service = service
    app.state.regression_root = root
    app.state.operator_host_allowlist = _operator_host_allowlist()
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def no_cache_api(request: Request, call_next):
        response = await call_next(request)
        # The operator dashboard is a live local tool.  Do not let a browser
        # keep an older HTML/JS/CSS bundle while the service has already moved
        # to a newer response schema.
        if request.url.path in {"/", "/cohorts"} or request.url.path.startswith(
            ("/api/", "/static/")
        ):
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

    def require_local_operator_request(
        request: Request,
        control_header: str = "simulation-policy-v1",
    ) -> None:
        """Permit bounded trusted-LAN operation while resisting Host/CSRF abuse."""
        client_host = request.client.host if request.client else ""
        try:
            client_address = ip_address(client_host)
            mapped = getattr(client_address, "ipv4_mapped", None)
            if mapped is not None:
                client_address = mapped
            if not (
                client_address.is_loopback
                or client_address in TRUSTED_OPERATOR_LAN
            ):
                raise ValueError("outside trusted operator network")
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail=(
                    "operator control is available only from loopback or "
                    "192.168.0.0/24"
                ),
            ) from exc
        host_header = request.headers.get("host", "").strip()
        host_name = urlsplit(f"//{host_header}").hostname or ""
        if host_name.lower() not in app.state.operator_host_allowlist:
            raise HTTPException(
                status_code=403,
                detail="operator control Host is not allowlisted",
            )
        if request.headers.get("x-mft-operator-control") != control_header:
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

    @app.get("/cohorts", response_class=HTMLResponse, include_in_schema=False)
    async def cohorts_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="cohorts.html",
            context={
                "title": "MFT 데이터 코호트 상세",
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

    @app.get("/api/pipeline")
    async def api_pipeline():
        return await invoke(service.continuous_pipeline.snapshot, "pipeline")

    @app.patch("/api/operator/simulation-policy")
    async def api_set_simulation_policy(request: Request):
        require_local_operator_request(request)
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        target = payload.get("desired_simulations")
        expected_revision = payload.get("expected_revision")
        if type(target) is not int:
            raise HTTPException(status_code=422, detail="desired_simulations must be an integer")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, (int, str))
            or not str(expected_revision).strip()
        ):
            raise HTTPException(status_code=422, detail="expected_revision is required")
        if payload.get("scale_down_mode") != "drain":
            raise HTTPException(status_code=422, detail="scale_down_mode must be drain")
        scheduler = getattr(service, "scheduler", None)
        setter = getattr(scheduler, "set_simulation_policy", None)
        if not callable(setter):
            raise HTTPException(status_code=503, detail="scheduler operator control is unavailable")
        try:
            def set_under_campaign_lock():
                CAMPAIGN_MUTATION_LOCK_PATH.parent.mkdir(
                    parents=True, exist_ok=True)
                with FileLock(
                        str(CAMPAIGN_MUTATION_LOCK_PATH), timeout=15 * 60):
                    current = scheduler.snapshot()
                    if current.get("policy_supported") is not True:
                        raise RuntimeError(
                            current.get("control_gate_reason")
                            or "scheduler simulation-policy is unavailable"
                        )
                    if current.get("control_enabled") is not True:
                        raise PermissionError(
                            current.get("control_gate_reason")
                            or "scheduler simulation-policy is gated"
                        )
                    if current.get("policy_revision") != expected_revision:
                        raise SimulationPolicyConflict(
                            "simulation policy changed; refresh and retry"
                        )
                    minimum = current.get("parallel_target_min")
                    maximum = current.get("parallel_target_max")
                    if (
                        type(minimum) is not int
                        or type(maximum) is not int
                        or not minimum <= target <= maximum
                    ):
                        raise ValueError(
                            f"desired_simulations must be between {minimum} and {maximum}"
                        )
                    return setter(
                        target, expected_revision=expected_revision
                    )

            result = await run_in_threadpool(set_under_campaign_lock)
        except SimulationPolicyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"scheduler target update failed: {type(exc).__name__}: {exc}",
            ) from exc
        LOGGER.info(
            "simulation_policy_update source=%s project=%s expected_revision=%s "
            "new_revision=%s desired_simulations=%s",
            request.client.host if request.client else "",
            result.get("project"),
            expected_revision,
            result.get("policy_revision"),
            target,
        )
        return {
            "schema_version": 2,
            "updated": True,
            **result,
        }

    @app.patch("/api/operator/campaign-demand")
    async def api_set_campaign_demand(request: Request):
        require_local_operator_request(request, "campaign-demand-v1")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        if set(payload) != {"total_simulations", "expected_revision"}:
            raise HTTPException(
                status_code=422,
                detail="only total_simulations and expected_revision are accepted",
            )
        target = payload.get("total_simulations")
        expected_revision = payload.get("expected_revision")
        if type(target) is not int:
            raise HTTPException(status_code=422, detail="total_simulations must be an integer")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, (int, str))
            or not str(expected_revision).strip()
        ):
            raise HTTPException(status_code=422, detail="expected_revision is required")
        scheduler = getattr(service, "scheduler", None)
        setter = getattr(scheduler, "set_campaign_demand", None)
        if not callable(setter):
            raise HTTPException(status_code=503, detail="campaign demand control is unavailable")
        try:
            def set_under_campaign_lock():
                # This is the same lock held from demand GET through feeder
                # submission, so a Web UI decrease cannot race a refill batch.
                CAMPAIGN_MUTATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(str(CAMPAIGN_MUTATION_LOCK_PATH), timeout=15 * 60):
                    current = scheduler.snapshot()
                    if current.get("campaign_demand_supported") is not True:
                        raise RuntimeError(
                            current.get("campaign_demand_error")
                            or "scheduler campaign-demand is unavailable"
                        )
                    if current.get("campaign_demand_control_enabled") is not True:
                        raise PermissionError("scheduler campaign-demand control is gated")
                    if current.get("demand_revision") != expected_revision:
                        raise CampaignDemandConflict(
                            "campaign demand changed; refresh and retry"
                        )
                    minimum = current.get("campaign_demand_min")
                    maximum = current.get("campaign_demand_max")
                    if (
                        type(minimum) is not int
                        or type(maximum) is not int
                        or not minimum <= target <= maximum
                    ):
                        raise ValueError(
                            f"total_simulations must be between {minimum} and {maximum}"
                        )
                    return setter(target, expected_revision=expected_revision)

            result = await run_in_threadpool(set_under_campaign_lock)
        except CampaignDemandConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"campaign demand update failed: {type(exc).__name__}: {exc}",
            ) from exc
        LOGGER.info(
            "campaign_demand_update source=%s project=%s expected_revision=%s "
            "new_revision=%s total_simulations=%s semantics=drain-no-cancel",
            request.client.host if request.client else "",
            result.get("project"),
            expected_revision,
            result.get("demand_revision"),
            target,
        )
        return {
            "schema_version": 1,
            "updated": True,
            "no_cancellation_performed": True,
            **result,
        }

    return app


app = create_app()
