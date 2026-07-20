"""Authenticate and load one corrected direct-model generation for Tier-1.

This module is intentionally additive.  The historical Tier-1 feedback-model
wrapper remains sealed to its all-11-temperature NSGA revision.  A corrected
generation instead contains one direct model bundle for each of the current 21
training targets.  This adapter authenticates that immutable generation and
loads the 20 models consumed by the current seven-temperature NSGA code.

The production quality result may be false: this is an explicitly experimental
Tier-1 input.  Recovery provenance may not be false or incomplete.  Candidate,
quality, report, dataset, profile, every target metadata file, and the complete
artifact inventory are bound before any fitted model is made available.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping


ADAPTER_SCHEMA = "mft-tier1-corrected-generation-adapter-v1"
RECOVERY_CONTRACT = "mft-capacitance-lc-inverse-v1"
RECOVERY_MAX_ABS_DELTA_F = 5.1e-11
RECOVERY_STATUS = "applied"

CURRENT_TEMPERATURE_TARGETS = (
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)

CORRECTED_GENERATION_TARGETS = (
    "Llt_phys",
    "k",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "P_winding_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
    "P_core_total",
    "P_core_plate_total",
    "P_wcp_total",
    "B_max_core",
    "B_mean_core",
    *CURRENT_TEMPERATURE_TARGETS,
)

CURRENT_REQUIRED_MODEL_TARGETS = (
    "Llt_phys",
    "k",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "P_winding_total",
    "P_core_total",
    "P_core_plate_total",
    "P_wcp_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
    "B_mean_core",
    *CURRENT_TEMPERATURE_TARGETS,
)

CURRENT_TEMPERATURE_CONTRACT = {
    "schema_version": "mft-tier1-current7-temperature-contract-v1",
    "semantic_version": "current7-robust-q90-half-width-t110-v1",
    "targets": list(CURRENT_TEMPERATURE_TARGETS),
    "target_count": len(CURRENT_TEMPERATURE_TARGETS),
    "side_winding_conditional_targets": [
        "Tprobe_Rx_side_leeward_max",
    ],
    "side_winding_activation": "finite_N2_side_gt_0",
    "side_winding_absent_behavior": (
        "finite_N2_side_eq_0_disables_with_negative_BIG"
    ),
    "side_winding_missing_behavior": (
        "missing_or_nonfinite_N2_side_fails_with_positive_BIG"
    ),
    "robust_upper_bound_C": 110.0,
    "formula": "surrogate_mu_plus_q90_conformal_half_width_le_limit",
    "predictor_conformal_argument": True,
    "additional_half_width_multiplier": 1.0,
    "constraint_name_format": "temperature_robust_limit:{target}",
    "source": "model_targets.SURROGATE_TEMPERATURE_TARGETS",
}

CURRENT_STAGE_HARD_CONTRACT = {
    "schema_version": "mft-tier1-current7-hard-constraint-contract-v2",
    "stage": "1200x1200x750-res15k-t110-core4-cw1-5-lmhalf",
    "temperature_limit_C": 110.0,
    "temperature_targets": list(CURRENT_TEMPERATURE_TARGETS),
    "uncertainty_contract": {
        "predictor_output": "q90_conformal_half_width_physical_v1",
        "additional_multiplier": 1.0,
    },
    "minimum_physical_insulation": {
        "minimum_mm": 40.0,
        "authority": "all_realized_clearances",
        "base_secondary_vertical_constraint_retained": True,
        "side_clearances_activation": "finite_N1_side_gt_0",
        "missing_or_nonfinite_behavior": "positive_BIG",
    },
    "self_resonance": {
        "operator": ">=",
        "minimum_Hz": 15_000.0,
        "aggregation": "min(f_res_tx_self_Hz,f_res_rx_self_Hz)",
        "magnetizing_inductance_factor": 0.5,
        "interwinding_resonance_is_not_part_of_this_minimum": True,
        "violation_formula": (
            "15000-min(f_res_tx_half_Lm_Hz,f_res_rx_half_Lm_Hz)"
        ),
    },
    "size_limits_mm": {"W": 1_200.0, "L": 1_200.0, "H": 750.0},
    "maximum_core_groups": 4,
    "primary_conductor_thickness_mm": 5.0,
    "primary_conductor_enforcement": (
        "fixed_cw1_consumed_inside_winding_budget_plus_derived_identity_attestation"
    ),
    "cooling_geometry": {
        "variable_trained_dimensions": [
            "core_plate_t",
            "wcp_t",
            "wcp_len_pct",
        ],
        "fixed_pad_thickness_mm": {
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
        },
        "simple_base_20mm_plate_clamp_superseded": True,
        "offspring_physics_projection_installed": True,
        "authoritative_hard_constraints_remain_physical_G": True,
    },
    "portability": {
        "mode": "source_local_or_authenticated_bundle_relocation_v2",
        "remote_relocation_supported": True,
        "immutable_source_evidence_rewritten": False,
        "remote_relocation_schema": "mft-tier1-current7-relocation-v1",
        "remote_packaging_requirement": (
            "seal original SHA identities and authenticate the explicit relocation "
            "map, bundle manifest, and bundle-relative evidence before execution"
        ),
    },
    "production_eligible": False,
    "automatic_promotion_allowed": False,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def training_profile_sha256(value: Any) -> str:
    """Match train_models' historical profile JSON canonicalization exactly."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


CURRENT_TEMPERATURE_CONTRACT_SHA256 = canonical_sha256(
    CURRENT_TEMPERATURE_CONTRACT
)
CURRENT_STAGE_HARD_CONTRACT_SHA256 = canonical_sha256(
    CURRENT_STAGE_HARD_CONTRACT
)
CURRENT_REQUIRED_MODEL_TARGETS_SHA256 = canonical_sha256(
    list(CURRENT_REQUIRED_MODEL_TARGETS)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _hex(value: Any, length: int, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{label} is not a {length}-character hexadecimal digest")
    return normalized


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a positive integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric != integer or integer <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return integer


def _recovery_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} recovery evidence is missing")
    if value.get("contract") != RECOVERY_CONTRACT:
        raise RuntimeError(f"{label} recovery contract mismatch")
    recovered = _positive_integer(
        value.get("recovered_row_count"), f"{label} recovered_row_count"
    )
    try:
        delta = float(value.get("max_observed_abs_delta_F"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} recovery maximum delta is invalid") from exc
    if (
        not math.isfinite(delta)
        or delta < 0.0
        or delta > RECOVERY_MAX_ABS_DELTA_F
    ):
        raise RuntimeError(f"{label} recovery maximum delta is out of bounds")
    return {
        "contract": RECOVERY_CONTRACT,
        "recovered_row_count": recovered,
        "max_observed_abs_delta_F": delta,
    }


def _training_recovery_identity(
    value: Any,
    label: str,
    *,
    strict_rows: int,
) -> dict[str, Any]:
    """Validate the complete recovery audit emitted into trained bundles."""

    identity = _recovery_identity(value, label)
    if not isinstance(value, Mapping):  # pragma: no cover - guarded above
        raise RuntimeError(f"{label} recovery evidence is missing")
    counts = {
        key: _positive_integer(value.get(key), f"{label} {key}")
        for key in (
            "row_count",
            "cap_enabled_row_count",
            "eligible_row_count",
            "recovered_row_count",
        )
    }
    if any(count != strict_rows for count in counts.values()):
        raise RuntimeError(f"{label} recovery row inventory is incomplete")
    try:
        maximum_allowed = float(value.get("max_allowed_abs_delta_F"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} recovery allowed delta is invalid") from exc
    if not math.isclose(
        maximum_allowed,
        RECOVERY_MAX_ABS_DELTA_F,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(f"{label} recovery allowed delta mismatch")
    if (
        value.get("status") != RECOVERY_STATUS
        or value.get("missing_columns") != []
    ):
        raise RuntimeError(f"{label} recovery was not completely applied")
    return {
        **identity,
        **counts,
        "max_allowed_abs_delta_F": maximum_allowed,
        "missing_columns": [],
        "status": RECOVERY_STATUS,
    }


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _path_below(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    root = root.resolve(strict=True)
    try:
        inside = os.path.commonpath([str(resolved), str(root)]) == str(root)
    except ValueError:
        inside = False
    if not inside or resolved == root:
        raise RuntimeError(f"{label} escapes its authenticated root")
    return resolved


def authenticate_code_root(code_root: Path, expected_revision: str) -> dict[str, Any]:
    """Require an exact, clean Git checkout without mutating global config."""

    root = code_root.resolve(strict=True)
    expected = _hex(expected_revision, 40, "expected code revision")
    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{count}"] = root.as_posix()
    environment["GIT_CONFIG_COUNT"] = str(count + 1)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if revision.returncode != 0:
        raise RuntimeError("current7 code root is not an authenticated Git checkout")
    actual = _hex(revision.stdout.strip(), 40, "current7 code revision")
    if actual != expected:
        raise RuntimeError(
            f"current7 code revision mismatch: expected {expected}, got {actual}"
        )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("current7 code checkout is dirty")
    return {"path": str(root), "revision": actual, "clean": True}


@dataclass(frozen=True)
class AuthenticatedCorrectedGeneration:
    generation: Path
    registry: Path
    report_path: Path
    report: dict[str, Any]
    candidate_path: Path
    candidate: dict[str, Any]
    quality_path: Path
    quality: dict[str, Any]
    dataset_path: Path
    profile_path: Path
    evidence: dict[str, Any]


def authenticate_corrected_generation(
    *,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
) -> AuthenticatedCorrectedGeneration:
    """Authenticate small-file provenance before the one full artifact pass."""

    generation = generation.resolve(strict=True)
    if not generation.is_dir() or generation.parent.name != "generations":
        raise RuntimeError("corrected generation must be below registry/generations")
    registry = generation.parent.parent.resolve(strict=True)
    generation_relative = generation.relative_to(registry).as_posix()
    report_path = generation / "train_report.json"
    report = read_json(report_path)
    report_sha = sha256_file(report_path)

    if report.get("schema_version") != 2:
        raise RuntimeError("corrected train report schema mismatch")
    run_id = str(report.get("training_run_id") or "")
    if not run_id or run_id != generation.name:
        raise RuntimeError("corrected generation training_run_id mismatch")
    dataset_sha = _hex(report.get("dataset_sha256"), 64, "dataset SHA-256")
    profile_sha = _hex(report.get("profile_sha256"), 64, "profile SHA-256")
    strict_rows = _positive_integer(report.get("strict_full_rows"), "strict rows")

    targets = report.get("targets")
    if targets != list(CORRECTED_GENERATION_TARGETS):
        raise RuntimeError("corrected generation target order/set mismatch")
    target_reports = report.get("report")
    if not isinstance(target_reports, Mapping) or set(target_reports) != set(targets):
        raise RuntimeError("corrected generation metric target inventory mismatch")
    artifacts = report.get("artifacts")
    expected_artifacts = {
        f"{target}/{filename}"
        for target in CORRECTED_GENERATION_TARGETS
        for filename in ("meta.json", "models.pkl")
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise RuntimeError("corrected generation artifact inventory mismatch")

    recovery = _training_recovery_identity(
        report.get("capacitance_recovery"),
        "train report",
        strict_rows=strict_rows,
    )

    artifact_sizes: dict[str, int] = {}
    for relative in sorted(expected_artifacts):
        expected_sha = _hex(artifacts.get(relative), 64, f"artifact SHA {relative}")
        artifact = _path_below(generation / relative, generation, "generation artifact")
        if not artifact.is_file():
            raise RuntimeError(f"corrected generation artifact is missing: {relative}")
        artifact_sizes[relative] = artifact.stat().st_size
        if relative.endswith("/meta.json"):
            if sha256_file(artifact) != expected_sha:
                raise RuntimeError(f"corrected target metadata SHA mismatch: {relative}")
            meta = read_json(artifact)
            target = relative.split("/", 1)[0]
            if (
                meta.get("target") != target
                or meta.get("training_run_id") != run_id
                or meta.get("dataset_sha256") != dataset_sha
                or meta.get("profile_sha256") != profile_sha
                or meta.get("strict_full_rows") != strict_rows
                or _training_recovery_identity(
                    meta.get("capacitance_recovery"),
                    relative,
                    strict_rows=strict_rows,
                ) != recovery
            ):
                raise RuntimeError(f"corrected target metadata identity mismatch: {target}")

    actual_directories = {
        item.name
        for item in generation.iterdir()
        if item.is_dir()
    }
    if actual_directories != set(CORRECTED_GENERATION_TARGETS):
        raise RuntimeError("corrected generation target directory inventory mismatch")

    dataset_path = Path(str(report.get("dataset_path") or "")).resolve(strict=True)
    if not dataset_path.is_file() or sha256_file(dataset_path) != dataset_sha:
        raise RuntimeError("corrected generation dataset fingerprint mismatch")
    profile_path = Path(str(report.get("profile_path") or "")).resolve(strict=True)
    profile = read_json(profile_path)
    if training_profile_sha256(profile) != profile_sha:
        raise RuntimeError("corrected generation profile fingerprint mismatch")

    candidate_path = candidate_path.resolve(strict=True)
    candidate = read_json(candidate_path)
    candidate_recovery = _training_recovery_identity(
        candidate.get("capacitance_recovery"),
        "candidate",
        strict_rows=strict_rows,
    )
    if (
        candidate.get("schema_version") != 2
        or candidate.get("training_run_id") != run_id
        or candidate.get("generation") != generation_relative
        or not _same_path(Path(str(candidate.get("generation_path") or "")), generation)
        or candidate.get("generation_report_sha256") != report_sha
        or candidate.get("dataset_sha256") != dataset_sha
        or candidate.get("strict_full_rows") != strict_rows
        or candidate_recovery != recovery
    ):
        raise RuntimeError("corrected candidate identity mismatch")

    quality_path = quality_path.resolve(strict=True)
    quality = read_json(quality_path)
    quality_recovery = quality.get("capacitance_recovery")
    expected_quality_recovery_identity = {
        key: recovery[key]
        for key in (
            "contract",
            "recovered_row_count",
            "max_observed_abs_delta_F",
        )
    }
    if not isinstance(quality.get("passed"), bool):
        raise RuntimeError("corrected quality status has no terminal pass/fail state")
    if quality.get("passed") is False and not quality.get("reasons"):
        raise RuntimeError("failed corrected quality status has no sealed blockers")
    if (
        quality.get("training_run_id") != run_id
        or quality.get("generation") != generation_relative
        or quality.get("generation_report_sha256") != report_sha
        or quality.get("dataset_sha256") != dataset_sha
        or quality.get("profile_sha256") != profile_sha
        or quality.get("strict_full_rows") != strict_rows
        or not isinstance(quality_recovery, Mapping)
        or quality_recovery.get("schema_version") != 1
        or quality_recovery.get("passed") is not True
        or quality_recovery.get("reasons") != []
        or _recovery_identity(quality_recovery, "quality recovery")
        != expected_quality_recovery_identity
        or quality_recovery.get("dataset_sha256") != dataset_sha
        or quality_recovery.get("profile_sha256") != profile_sha
    ):
        raise RuntimeError("corrected quality/recovery identity mismatch")
    recovery_targets = quality_recovery.get("targets")
    if not isinstance(recovery_targets, Mapping) or set(recovery_targets) != set(targets):
        raise RuntimeError("corrected quality recovery target inventory mismatch")
    for target, status in recovery_targets.items():
        if (
            not isinstance(status, Mapping)
            or status.get("passed") is not True
            or status.get("reasons") != []
        ):
            raise RuntimeError(f"corrected recovery guard failed for target: {target}")

    quality_targets = quality.get("targets")
    if not isinstance(quality_targets, Mapping) or set(quality_targets) != set(targets):
        raise RuntimeError("corrected quality metric target inventory mismatch")

    evidence = {
        "schema_version": ADAPTER_SCHEMA,
        "generation": str(generation),
        "registry": str(registry),
        "generation_relative": generation_relative,
        "training_run_id": run_id,
        "train_report": {"path": str(report_path), "sha256": report_sha},
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "quality_status": {
            "path": str(quality_path),
            "sha256": sha256_file(quality_path),
            "passed": quality["passed"],
            "blocking_reason_count": len(quality.get("reasons") or []),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha,
            "strict_full_rows": strict_rows,
        },
        "profile": {
            "path": str(profile_path),
            "canonical_sha256": profile_sha,
            "canonicalization": (
                "json_sort_keys_compact_ensure_ascii_true_train_models_v1"
            ),
        },
        "capacitance_recovery": {
            **recovery,
            "guard_passed": True,
            "guard_target_count": len(recovery_targets),
            "guard_passed_target_count": sum(
                status.get("passed") is True for status in recovery_targets.values()
            ),
        },
        "generation_targets": list(CORRECTED_GENERATION_TARGETS),
        "generation_target_count": len(CORRECTED_GENERATION_TARGETS),
        "required_model_targets": list(CURRENT_REQUIRED_MODEL_TARGETS),
        "required_model_targets_sha256": CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        "artifact_count": len(artifacts),
        "artifact_sizes_bytes": artifact_sizes,
        "full_artifact_hash_pass_deferred_to_single_model_load": True,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    return AuthenticatedCorrectedGeneration(
        generation=generation,
        registry=registry,
        report_path=report_path,
        report=report,
        candidate_path=candidate_path,
        candidate=candidate,
        quality_path=quality_path,
        quality=quality,
        dataset_path=dataset_path,
        profile_path=profile_path,
        evidence=evidence,
    )


class CorrectedGenerationModelCache:
    """Load and authenticate each current7 model once in one process."""

    def __init__(
        self,
        authenticated: AuthenticatedCorrectedGeneration,
        *,
        train_models_module: Any,
        predictor_class: Any,
        required_targets: tuple[str, ...] = CURRENT_REQUIRED_MODEL_TARGETS,
    ) -> None:
        self.authenticated = authenticated
        self.train_models_module = train_models_module
        self.predictor_class = predictor_class
        self.required_targets = tuple(required_targets)
        if self.required_targets != CURRENT_REQUIRED_MODEL_TARGETS:
            raise RuntimeError("current7 required-model target contract mismatch")
        self._models: Mapping[str, Any] | None = None
        self._load_error: BaseException | None = None
        self.load_calls = 0
        self.full_generation_authentication_passes = 0

    def load(self) -> Mapping[str, Any]:
        if self._models is not None:
            return self._models
        if self._load_error is not None:
            raise RuntimeError(
                "corrected generation model load previously failed; refusing a "
                "second full artifact pass in this process"
            ) from self._load_error
        self.load_calls += 1
        try:
            record = self.train_models_module.load_generation(
                str(self.authenticated.registry),
                self.authenticated.evidence["generation_relative"],
                require_accepted=False,
            )
            self.full_generation_authentication_passes += 1
            if (
                Path(record.get("generation", "")).resolve()
                != self.authenticated.generation
                or record.get("generation_report_sha256")
                != self.authenticated.evidence["train_report"]["sha256"]
                or record.get("report") != self.authenticated.report
            ):
                raise RuntimeError(
                    "one-pass generation loader returned a different identity"
                )

            report = self.authenticated.report
            expected_recovery = _training_recovery_identity(
                report.get("capacitance_recovery"),
                "train report",
                strict_rows=int(report["strict_full_rows"]),
            )
            report_features = list(report.get("features") or [])
            if not report_features or len(report_features) != len(set(report_features)):
                raise RuntimeError("corrected generation feature schema is invalid")
            models: dict[str, Any] = {}
            for target in self.required_targets:
                predictor = self.predictor_class._load_record(target, record)
                bundle = getattr(predictor, "bundle", None)
                if not isinstance(bundle, Mapping):
                    raise RuntimeError(f"loaded model bundle is unavailable: {target}")
                try:
                    q90 = float(bundle.get("q90"))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"loaded model conformal calibration is invalid: {target}"
                    ) from exc
                if not math.isfinite(q90) or q90 <= 0.0:
                    raise RuntimeError(
                        f"loaded model conformal calibration is invalid: {target}"
                    )
                if (
                    bundle.get("target") != target
                    or bundle.get("training_run_id") != report.get("training_run_id")
                    or bundle.get("dataset_sha256") != report.get("dataset_sha256")
                    or bundle.get("profile_sha256") != report.get("profile_sha256")
                    or bundle.get("strict_full_rows") != report.get("strict_full_rows")
                    or list(bundle.get("features") or []) != report_features
                    or list(bundle.get("feature_schema") or []) != report_features
                    or _training_recovery_identity(
                        bundle.get("capacitance_recovery"),
                        target,
                        strict_rows=int(report["strict_full_rows"]),
                    ) != expected_recovery
                ):
                    raise RuntimeError(
                        f"loaded model bundle identity mismatch: {target}"
                    )
                models[target] = predictor
            if tuple(models) != self.required_targets:
                raise RuntimeError("current7 loaded model order/set mismatch")
            self._models = MappingProxyType(models)
            return self._models
        except BaseException as exc:
            self._load_error = exc
            raise

    @property
    def loaded_once(self) -> bool:
        return (
            self._models is not None
            and self.load_calls == 1
            and self.full_generation_authentication_passes == 1
        )


_PROCESS_MODEL_CACHES: dict[
    tuple[str, str, str], CorrectedGenerationModelCache
] = {}


def process_model_cache(
    authenticated: AuthenticatedCorrectedGeneration,
    *,
    train_models_module: Any,
    predictor_class: Any,
) -> CorrectedGenerationModelCache:
    """Return the one cache authorized for this generation in this process."""

    key = (
        str(authenticated.generation),
        str(authenticated.evidence["train_report"]["sha256"]),
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
    )
    existing = _PROCESS_MODEL_CACHES.get(key)
    if existing is not None:
        if (
            existing.train_models_module is not train_models_module
            or existing.predictor_class is not predictor_class
        ):
            raise RuntimeError(
                "process model cache identity conflicts with already loaded code"
            )
        return existing
    created = CorrectedGenerationModelCache(
        authenticated,
        train_models_module=train_models_module,
        predictor_class=predictor_class,
    )
    _PROCESS_MODEL_CACHES[key] = created
    return created


def adapter_manifest(
    authenticated: AuthenticatedCorrectedGeneration,
    *,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if code_identity.get("clean") is not True:
        raise RuntimeError("current7 code identity is not clean")
    revision = _hex(code_identity.get("revision"), 40, "current7 code revision")
    root = Path(str(code_identity.get("path") or "")).resolve(strict=True)
    return {
        **authenticated.evidence,
        "code": {"path": str(root), "revision": revision, "clean": True},
        "temperature_contract": CURRENT_TEMPERATURE_CONTRACT,
        "temperature_contract_sha256": CURRENT_TEMPERATURE_CONTRACT_SHA256,
        "hard_constraint_contract": CURRENT_STAGE_HARD_CONTRACT,
        "hard_constraint_contract_sha256": CURRENT_STAGE_HARD_CONTRACT_SHA256,
        "model_loading": {
            "process_scope": "single_local_process",
            "cache_policy": "one_generation_authentication_and_unpickle_pass",
            "generation_copy_performed": False,
            "legacy_feedback_wrapper_used": False,
        },
        "portability": CURRENT_STAGE_HARD_CONTRACT["portability"],
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "canonical_pointer_write_performed": False,
    }


def validate_adapter_manifest(
    value: Any,
    *,
    source_paths_required: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("corrected adapter manifest must be an object")
    code = value.get("code") or {}
    model_loading = value.get("model_loading") or {}
    dataset = value.get("dataset") or {}
    recovery = value.get("capacitance_recovery") or {}
    artifact_sizes = value.get("artifact_sizes_bytes") or {}
    expected_artifacts = {
        f"{target}/{filename}"
        for target in CORRECTED_GENERATION_TARGETS
        for filename in ("meta.json", "models.pkl")
    }
    try:
        code_path = str(code.get("path") or "")
        if not code_path:
            raise RuntimeError("current7 code path is missing")
        code_root = (
            Path(code_path).resolve(strict=True)
            if source_paths_required
            else Path(code_path)
        )
        code_revision = _hex(code.get("revision"), 40, "current7 code revision")
        strict_rows = _positive_integer(
            dataset.get("strict_full_rows"), "manifest strict rows"
        )
        _hex(dataset.get("sha256"), 64, "manifest dataset SHA-256")
        _hex(
            (value.get("train_report") or {}).get("sha256"),
            64,
            "manifest train report SHA-256",
        )
        _hex(
            (value.get("candidate") or {}).get("sha256"),
            64,
            "manifest candidate SHA-256",
        )
        _hex(
            (value.get("quality_status") or {}).get("sha256"),
            64,
            "manifest quality SHA-256",
        )
        _training_recovery_identity(
            recovery,
            "manifest",
            strict_rows=strict_rows,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("corrected adapter manifest identity is invalid") from exc
    if (
        value.get("schema_version") != ADAPTER_SCHEMA
        or value.get("generation_targets") != list(CORRECTED_GENERATION_TARGETS)
        or value.get("generation_target_count") != len(CORRECTED_GENERATION_TARGETS)
        or value.get("required_model_targets") != list(CURRENT_REQUIRED_MODEL_TARGETS)
        or value.get("required_model_targets_sha256")
        != CURRENT_REQUIRED_MODEL_TARGETS_SHA256
        or value.get("temperature_contract") != CURRENT_TEMPERATURE_CONTRACT
        or value.get("temperature_contract_sha256")
        != CURRENT_TEMPERATURE_CONTRACT_SHA256
        or value.get("hard_constraint_contract") != CURRENT_STAGE_HARD_CONTRACT
        or value.get("hard_constraint_contract_sha256")
        != CURRENT_STAGE_HARD_CONTRACT_SHA256
        or value.get("portability")
        != CURRENT_STAGE_HARD_CONTRACT["portability"]
        or value.get("artifact_count") != 2 * len(CORRECTED_GENERATION_TARGETS)
        or set(artifact_sizes) != expected_artifacts
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in artifact_sizes.values()
        )
        or recovery.get("guard_passed") is not True
        or recovery.get("guard_target_count")
        != len(CORRECTED_GENERATION_TARGETS)
        or recovery.get("guard_passed_target_count")
        != len(CORRECTED_GENERATION_TARGETS)
        or value.get("full_artifact_hash_pass_deferred_to_single_model_load")
        is not True
        or code.get("clean") is not True
        or (
            source_paths_required
            and str(code_root) != str(Path(str(code.get("path"))).resolve())
        )
        or code_revision != str(code.get("revision")).lower()
        or model_loading != {
            "process_scope": "single_local_process",
            "cache_policy": "one_generation_authentication_and_unpickle_pass",
            "generation_copy_performed": False,
            "legacy_feedback_wrapper_used": False,
        }
        or not isinstance((value.get("quality_status") or {}).get("passed"), bool)
        or (
            (value.get("quality_status") or {}).get("passed") is False
            and _positive_integer(
                (value.get("quality_status") or {}).get("blocking_reason_count"),
                "manifest quality blocker count",
            ) < 1
        )
        or value.get("production_eligible") is not False
        or value.get("automatic_promotion_allowed") is not False
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "slurm_submission_performed",
                "canonical_pointer_write_performed",
            )
        )
    ):
        raise RuntimeError("corrected adapter manifest contract mismatch")
    return value
