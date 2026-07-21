"""Fail-closed adapter-receipt mapping for corrected current7 Slurm lanes.

Only this module understands the local corrected-generation adapter receipt.
The bundle planner and the remote runner consume the normalized identity
returned here, so an additive adapter receipt revision can be accommodated
without weakening the Slurm task or bundle contracts.

Paths recorded by the local adapter are documentary evidence.  They are never
resolved by the Linux runner; the immutable bundle relocation contract binds
the corresponding bytes to bundle-relative paths instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

try:
    from tier1_corrected_generation_adapter import (
        ADAPTER_SCHEMA,
        CORRECTED_GENERATION_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        canonical_sha256,
        training_profile_sha256,
    )
    from tier1_corrected_generation_preflight import (
        RECEIPT_SCHEMA as SMOKE_RECEIPT_SCHEMA,
        SUPPORTED_FIXED_PRIMARY_TURNS,
        stage_constraint_names,
        stage_hard_constraint_contract,
        stage_temperature_contract,
        validate_smoke_receipt as authoritative_validate_smoke_receipt,
        validate_stage_spec,
    )
except ImportError:  # pragma: no cover - repository package path
    from tools.tier1_corrected_generation_adapter import (
        ADAPTER_SCHEMA,
        CORRECTED_GENERATION_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        canonical_sha256,
        training_profile_sha256,
    )
    from tools.tier1_corrected_generation_preflight import (
        RECEIPT_SCHEMA as SMOKE_RECEIPT_SCHEMA,
        SUPPORTED_FIXED_PRIMARY_TURNS,
        stage_constraint_names,
        stage_hard_constraint_contract,
        stage_temperature_contract,
        validate_smoke_receipt as authoritative_validate_smoke_receipt,
        validate_stage_spec,
    )

LEGACY_ALL11_TARGETS = frozenset(
    ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core")
)

__all__ = (
    "ADAPTER_SCHEMA",
    "CORRECTED_GENERATION_TARGETS",
    "CURRENT_REQUIRED_MODEL_TARGETS",
    "CURRENT_REQUIRED_MODEL_TARGETS_SHA256",
    "CURRENT_TEMPERATURE_TARGETS",
    "SMOKE_RECEIPT_SCHEMA",
    "CorrectedReceiptIdentity",
    "adapter_manifest_view",
    "canonical_sha256",
    "expected_generation_artifacts",
    "training_profile_sha256",
    "validate_adapter_receipt",
)


def expected_generation_artifacts() -> tuple[str, ...]:
    return tuple(
        f"{target}/{filename}"
        for target in CORRECTED_GENERATION_TARGETS
        for filename in ("meta.json", "models.pkl")
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _hex(value: Any, length: int, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(
            f"{label} must be a {length}-character hexadecimal digest"
        )
    return normalized


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be a positive integer") from exc
    if integer <= 0 or value != integer:
        raise RuntimeError(f"{label} must be a positive integer")
    return integer


def _adapter_view(
    receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Return the adapter manifest and whether a model-load smoke was sealed.

    This is the intentionally isolated field-mapping seam.  The first adapter
    implementation emits its manifest directly.  Its smoke wrapper nests that
    exact manifest under ``adapter_manifest`` and adds a one-process load seal.
    """

    schema = receipt.get("schema_version")
    if schema == ADAPTER_SCHEMA:
        return receipt, None
    if schema != SMOKE_RECEIPT_SCHEMA:
        raise RuntimeError("unsupported corrected adapter receipt schema")
    # The producer owns all v2 dual-stratum semantics.  This adapter deliberately
    # does not duplicate that validator; it extracts only a sealed Slurm view.
    validated = authoritative_validate_smoke_receipt(
        dict(receipt), relocated_source_evidence=True
    )
    adapter = _mapping(validated.get("adapter_manifest"), "adapter_manifest")
    return adapter, validated


def adapter_manifest_view(value: Any) -> Mapping[str, Any]:
    """Expose the normalized adapter object for documentary-path sealing only."""

    receipt = _mapping(value, "corrected adapter receipt")
    adapter, _smoke = _adapter_view(receipt)
    return adapter


@dataclass(frozen=True)
class CorrectedReceiptIdentity:
    receipt_schema_version: str
    adapter_schema_version: str
    adapter_manifest_sha256: str
    training_run_id: str
    generation_relative: str
    train_report_sha256: str
    candidate_sha256: str
    quality_status_sha256: str
    quality_passed: bool
    quality_blocking_reason_count: int
    dataset_sha256: str
    strict_full_rows: int
    profile_canonical_sha256: str
    artifact_count: int
    artifact_sizes_bytes: dict[str, int]
    required_model_targets_sha256: str
    temperature_contract_sha256: str
    hard_constraint_contract_sha256: str
    adapter_code_revision: str
    local_model_load_smoke_passed: bool
    optimizer_repair_contract_sha256_by_fixed_primary_turns: dict[str, str]
    offspring_physics_repair: bool
    fixed_primary_turns_supported: tuple[int, ...]
    initial_repair_attested: bool
    warm_repair_attested: bool
    every_offspring_decode_repair_attested: bool
    terminal_physical_replay_attested: bool
    constraint_version: str | None
    hard_spec: dict[str, Any] | None
    hard_spec_sha256: str | None
    constraint_names: tuple[str, ...]
    launch_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fixed_primary_turns_supported"] = list(
            self.fixed_primary_turns_supported
        )
        value["constraint_names"] = list(self.constraint_names)
        return value


def validate_adapter_receipt(value: Any) -> CorrectedReceiptIdentity:
    """Validate one local receipt without resolving any recorded path."""

    receipt = _mapping(value, "corrected adapter receipt")
    adapter, smoke_receipt = _adapter_view(receipt)
    is_smoke = smoke_receipt is not None
    if adapter.get("schema_version") != ADAPTER_SCHEMA:
        raise RuntimeError("corrected adapter manifest schema mismatch")

    generation_targets = adapter.get("generation_targets")
    required_targets = adapter.get("required_model_targets")
    if generation_targets != list(CORRECTED_GENERATION_TARGETS):
        raise RuntimeError("corrected generation target order/set mismatch")
    if required_targets != list(CURRENT_REQUIRED_MODEL_TARGETS):
        raise RuntimeError("current7 required-model target order/set mismatch")
    if LEGACY_ALL11_TARGETS.intersection(generation_targets or ()):
        raise RuntimeError("legacy all11 temperature target leaked into current7")
    if LEGACY_ALL11_TARGETS.intersection(required_targets or ()):
        raise RuntimeError("legacy all11 required model leaked into current7")
    if adapter.get("generation_target_count") != len(
        CORRECTED_GENERATION_TARGETS
    ):
        raise RuntimeError("corrected generation target count mismatch")
    if adapter.get("required_model_targets_sha256") != (
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256
    ):
        raise RuntimeError("current7 required-model target digest mismatch")

    sizes = _mapping(
        adapter.get("artifact_sizes_bytes"), "artifact_sizes_bytes"
    )
    expected_artifacts = set(expected_generation_artifacts())
    if set(sizes) != expected_artifacts:
        raise RuntimeError("corrected generation artifact-size inventory mismatch")
    normalized_sizes = {
        str(relative): _positive_integer(size, f"artifact size {relative}")
        for relative, size in sizes.items()
    }
    if adapter.get("artifact_count") != len(expected_artifacts):
        raise RuntimeError("corrected generation artifact count mismatch")

    recovery = _mapping(
        adapter.get("capacitance_recovery"), "capacitance_recovery"
    )
    if (
        recovery.get("guard_passed") is not True
        or recovery.get("guard_target_count") != len(
            CORRECTED_GENERATION_TARGETS
        )
        or recovery.get("guard_passed_target_count") != len(
            CORRECTED_GENERATION_TARGETS
        )
    ):
        raise RuntimeError("corrected capacitance-recovery guard mismatch")

    model_loading = _mapping(adapter.get("model_loading"), "model_loading")
    if (
        model_loading.get("process_scope") != "single_local_process"
        or model_loading.get("cache_policy")
        != "one_generation_authentication_and_unpickle_pass"
        or model_loading.get("generation_copy_performed") is not False
        or model_loading.get("legacy_feedback_wrapper_used") is not False
    ):
        raise RuntimeError("corrected adapter model-loading policy mismatch")

    if any(
        adapter.get(field) is not False
        for field in (
            "production_eligible",
            "automatic_promotion_allowed",
            "scheduler_write_performed",
            "slurm_submission_performed",
            "canonical_pointer_write_performed",
        )
    ):
        raise RuntimeError("corrected adapter fail-closed policy mismatch")

    repair_contract_by_turns: dict[str, str] = {}
    repair_gate = False
    constraint_version: str | None = None
    hard_spec: dict[str, Any] | None = None
    hard_spec_sha: str | None = None
    constraint_names: tuple[str, ...] = ()
    staged_temperature_contract_sha: str | None = None
    staged_hard_contract_sha: str | None = None
    if smoke_receipt is not None:
        for turns in SUPPORTED_FIXED_PRIMARY_TURNS:
            key = str(turns)
            stratum = smoke_receipt["strata"][key]
            repair_contract_by_turns[key] = _hex(
                stratum["optimizer_repair"]["contract_sha256"],
                64,
                f"N1={turns} optimizer repair contract SHA-256",
            )
        problem = smoke_receipt["problem_contract"]
        hard_spec = validate_stage_spec(problem["stage_spec"])
        hard_spec_sha = _hex(
            problem["stage_spec_sha256"], 64, "hard spec SHA-256"
        )
        constraint_names = tuple(problem["constraint_names"])
        hard_contract = stage_hard_constraint_contract(hard_spec)
        temperature_contract = stage_temperature_contract(hard_spec)
        constraint_version = str(hard_contract["stage"])
        staged_temperature_contract_sha = canonical_sha256(
            temperature_contract
        )
        staged_hard_contract_sha = canonical_sha256(hard_contract)
        repair_gate = bool(
            hard_spec_sha == canonical_sha256(hard_spec)
            and constraint_names == stage_constraint_names(hard_spec)
            and problem.get("temperature_contract") == temperature_contract
            and problem.get("temperature_contract_sha256")
            == staged_temperature_contract_sha
            and problem.get("hard_constraint_contract") == hard_contract
            and problem.get("hard_constraint_contract_sha256")
            == staged_hard_contract_sha
            and set(repair_contract_by_turns) == {"5", "6"}
        )

    report = _mapping(adapter.get("train_report"), "train_report")
    candidate = _mapping(adapter.get("candidate"), "candidate")
    quality = _mapping(adapter.get("quality_status"), "quality_status")
    dataset = _mapping(adapter.get("dataset"), "dataset")
    profile = _mapping(adapter.get("profile"), "profile")
    code = _mapping(adapter.get("code"), "code")
    if not isinstance(quality.get("passed"), bool):
        raise RuntimeError("corrected quality terminal state is missing")
    blocker_count = int(quality.get("blocking_reason_count", -1))
    if blocker_count < 0 or (quality.get("passed") is False and blocker_count < 1):
        raise RuntimeError("corrected failed quality has no sealed blockers")
    if code.get("clean") is not True:
        raise RuntimeError("local adapter code identity was not clean")

    run_id = str(adapter.get("training_run_id") or "").strip()
    generation_relative = str(adapter.get("generation_relative") or "").strip()
    if not run_id or generation_relative != f"generations/{run_id}":
        raise RuntimeError("corrected generation relative identity mismatch")

    return CorrectedReceiptIdentity(
        receipt_schema_version=str(receipt.get("schema_version")),
        adapter_schema_version=ADAPTER_SCHEMA,
        adapter_manifest_sha256=canonical_sha256(adapter),
        training_run_id=run_id,
        generation_relative=generation_relative,
        train_report_sha256=_hex(
            report.get("sha256"), 64, "train report SHA-256"
        ),
        candidate_sha256=_hex(candidate.get("sha256"), 64, "candidate SHA-256"),
        quality_status_sha256=_hex(
            quality.get("sha256"), 64, "quality status SHA-256"
        ),
        quality_passed=bool(quality["passed"]),
        quality_blocking_reason_count=blocker_count,
        dataset_sha256=_hex(dataset.get("sha256"), 64, "dataset SHA-256"),
        strict_full_rows=_positive_integer(
            dataset.get("strict_full_rows"), "strict-full rows"
        ),
        profile_canonical_sha256=_hex(
            profile.get("canonical_sha256"), 64, "profile canonical SHA-256"
        ),
        artifact_count=len(expected_artifacts),
        artifact_sizes_bytes=normalized_sizes,
        required_model_targets_sha256=CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        temperature_contract_sha256=_hex(
            staged_temperature_contract_sha
            or adapter.get("temperature_contract_sha256"),
            64,
            "temperature contract SHA-256",
        ),
        hard_constraint_contract_sha256=_hex(
            staged_hard_contract_sha
            or adapter.get("hard_constraint_contract_sha256"),
            64,
            "hard constraint contract SHA-256",
        ),
        adapter_code_revision=_hex(
            code.get("revision"), 40, "adapter code revision"
        ),
        local_model_load_smoke_passed=is_smoke,
        optimizer_repair_contract_sha256_by_fixed_primary_turns=(
            repair_contract_by_turns
        ),
        offspring_physics_repair=bool(is_smoke and repair_gate),
        fixed_primary_turns_supported=(
            tuple(SUPPORTED_FIXED_PRIMARY_TURNS) if is_smoke else ()
        ),
        initial_repair_attested=bool(is_smoke and repair_gate),
        warm_repair_attested=bool(is_smoke and repair_gate),
        every_offspring_decode_repair_attested=bool(is_smoke and repair_gate),
        terminal_physical_replay_attested=bool(is_smoke and repair_gate),
        constraint_version=constraint_version,
        hard_spec=hard_spec,
        hard_spec_sha256=hard_spec_sha,
        constraint_names=constraint_names,
        launch_eligible=bool(is_smoke and repair_gate),
    )
