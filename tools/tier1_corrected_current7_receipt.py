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
import hashlib
import json
from typing import Any, Mapping


ADAPTER_SCHEMA = "mft-tier1-corrected-generation-adapter-v1"
SMOKE_RECEIPT_SCHEMA = "mft-tier1-corrected-generation-smoke-receipt-v1"

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

LEGACY_ALL11_TARGETS = frozenset(
    ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core")
)


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


CURRENT_REQUIRED_MODEL_TARGETS_SHA256 = canonical_sha256(
    list(CURRENT_REQUIRED_MODEL_TARGETS)
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


def _adapter_view(receipt: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Return the adapter manifest and whether a model-load smoke was sealed.

    This is the intentionally isolated field-mapping seam.  The first adapter
    implementation emits its manifest directly.  Its smoke wrapper nests that
    exact manifest under ``adapter_manifest`` and adds a one-process load seal.
    """

    schema = receipt.get("schema_version")
    if schema == ADAPTER_SCHEMA:
        return receipt, False
    if schema != SMOKE_RECEIPT_SCHEMA:
        raise RuntimeError("unsupported corrected adapter receipt schema")
    if receipt.get("status") not in {
        "passed",
        "authenticated_model_smoke_passed_launch_blocked",
        "authenticated_model_smoke_passed_launch_eligible",
    }:
        raise RuntimeError("corrected adapter smoke receipt did not pass")
    if "payload_sha256" in receipt:
        unsigned = {
            key: item for key, item in receipt.items() if key != "payload_sha256"
        }
        if receipt.get("payload_sha256") != canonical_sha256(unsigned):
            raise RuntimeError("corrected adapter smoke payload SHA-256 mismatch")
    adapter = _mapping(receipt.get("adapter_manifest"), "adapter_manifest")
    if adapter.get("schema_version") != ADAPTER_SCHEMA:
        raise RuntimeError("smoke receipt nested adapter schema mismatch")
    if receipt.get("adapter_manifest_sha256") != canonical_sha256(adapter):
        raise RuntimeError("smoke receipt adapter-manifest SHA-256 mismatch")
    return adapter, True


def adapter_manifest_view(value: Any) -> Mapping[str, Any]:
    """Expose the normalized adapter object for documentary-path sealing only."""

    receipt = _mapping(value, "corrected adapter receipt")
    adapter, _is_smoke = _adapter_view(receipt)
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
    optimizer_repair_contract_sha256: str | None
    offspring_physics_repair: bool
    fixed_primary_turns_supported: tuple[int, ...]
    initial_repair_attested: bool
    warm_repair_attested: bool
    every_offspring_decode_repair_attested: bool
    terminal_physical_replay_attested: bool
    launch_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fixed_primary_turns_supported"] = list(
            self.fixed_primary_turns_supported
        )
        return value


def validate_adapter_receipt(value: Any) -> CorrectedReceiptIdentity:
    """Validate one local receipt without resolving any recorded path."""

    receipt = _mapping(value, "corrected adapter receipt")
    adapter, is_smoke = _adapter_view(receipt)
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

    repair_contract: Mapping[str, Any] = {}
    repair_contract_sha: str | None = None
    repair_gate = False
    if is_smoke:
        if isinstance(receipt.get("model_loading"), Mapping):
            smoke = receipt["model_loading"]
            smoke_passed = bool(
                smoke.get("required_targets")
                == list(CURRENT_REQUIRED_MODEL_TARGETS)
                and smoke.get("required_targets_sha256")
                == CURRENT_REQUIRED_MODEL_TARGETS_SHA256
                and smoke.get("loaded_target_count")
                == len(CURRENT_REQUIRED_MODEL_TARGETS)
                and smoke.get("cache_load_calls") == 1
                and smoke.get("full_generation_authentication_passes") == 1
                and smoke.get("models_loaded_once_per_process") is True
                and smoke.get("generation_copy_performed") is False
                and smoke.get("model_smoke_completed") is True
            )
        else:
            smoke = _mapping(receipt.get("model_load"), "model_load")
            smoke_passed = bool(
                smoke.get("process_scope") == "single_local_process"
                and smoke.get("local_process_count") == 1
                and smoke.get("cache_loaded_once") is True
                and smoke.get("load_calls") == 1
                and smoke.get("full_generation_authentication_passes") == 1
                and smoke.get("loaded_model_count")
                == len(CURRENT_REQUIRED_MODEL_TARGETS)
                and smoke.get("loaded_model_targets_sha256")
                == CURRENT_REQUIRED_MODEL_TARGETS_SHA256
            )
        if not smoke_passed:
            raise RuntimeError("corrected adapter one-process smoke gate mismatch")
        if any(
            receipt.get(field) is not False
            for field in (
                "production_eligible",
                "automatic_promotion_allowed",
                "scheduler_write_performed",
                "slurm_submission_performed",
                "canonical_pointer_write_performed",
            )
        ):
            raise RuntimeError("corrected adapter smoke fail-closed policy mismatch")
        candidate_repair = receipt.get("physics_repair")
        if not isinstance(candidate_repair, Mapping):
            candidate_repair = receipt.get("optimizer_repair")
        if isinstance(candidate_repair, Mapping):
            repair_contract = candidate_repair
            unsigned = {
                key: item
                for key, item in repair_contract.items()
                if key != "sha256"
            }
            repair_contract_sha = canonical_sha256(unsigned)
            has_seal = repair_contract.get("sha256") == repair_contract_sha
            if repair_contract.get("sha256") is not None and not has_seal:
                raise RuntimeError("optimizer repair contract SHA-256 mismatch")
            stages = repair_contract.get("stages")
            if not isinstance(stages, Mapping):
                stages = {
                    "initial_population_repair": repair_contract.get(
                        "initial_population_repair"
                    ),
                    "warm_start_repair": repair_contract.get(
                        "warm_start_repair"
                    ),
                    "every_offspring_decode_repair": repair_contract.get(
                        "every_offspring_decode_repair",
                        repair_contract.get("offspring_physics_repair"),
                    ),
                    "terminal_physical_replay": repair_contract.get(
                        "terminal_physical_replay"
                    ),
                }
            fixed_supported = repair_contract.get(
                "fixed_primary_turns_supported"
            )
            if fixed_supported is None:
                fixed_supported = repair_contract.get("fixed_primary_turns")
            if isinstance(fixed_supported, int):
                fixed_supported = [fixed_supported]
            runner_contract = receipt.get("runner") or {}
            repair_gate = bool(
                has_seal
                and repair_contract.get("schema_version")
                == "mft-tier1-current7-physics-repair-attestation-v1"
                and repair_contract.get("offspring_physics_repair") is True
                and repair_contract.get("fixed_primary_turns_repair") is True
                and fixed_supported == [5, 6]
                and stages.get("initial_population_repair") is True
                and stages.get("warm_start_repair") is True
                and stages.get("every_offspring_decode_repair") is True
                and stages.get("terminal_physical_replay") is True
                and repair_contract.get("warm_coordinates_are_donors_only") is True
                and repair_contract.get(
                    "source_prediction_or_pass_classification_inherited"
                )
                is False
                and repair_contract.get("launch_eligible") is True
                and (
                    not runner_contract
                    or runner_contract.get("launch_eligible") is True
                )
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
            adapter.get("temperature_contract_sha256"),
            64,
            "temperature contract SHA-256",
        ),
        hard_constraint_contract_sha256=_hex(
            adapter.get("hard_constraint_contract_sha256"),
            64,
            "hard constraint contract SHA-256",
        ),
        adapter_code_revision=_hex(
            code.get("revision"), 40, "adapter code revision"
        ),
        local_model_load_smoke_passed=is_smoke,
        optimizer_repair_contract_sha256=repair_contract_sha,
        offspring_physics_repair=bool(
            repair_contract.get("offspring_physics_repair") is True
        ),
        fixed_primary_turns_supported=tuple(
            (
                repair_contract.get("fixed_primary_turns_supported")
                or repair_contract.get("fixed_primary_turns")
                or ()
            )
            if not isinstance(
                repair_contract.get("fixed_primary_turns_supported")
                or repair_contract.get("fixed_primary_turns"),
                int,
            )
            else [
                repair_contract.get("fixed_primary_turns_supported")
                or repair_contract.get("fixed_primary_turns")
            ]
        ),
        initial_repair_attested=bool(
            (repair_contract.get("stages") or repair_contract).get(
                "initial_population_repair"
            )
            is True
        ),
        warm_repair_attested=bool(
            (repair_contract.get("stages") or repair_contract).get(
                "warm_start_repair"
            )
            is True
        ),
        every_offspring_decode_repair_attested=bool(
            (repair_contract.get("stages") or {}).get(
                "every_offspring_decode_repair"
            )
            is True
            or repair_contract.get("every_offspring_decode_repair") is True
            or repair_contract.get("offspring_physics_repair") is True
        ),
        terminal_physical_replay_attested=bool(
            (repair_contract.get("stages") or repair_contract).get(
                "terminal_physical_replay"
            )
            is True
        ),
        launch_eligible=bool(is_smoke and repair_gate),
    )
