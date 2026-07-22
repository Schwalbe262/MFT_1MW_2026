from __future__ import annotations

import copy
import hashlib
from pathlib import Path, PurePosixPath

import pytest

from tools import tier1_final1000_resource2_canary as canary
from tools import tier1_resource2_cgroup_memory as cgroup


ACTUAL_V1_CGROUP = """12:devices:/slurm_n012/uid_1634/job_799377/step_347/task_0
11:hugetlb:/
10:blkio:/system.slice/slurmd.service
9:rdma:/
8:freezer:/slurm_n012/uid_1634/job_799377/step_347
7:perf_event:/
6:memory:/slurm_n012/system
5:pids:/system.slice/slurmd.service
4:cpuset:/slurm_n012/uid_1634/job_799377/step_347
3:cpu,cpuacct:/system.slice/slurmd.service
2:net_cls,net_prio:/
1:name=systemd:/system.slice/slurmd.service
"""

ACTUAL_V1_MOUNTINFO = """33 32 0:26 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime shared:5 - cgroup cgroup rw,xattr,release_agent=/usr/lib/systemd/systemd-cgroups-agent,name=systemd
39 32 0:32 / /sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,cpuset
41 32 0:34 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:18 - cgroup cgroup rw,memory
47 32 0:40 / /sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime shared:24 - cgroup cgroup rw,devices
"""


def _write_triplet(
    directory: Path,
    *,
    version: str,
    limit: str,
    current: int,
    peak: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "v1": (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
        ),
        "v2": ("memory.max", "memory.current", "memory.peak"),
    }[version]
    (directory / filenames[0]).write_text(limit + "\n", encoding="ascii")
    (directory / filenames[1]).write_text(str(current) + "\n", encoding="ascii")
    (directory / filenames[2]).write_text(str(peak) + "\n", encoding="ascii")


def test_actual_n012_v1_memory_membership_and_mount_mapping(tmp_path: Path):
    root = tmp_path / "cgroup"
    leaf = root / "memory" / "slurm_n012" / "system"
    _write_triplet(
        leaf,
        version="v1",
        limit="30064771072",
        current=4 * 1024**3,
        peak=5 * 1024**3,
    )
    snapshot = cgroup.cgroup_snapshot_from_text(
        ACTUAL_V1_CGROUP, ACTUAL_V1_MOUNTINFO, root
    )
    sealed = cgroup.validate_cgroup_snapshot(snapshot)
    assert sealed["cgroup_version"] == "v1"
    assert sealed["hierarchy_id"] == 6
    assert sealed["membership_path"] == "/slurm_n012/system"
    assert sealed["mount_relative_path"] == "memory"
    assert sealed["leaf_relative_path"] == "memory/slurm_n012/system"
    assert sealed["selected_finite_ancestor"]["memory_limit_bytes"] == 30_064_771_072


def test_actual_n012_v1_stats_select_finite_leaf_and_normalize_parent_sentinel(
    tmp_path: Path,
):
    """Seal the bounded task-86140 probe values that motivated the v1 fix."""

    root = tmp_path / "cgroup"
    mount = root / "memory"
    parent = mount / "slurm_n012"
    leaf = parent / "system"
    _write_triplet(
        leaf,
        version="v1",
        limit="786432000000",
        current=364_038_721_536,
        peak=775_322_603_520,
    )
    _write_triplet(
        parent,
        version="v1",
        limit="9223372036854771712",
        current=364_039_028_736,
        peak=775_322_714_112,
    )
    _write_triplet(
        mount,
        version="v1",
        limit="9223372036854771712",
        current=392_483_864_576,
        peak=795_393_523_712,
    )

    snapshot = cgroup.cgroup_snapshot_from_text(
        ACTUAL_V1_CGROUP, ACTUAL_V1_MOUNTINFO, root
    )
    sealed = cgroup.validate_cgroup_snapshot(snapshot)

    assert sealed["nearest_accounting_limit_unbounded"] is False
    assert sealed["selected_finite_ancestor"] == sealed["ancestors"][0]
    assert sealed["selected_finite_ancestor"]["memory_limit_bytes"] == 786_432_000_000
    assert sealed["selected_finite_ancestor"]["memory_current_bytes"] == 364_038_721_536
    assert sealed["selected_finite_ancestor"]["memory_peak_bytes"] == 775_322_603_520
    assert [record["memory_limit_bytes"] for record in sealed["ancestors"]] == [
        786_432_000_000,
        None,
        None,
    ]
    assert [record["memory_max_raw"] for record in sealed["ancestors"]] == [
        "786432000000",
        "9223372036854771712",
        "9223372036854771712",
    ]


def test_v1_unlimited_leaf_selects_nearest_finite_parent(tmp_path: Path):
    root = tmp_path / "cgroup"
    leaf = root / "memory" / "slurm_n012" / "system"
    parent = leaf.parent
    _write_triplet(
        leaf,
        version="v1",
        limit="9223372036854771712",
        current=1024,
        peak=2048,
    )
    _write_triplet(
        parent,
        version="v1",
        limit="68719476736",
        current=4096,
        peak=8192,
    )
    snapshot = cgroup.cgroup_snapshot_from_text(
        "6:memory:/slurm_n012/system\n", ACTUAL_V1_MOUNTINFO, root
    )
    assert snapshot["nearest_accounting_limit_unbounded"] is True
    assert snapshot["selected_finite_ancestor"]["depth_from_leaf"] == 1
    assert snapshot["selected_finite_ancestor"]["memory_limit_bytes"] == 68_719_476_736
    cgroup.validate_cgroup_snapshot(snapshot)


def test_v2_unbounded_leaf_and_finite_parent_are_supported(tmp_path: Path):
    root = tmp_path / "cgroup"
    leaf = root / "slurm" / "job" / "task"
    _write_triplet(leaf, version="v2", limit="max", current=100, peak=200)
    _write_triplet(
        leaf.parent,
        version="v2",
        limit="30064771072",
        current=300,
        peak=400,
    )
    snapshot = cgroup.cgroup_snapshot_from_text(
        "0::/slurm/job/task\n",
        "30 29 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        root,
    )
    assert snapshot["cgroup_version"] == "v2"
    assert snapshot["nearest_accounting_limit_unbounded"] is True
    assert snapshot["selected_finite_ancestor"]["depth_from_leaf"] == 1
    cgroup.validate_cgroup_snapshot(snapshot)


def test_hybrid_selects_only_memory_capable_finite_hierarchy(tmp_path: Path):
    root = tmp_path / "cgroup"
    (root / "unified" / "ignored").mkdir(parents=True)
    _write_triplet(
        root / "memory" / "slurm_n012" / "system",
        version="v1",
        limit="30064771072",
        current=100,
        peak=200,
    )
    snapshot = cgroup.cgroup_snapshot_from_text(
        "0::/ignored\n6:memory:/slurm_n012/system\n",
        (
            "30 29 0:26 / /sys/fs/cgroup/unified rw - cgroup2 cgroup rw\n"
            + ACTUAL_V1_MOUNTINFO
        ),
        root,
    )
    assert snapshot["cgroup_version"] == "v1"


def test_hybrid_with_two_finite_memory_hierarchies_is_rejected(tmp_path: Path):
    root = tmp_path / "cgroup"
    _write_triplet(
        root / "unified" / "unified_task",
        version="v2",
        limit="30064771072",
        current=100,
        peak=200,
    )
    _write_triplet(
        root / "memory" / "slurm_n012" / "system",
        version="v1",
        limit="30064771072",
        current=100,
        peak=200,
    )
    with pytest.raises(RuntimeError, match="unique finite"):
        cgroup.cgroup_snapshot_from_text(
            "0::/unified_task\n6:memory:/slurm_n012/system\n",
            (
                "30 29 0:26 / /sys/fs/cgroup/unified rw - cgroup2 cgroup rw\n"
                + ACTUAL_V1_MOUNTINFO
            ),
            root,
        )


@pytest.mark.parametrize(
    "membership",
    [
        "6:memory:/slurm/../secret\n",
        "6:memory:/slurm//system\n",
        "6:memory:relative/path\n",
        "6:memory:/one\n7:memory:/two\n",
        "not:a:valid:record\n",
        "0:memory:/wrong-v1-zero\n",
    ],
)
def test_malformed_or_ambiguous_membership_is_rejected(
    tmp_path: Path, membership: str
):
    root = tmp_path / "cgroup"
    root.mkdir()
    with pytest.raises(RuntimeError):
        cgroup.cgroup_snapshot_from_text(membership, ACTUAL_V1_MOUNTINFO, root)


def test_mount_outside_sysfs_and_membership_outside_mount_root_are_rejected(
    tmp_path: Path,
):
    root = tmp_path / "cgroup"
    root.mkdir()
    with pytest.raises(RuntimeError, match="escaped /sys/fs/cgroup"):
        cgroup.cgroup_snapshot_from_text(
            "6:memory:/slurm/system\n",
            "41 32 0:34 / /tmp/memory rw - cgroup cgroup rw,memory\n",
            root,
        )
    with pytest.raises(RuntimeError, match="unique finite"):
        cgroup.cgroup_snapshot_from_text(
            "6:memory:/other/system\n",
            "41 32 0:34 /slurm /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
            root,
        )


def test_partial_accounting_and_peak_below_usage_fail_closed(tmp_path: Path):
    root = tmp_path / "cgroup"
    leaf = root / "memory" / "slurm_n012" / "system"
    leaf.mkdir(parents=True)
    (leaf / "memory.limit_in_bytes").write_text("30064771072\n")
    with pytest.raises(RuntimeError, match="partial"):
        cgroup.cgroup_snapshot_from_text(
            "6:memory:/slurm_n012/system\n", ACTUAL_V1_MOUNTINFO, root
        )
    (leaf / "memory.usage_in_bytes").write_text("200\n")
    (leaf / "memory.max_usage_in_bytes").write_text("100\n")
    with pytest.raises(RuntimeError, match="below current"):
        cgroup.cgroup_snapshot_from_text(
            "6:memory:/slurm_n012/system\n", ACTUAL_V1_MOUNTINFO, root
        )
    (leaf / "memory.max_usage_in_bytes").write_text("1" * 129, encoding="ascii")
    with pytest.raises(RuntimeError, match="exceeds 128 bytes"):
        cgroup.cgroup_snapshot_from_text(
            "6:memory:/slurm_n012/system\n", ACTUAL_V1_MOUNTINFO, root
        )


def test_snapshot_validator_rejects_path_hierarchy_and_limit_mutations(
    tmp_path: Path,
):
    root = tmp_path / "cgroup"
    leaf = root / "memory" / "slurm_n012" / "system"
    _write_triplet(
        leaf,
        version="v1",
        limit="30064771072",
        current=100,
        peak=200,
    )
    good = cgroup.cgroup_snapshot_from_text(
        "6:memory:/slurm_n012/system\n", ACTUAL_V1_MOUNTINFO, root
    )
    for mutation in (
        lambda value: value.__setitem__("leaf_relative_path", "../escape"),
        lambda value: value.__setitem__("membership_path", "/slurm/../escape"),
        lambda value: value["ancestors"][0].__setitem__("depth_from_leaf", 3),
        lambda value: value["ancestors"][0].__setitem__("memory_max_raw", "max"),
        lambda value: value["ancestors"][0].__setitem__("memory_peak_bytes", 99),
    ):
        bad = copy.deepcopy(good)
        mutation(bad)
        with pytest.raises(RuntimeError):
            cgroup.validate_cgroup_snapshot(bad)


def test_bounded_raw_diagnostics_are_sealed_and_truncation_is_rejected(
    tmp_path: Path,
):
    cgroup_path = tmp_path / "cgroup"
    mountinfo_path = tmp_path / "mountinfo"
    cgroup_path.write_text("6:memory:/slurm_n012/system\n", encoding="utf-8")
    mountinfo_path.write_text(ACTUAL_V1_MOUNTINFO, encoding="utf-8")
    diagnostics = cgroup.capture_cgroup_diagnostics(
        cgroup_path=cgroup_path,
        mountinfo_path=mountinfo_path,
        sysfs_root=Path("/sys/fs/cgroup"),
    )
    # Production validation pins the canonical procfs path names.  Rebind only
    # those two audit labels while preserving the actually captured bytes.
    diagnostics["proc_self_cgroup"]["path"] = "/proc/self/cgroup"
    diagnostics["proc_self_mountinfo"]["path"] = "/proc/self/mountinfo"
    cgroup.validate_cgroup_diagnostics(diagnostics)
    bad = copy.deepcopy(diagnostics)
    bad["proc_self_cgroup"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="unsealed"):
        cgroup.validate_cgroup_diagnostics(bad)

    cgroup_path.write_bytes(b"x" * 17)
    bounded = cgroup.capture_cgroup_diagnostics(
        cgroup_path=cgroup_path,
        mountinfo_path=mountinfo_path,
        maximum_bytes=16,
    )
    assert bounded["proc_self_cgroup"]["bytes_captured"] == 16
    assert bounded["proc_self_cgroup"]["truncated"] is True
    with pytest.raises(RuntimeError, match="truncated"):
        cgroup.cgroup_snapshot_from_diagnostics(bounded, tmp_path)


def test_remote_wrapper_embeds_the_exact_tested_parser_and_captures_before_parse():
    source = cgroup.embedded_runtime_source()
    assert source in canary._INLINE_WRAPPER
    assert "memory.limit_in_bytes" in source
    assert "memory.max_usage_in_bytes" in source
    assert "memory.max" in source
    assert canary._INLINE_WRAPPER.index(
        'cgroup_diagnostics["before"] = capture_cgroup_diagnostics()'
    ) < canary._INLINE_WRAPPER.index(
        'cgroup_before = cgroup_snapshot_from_diagnostics(cgroup_diagnostics["before"])'
    )
    namespace = {
        "Path": Path,
        "PurePosixPath": PurePosixPath,
        "hashlib": hashlib,
    }
    exec(source, namespace)
    assert callable(namespace["cgroup_snapshot_from_text"])
