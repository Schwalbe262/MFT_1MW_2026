"""Content-addressed immutable generation storage.

Writers stage a complete directory, verify every copied byte, write the
manifest, and install ``COMPLETED`` last.  Readers reject directories without
that marker or whose files no longer match the manifest.  The identity omits
timestamps, so replaying the same inputs is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Iterable, Mapping


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETED_NAME = "COMPLETED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str | os.PathLike[str]) -> str:
    raw = os.fspath(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or raw in (MANIFEST_NAME, COMPLETED_NAME)
    ):
        raise ValueError(f"unsafe generation artifact path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class PublishedGeneration:
    kind: str
    generation_id: str
    path: Path
    manifest: dict


class GenerationStore:
    """Publish and authenticate immutable artifact generations."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_kind(kind: str) -> str:
        value = str(kind or "").strip()
        if not value or value in (".", "..") or any(
            char in value for char in "/\\:"
        ):
            raise ValueError(f"invalid generation kind: {kind!r}")
        return value

    def publish_files(
        self,
        kind: str,
        files: Mapping[str, str | os.PathLike[str]],
        *,
        metadata: Mapping[str, object] | None = None,
        parents: Iterable[str] = (),
    ) -> PublishedGeneration:
        kind = self._validate_kind(kind)
        normalized: dict[str, Path] = {}
        inventory: dict[str, dict[str, object]] = {}
        for relative, source in sorted(files.items()):
            rel = _safe_relative(relative)
            source_path = Path(source).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            normalized[rel] = source_path
            inventory[rel] = {
                "sha256": sha256_file(source_path),
                "size": source_path.stat().st_size,
            }
        if not normalized:
            raise ValueError("a generation must contain at least one artifact")

        identity = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "parents": sorted({str(parent) for parent in parents}),
            "metadata": dict(metadata or {}),
            "artifacts": inventory,
        }
        generation_id = hashlib.sha256(canonical_json(identity)).hexdigest()
        kind_root = self.root / kind
        target = kind_root / generation_id
        kind_root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return self.load(target)

        staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=kind_root))
        try:
            for relative, source in normalized.items():
                destination = staging.joinpath(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                expected = inventory[relative]
                if (
                    destination.stat().st_size != expected["size"]
                    or sha256_file(destination) != expected["sha256"]
                ):
                    raise RuntimeError(f"generation copy verification failed: {relative}")

            manifest = {
                **identity,
                "generation_id": generation_id,
                "created_at": _utc_now(),
            }
            manifest_path = staging / MANIFEST_NAME
            manifest_path.write_bytes(canonical_json(manifest) + b"\n")
            marker = {
                "generation_id": generation_id,
                "manifest_sha256": sha256_file(manifest_path),
            }
            # The marker is deliberately the final write inside staging.
            (staging / COMPLETED_NAME).write_bytes(canonical_json(marker) + b"\n")
            try:
                os.replace(staging, target)
            except OSError:
                # Concurrent identical publishers are harmless.  Only accept
                # the winner after fully authenticating it.
                if not target.exists():
                    raise
            return self.load(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def publish_tree(
        self,
        kind: str,
        source: str | os.PathLike[str],
        *,
        metadata: Mapping[str, object] | None = None,
        parents: Iterable[str] = (),
    ) -> PublishedGeneration:
        source_path = Path(source).resolve()
        if not source_path.is_dir():
            raise NotADirectoryError(source_path)
        files = {
            path.relative_to(source_path).as_posix(): path
            for path in source_path.rglob("*")
            if path.is_file()
            and path.name not in (MANIFEST_NAME, COMPLETED_NAME)
        }
        return self.publish_files(
            kind, files, metadata=metadata, parents=parents
        )

    def load(
        self,
        path_or_kind: str | os.PathLike[str],
        generation_id: str | None = None,
    ) -> PublishedGeneration:
        if generation_id is None:
            path = Path(path_or_kind).resolve()
        else:
            kind = self._validate_kind(os.fspath(path_or_kind))
            path = (self.root / kind / generation_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RuntimeError("generation escapes store root") from exc
        manifest_path = path / MANIFEST_NAME
        marker_path = path / COMPLETED_NAME
        if not manifest_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"incomplete generation: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        identity = {
            key: manifest.get(key)
            for key in (
                "schema_version", "kind", "parents", "metadata", "artifacts"
            )
        }
        expected_id = hashlib.sha256(canonical_json(identity)).hexdigest()
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("generation_id") != expected_id
            or path.name != expected_id
            or marker.get("generation_id") != expected_id
            or marker.get("manifest_sha256") != sha256_file(manifest_path)
        ):
            raise RuntimeError(f"generation identity mismatch: {path}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise RuntimeError(f"generation has no artifact inventory: {path}")
        for relative, record in artifacts.items():
            rel = _safe_relative(relative)
            artifact = path.joinpath(*PurePosixPath(rel).parts)
            if (
                not artifact.is_file()
                or artifact.stat().st_size != record.get("size")
                or sha256_file(artifact) != record.get("sha256")
            ):
                raise RuntimeError(f"generation artifact mismatch: {relative}")
        return PublishedGeneration(
            kind=manifest["kind"],
            generation_id=expected_id,
            path=path,
            manifest=manifest,
        )

    def generations(self, kind: str) -> list[PublishedGeneration]:
        kind = self._validate_kind(kind)
        root = self.root / kind
        if not root.is_dir():
            return []
        output = []
        for path in root.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                output.append(self.load(path))
        return sorted(
            output,
            key=lambda item: (
                item.manifest.get("created_at", ""), item.generation_id
            ),
        )
