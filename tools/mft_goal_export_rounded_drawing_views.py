"""Export drawing-oriented views from a retained rounded AEDT project.

This is deliberately a post-retention utility.  ``export_model_picture`` is
an AEDT graphical-mode API, so the final FEA task must not depend on it.  Run
this utility after the retained ``.aedt`` bundle has been collected, either on
the workstation or in a separate graphical AEDT session.

The exporter restores the drawing colour convention (red primary, blue
secondary, green plates, grey core), writes five PNG views, and seals a
machine-readable manifest.  No solve, Scheduler request, or project save is
performed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any


VIEW_EXPORT_SCHEMA = "mft-goal-rounded-drawing-views-v1"
DEFAULT_DESIGN = "maxwell_loss"
DEFAULT_WIDTH_PX = 2400
DEFAULT_HEIGHT_PX = 1600

TX_COLOR = (255, 10, 10)
RX_COLOR = (10, 10, 255)
PLATE_COLOR = (144, 190, 144)
PAD_COLOR = (190, 220, 190)
CORE_COLOR = (112, 112, 112)

_TX_TURN = re.compile(r"^Tx_(?:main|side2?|side)_\d+_\d+$")
_RX_TURN = re.compile(r"^Rx_(?:main|side2?|side)_\d+_\d+$")
_CENTER_TURN = re.compile(r"^(?:Tx_main|Rx_main)_\d+_\d+$")
_SIDE_TURN = re.compile(r"^Rx_side2?_\d+_\d+$")


class ViewExportError(RuntimeError):
    """Raised when one required drawing view cannot be exported."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        staging = Path(stream.name)
        json.dump(
            value,
            stream,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
        stream.flush()
    staging.replace(path)
    return path


def _object_names(design: Any) -> list[str]:
    try:
        names = [str(name) for name in design.modeler.object_names]
    except Exception as exc:
        raise ViewExportError("AEDT object inventory is unavailable") from exc
    if not names:
        raise ViewExportError("AEDT object inventory is empty")
    return sorted(set(names))


def drawing_color_for_object(name: str) -> tuple[int, int, int] | None:
    """Return the requested drawing colour for one AEDT object name."""

    if name.startswith("Tx_main_wcp"):
        return PAD_COLOR if "_pad_" in name else PLATE_COLOR
    if name.startswith("core_plate_pad_"):
        return PAD_COLOR
    if name.startswith("core_plate_"):
        return PLATE_COLOR
    if name.startswith("core_"):
        return CORE_COLOR
    if _TX_TURN.fullmatch(name):
        return TX_COLOR
    if _RX_TURN.fullmatch(name):
        return RX_COLOR
    return None


def apply_drawing_colors(design: Any, names: Sequence[str]) -> dict[str, int]:
    """Apply colours in-memory without saving or mutating the source file."""

    counts = {
        "primary_red": 0,
        "secondary_blue": 0,
        "plates_green": 0,
        "pads_light_green": 0,
        "core_grey": 0,
    }
    count_by_color = {
        TX_COLOR: "primary_red",
        RX_COLOR: "secondary_blue",
        PLATE_COLOR: "plates_green",
        PAD_COLOR: "pads_light_green",
        CORE_COLOR: "core_grey",
    }
    for name in names:
        color = drawing_color_for_object(name)
        if color is None:
            continue
        try:
            obj = design.modeler[name]
            obj.color = list(color)
            if hasattr(obj, "transparency"):
                obj.transparency = 0.0
        except Exception as exc:
            raise ViewExportError(
                f"failed to apply drawing colour to {name!r}"
            ) from exc
        counts[count_by_color[color]] += 1
    return counts


def view_specs(names: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Return the complete overall/detail view contract."""

    center = [
        name
        for name in names
        if _CENTER_TURN.fullmatch(name)
        or name.startswith("Tx_main_wcp")
    ]
    side = [name for name in names if _SIDE_TURN.fullmatch(name)]
    return (
        {
            "name": "overall_top",
            "orientation": "top",
            "selections": None,
        },
        {
            "name": "overall_front",
            "orientation": "front",
            "selections": None,
        },
        {
            "name": "overall_isometric",
            "orientation": "isometric",
            "selections": None,
        },
        {
            "name": "center_winding_detail_top",
            "orientation": "top",
            "selections": center,
        },
        {
            "name": "side_winding_detail_top",
            "orientation": "top",
            "selections": side,
        },
    )


def _background_distance(
    pixel: Sequence[int], reference: Sequence[int]
) -> int:
    return max(
        abs(int(pixel[index]) - int(reference[index])) for index in range(3)
    )


def convert_jpg_to_png(
    source: Path,
    destination: Path,
    *,
    background: str,
    tolerance: int = 12,
) -> Path:
    """Convert AEDT's required JPG output to a white/transparent PNG."""

    if background not in {"white", "transparent", "native"}:
        raise ViewExportError(f"unsupported background mode: {background}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ViewExportError(
            "Pillow is required for deterministic PNG conversion"
        ) from exc
    try:
        with Image.open(source) as opened:
            rgb = opened.convert("RGB")
            if background == "native":
                output = rgb
            else:
                corner_pixels = (
                    rgb.getpixel((0, 0)),
                    rgb.getpixel((rgb.width - 1, 0)),
                    rgb.getpixel((0, rgb.height - 1)),
                    rgb.getpixel((rgb.width - 1, rgb.height - 1)),
                )
                reference = tuple(
                    round(sum(pixel[index] for pixel in corner_pixels) / 4)
                    for index in range(3)
                )
                if background == "white":
                    output = rgb.copy()
                    output.putdata(
                        [
                            (255, 255, 255)
                            if _background_distance(pixel, reference)
                            <= tolerance
                            else pixel
                            for pixel in rgb.getdata()
                        ]
                    )
                else:
                    output = rgb.convert("RGBA")
                    output.putdata(
                        [
                            (*pixel, 0)
                            if _background_distance(pixel, reference)
                            <= tolerance
                            else (*pixel, 255)
                            for pixel in rgb.getdata()
                        ]
                    )
            destination.parent.mkdir(parents=True, exist_ok=True)
            output.save(destination, format="PNG", optimize=True)
    except (OSError, ValueError) as exc:
        raise ViewExportError(
            f"failed to convert AEDT image {source}"
        ) from exc
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise ViewExportError(f"PNG conversion produced no bytes: {destination}")
    return destination


PictureExporter = Callable[..., Any]


def export_views(
    design: Any,
    output: Path,
    *,
    source_project: Path,
    background: str = "white",
    width: int = DEFAULT_WIDTH_PX,
    height: int = DEFAULT_HEIGHT_PX,
    picture_exporter: PictureExporter | None = None,
) -> Path:
    """Export all required views and return the sealed manifest path."""

    if width < 640 or height < 480:
        raise ViewExportError("drawing view resolution is too small")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = _object_names(design)
    colors = apply_drawing_colors(design, names)
    exporter = picture_exporter
    if exporter is None:
        post = getattr(design, "post", None)
        exporter = getattr(post, "export_model_picture", None)
    if not callable(exporter):
        raise ViewExportError(
            "AEDT graphical export_model_picture API is unavailable"
        )

    records: list[dict[str, Any]] = []
    for spec in view_specs(names):
        selections = spec["selections"]
        if selections == []:
            raise ViewExportError(
                f"{spec['name']} has no matching retained objects"
            )
        jpg = output / f".{spec['name']}.aedt.jpg"
        png = output / f"{spec['name']}.png"
        try:
            returned = exporter(
                full_name=str(jpg),
                show_axis=False,
                show_grid=False,
                show_ruler=False,
                show_region=False,
                selections=selections,
                orientation=spec["orientation"],
                width=width,
                height=height,
            )
        except Exception as exc:
            raise ViewExportError(
                f"AEDT failed to export {spec['name']}"
            ) from exc
        if not jpg.is_file() or jpg.stat().st_size <= 0:
            raise ViewExportError(
                f"AEDT returned no JPG for {spec['name']}: {returned!r}"
            )
        convert_jpg_to_png(jpg, png, background=background)
        jpg.unlink()
        records.append(
            {
                "name": spec["name"],
                "orientation": spec["orientation"],
                "selection_mode": (
                    "all_model_objects"
                    if selections is None
                    else "exact_object_names"
                ),
                "selection_count": (
                    len(names) if selections is None else len(selections)
                ),
                "selections": selections,
                "file": {
                    "path": png.name,
                    "sha256": _sha256(png),
                    "size_bytes": png.stat().st_size,
                },
            }
        )

    manifest = {
        "schema_version": VIEW_EXPORT_SCHEMA,
        "source_project": {
            "path": str(source_project.resolve()),
            "sha256": _sha256(source_project),
            "size_bytes": source_project.stat().st_size,
        },
        "design_name": str(
            getattr(design, "design_name", DEFAULT_DESIGN)
        ),
        "render_contract": {
            "graphical_aedt_required": True,
            "solver_invoked": False,
            "project_save_performed": False,
            "width_px": width,
            "height_px": height,
            "background": background,
            "native_intermediate_format": "jpg",
            "delivered_format": "png",
        },
        "drawing_colors_rgb": {
            "primary": list(TX_COLOR),
            "secondary": list(RX_COLOR),
            "plates": list(PLATE_COLOR),
            "pads": list(PAD_COLOR),
            "core": list(CORE_COLOR),
        },
        "recolored_object_counts": colors,
        "object_inventory_count": len(names),
        "views": records,
    }
    return _atomic_json(output / "drawing_views_manifest.json", manifest)


def _open_design(
    project: Path, design_name: str, *, version: str | None
) -> Any:
    try:
        from ansys.aedt.core import Maxwell3d
    except ImportError as exc:
        raise ViewExportError(
            "PyAEDT is unavailable; use the pyaedt2026v1 environment"
        ) from exc
    kwargs: dict[str, Any] = {
        "project": str(project),
        "design": design_name,
        "non_graphical": False,
        "new_desktop": True,
        "close_on_exit": True,
    }
    if version:
        kwargs["version"] = version
    try:
        return Maxwell3d(**kwargs)
    except Exception as exc:
        raise ViewExportError(
            f"failed to open {project} / {design_name} in graphical AEDT"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--design", default=DEFAULT_DESIGN)
    parser.add_argument(
        "--background",
        choices=("white", "transparent", "native"),
        default="white",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH_PX)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT_PX)
    parser.add_argument("--version", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project = args.project.resolve(strict=True)
    if not project.is_file() or project.suffix.lower() != ".aedt":
        raise ViewExportError("source project must be one retained .aedt file")
    app = _open_design(project, args.design, version=args.version)
    try:
        manifest = export_views(
            app,
            args.output,
            source_project=project,
            background=args.background,
            width=args.width,
            height=args.height,
        )
    finally:
        try:
            app.release_desktop(
                close_projects=False,
                close_on_exit=True,
            )
        except Exception:
            pass
    print(
        json.dumps(
            {
                "event": "rounded_drawing_views_exported",
                "manifest": str(manifest),
                "solver_invoked": False,
                "project_save_performed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ViewExportError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_drawing_view_export_error",
                    "error": str(exc),
                    "solver_invoked": False,
                    "project_save_performed": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2) from exc
