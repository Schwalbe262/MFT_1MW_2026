import json
import os
import sys
from pathlib import Path


def _run_notebook_until_modeling(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    g = {"__name__": "__main__"}

    marker = "# ======== 여기까지가 현재 작업한 코드고 밑에는 예전 코드 입니다 ========"

    for idx, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        src = "".join(cell.get("source", []))
        if not src.strip():
            continue

        if marker in src:
            print(f"[INFO] Stop at marker cell #{idx}")
            break

        skip_tokens = [
            "analyze(",
            "run_second_maxwell_for_coreloss(",
            "get_magnetic_parameter(",
            "get_calculator_parameter(",
            "second_pass =",
        ]
        if any(token in src for token in skip_tokens):
            print(f"[INFO] Skip non-modeling cell #{idx}")
            continue

        print(f"[INFO] Execute cell #{idx}")
        exec(compile(src, f"<nb-cell-{idx}>", "exec"), g, g)

    return g


def _save_snapshots(g, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = g.get("sim")
    if sim is None:
        raise RuntimeError("'sim' not found after notebook execution")

    design = sim.design1
    modeler = design.modeler

    saved = []

    # Try AEDT-native preview export (embedded preview image).
    preview_path = out_dir / "design_preview.jpg"
    try:
        if hasattr(design, "export_design_preview_to_jpg"):
            ok = design.export_design_preview_to_jpg(output_file=str(preview_path))
            print(f"[INFO] export_design_preview_to_jpg: {ok}")
            if ok and preview_path.exists():
                saved.append(preview_path)
    except Exception as e:
        print(f"[WARN] export_design_preview_to_jpg failed: {e}")

    # Preferred: AEDT model window export (does not require pyvista).
    for orientation, name in [("isometric", "model_isometric.jpg"), ("top", "model_top.jpg")]:
        out_path = out_dir / name
        try:
            if hasattr(design, "post") and hasattr(design.post, "export_model_picture"):
                result = design.post.export_model_picture(
                    full_name=str(out_path),
                    show_axis=True,
                    show_grid=False,
                    show_ruler=False,
                    show_region=False,
                    orientation=orientation,
                )
                print(f"[INFO] export_model_picture({orientation}): {result}")
                if out_path.exists():
                    saved.append(out_path)
        except Exception as e:
            print(f"[WARN] export_model_picture failed ({orientation}): {e}")

    # Optional: app-level plot export (requires pyvista).
    plot_path = out_dir / "model_plot.png"
    try:
        if hasattr(design, "plot"):
            result = design.plot(show=False, output_file=str(plot_path))
            print(f"[INFO] design.plot result: {result}")
            if plot_path.exists():
                saved.append(plot_path)
    except Exception as e:
        print(f"[WARN] design.plot failed: {e}")

    # Try object-level export_image using representative objects.
    for key, file_name in [
        ("core", "core_export.png"),
        ("Tx_windings", "tx_export.png"),
        ("Rx_windings", "rx_export.png"),
    ]:
        target = g.get(key)
        if target is None:
            continue

        obj = target[0] if isinstance(target, list) and target else target
        if isinstance(obj, list):
            continue
        path = out_dir / file_name
        try:
            if hasattr(obj, "export_image"):
                obj.export_image(str(path))
                if path.exists():
                    saved.append(path)
        except Exception as e:
            print(f"[WARN] export_image failed for {key}: {e}")

    # Save model object names as fallback evidence.
    names_path = out_dir / "model_objects.txt"
    try:
        names = list(modeler.object_names)
        names_path.write_text("\n".join(names), encoding="utf-8")
        saved.append(names_path)
    except Exception as e:
        print(f"[WARN] failed to dump object names: {e}")

    return saved


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    nb_path = root / "test_simulation.ipynb"
    out_dir = root / "picture" / "debug_snapshot"

    g = None
    try:
        g = _run_notebook_until_modeling(nb_path)
        saved = _save_snapshots(g, out_dir)

        print("[INFO] Saved files:")
        for p in saved:
            print(f" - {p}")
    finally:
        # Explicitly close AEDT so the process does not stay alive after modeling.
        try:
            if g is not None:
                sim_obj = g.get("sim")
                desktop_obj = getattr(sim_obj, "desktop", None)
                if desktop_obj is not None and hasattr(desktop_obj, "release_desktop"):
                    desktop_obj.release_desktop(close_projects=True, close_on_exit=True)
                    print("[INFO] Desktop released via sim.desktop.release_desktop")
        except Exception as e:
            print(f"[WARN] Failed to release desktop cleanly: {e}")


if __name__ == "__main__":
    main()
