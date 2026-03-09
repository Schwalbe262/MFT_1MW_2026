# Modeling Capture Workflow

## Goal
- Run only modeling cells from `test_simulation.ipynb`.
- Skip solve/report cells.
- Export geometry screenshots for debugging.
- Always release AEDT desktop session at the end.

## Runtime
- Conda env: `pyaedt2026v1`
- PyAEDT: `0.24.1`
- AEDT: `2025 R2`

## Runner
- Script: `tools/run_modeling_snapshot.py`
- Command:

```bash
conda run -n pyaedt2026v1 python Y:\git\MFT_1MW_2026\tools\run_modeling_snapshot.py
```

## What the runner does
1. Loads `test_simulation.ipynb` as JSON and executes code cells until the marker comment.
2. Skips expensive/non-modeling cells (`analyze`, second-pass/report extraction).
3. Exports images using AEDT-native methods (no PyVista dependency):
   - `design.post.export_model_picture(..., orientation="isometric")`
   - `design.post.export_model_picture(..., orientation="top")`
4. Writes fallback model object list to `model_objects.txt`.
5. Calls `sim.desktop.release_desktop(close_projects=True, close_on_exit=True)` in `finally`.

## Output folder
- `picture/debug_snapshot/`
  - `model_isometric.jpg`
  - `model_top.jpg`
  - `model_objects.txt`

## Notes
- `export_design_preview_to_jpg` can return `False` depending on project preview availability.
- `design.plot` may require PyVista; AEDT-native export is the primary path.
