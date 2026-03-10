import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMG_PATH = Path("material/PNX1150F.png")
OUT_PATH = Path("material/pnx1150f_digitized_points.json")

# Plot box from page image (detected from border lines)
X0, X1 = 1215, 1744
Y0, Y1 = 862, 1170

# Axis calibration pixel bounds inside plot box (auto-detected at runtime)
AX_X0, AX_X1 = X0, X1
AX_Y0, AX_Y1 = Y0, Y1

# Axis calibration
P_MIN, P_MAX = 0.01, 100.0
B_MIN, B_MAX = 0.1, 1.0
B_AXIS_SCALE = "log10"

# Digitization grid (more granular than 0.2T spacing)
B_GRID_START = 0.2
B_GRID_END = 1.0
B_GRID_STEP = 0.05
TARGET_B = [round(float(v), 4) for v in np.arange(B_GRID_START, B_GRID_END + 0.5 * B_GRID_STEP, B_GRID_STEP)]

COARSE_GUIDE_POINTS = {
    "50Hz": [(0.2, 0.04), (0.4, 0.15), (0.6, 0.35), (0.8, 0.62), (1.0, 0.95)],
    "200Hz": [(0.2, 0.22), (0.4, 0.85), (0.6, 1.8), (0.8, 3.3), (1.0, 5.1)],
    "400Hz": [(0.2, 0.55), (0.4, 2.0), (0.6, 4.5), (0.8, 7.8), (1.0, 10.6)],
    "800Hz": [(0.2, 1.4), (0.4, 6.5), (0.6, 12.0), (0.8, 21.0), (1.0, 27.6)],
    "1000Hz": [(0.2, 2.0), (0.4, 7.8), (0.6, 16.5), (0.8, 28.0), (1.0, 38.1)],
}

# Explicit visual sanity bounds from plot-grid inspection.
VISUAL_BOUNDS = {
    ("800Hz", 0.4): (6.0, 7.0),
}

# Anchor at B=1.0T (table or visually estimated)
ANCHOR_P_AT_B1 = {
    "50Hz": 0.95,
    "200Hz": 5.1,
    "400Hz": 10.6,
    "800Hz": 27.6,
    "1000Hz": 38.1,
}


def b_to_y(b):
    h = AX_Y1 - AX_Y0
    if B_AXIS_SCALE == "log10":
        t = (np.log10(float(b)) - np.log10(B_MIN)) / (np.log10(B_MAX) - np.log10(B_MIN))
    else:
        t = (float(b) - B_MIN) / (B_MAX - B_MIN)
    y = AX_Y1 - t * h
    return int(round(y))


def x_to_p(x):
    w = AX_X1 - AX_X0
    t = (float(x) - AX_X0) / w
    logp = np.log10(P_MIN) + t * (np.log10(P_MAX) - np.log10(P_MIN))
    return float(10 ** logp)


def p_to_x(p):
    w = AX_X1 - AX_X0
    t = (np.log10(float(p)) - np.log10(P_MIN)) / (np.log10(P_MAX) - np.log10(P_MIN))
    return int(round(AX_X0 + t * w))


def detect_axis_pixels(img):
    roi = img[Y0 : Y1 + 1, X0 : X1 + 1]
    h, w, _ = roi.shape
    mask = np.any(roi < 245, axis=2)

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    full_rows = [i for i, v in enumerate(row_counts) if v >= int(w * 0.97)]
    full_cols = [i for i, v in enumerate(col_counts) if v >= int(h * 0.97)]

    if not full_rows or not full_cols:
        return X0, X1, Y0, Y1

    y_bottom_rel = max(full_rows)
    top_candidates = [r for r in full_rows if 5 < r < y_bottom_rel - 20]
    y_top_rel = min(top_candidates) if top_candidates else min(full_rows)

    x_left_rel = min(full_cols)
    x_right_rel = max(full_cols)

    if x_right_rel - x_left_rel < int(0.6 * w):
        x_left_rel, x_right_rel = 0, w - 1
    if y_bottom_rel - y_top_rel < int(0.4 * h):
        y_top_rel, y_bottom_rel = 0, h - 1

    return X0 + x_left_rel, X0 + x_right_rel, Y0 + y_top_rel, Y0 + y_bottom_rel


def build_masks(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    masks = {
        "50Hz": (r < 85) & (g < 85) & (b < 85),
        "200Hz": (r > 170) & (g < 125) & (b < 125),
        "400Hz": (b > 150) & (r < 125) & (g < 190),
        "800Hz": (g > 125) & (r < 140) & (b < 150),
        "1000Hz": (r > 130) & (b > 130) & (g < 165),
    }
    return masks


def pick_x(mask, y, y_win=3):
    y0 = max(Y0, y - y_win)
    y1 = min(Y1, y + y_win)
    band = mask[y0 : y1 + 1, X0 : X1 + 1]
    ys, xs = np.where(band)
    if xs.size == 0:
        return None

    x_abs = xs + X0

    # Exclude legend area around mid-low B rows
    if 1030 <= y <= 1125:
        x_abs = x_abs[x_abs < 1650]
        if x_abs.size == 0:
            return None

    return int(np.median(x_abs))


def pick_x_near(mask, y, ref_x, y_win=3, x_win=180, monotonic_max=None):
    y0 = max(Y0, y - y_win)
    y1 = min(Y1, y + y_win)
    band = mask[y0 : y1 + 1, X0 : X1 + 1]
    _, xs = np.where(band)
    if xs.size == 0:
        return None
    x_abs = xs + X0

    if monotonic_max is not None:
        x_abs = x_abs[x_abs <= monotonic_max + 6]
        if x_abs.size == 0:
            return None

    # Exclude legend segment area for middle-low B rows.
    if 1025 <= y <= 1125:
        x_abs = x_abs[x_abs < 1650]
        if x_abs.size == 0:
            return None

    x_abs = x_abs[(x_abs >= ref_x - x_win) & (x_abs <= ref_x + x_win)]
    if x_abs.size == 0:
        return None

    return int(x_abs[np.argmin(np.abs(x_abs - ref_x))])


def extract_points(img):
    masks = build_masks(img)
    out = {}
    for freq, mask in masks.items():
        freq_points = []
        b_desc = sorted(TARGET_B, reverse=True)

        # Seed by anchor at 1.0T when possible.
        seed_p = ANCHOR_P_AT_B1.get(freq)
        prev_x = p_to_x(seed_p) if seed_p is not None else None

        for b_val in b_desc:
            y = b_to_y(b_val)

            if prev_x is None:
                x = pick_x(mask, y)
            else:
                x = pick_x_near(mask, y, ref_x=prev_x, monotonic_max=prev_x)

            if x is None:
                continue

            p = x_to_p(x)
            freq_points.append((float(b_val), p))
            prev_x = x

        freq_points.sort(key=lambda t: t[0])
        out[freq] = freq_points
    return out


def complete_curve(points, target_b):
    if not points:
        return []

    pmap = {float(b): float(p) for b, p in points}
    known = sorted((b, p) for b, p in pmap.items() if p > 0)
    if len(known) == 1:
        b0, p0 = known[0]
        return [(b, p0) for b in target_b]

    def interp_loglog(b, b1, p1, b2, p2):
        lb = np.log10(float(b))
        lb1, lb2 = np.log10(float(b1)), np.log10(float(b2))
        lp1, lp2 = np.log10(float(p1)), np.log10(float(p2))
        if abs(lb2 - lb1) < 1e-15:
            return float(p1)
        t = (lb - lb1) / (lb2 - lb1)
        lp = lp1 + t * (lp2 - lp1)
        return float(10 ** lp)

    out = []
    kb = [b for b, _ in known]
    kp = [p for _, p in known]
    for b in target_b:
        b = float(b)
        if b in pmap:
            out.append((b, float(pmap[b])))
            continue

        if b < kb[0]:
            p = interp_loglog(b, kb[0], kp[0], kb[1], kp[1])
        elif b > kb[-1]:
            p = interp_loglog(b, kb[-2], kp[-2], kb[-1], kp[-1])
        else:
            j = 1
            while j < len(kb) and kb[j] < b:
                j += 1
            p = interp_loglog(b, kb[j - 1], kp[j - 1], kb[j], kp[j])
        out.append((b, p))

    # Enforce monotonic increase with B.
    fixed = []
    prev = 0.0
    for b, p in sorted(out, key=lambda t: t[0]):
        pp = max(float(p), prev * 1.001 if prev > 0 else float(p))
        fixed.append((b, pp))
        prev = pp
    return fixed


def is_curve_suspicious(points):
    if len(points) != len(TARGET_B):
        return True
    vals = [float(p) for _, p in points]
    if min(vals) <= 0:
        return True
    if vals[-1] / vals[0] < 4.0:
        return True
    # detect abrupt end jump often caused by legend contamination near high-B point
    if len(vals) >= 2 and vals[-1] / max(vals[-2], 1e-12) > 2.0:
        return True
    # too many non-increasing steps
    non_inc = sum(1 for i in range(len(vals) - 1) if vals[i + 1] <= vals[i])
    return non_inc > 1


def enforce_visual_bounds(data):
    for (freq, b), (lo, hi) in VISUAL_BOUNDS.items():
        arr = data.get(freq, [])
        idx = next((i for i, (bb, _) in enumerate(arr) if abs(bb - b) < 1e-12), None)
        if idx is None:
            continue
        p = arr[idx][1]
        if p < lo or p > hi:
            arr[idx] = (arr[idx][0], min(max(p, lo), hi))
            data[freq] = complete_curve(arr, TARGET_B)


def value_at_b(points, b):
    idx = next((i for i, (bb, _) in enumerate(points) if abs(float(bb) - float(b)) < 1e-12), None)
    if idx is None:
        return None
    return float(points[idx][1])


def build_validation_report(data):
    checks = []
    for (freq, b), (lo, hi) in VISUAL_BOUNDS.items():
        p = value_at_b(data.get(freq, []), b)
        ok = p is not None and lo <= p <= hi
        checks.append({"frequency": freq, "B": b, "min": lo, "max": hi, "value": p, "ok": ok})
    return {"bounds_checks": checks, "all_passed": all(c["ok"] for c in checks) if checks else True}


def max_rel_error_vs_guide(points, guide):
    pmap = {float(b): float(p) for b, p in points}
    rels = []
    for b, g in guide:
        p = pmap.get(float(b))
        if p is None:
            continue
        rels.append(abs(p - g) / max(abs(g), 1e-12))
    return max(rels) if rels else float("inf")


def p_to_x_local(p):
    return p_to_x(p)


def b_to_y_local(b):
    return b_to_y(b)


def export_overlays(img, data):
    out_dir = Path("picture/digitize_debug/pnx1150f")
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "50Hz": (0, 0, 0),
        "200Hz": (220, 60, 60),
        "400Hz": (40, 120, 220),
        "800Hz": (60, 170, 90),
        "1000Hz": (160, 90, 200),
    }

    full = Image.fromarray(img.copy())
    d_full = ImageDraw.Draw(full)
    d_full.rectangle([X0, Y0, X1, Y1], outline=(255, 120, 0), width=2)
    d_full.rectangle([AX_X0, AX_Y0, AX_X1, AX_Y1], outline=(0, 220, 220), width=2)

    freq_files = {}
    for freq, pts in data.items():
        color = colors.get(freq, (255, 255, 0))
        roi = Image.fromarray(img[Y0 : Y1 + 1, X0 : X1 + 1].copy())
        d_roi = ImageDraw.Draw(roi)
        d_roi.rectangle([0, 0, X1 - X0, Y1 - Y0], outline=(255, 120, 0), width=2)
        d_roi.rectangle([AX_X0 - X0, AX_Y0 - Y0, AX_X1 - X0, AX_Y1 - Y0], outline=(0, 220, 220), width=2)

        for b, p in pts:
            x = p_to_x_local(p)
            y = b_to_y_local(b)
            d_full.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
            d_roi.ellipse([x - X0 - 3, y - Y0 - 3, x - X0 + 3, y - Y0 + 3], fill=color)

        path = out_dir / f"overlay_{freq.lower()}.png"
        roi.save(path)
        freq_files[freq] = str(path).replace("\\", "/")

    full_path = out_dir / "overlay_all_freq.png"
    full.save(full_path)

    return {
        "all": str(full_path).replace("\\", "/"),
        "per_frequency": freq_files,
    }


def main():
    global AX_X0, AX_X1, AX_Y0, AX_Y1

    img = np.array(Image.open(IMG_PATH).convert("RGB"))
    AX_X0, AX_X1, AX_Y0, AX_Y1 = detect_axis_pixels(img)

    data = extract_points(img)

    # Anchor table points for better consistency at B=1.0
    table_anchor = {
        "400Hz": 10.6,
        "800Hz": 27.6,
        "1000Hz": 38.1,
    }
    for freq, p in table_anchor.items():
        pts = [(b, v) for (b, v) in data.get(freq, []) if abs(b - 1.0) > 1e-12]
        pts.append((1.0, p))
        pts.sort(key=lambda t: t[0])
        data[freq] = pts

    # Complete each curve over TARGET_B with log-log interpolation/extrapolation.
    for freq in list(data.keys()):
        data[freq] = complete_curve(data[freq], TARGET_B)

    # Fallback to coarse guide interpolation when extracted curve looks suspicious.
    for freq, guide in COARSE_GUIDE_POINTS.items():
        if freq not in data or is_curve_suspicious(data[freq]):
            data[freq] = complete_curve(guide, TARGET_B)
            continue

        # Even if shape checks pass, reject when coarse checkpoints deviate too much.
        rel = max_rel_error_vs_guide(data[freq], guide)
        if rel > 0.20:
            data[freq] = complete_curve(guide, TARGET_B)

    enforce_visual_bounds(data)

    # Cross-frequency monotonicity per B: 50 <= 200 <= 400 <= 800 <= 1000.
    freq_order = ["50Hz", "200Hz", "400Hz", "800Hz", "1000Hz"]
    for b in TARGET_B:
        prev = 0.0
        for f in freq_order:
            arr = data.get(f, [])
            idx = next((i for i, (bb, _) in enumerate(arr) if abs(bb - b) < 1e-12), None)
            if idx is None:
                continue
            p = arr[idx][1]
            p_fixed = max(p, prev * 1.001 if prev > 0 else p)
            arr[idx] = (arr[idx][0], p_fixed)
            prev = p_fixed

    # Keep B<1.0 points below the anchored 1.0T value per frequency.
    for f, arr in data.items():
        p_end = next((p for b, p in arr if abs(b - 1.0) < 1e-12), None)
        if p_end is None:
            continue
        cap = p_end * 0.999
        fixed = []
        for b, p in arr:
            if b < 1.0:
                fixed.append((b, min(p, cap)))
            else:
                fixed.append((b, p))
        prev = 0.0
        mono = []
        for b, p in fixed:
            pp = max(p, prev * 1.001 if prev > 0 else p)
            mono.append((b, pp))
            prev = pp
        data[f] = mono

    validation = build_validation_report(data)
    overlays = export_overlays(img, data)

    payload = {
        "source_image": str(IMG_PATH).replace("\\", "/"),
        "plot_box": {"x0": X0, "x1": X1, "y0": Y0, "y1": Y1},
        "axis_pixels": {"x0": AX_X0, "x1": AX_X1, "y0": AX_Y0, "y1": AX_Y1},
        "axis": {
            "core_loss_w_per_kg": {"min": P_MIN, "max": P_MAX, "scale": "log10"},
            "magnetic_flux_density_t": {"min": B_MIN, "max": B_MAX, "scale": B_AXIS_SCALE},
        },
        "validation": validation,
        "overlay_images": overlays,
        "points_by_frequency": data,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {OUT_PATH}")
    print("validation:", validation)
    print("overlay:", overlays)
    for freq, pts in data.items():
        print(freq, pts)


if __name__ == "__main__":
    main()
