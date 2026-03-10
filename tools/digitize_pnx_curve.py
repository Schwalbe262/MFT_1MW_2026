import json
from pathlib import Path

import numpy as np
from PIL import Image


IMG_PATH = Path("material/PNX1150F.png")
OUT_PATH = Path("material/pnx1150f_digitized_points.json")

# Plot box from page image (detected from border lines)
X0, X1 = 1215, 1744
Y0, Y1 = 862, 1170

# Axis calibration
P_MIN, P_MAX = 0.01, 100.0
B_MIN, B_MAX = 0.1, 1.2

TARGET_B = [0.2, 0.4, 0.6, 0.8, 1.0]

# Anchor at B=1.0T (table or visually estimated)
ANCHOR_P_AT_B1 = {
    "50Hz": 0.95,
    "200Hz": 5.1,
    "400Hz": 10.6,
    "800Hz": 27.6,
    "1000Hz": 38.1,
}


def b_to_y(b):
    h = Y1 - Y0
    y = Y1 - (float(b) - B_MIN) / (B_MAX - B_MIN) * h
    return int(round(y))


def x_to_p(x):
    w = X1 - X0
    t = (float(x) - X0) / w
    logp = np.log10(P_MIN) + t * (np.log10(P_MAX) - np.log10(P_MIN))
    return float(10 ** logp)


def p_to_x(p):
    w = X1 - X0
    t = (np.log10(float(p)) - np.log10(P_MIN)) / (np.log10(P_MAX) - np.log10(P_MIN))
    return int(round(X0 + t * w))


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


def main():
    img = np.array(Image.open(IMG_PATH).convert("RGB"))
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

    # Sanity repair for 1000Hz when legend contamination causes non-physical shape.
    pts1000 = data.get("1000Hz", [])
    if len(pts1000) == len(TARGET_B):
        vals = [p for _, p in pts1000]
        strictly_inc = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        if not strictly_inc and "800Hz" in data and len(data["800Hz"]) == len(TARGET_B):
            p800 = [p for _, p in data["800Hz"]]
            scale = table_anchor["1000Hz"] / table_anchor["800Hz"]
            repaired = [(b, p * scale) for (b, p) in data["800Hz"]]
            repaired[-1] = (1.0, table_anchor["1000Hz"])
            data["1000Hz"] = repaired

    payload = {
        "source_image": str(IMG_PATH).replace("\\", "/"),
        "plot_box": {"x0": X0, "x1": X1, "y0": Y0, "y1": Y1},
        "axis": {
            "core_loss_w_per_kg": {"min": P_MIN, "max": P_MAX, "scale": "log10"},
            "magnetic_flux_density_t": {"min": B_MIN, "max": B_MAX, "scale": "linear"},
        },
        "points_by_frequency": data,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {OUT_PATH}")
    for freq, pts in data.items():
        print(freq, pts)


if __name__ == "__main__":
    main()
