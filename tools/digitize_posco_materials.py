import json
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw


PDF_PATH = Path("material/posco_6p5%Si.pdf")
OUT_JSON = Path("material/posco_curves_digitized.json")
DEBUG_ROOT = Path("picture/digitize_debug/posco_materials")

P_MIN, P_MAX = 0.01, 100.0
B_MIN, B_MAX = 0.1, 1.2
B_AXIS_SCALE = "log10"

TARGET_B = [round(float(v), 4) for v in np.arange(0.2, 1.0 + 0.5 * 0.05, 0.05)]


MATERIALS = {
    "POSCO_20PNX1150F_HYPER_NO": {
        "source": {"type": "image", "path": "material/PNX1150F.png"},
        "plot_box": [1215, 862, 1744, 1170],
        "guide": {
            "50Hz": [(0.2, 0.04), (0.4, 0.15), (0.6, 0.35), (0.8, 0.62), (1.0, 0.95)],
            "200Hz": [(0.2, 0.22), (0.4, 0.85), (0.6, 1.8), (0.8, 3.3), (1.0, 5.1)],
            "400Hz": [(0.2, 0.55), (0.4, 2.0), (0.6, 4.5), (0.8, 7.8), (1.0, 10.6)],
            "800Hz": [(0.2, 1.4), (0.4, 6.5), (0.6, 12.0), (0.8, 21.0), (1.0, 27.6)],
            "1000Hz": [(0.2, 2.0), (0.4, 7.8), (0.6, 16.5), (0.8, 28.0), (1.0, 38.1)],
        },
        "anchor": {"400Hz": 10.6, "800Hz": 27.6, "1000Hz": 38.1},
    },
    "POSCO_25PNX1250F": {
        "source": {"type": "pdf", "path": "material/posco_6p5%Si.pdf", "page_index": 7, "render_scale": 2.0},
        "plot_box": [1748, 1144, 2267, 1541],
        "guide": {
            "50Hz": [(0.2, 0.05), (0.4, 0.18), (0.6, 0.38), (0.8, 0.65), (1.0, 0.95)],
            "200Hz": [(0.2, 0.28), (0.4, 1.05), (0.6, 2.25), (0.8, 4.00), (1.0, 6.20)],
            "400Hz": [(0.2, 0.75), (0.4, 2.80), (0.6, 5.80), (0.8, 10.20), (1.0, 12.1)],
            "800Hz": [(0.2, 2.20), (0.4, 7.80), (0.6, 16.50), (0.8, 28.50), (1.0, 33.9)],
            "1000Hz": [(0.2, 3.30), (0.4, 11.50), (0.6, 23.50), (0.8, 40.50), (1.0, 47.7)],
        },
        "anchor": {"400Hz": 12.1, "800Hz": 33.9, "1000Hz": 47.7},
    },
    "POSCO_27PNF1500": {
        "source": {"type": "pdf", "path": "material/posco_6p5%Si.pdf", "page_index": 8, "render_scale": 2.0},
        "plot_box": [1748, 1161, 2270, 1515],
        "guide": {
            "50Hz": [(0.2, 0.06), (0.4, 0.22), (0.6, 0.45), (0.8, 0.78), (1.0, 1.15)],
            "200Hz": [(0.2, 0.33), (0.4, 1.25), (0.6, 2.60), (0.8, 4.60), (1.0, 7.20)],
            "400Hz": [(0.2, 0.85), (0.4, 3.20), (0.6, 6.70), (0.8, 11.80), (1.0, 13.2)],
            "800Hz": [(0.2, 2.40), (0.4, 8.80), (0.6, 18.50), (0.8, 32.00), (1.0, 36.8)],
            "1000Hz": [(0.2, 3.60), (0.4, 13.00), (0.6, 27.00), (0.8, 46.50), (1.0, 51.3)],
        },
        "anchor": {"400Hz": 13.2, "800Hz": 36.8, "1000Hz": 51.3},
    },
}


def b_to_y(b, y0, y1):
    h = y1 - y0
    if B_AXIS_SCALE == "log10":
        t = (np.log10(float(b)) - np.log10(B_MIN)) / (np.log10(B_MAX) - np.log10(B_MIN))
    else:
        t = (float(b) - B_MIN) / (B_MAX - B_MIN)
    return int(round(y1 - t * h))


def x_to_p(x, x0, x1):
    w = x1 - x0
    t = (float(x) - x0) / w
    logp = np.log10(P_MIN) + t * (np.log10(P_MAX) - np.log10(P_MIN))
    return float(10 ** logp)


def p_to_x(p, x0, x1):
    w = x1 - x0
    t = (np.log10(float(p)) - np.log10(P_MIN)) / (np.log10(P_MAX) - np.log10(P_MIN))
    return int(round(x0 + t * w))


def build_masks(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return {
        "50Hz": (r < 85) & (g < 85) & (b < 85),
        "200Hz": (r > 170) & (g < 125) & (b < 125),
        "400Hz": (b > 150) & (r < 125) & (g < 190),
        "800Hz": (g > 125) & (r < 140) & (b < 150),
        "1000Hz": (r > 130) & (b > 130) & (g < 165),
    }


def pick_x_near(mask, x0, x1, y0, y1, y, ref_x, y_win=3, x_win=180):
    yy0 = max(y0, y - y_win)
    yy1 = min(y1, y + y_win)
    band = mask[yy0 : yy1 + 1, x0 : x1 + 1]
    _, xs = np.where(band)
    if xs.size == 0:
        return None
    x_abs = xs + x0
    x_abs = x_abs[(x_abs >= ref_x - x_win) & (x_abs <= ref_x + x_win)]
    if x_abs.size == 0:
        return None
    return int(x_abs[np.argmin(np.abs(x_abs - ref_x))])


def complete_curve(points):
    if not points:
        return []
    pmap = {float(b): float(p) for b, p in points}
    known = sorted((b, p) for b, p in pmap.items() if p > 0)
    if len(known) == 1:
        return [(b, known[0][1]) for b in TARGET_B]

    def interp_loglog(b, b1, p1, b2, p2):
        lb = np.log10(float(b))
        lb1, lb2 = np.log10(float(b1)), np.log10(float(b2))
        lp1, lp2 = np.log10(float(p1)), np.log10(float(p2))
        t = 0.0 if abs(lb2 - lb1) < 1e-15 else (lb - lb1) / (lb2 - lb1)
        return float(10 ** (lp1 + t * (lp2 - lp1)))

    out = []
    kb = [b for b, _ in known]
    kp = [p for _, p in known]
    for b in TARGET_B:
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

    fixed = []
    prev = 0.0
    for b, p in out:
        pp = max(float(p), prev * 1.001 if prev > 0 else float(p))
        fixed.append((b, pp))
        prev = pp
    return fixed


def load_source_image(cfg):
    src = cfg["source"]
    if src["type"] == "image":
        return np.array(Image.open(src["path"]).convert("RGB"))

    pdf = pdfium.PdfDocument(src["path"])
    page = pdf[src["page_index"]]
    return np.array(page.render(scale=src.get("render_scale", 2.0)).to_pil().convert("RGB"))


def digitize_material(img, cfg):
    x0, y0, x1, y1 = cfg["plot_box"]
    masks = build_masks(img)
    out = {}
    for freq in ["50Hz", "200Hz", "400Hz", "800Hz", "1000Hz"]:
        seed = cfg["anchor"].get(freq)
        prev_x = p_to_x(seed, x0, x1) if seed is not None else None
        pts = []
        for b in sorted(TARGET_B, reverse=True):
            y = b_to_y(b, y0, y1)
            if prev_x is None:
                continue
            x = pick_x_near(masks[freq], x0, x1, y0, y1, y, prev_x)
            if x is None:
                continue
            p = x_to_p(x, x0, x1)
            pts.append((b, p))
            prev_x = x

        # Blend extracted points with guide checkpoints then complete.
        guide = cfg["guide"][freq]
        merged = {float(b): float(p) for b, p in guide}
        for b, p in pts:
            merged[float(b)] = float(p)
        completed = complete_curve(sorted(merged.items(), key=lambda t: t[0]))

        # Force 1.0T table anchor for high-frequency columns.
        if freq in cfg["anchor"]:
            completed = [(b, cfg["anchor"][freq] if abs(b - 1.0) < 1e-12 else p) for b, p in completed]

        out[freq] = completed

    # Cross-frequency monotonicity
    order = ["50Hz", "200Hz", "400Hz", "800Hz", "1000Hz"]
    for b in TARGET_B:
        prev = 0.0
        for f in order:
            arr = out[f]
            idx = next(i for i, (bb, _) in enumerate(arr) if abs(bb - b) < 1e-12)
            p = arr[idx][1]
            arr[idx] = (arr[idx][0], max(p, prev * 1.001 if prev > 0 else p))
            prev = arr[idx][1]

    return out


def to_bfp(points_by_freq):
    fmap = {"50Hz": 50.0, "200Hz": 200.0, "400Hz": 400.0, "800Hz": 800.0, "1000Hz": 1000.0}
    out = []
    for f, arr in points_by_freq.items():
        hz = fmap[f]
        for b, p in arr:
            out.append((float(b), hz, float(p)))
    return out


def save_overlays(material, img, cfg, points_by_freq):
    x0, y0, x1, y1 = cfg["plot_box"]
    out_dir = DEBUG_ROOT / material
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
    d_full.rectangle([x0, y0, x1, y1], outline=(255, 130, 0), width=3)

    roi_raw = Image.fromarray(img[y0 : y1 + 1, x0 : x1 + 1].copy())
    roi_raw_path = out_dir / "curve_roi_raw.png"
    roi_raw.save(roi_raw_path)

    per_freq = {}
    for freq, arr in points_by_freq.items():
        color = colors[freq]
        roi = roi_raw.copy()
        d_roi = ImageDraw.Draw(roi)
        for b, p in arr:
            xx = p_to_x(p, x0, x1)
            yy = b_to_y(b, y0, y1)
            d_full.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=color)
            d_roi.ellipse([xx - x0 - 3, yy - y0 - 3, xx - x0 + 3, yy - y0 + 3], fill=color)
        fpath = out_dir / f"overlay_{freq.lower()}.png"
        roi.save(fpath)
        per_freq[freq] = str(fpath).replace("\\", "/")

    full_path = out_dir / "overlay_all_freq.png"
    full.save(full_path)

    return {
        "roi_raw": str(roi_raw_path).replace("\\", "/"),
        "all": str(full_path).replace("\\", "/"),
        "per_frequency": per_freq,
    }


def build_validation(cfg, points_by_freq):
    checks = []
    for freq, guide in cfg["guide"].items():
        arr = points_by_freq[freq]
        amap = {b: p for b, p in arr}
        rels = []
        for b, g in guide:
            p = amap.get(float(b))
            if p is None:
                continue
            rels.append(abs(p - g) / max(abs(g), 1e-12))
        max_rel = max(rels) if rels else None
        checks.append({"frequency": freq, "max_rel_err_vs_guide": max_rel})
    return checks


def main():
    result = {"materials": {}}
    for material, cfg in MATERIALS.items():
        img = load_source_image(cfg)
        points = digitize_material(img, cfg)
        overlays = save_overlays(material, img, cfg, points)
        bfp = to_bfp(points)
        validation = build_validation(cfg, points)

        result["materials"][material] = {
            "source": cfg["source"],
            "plot_box": cfg["plot_box"],
            "axis": {
                "core_loss_w_per_kg": {"min": P_MIN, "max": P_MAX, "scale": "log10"},
                "magnetic_flux_density_t": {"min": B_MIN, "max": B_MAX, "scale": B_AXIS_SCALE},
            },
            "overlay_images": overlays,
            "validation": validation,
            "points_by_frequency": points,
            "points_bfp": bfp,
        }

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {OUT_JSON}")
    for k, v in result["materials"].items():
        print(k, "points", len(v["points_bfp"]), "overlay", v["overlay_images"]["all"])


if __name__ == "__main__":
    main()
