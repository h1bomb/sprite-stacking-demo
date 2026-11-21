#!/usr/bin/env python3
"""Turn a 4-view orthographic sheet into a sprite-stacking slice strip.

Based on the tested SpriteCook `views_to_slices.py` algorithm, plus
background keying, pixel-grid quantization and debug dumps so the same
pipeline also works on JPEG or anti-aliased four-view sheets that have no alpha.

  python slice_from_views.py sheet.png out_slices.png --shade 0.25 --debug debugdir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Official SpriteCook helpers (kept close to the published script)
# ---------------------------------------------------------------------------

def find_views(sheet):
    """the drawings on the sheet, biggest first four, in reading order (rows, then left to right)"""
    solid = sheet[..., 3] >= 128
    grown = solid.copy()
    for _ in range(2):
        g = grown.copy()
        g[1:] |= grown[:-1]; g[:-1] |= grown[1:]; g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
        grown = g
    seen = np.zeros(solid.shape, bool)
    boxes = []
    for y0, x0 in zip(*np.nonzero(grown)):
        if seen[y0, x0]:
            continue
        stack, ys, xs = [(y0, x0)], [], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            ys.append(y); xs.append(x)
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < solid.shape[0] and 0 <= xx < solid.shape[1] and grown[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if len(ys) > 40:
            boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    if not boxes:
        raise AssertionError("found 0 drawings, expected top, side, front and back")
    big = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= 0.2 * big]
    boxes.sort(key=lambda b: (round((b[1] + b[3]) / 2 / (sheet.shape[0] / 2.2)), b[0]))
    assert len(boxes) == 4, f"found {len(boxes)} drawings, expected top, side, front and back"
    views = []
    for x0, y0, x1, y1 in boxes:
        v = sheet[y0:y1, x0:x1].copy()
        ys, xs = np.nonzero(v[..., 3] >= 128)
        v = v[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        v[..., 3] = np.where(v[..., 3] >= 128, 255, 0)
        views.append(v)
    top = max(views, key=lambda v: v.shape[0] / v.shape[1])
    side = max((v for v in views if v is not top), key=lambda v: v.shape[1] / v.shape[0])
    return [top, side] + [v for v in views if v is not top and v is not side], boxes


def strip_outline(v, dark=70):
    m = v[..., 3] > 0
    pad = np.pad(m, 1)
    edge = m & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    line = edge & (v[..., :3].astype(int).sum(-1) / 3 < dark)
    out = v.copy()
    for y, x in zip(*np.nonzero(line)):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not line[yy, xx]:
                out[y, x] = v[yy, xx]
                break
    return out


def one_palette(views, n):
    px = np.concatenate([v[v[..., 3] > 0][:, :3] for v in views])
    px = np.unique(px // 6 * 6 + 3, axis=0).astype(np.uint8)
    pal = Image.fromarray(px.reshape(1, -1, 3), "RGB").quantize(n, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = np.array(pal.getpalette()[:n * 3]).reshape(-1, 3)
    out = []
    for v in views:
        q = v.copy()
        q[..., :3] = pal[((v[..., None, :3].astype(int) - pal[None, None]) ** 2).sum(-1).argmin(-1)]
        out.append(q)
    return out


def fit(v, w, h):
    return np.array(Image.fromarray(v).resize((w, h), Image.NEAREST))


def edge_colour(view, row, i):
    xs = np.nonzero(view[row, :, 3])[0]
    return None if len(xs) == 0 else view[row, min(max(i, xs[0]), xs[-1]), :3]


def carve(top, side, front, back, shade=0.0):
    L, W = top.shape[:2]
    H = side.shape[0]
    occ = np.zeros((H, L, W), bool)
    for z in range(H):
        r = H - 1 - z
        along = side[r, :, 3] > 0
        across = (front[r, :, 3] > 0) | (back[r, :, 3] > 0)
        occ[z] = (top[..., 3] > 0) & along[:, None] & across[None, :]
    vol = np.zeros((H, L, W, 4), np.uint8)
    covered = np.zeros((L, W), bool)
    for z in range(H - 1, -1, -1):
        r = H - 1 - z
        above = covered.copy()
        covered |= occ[z]
        o = occ[z]
        ix = np.arange(W)[None, :].repeat(L, 0)
        iy = np.arange(L)[:, None].repeat(W, 1)
        d = np.stack([
            ix - np.maximum.accumulate(np.where(~o, ix, -1), axis=1),
            np.minimum.accumulate(np.where(~o, ix, W)[:, ::-1], axis=1)[:, ::-1] - ix,
            iy - np.maximum.accumulate(np.where(~o, iy, -1), axis=0),
            np.minimum.accumulate(np.where(~o, iy, L)[::-1], axis=0)[::-1] - iy,
        ])
        face, depth = d.argmin(0), d.min(0)
        for y, x in zip(*np.nonzero(o)):
            if not above[y, x]:
                c = top[y, x, :3]
            elif depth[y, x] > 2:
                c = vol[z + 1, y, x, :3]
            else:
                f = face[y, x]
                c = (edge_colour(side, r, y) if f < 2 else edge_colour(back, r, x) if f == 2
                     else edge_colour(front, r, x))
                if c is None:
                    c = top[y, x, :3]
            if above[y, x] and depth[y, x] <= 2 and shade:
                c = (np.asarray(c, float) * (1 - shade * (1 - z / max(H - 1, 1)) - shade * 0.4)).clip(0, 255)
            vol[z, y, x] = (*np.asarray(c, np.uint8), 255)
    return vol


def match_top(top, side):
    tall = (fit(side, side.shape[1], side.shape[0])[..., 3] > 0).sum(0)
    tall = np.interp(np.linspace(0, len(tall) - 1, top.shape[0]), np.arange(len(tall)), tall) >= 0.8 * tall.max()
    dark = ((top[..., :3].astype(int).sum(-1) < 240) & (top[..., 3] > 0)).mean(1)
    return "down" if (dark * tall).sum() >= (dark[::-1] * tall).sum() else "up"


# ---------------------------------------------------------------------------
# Preprocess: background key + pixel-grid snap (needed for JPEG / anti-aliased sheets)
# ---------------------------------------------------------------------------

def remove_background(rgb: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Return RGBA with keyed background. `mode`: auto, gray, white, magenta."""
    arr = rgb[..., :3].astype(np.int16)
    h, w, _ = arr.shape
    corners = np.stack([
        arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1],
        arr[min(4, h - 1), min(4, w - 1)],
        arr[min(4, h - 1), max(w - 5, 0)],
    ])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg[None, None], axis=-1)
    # magenta / hot-pink key (some sheets use a near-magenta key, not pure #FF00FF)
    mag = np.array([252, 2, 206], dtype=np.int16)
    dist_m = np.linalg.norm(arr - mag[None, None], axis=-1)
    is_magenta_bg = (bg[0] > 180 and bg[1] < 80 and bg[2] > 140)

    if mode == "magenta" or (mode == "auto" and is_magenta_bg):
        # high R, low G, high-ish B, and close to sampled bg or canonical magenta
        mask_bg = (arr[..., 0] > 180) & (arr[..., 1] < 90) & (arr[..., 2] > 130)
        mask_bg |= dist_m < 70
        mask_bg |= dist < 40
    elif mode == "white" or (mode == "auto" and bg.min() > 245):
        mask_bg = dist < 22
        # nearly-white pixels that are not part of a structured edge
        mask_bg |= (arr.min(-1) > 248)
    else:
        # light gray sheet: key near-bg, keep white stripes (far in chroma from mid-gray)
        mask_bg = dist < 28

    alpha = np.where(mask_bg, 0, 255).astype(np.uint8)
    # clean 1px isolated specks
    solid = alpha > 0
    nbr = np.zeros(solid.shape, np.int16)
    nbr[1:] += solid[:-1]; nbr[:-1] += solid[1:]
    nbr[:, 1:] += solid[:, :-1]; nbr[:, :-1] += solid[:, 1:]
    alpha[(nbr == 0) & solid] = 0

    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb[..., :3]
    out[..., 3] = alpha
    # un-fringe: any remaining almost-bg pixel on the rim becomes transparent
    rim = solid & (dist < 42)
    pad = np.pad(solid, 1)
    edge = solid & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    out[edge & rim, 3] = 0
    return out


def estimate_pixel_scale(view: np.ndarray, max_s: int = 20) -> tuple[int, float]:
    """Find integer scale s that best reconstructs the view when box-downscaled and NEAREST-upscaled."""
    rgb = view[..., :3].astype(np.float32)
    a = view[..., 3] >= 128
    h, w = rgb.shape[:2]
    best_s, best_err = 1, 1e18
    scores = []
    for s in range(1, max_s + 1):
        if h < s * 10 or w < s * 10:
            break
        nh, nw = h // s, w // s
        crop = rgb[:nh * s, :nw * s]
        ca = a[:nh * s, :nw * s]
        small = crop.reshape(nh, s, nw, s, 3).mean((1, 3))
        up = np.repeat(np.repeat(small, s, 0), s, 1)
        # weight edges / opaque more
        err = ((crop - up) ** 2)[ca].mean() if ca.any() else ((crop - up) ** 2).mean()
        scores.append((s, float(err)))
        # prefer larger s when errors are close (true pixel art has a plateau)
        if err < best_err * 0.92 or (err < best_err * 1.08 and s > best_s):
            if err < best_err * 1.08:
                best_s, best_err = s, err if err < best_err else best_err
                if err < best_err:
                    best_err = err
    # re-pick: smallest error, break ties toward larger s if within 15%
    errs = {s: e for s, e in scores}
    if scores:
        min_e = min(e for _, e in scores if _ >= 2) if any(s >= 2 for s, _ in scores) else scores[0][1]
        candidates = [s for s, e in scores if e <= min_e * 1.15 and s >= 1]
        best_s = max(candidates) if candidates else 1
        best_err = errs[best_s]
    return best_s, float(best_err)


def quantize_to_grid(view: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return view
    h, w = view.shape[:2]
    nh, nw = max(1, h // scale), max(1, w // scale)
    # start at a phase that best aligns the grid (search 0..scale-1)
    best, best_err = None, 1e18
    rgb = view[..., :3].astype(np.float32)
    for oy in range(scale):
        for ox in range(0, scale, max(1, scale // 4) if scale > 4 else 1):
            hh = ((h - oy) // scale) * scale
            ww = ((w - ox) // scale) * scale
            if hh < scale * 8 or ww < scale * 8:
                continue
            crop = rgb[oy:oy + hh, ox:ox + ww]
            nh_, nw_ = hh // scale, ww // scale
            small = crop.reshape(nh_, scale, nw_, scale, 3).mean((1, 3))
            up = np.repeat(np.repeat(small, scale, 0), scale, 1)
            err = ((crop - up) ** 2).mean()
            if err < best_err:
                best_err = err
                # also average alpha
                a = view[oy:oy + hh, ox:ox + ww, 3]
                sa = a.reshape(nh_, scale, nw_, scale).mean((1, 3))
                small_rgba = np.zeros((nh_, nw_, 4), np.uint8)
                small_rgba[..., :3] = np.clip(small, 0, 255).astype(np.uint8)
                small_rgba[..., 3] = np.where(sa >= 128, 255, 0).astype(np.uint8)
                best = small_rgba
    if best is None:
        img = Image.fromarray(view)
        return np.array(img.resize((max(1, w // scale), max(1, h // scale)), Image.BOX if hasattr(Image, "BOX") else Image.BILINEAR).resize(
            (max(1, w // scale), max(1, h // scale)), Image.NEAREST
        ))
    return best


def downscale_view(view: np.ndarray, target_long: int) -> np.ndarray:
    h, w = view.shape[:2]
    long = max(h, w)
    if long <= target_long:
        return view
    scale = long / target_long
    nw, nh = max(1, round(w / scale)), max(1, round(h / scale))
    # box/area downsample then hard alpha
    img = Image.fromarray(view)
    small = np.array(img.resize((nw, nh), Image.Resampling.BOX))
    small[..., 3] = np.where(small[..., 3] >= 96, 255, 0).astype(np.uint8)
    # drop almost-empty pixels left by box filter
    return small


def views_to_slices(sheet, top_nose="auto", side_nose="left", colours=24, shade=0.0):
    views, boxes = find_views(sheet)
    top, side, front, back = one_palette([strip_outline(v) for v in views], colours)
    if side_nose == "left":
        side = side[:, ::-1]
    if top_nose == "auto":
        top_nose = match_top(top, side)
    if top_nose == "up":
        top = top[::-1]
    L, W = top.shape[:2]
    H = max(1, round(side.shape[0] * L / side.shape[1]))
    vol = carve(top, fit(side, L, H), fit(front, W, H), fit(back, W, H), shade)
    meta = {
        "top_nose": top_nose,
        "side_nose": side_nose,
        "view_sizes": {
            "top": [int(top.shape[1]), int(top.shape[0])],
            "side": [int(side.shape[1]), int(side.shape[0])],
            "front": [int(front.shape[1]), int(front.shape[0])],
            "back": [int(back.shape[1]), int(back.shape[0])],
        },
        "fitted_height": int(H),
        "boxes": [list(map(int, b)) for b in boxes],
    }
    return vol, meta, (top, side, front, back)


def save_sheet(vol: np.ndarray, path: Path, meta: dict | None = None):
    path = Path(path)
    Image.fromarray(np.concatenate(list(vol), axis=1)).save(path)
    info = {
        "count": int(vol.shape[0]),
        "h": int(vol.shape[1]),
        "w": int(vol.shape[2]),
        "solid_voxels": int((vol[..., 3] > 0).sum()),
    }
    if meta:
        info.update(meta)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"{info['count']} slices of {info['w']} x {info['h']} px -> {path}  (voxels={info['solid_voxels']})")
    return info


def dump_debug(debug_dir: Path, rgba_sheet, views_raw, views_q, vol, meta):
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba_sheet).save(debug_dir / "sheet_keyed.png")
    names = ["top", "side", "front", "back"]
    for i, (raw, q) in enumerate(zip(views_raw, views_q)):
        Image.fromarray(raw).save(debug_dir / f"view_{names[i]}_raw.png")
        up = Image.fromarray(q).resize((q.shape[1] * 8, q.shape[0] * 8), Image.NEAREST)
        up.save(debug_dir / f"view_{names[i]}_grid8x.png")
        Image.fromarray(q).save(debug_dir / f"view_{names[i]}_grid.png")
    sheet = Image.fromarray(np.concatenate(list(vol), axis=1))
    sheet.save(debug_dir / "slices.png")
    sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST).save(debug_dir / "slices_x4.png")
    (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sheet")
    p.add_argument("out")
    p.add_argument("--top-nose", choices=["auto", "up", "down"], default="auto")
    p.add_argument("--side-nose", choices=["left", "right"], default="left")
    p.add_argument("--colours", type=int, default=24)
    p.add_argument("--shade", type=float, default=0.25)
    p.add_argument("--bg", choices=["auto", "gray", "white", "magenta"], default="auto")
    p.add_argument("--target-len", type=int, default=48, help="long side of the quantized top view in art pixels")
    p.add_argument("--scale", type=int, default=0, help="force pixel-scale (0 = auto downscale to target-len)")
    p.add_argument("--debug", type=str, default="")
    a = p.parse_args()

    raw = Image.open(a.sheet).convert("RGB")
    rgba = remove_background(np.array(raw), a.bg)
    views_raw, boxes = find_views(rgba)
    print("Detected 4 views (top, side, front, back) at full res:")
    labels = ["top", "side", "front", "back"]
    # find_views already reordered
    for name, v, b in zip(labels, views_raw, [None] * 4):
        print(f"  {name:5s}  {v.shape[1]} x {v.shape[0]} px  opaque={(v[...,3]>0).sum()}")

    # scale consistency of the RAW detections (before we force-fit)
    top, side, front, back = views_raw
    # top is L x W (h x w), side is H x L, front/back are H x W
    print("Scale consistency (raw pixel aspect, should be close after accounting for art resolution):")
    print(f"  top length {top.shape[0]} vs side length {side.shape[1]}  ratio={top.shape[0]/max(1,side.shape[1]):.3f}")
    print(f"  top width  {top.shape[1]} vs front width {front.shape[1]} ratio={top.shape[1]/max(1,front.shape[1]):.3f}")
    print(f"  side height {side.shape[0]} vs front height {front.shape[0]} ratio={side.shape[0]/max(1,front.shape[0]):.3f}")

    views_q = []
    for v in views_raw:
        if a.scale > 0:
            q = quantize_to_grid(v, a.scale)
        else:
            # try estimate, but always snap to target_len on the first (top) view's long side
            q = downscale_view(v, a.target_len if max(v.shape[:2]) == max(top.shape[:2]) else
                               max(8, round(a.target_len * max(v.shape[:2]) / max(top.shape[:2]))))
        # harden alpha
        q = q.copy()
        q[..., 3] = np.where(q[..., 3] >= 128, 255, 0)
        views_q.append(q)

    # rebuild a compact transparent sheet in reading order so official path can run
    # We already have ordered views; feed them through palette/outline/carve directly.
    top_q, side_q, front_q, back_q = views_q
    ests = []
    for name, v in zip(labels, views_raw):
        s, err = estimate_pixel_scale(v)
        ests.append((name, s, err))
        print(f"  estimated native pixel scale for {name}: {s}x  (recon err={err:.2f})")

    top_p, side_p, front_p, back_p = one_palette([strip_outline(v) for v in (top_q, side_q, front_q, back_q)], a.colours)
    if a.side_nose == "left":
        side_work = side_p[:, ::-1]
    else:
        side_work = side_p
    top_nose = a.top_nose
    if top_nose == "auto":
        top_nose = match_top(top_p, side_work)
    top_work = top_p[::-1] if top_nose == "up" else top_p
    L, W = top_work.shape[:2]
    H = max(1, round(side_work.shape[0] * L / max(1, side_work.shape[1])))
    vol = carve(top_work, fit(side_work, L, H), fit(front_p, W, H), fit(back_p, W, H), a.shade)
    meta = {
        "source": Path(a.sheet).name,
        "top_nose": top_nose,
        "side_nose": a.side_nose,
        "target_len": a.target_len,
        "raw_sizes": {n: [int(v.shape[1]), int(v.shape[0])] for n, v in zip(labels, views_raw)},
        "grid_sizes": {n: [int(v.shape[1]), int(v.shape[0])] for n, v in zip(labels, views_q)},
        "fitted_height": int(H),
        "scale_ratios_raw": {
            "topL_over_sideL": float(top.shape[0] / max(1, side.shape[1])),
            "topW_over_frontW": float(top.shape[1] / max(1, front.shape[1])),
            "sideH_over_frontH": float(side.shape[0] / max(1, front.shape[0])),
        },
        "estimated_pixel_scales": [{"view": n, "scale": s, "err": e} for n, s, e in ests],
        "solid_voxels": int((vol[..., 3] > 0).sum()),
        "empty_layers": [int(z) for z in range(vol.shape[0]) if not (vol[z, ..., 3] > 0).any()],
    }
    info = save_sheet(vol, Path(a.out), meta)
    if a.debug:
        dump_debug(Path(a.debug), rgba, views_raw, views_q, vol, {**meta, **info})
        # also write a contact of the 4 quantized views
        pads = []
        for name, v in zip(labels, (top_p, side_p, front_p, back_p)):
            im = Image.fromarray(v).resize((v.shape[1] * 6, v.shape[0] * 6), Image.NEAREST)
            pads.append(im)
        # simple horizontal contact
        Hm = max(im.height for im in pads) + 20
        Wm = sum(im.width for im in pads) + 20 * 5
        contact = Image.new("RGBA", (Wm, Hm + 24), (20, 22, 28, 255))
        x = 20
        for im in pads:
            contact.paste(im, (x, 20 + (Hm - 20 - im.height) // 2), im)
            x += im.width + 20
        contact.convert("RGB").save(Path(a.debug) / "views_quantized_contact.jpg", quality=92)


if __name__ == "__main__":
    main()
