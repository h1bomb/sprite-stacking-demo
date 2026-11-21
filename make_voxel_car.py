#!/usr/bin/env python3
"""Build a crisp procedural voxel retro sports car and export a slice sheet.

The car is 24 x 43 x 15 voxels (width x length x height), matching the
SpriteCook reference proportions. Colors get baked directional lighting
plus a little ambient occlusion, then each Z layer is written left-to-right
into code_car_slices.png (ground slice first).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

W, L, H = 24, 43, 15  # x, y, z
# Palette (flat, punchy, pixel-art)
BODY        = (42, 112, 230)
BODY_HI     = (78, 150, 250)
BODY_LO     = (28, 78, 175)
STRIPE      = (245, 247, 252)
STRIPE_LO   = (210, 214, 228)
WINDOW      = (28, 36, 55)
WINDOW_HI   = (58, 78, 112)
TIRE        = (22, 22, 26)
TIRE_LO     = (14, 14, 16)
RIM         = (140, 144, 152)
HUB         = (210, 212, 218)
RUBBER      = (36, 36, 40)
LIGHT_F     = (255, 232, 150)
LIGHT_F2    = (255, 250, 220)
LIGHT_R     = (214, 36, 42)
LIGHT_R2    = (150, 22, 28)
GRILLE      = (22, 24, 28)
BUMPER      = (32, 36, 44)
EXHAUST     = (86, 86, 92)
MIRROR      = (36, 40, 50)
GLASS_TRIM  = (20, 24, 32)
HOOD_VENT   = (24, 28, 38)


def inb(x, y, z):
    return 0 <= x < W and 0 <= y < L and 0 <= z < H


def fill_box(occ, col, x0, x1, y0, y1, z0, z1, color):
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(L, y1)
    z0, z1 = max(0, z0), min(H, z1)
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return
    occ[z0:z1, y0:y1, x0:x1] = True
    col[z0:z1, y0:y1, x0:x1] = color


def body_half_w(y: int) -> int:
    """Half-width of the painted body (not including protruding wheels)."""
    t = y / (L - 1)
    if y <= 1:
        return 7
    if y <= 5:
        return 9
    if y <= 29:
        return 10
    if y <= 35:
        return 9
    if y <= 39:
        return 8
    return 6


def body_z_lo(y: int) -> int:
    return 2  # ground clearance


def body_z_hi(y: int) -> int:
    """Inclusive top z of the body at this y (length) station."""
    if y <= 2:
        return 8          # rear lip / spoiler
    if y <= 6:
        return 8          # trunk deck
    if y <= 11:
        # climb the rear window
        return 8 + int(round((y - 6) * (13 - 8) / 5))
    if y <= 23:
        return 14         # cabin roof (z 0..14)
    if y <= 29:
        # windshield slope 14 -> 7
        return 14 - int(round((y - 23) * (14 - 7) / 6))
    if y <= 38:
        return 7          # hood
    return 6              # nose


def is_wheel_station(y: int) -> bool:
    return (6 <= y <= 11) or (30 <= y <= 35)


def build():
    occ = np.zeros((H, L, W), bool)
    col = np.zeros((H, L, W, 3), np.uint8)

    cx = (W - 1) / 2.0  # 11.5

    # --- main body hull ---
    for y in range(L):
        hw = body_half_w(y)
        z0, z1 = body_z_lo(y), body_z_hi(y) + 1
        # taper the top 2 layers inward for a slightly rounded cabin/hood
        for z in range(z0, z1):
            shrink = 0
            if z >= z1 - 1 and (z1 - z0) > 4:
                shrink = 1
            if z >= 13:
                shrink = 1 if y < 26 else 2
            lo = int(np.floor(cx - hw + shrink))
            hi = int(np.ceil(cx + hw + 1 - shrink))
            # wheel wells: don't fill the outer 3 columns at wheel height
            if is_wheel_station(y) and z <= 4:
                lo = max(lo, 4)
                hi = min(hi, 20)
            fill_box(occ, col, lo, hi, y, y + 1, z, z + 1, BODY)

    # rocker / skirt slightly darker (applied later via color overwrite)
    for y in range(2, 40):
        hw = body_half_w(y)
        lo = int(np.floor(cx - hw))
        hi = int(np.ceil(cx + hw + 1))
        if is_wheel_station(y):
            lo, hi = 4, 20
        fill_box(occ, col, lo, hi, y, y + 1, 2, 3, BODY_LO)

    # --- rear spoiler lip ---
    fill_box(occ, col, 6, 18, 0, 3, 8, 10, BODY)
    fill_box(occ, col, 7, 17, 0, 2, 9, 11, BODY_LO)

    # --- bumpers ---
    fill_box(occ, col, 7, 17, 0, 2, 2, 6, BUMPER)          # rear
    fill_box(occ, col, 7, 17, 41, 43, 2, 5, BUMPER)        # front
    # front lower lip
    fill_box(occ, col, 8, 16, 42, 43, 2, 3, BUMPER)

    # --- white racing stripe (top-facing cells + a bit down the nose/tail) ---
    for y in range(1, 42):
        z_top = body_z_hi(y)
        # skip windshield / rear window area — those stay glass
        is_glass = (12 <= y <= 15) or (24 <= y <= 28)
        if not is_glass:
            fill_box(occ, col, 10, 14, y, y + 1, z_top, z_top + 1, STRIPE)
            if z_top - 1 >= body_z_lo(y) and y <= 5 or y >= 38:
                fill_box(occ, col, 10, 14, y, y + 1, max(body_z_lo(y), z_top - 1), z_top, STRIPE_LO)

    # stripe wraps the spoiler and front bumper
    fill_box(occ, col, 10, 14, 0, 3, 9, 11, STRIPE)
    fill_box(occ, col, 10, 14, 41, 43, 5, 7, STRIPE)

    # --- glass (windshield, roof does NOT get glass, rear window, side windows) ---
    # rear window (top + slope)
    for y in range(12, 16):
        z_top = body_z_hi(y)
        fill_box(occ, col, 6, 18, y, y + 1, z_top, z_top + 1, WINDOW)
        # keep a thin body frame
        fill_box(occ, col, 6, 7, y, y + 1, z_top, z_top + 1, GLASS_TRIM)
        fill_box(occ, col, 17, 18, y, y + 1, z_top, z_top + 1, GLASS_TRIM)
    # windshield
    for y in range(24, 29):
        z_top = body_z_hi(y)
        fill_box(occ, col, 6, 18, y, y + 1, max(z_top - 1, 7), z_top + 1, WINDOW)
        fill_box(occ, col, 6, 7, y, y + 1, max(z_top - 1, 7), z_top + 1, GLASS_TRIM)
        fill_box(occ, col, 17, 18, y, y + 1, max(z_top - 1, 7), z_top + 1, GLASS_TRIM)
    # side windows (left / right cabin walls)
    for y in range(16, 26):
        for z in range(9, 13):
            fill_box(occ, col, 2, 4, y, y + 1, z, z + 1, WINDOW)
            fill_box(occ, col, 20, 22, y, y + 1, z, z + 1, WINDOW)

    # A-pillar / C-pillar body color to split the glass
    fill_box(occ, col, 2, 4, 15, 17, 8, 14, BODY)
    fill_box(occ, col, 20, 22, 15, 17, 8, 14, BODY)
    fill_box(occ, col, 2, 4, 25, 27, 7, 13, BODY)
    fill_box(occ, col, 20, 22, 25, 27, 7, 13, BODY)

    # --- hood vents ---
    fill_box(occ, col, 7, 9, 34, 38, 7, 8, HOOD_VENT)
    fill_box(occ, col, 15, 17, 34, 38, 7, 8, HOOD_VENT)

    # --- headlights / grille / indicators ---
    fill_box(occ, col, 5, 9, 41, 43, 4, 6, LIGHT_F)
    fill_box(occ, col, 15, 19, 41, 43, 4, 6, LIGHT_F)
    fill_box(occ, col, 6, 8, 42, 43, 4, 6, LIGHT_F2)
    fill_box(occ, col, 16, 18, 42, 43, 4, 6, LIGHT_F2)
    fill_box(occ, col, 9, 15, 42, 43, 3, 5, GRILLE)

    # --- taillights + exhaust ---
    fill_box(occ, col, 5, 9, 0, 1, 4, 7, LIGHT_R)
    fill_box(occ, col, 15, 19, 0, 1, 4, 7, LIGHT_R)
    fill_box(occ, col, 6, 8, 0, 1, 5, 7, LIGHT_R2)
    fill_box(occ, col, 16, 18, 0, 1, 5, 7, LIGHT_R2)
    fill_box(occ, col, 7, 9, 0, 1, 2, 3, EXHAUST)
    fill_box(occ, col, 15, 17, 0, 1, 2, 3, EXHAUST)

    # --- mirrors ---
    fill_box(occ, col, 1, 3, 26, 28, 7, 9, MIRROR)
    fill_box(occ, col, 21, 23, 26, 28, 7, 9, MIRROR)
    fill_box(occ, col, 1, 2, 26, 28, 8, 9, WINDOW)
    fill_box(occ, col, 22, 23, 26, 28, 8, 9, WINDOW)

    # --- wheels: 5x5x5-ish cylinders with rims on the outer face ---
    def add_wheel(y0, left: bool):
        xs = range(0, 4) if left else range(20, 24)
        outer_x = 0 if left else 23
        inner_x = 3 if left else 20
        for y in range(y0, y0 + 6):
            for z in range(0, 5):
                # circle in the (y,z) plane, center (y0+2.5, 2)
                cy, cz = y0 + 2.5, 2.0
                if (y + 0.5 - cy) ** 2 + (z + 0.5 - cz) ** 2 <= 2.7 ** 2:
                    for x in xs:
                        occ[z, y, x] = True
                        col[z, y, x] = TIRE if (y + 0.5 - cy) ** 2 + (z + 0.5 - cz) ** 2 > 1.5 ** 2 else RIM
                    # hub on outer face
                    occ[z, y, outer_x] = True
                    r2 = (y + 0.5 - cy) ** 2 + (z + 0.5 - cz) ** 2
                    if r2 <= 0.7 ** 2:
                        col[z, y, outer_x] = HUB
                    elif r2 <= 1.6 ** 2:
                        col[z, y, outer_x] = RIM
                    else:
                        col[z, y, outer_x] = TIRE_LO
                    # inner face darker rubber
                    col[z, y, inner_x] = TIRE

    add_wheel(6, True); add_wheel(6, False)
    add_wheel(30, True); add_wheel(30, False)

    return occ, col


def bake(occ, col):
    """Directional light + cheap AO, posterized a little so it stays pixel-crisp."""
    light = np.array([-0.42, 0.32, 0.85], dtype=np.float64)
    light /= np.linalg.norm(light)
    out = np.zeros((H, L, W, 4), np.uint8)
    # precompute 6-neighborhood emptiness
    for z, y, x in zip(*np.nonzero(occ)):
        n = np.zeros(3, dtype=np.float64)
        faces = 0
        for dz, dy, dx, vec in (
            (1, 0, 0, (0, 0, 1)), (-1, 0, 0, (0, 0, -1)),
            (0, 1, 0, (0, 1, 0)), (0, -1, 0, (0, -1, 0)),
            (0, 0, 1, (1, 0, 0)), (0, 0, -1, (-1, 0, 0)),
        ):
            zz, yy, xx = z + dz, y + dy, x + dx
            if not inb(xx, yy, zz) or not occ[zz, yy, xx]:
                n += np.array(vec, dtype=np.float64)
                faces += 1
        if faces == 0:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n /= max(np.linalg.norm(n), 1e-6)
        # AO: 26-neighborhood occupancy
        occ_n, tot = 0, 0
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    tot += 1
                    zz, yy, xx = z + dz, y + dy, x + dx
                    if inb(xx, yy, zz) and occ[zz, yy, xx]:
                        occ_n += 1
        ao = occ_n / tot
        shade = 0.58 + 0.50 * max(0.0, float(n.dot(light))) - 0.22 * ao
        # keep windows / tires from washing out
        base = col[z, y, x].astype(np.float64)
        luma = base.mean()
        if luma < 50:
            shade = 0.75 + 0.25 * (shade - 0.6)
        c = np.clip(base * shade, 0, 255)
        # posterize to 8-level steps so magnification stays chunky
        c = (np.round(c / 8) * 8).clip(0, 255)
        out[z, y, x] = (*c.astype(np.uint8), 255)
    return out


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Build a procedural voxel sports car and export a slice sheet.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: this script's directory)",
    )
    args = p.parse_args(argv)
    root = (args.outdir or Path(__file__).resolve().parent).resolve()
    out_dir = root / "code_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    occ, col = build()
    vol = bake(occ, col)
    sheet = Image.fromarray(np.concatenate(list(vol), axis=1))
    sheet_path = out_dir / "code_car_slices.png"
    # also copy to demo root for the HTML loader
    root_sheet = root / "code_car_slices.png"
    sheet.save(sheet_path)
    sheet.save(root_sheet)
    sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST).save(out_dir / "code_car_slices_x4.png")
    info = {
        "count": int(vol.shape[0]),
        "h": int(vol.shape[1]),
        "w": int(vol.shape[2]),
        "solid_voxels": int((vol[..., 3] > 0).sum()),
        "source": "procedural voxel sports car",
    }
    sheet_path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    (root / "code_car_slices.json").write_text(json.dumps(info, indent=2))
    print(f"{info['count']} slices of {info['w']} x {info['h']} px, voxels={info['solid_voxels']}")
    print(f"wrote {sheet_path} and {root_sheet}")
    # also dump a top-view of the highest occupied voxel color for sanity
    top = np.zeros((L, W, 4), np.uint8)
    for y in range(L):
        for x in range(W):
            zs = np.where(vol[:, y, x, 3] > 0)[0]
            if len(zs):
                top[y, x] = vol[zs[-1], y, x]
    Image.fromarray(top).resize((W * 12, L * 12), Image.NEAREST).save(out_dir / "preview_top_x12.png")


if __name__ == "__main__":
    main()
