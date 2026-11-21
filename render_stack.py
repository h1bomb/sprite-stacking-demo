#!/usr/bin/env python3
"""Nearest-neighbor sprite-stacking renderer + 9:16 demo exporter.

Matches the SpriteCook draw order:
  translate(pos) * translate(0, -round(z*spacing*scale)) * scale(1, squash) * rotate(turn)
  nearest-neighbor sampling, whole-pixel rounding, optional unified silhouette outline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def load_sheet(path: Path):
    path = Path(path)
    im = Image.open(path).convert("RGBA")
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        count, w, h = int(meta["count"]), int(meta["w"]), int(meta["h"])
    else:
        # infer: assume slice height == image height, try common counts
        h = im.height
        for c in range(8, 65):
            if im.width % c == 0:
                w = im.width // c
                # prefer nearly-square-or-portrait slices (cars are taller than wide in top view)
                if 0.3 * h <= w <= 2.5 * h:
                    count = c
                    break
        else:
            raise ValueError(f"cannot infer slice count from {im.size}")
        meta = {"count": count, "w": w, "h": h}
    arr = np.array(im)
    assert arr.shape[1] == w * count and arr.shape[0] == h, f"sheet {arr.shape} != {count}x{w}x{h}"
    vol = np.stack([arr[:, z * w:(z + 1) * w] for z in range(count)], axis=0)
    return vol, meta


def blit_slice(dst, src, cx, cy, angle, squash, scale):
    """Inverse-map a single art-pixel slice into dst with nearest-neighbor sampling."""
    sh, sw = src.shape[:2]
    rad = 0.5 * math.hypot(sw, sh) * scale + 2
    yrad = rad * max(squash, 0.05)
    H, W = dst.shape[:2]
    x0 = max(0, int(math.floor(cx - rad - 1)))
    x1 = min(W, int(math.ceil(cx + rad + 1)))
    y0 = max(0, int(math.floor(cy - yrad - 1)))
    y1 = min(H, int(math.ceil(cy + yrad + 1)))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    rx = xs + 0.5 - cx
    ry = (ys + 0.5 - cy) / max(squash, 1e-4)
    c, s = math.cos(angle), math.sin(angle)
    ux = (c * rx + s * ry) / scale
    uy = (-s * rx + c * ry) / scale
    sx = np.floor(ux + sw / 2.0).astype(np.int32)
    sy = np.floor(uy + sh / 2.0).astype(np.int32)
    valid = (sx >= 0) & (sx < sw) & (sy >= 0) & (sy < sh)
    if not valid.any():
        return
    pix = src[sy[valid], sx[valid]]
    opaque = pix[:, 3] > 0
    if not opaque.any():
        return
    yy = ys[valid][opaque]
    xx = xs[valid][opaque]
    dst[yy, xx] = pix[opaque]


def add_outline(dst, width_px: int, color=(16, 18, 22, 255)):
    if width_px < 1:
        return dst
    alpha = dst[..., 3]
    aimg = Image.fromarray(alpha)
    # MaxFilter size must be odd
    k = width_px * 2 + 1
    dil = np.array(aimg.filter(ImageFilter.MaxFilter(size=k)))
    ring = (dil > 0) & (alpha == 0)
    out = dst.copy()
    out[ring] = color
    # put original pixels back on top (already there except we wrote only ring)
    return out


def render_stack(
    vol,
    turn=0.0,
    squash=0.75,
    spacing=1.0,
    scale=10,
    outline=True,
    drop=None,
    canvas=None,
    origin=None,
    shadow=True,
    bg=None,
):
    """Render one stack into an RGBA array.

    drop: optional list/array of extra *upward* art-pixel offsets per slice
          (used for the falling-assembly animation).
    canvas: (H, W) of the output; default is a tight padded box (not used if given).
    origin: (cx, cy) ground-contact in output pixels. Default: centered.
    """
    count, L, W = vol.shape[:3]
    if drop is None:
        drop = np.zeros(count, dtype=np.float64)
    else:
        drop = np.asarray(drop, dtype=np.float64)

    # estimate bounds
    rad = 0.5 * math.hypot(W, L) * scale
    lifts = [int(round(z * spacing * scale)) + int(round(float(drop[z]) * scale)) for z in range(count)]
    max_lift = max(lifts) if lifts else 0
    if canvas is None:
        out_w = int(math.ceil(rad * 2 + 8 * scale))
        out_h = int(math.ceil(rad * 2 * max(squash, 0.2) + max_lift + 8 * scale))
    else:
        out_h, out_w = canvas
    dst = np.zeros((out_h, out_w, 4), np.uint8)
    if origin is None:
        cx = out_w / 2.0
        cy = min(out_h * 0.62, out_h - rad * squash - 2 * scale)
    else:
        cx, cy = origin

    if shadow:
        # soft-ish hard-edge ellipse under the ground contact, squashed with turn
        sh = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(sh)
        rw = (0.42 * L + 0.42 * W) * scale
        rh = rw * 0.28 * squash + scale
        # rotate bounding ellipse by projecting width/length
        rw = (abs(math.cos(turn)) * W + abs(math.sin(turn)) * L) * 0.52 * scale
        rh = (abs(math.sin(turn)) * W + abs(math.cos(turn)) * L) * 0.22 * scale * squash + 1.2 * scale
        bbox = [cx - rw, cy - rh, cx + rw, cy + rh]
        d.ellipse(bbox, fill=(8, 10, 14, 110))
        dst = np.array(sh)

    for z in range(count):
        lift = int(round(z * spacing * scale)) + int(round(float(drop[z]) * scale))
        blit_slice(dst, vol[z], cx, cy - lift, turn, squash, scale)

    if outline:
        dst = add_outline(dst, max(1, scale // 8 + (1 if scale >= 6 else 0)), color=(14, 16, 20, 255))

    if bg is not None:
        canvas_bg = np.zeros_like(dst)
        canvas_bg[..., :3] = np.array(bg, dtype=np.uint8)
        canvas_bg[..., 3] = 255
        a = dst[..., 3:4].astype(np.float32) / 255.0
        canvas_bg = (dst.astype(np.float32) * a + canvas_bg.astype(np.float32) * (1 - a)).astype(np.uint8)
        canvas_bg[..., 3] = 255
        return canvas_bg, (cx, cy)
    return dst, (cx, cy)


def _font(size, bold=True):
    # Prefer Noto Sans CJK so Chinese phase labels render (DejaVu has no CJK).
    candidates = [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", None),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None),
    ]
    for path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            kw = {"size": size}
            if index is not None:
                kw["index"] = index
            return ImageFont.truetype(path, **kw)
        except OSError:
            continue
    return ImageFont.load_default()


def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t


def demo_pose(t: float, count: int):
    """Deterministic timeline (12.0s) -> turn, squash, spacing, drop[], label, subtitle."""
    # assemble 0-2.5, spin 2.5-6.5, explode 6.5-9.0, squash compare 9.0-12.0
    turn0 = math.radians(38)
    drop = np.zeros(count, dtype=np.float64)
    if t < 2.5:
        u = ease_out_cubic(t / 2.5)
        # slices fall from 70 art-px above, staggered from the ground up
        for z in range(count):
            delay = z / max(count - 1, 1) * 0.45
            local = ease_out_cubic(max(0.0, min(1.0, (t / 2.5 - delay) / (1 - 0.45))))
            drop[z] = (1 - local) * 52
        return turn0, 0.75, 1.0, drop, "01  切片降落组装", "SLICE-DROP ASSEMBLY"
    if t < 6.5:
        u = (t - 2.5) / 4.0
        turn = turn0 + u * math.tau
        return turn, 0.75, 1.0, drop, "02  360° 旋转", "NEAREST-NEIGHBOR TURN"
    if t < 9.0:
        u = (t - 6.5) / 2.5
        # 1 -> 7 -> 1
        if u < 0.5:
            spacing = 1.0 + ease_in_out(u / 0.5) * 6.0
        else:
            spacing = 7.0 - ease_in_out((u - 0.5) / 0.5) * 6.0
        return turn0 + math.tau, 0.75, spacing, drop, "03  层距拉开 / 爆炸视图", "SLICE GAP  ·  EXPLODED VIEW"
    # squash compare
    u = (t - 9.0) / 3.0
    if u < 0.28:
        squash = 1.0
        tag = "squash = 1.00  正俯视"
    elif u < 0.62:
        k = ease_in_out((u - 0.28) / 0.34)
        squash = 1.0 - 0.25 * k
        tag = f"squash = {squash:.2f}  俯仰切换"
    else:
        squash = 0.75
        tag = "squash = 0.75  斜俯视"
    return turn0 + math.tau, squash, 1.0, drop, "04  俯仰压缩对比", tag


def draw_frame(vol, t, width=1080, height=1920, scale=10, title="SPRITE STACKING", source_tag=""):
    count = vol.shape[0]
    turn, squash, spacing, drop, label, sub = demo_pose(t, count)
    bg = (16, 18, 24)
    canvas = np.zeros((height, width, 4), np.uint8)
    canvas[..., :3] = bg
    canvas[..., 3] = 255

    # faint pixel grid
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img, "RGBA")
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 8))
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 8))

    # header / footer bars
    draw.rectangle([0, 0, width, 210], fill=(12, 13, 18, 255))
    draw.rectangle([0, height - 240, width, height], fill=(12, 13, 18, 255))
    f_title = _font(54)
    f_sub = _font(28)
    f_sm = _font(22)
    f_tiny = _font(18)
    draw.text((width // 2, 46), title, font=f_title, fill=(236, 238, 244, 255), anchor="mt")
    draw.text((width // 2, 112), "精灵堆叠  ·  纯 2D 切片错位叠层", font=f_sub, fill=(150, 160, 185, 255), anchor="mt")
    draw.text((width // 2, 160), label, font=f_sub, fill=(120, 200, 255, 255), anchor="mt")

    canvas = np.array(img)
    # render stack into the middle
    origin = (width / 2.0, height * 0.55)
    stack, _ = render_stack(
        vol, turn=turn, squash=squash, spacing=spacing, scale=scale,
        outline=True, drop=drop, canvas=(height, width), origin=origin,
        shadow=True, bg=None,
    )
    # composite stack over canvas (stack may be same size)
    a = stack[..., 3:4].astype(np.float32) / 255.0
    canvas = (stack.astype(np.float32) * a + canvas.astype(np.float32) * (1 - a)).astype(np.uint8)
    canvas[..., 3] = 255

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img, "RGBA")
    # Chrome goes on TOP so falling slices cannot cover the titles.
    draw.rectangle([0, 0, width, 210], fill=(12, 13, 18, 255))
    draw.rectangle([0, height - 240, width, height], fill=(12, 13, 18, 255))
    draw.text((width // 2, 46), title, font=f_title, fill=(236, 238, 244, 255), anchor="mt")
    draw.text((width // 2, 112), "精灵堆叠  ·  纯 2D 切片错位叠层", font=f_sub, fill=(150, 160, 185, 255), anchor="mt")
    draw.text((width // 2, 160), label, font=f_sub, fill=(120, 200, 255, 255), anchor="mt")
    readout = f"turn {math.degrees(turn) % 360:6.1f}°    squash {squash:.2f}    spacing {spacing:.2f}    slices {count}"
    draw.text((width // 2, height - 170), sub, font=f_sub, fill=(230, 232, 240, 255), anchor="mt")
    draw.text((width // 2, height - 120), readout, font=f_sm, fill=(140, 150, 170, 255), anchor="mt")
    if source_tag:
        draw.text((width // 2, height - 70), source_tag, font=f_tiny, fill=(100, 110, 130, 255), anchor="mt")
    return img.convert("RGB")


def export_mp4(vol, out_path, duration=12.0, fps=30, scale=10, source_tag="", title="SPRITE STACKING"):
    out_path = Path(out_path)
    frames_dir = out_path.parent / "frames" / out_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    n = int(round(duration * fps))
    paths = []
    for i in range(n):
        t = i / fps
        im = draw_frame(vol, t, scale=scale, source_tag=source_tag, title=title)
        fp = frames_dir / f"f_{i:04d}.png"
        im.save(fp)
        paths.append(fp)
        if i % 30 == 0 or i == n - 1:
            print(f"  frame {i+1}/{n}  t={t:.2f}s")
    # ffmpeg
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {out_path}")
    return frames_dir


def still(vol, turn, squash, spacing, scale, pad=40, outline=True):
    dst, _ = render_stack(vol, turn=turn, squash=squash, spacing=spacing, scale=scale, outline=outline, shadow=True)
    # trim transparent, keep pad
    a = dst[..., 3]
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return Image.fromarray(dst)
    y0, y1 = max(0, ys.min() - pad), min(dst.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(dst.shape[1], xs.max() + 1 + pad)
    return Image.fromarray(dst[y0:y1, x0:x1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sheet")
    p.add_argument("--out", default="")
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--tag", default="")
    p.add_argument("--title", default="SPRITE STACKING")
    p.add_argument("--stills", default="", help="directory for angle stills")
    a = p.parse_args()
    vol, meta = load_sheet(Path(a.sheet))
    print(f"loaded {meta}")
    if a.stills:
        d = Path(a.stills)
        d.mkdir(parents=True, exist_ok=True)
        poses = {
            "front_34": (math.radians(38), 0.75, 1.0),
            "side": (math.radians(90), 0.75, 1.0),
            "back_34": (math.radians(218), 0.75, 1.0),
            "top_squash1": (math.radians(38), 1.0, 1.0),
            "exploded": (math.radians(38), 0.75, 6.0),
            "assemble_mid": (math.radians(38), 0.75, 1.0),
        }
        for name, (turn, squash, spacing) in poses.items():
            drop = None
            if name == "assemble_mid":
                drop = np.array([72 * max(0, 1 - (z + 3) / vol.shape[0]) for z in range(vol.shape[0])])
            im = still(vol, turn, squash, spacing, a.scale) if name != "assemble_mid" else Image.fromarray(
                render_stack(vol, turn, squash, spacing, a.scale, True, drop, shadow=True)[0]
            )
            # trim assemble too
            if name == "assemble_mid":
                arr = np.array(im)
                ys, xs = np.where(arr[..., 3] > 0)
                if len(ys):
                    im = Image.fromarray(arr[max(0, ys.min()-40):ys.max()+41, max(0, xs.min()-40):xs.max()+41])
            im.save(d / f"{name}.png")
        print(f"stills -> {d}")
    if a.out:
        export_mp4(vol, Path(a.out), a.duration, a.fps, a.scale, a.tag, a.title)


if __name__ == "__main__":
    main()
