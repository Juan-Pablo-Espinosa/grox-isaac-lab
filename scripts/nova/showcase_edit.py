# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HUD overlay + edit of the NOVA showcase (frames from scripts/nova/showcase_blender.py) into an H.264 video.

Plain Python (PIL + numpy) piping raw RGB frames into ffmpeg; no Isaac Sim needed::

    python scripts/nova/showcase_edit.py --root ~/nova_showcase --preview 800:1300   # -> preview.mp4
    python scripts/nova/showcase_edit.py --root ~/nova_showcase                       # -> showcase_1080p.mp4
    python scripts/nova/showcase_edit.py --hero_dir hero_v2 --out showcase_1080p_v2.mp4

Expected layout under --root: data/{hero,lineup}.npz, frames/{hero,lineup}/#####.png (+ screen.json).
Full edit: title card 3 s + hero (HUD lower third) + lineup (label under each robot) + end card 3 s, with a
persistent corner tag. 1920x1080, 50 fps, yuv420p.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 50
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
CRIMSON = (172, 43, 55)
INK = (28, 30, 34)
TAG = "Physics simulation (Isaac Lab) · rendered offline"
TITLE = "NOVA — Morphology-Agnostic Locomotion"
SUBTITLE = "One learned policy · any leg length, even while it changes"
BRAND = "WPI · ALMaS Research Group"
LEN_MIN_MM, LEN_MAX_MM = 5.0, 95.0


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


F_LABEL, F_VALUE, F_SMALL, F_TAG = font(22), font(28), font(19), font(20)


def find_ffmpeg() -> str:
    for cand in (shutil.which("ffmpeg"), os.path.expanduser("~/tools/ffmpeg")):
        if cand and os.path.exists(cand):
            return cand
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average (zero phase) with edge padding, along axis 0."""
    pad = np.pad(x, [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    c = np.cumsum(np.insert(pad, 0, 0.0, axis=0), axis=0)
    return (c[n:] - c[:-n]) / n


# --------------------------------------------------------------------------------------------- overlays


def corner_tag(img: Image.Image):
    d = ImageDraw.Draw(img, "RGBA")
    tw = d.textlength(TAG, font=F_TAG)
    x, y = W - tw - 40, 30
    d.rounded_rectangle((x - 14, y - 7, x + tw + 14, y + 30), radius=8, fill=(255, 255, 255, 150))
    d.text((x, y), TAG, font=F_TAG, fill=(70, 72, 78, 255))


CMD_ZERO = 0.05
"""Commands with |value| below this [m/s or rad/s] are shown as 0."""
SHORT_GAP_S = 0.5
"""An all-zero command gap shorter than this keeps the surrounding mode label (e.g. a turn reversing through 0)."""


def command_mode(c: np.ndarray) -> str | None:
    """'turn' | 'sideways' | 'forward' (incl. backward), or None when every component is (shown as) 0."""
    vx, vy, wz = (0.0 if abs(float(v)) < CMD_ZERO else float(v) for v in c)
    if vx == 0.0 and vy == 0.0 and wz == 0.0:
        return None
    if wz != 0.0 and vx == 0.0 and vy == 0.0:
        return "turn"
    return "sideways" if abs(vy) > abs(vx) else "forward"


def command_labels(cmd: np.ndarray) -> list[tuple[str, str]]:
    """Per frame (command string, measured quantity 'speed' | 'yaw'). |values| < CMD_ZERO show as 0; all-zero frames
    read "standing", except short gaps between moving segments, which keep the mode with a 0 value."""
    modes = [command_mode(c) for c in cmd]
    k, n, gap = 0, len(modes), int(SHORT_GAP_S * FPS)
    while k < n:  # fill short all-zero gaps with the previous mode
        if modes[k] is None:
            j = k
            while j < n and modes[j] is None:
                j += 1
            if k > 0 and j < n and j - k < gap:
                modes[k:j] = [modes[k - 1]] * (j - k)
            k = j
        else:
            k += 1
    out = []
    for c, mode in zip(cmd, modes):
        vx, vy, wz = (0.0 if abs(float(v)) < CMD_ZERO else float(v) for v in c)
        if mode is None:
            out.append(("standing", "speed"))
        elif mode == "turn":
            side = "" if wz == 0.0 else (" left" if wz > 0 else " right")
            out.append((f"turn {abs(wz):.2f} rad/s{side}", "yaw"))
        elif mode == "sideways":
            side = "" if vy == 0.0 else (" (left)" if vy > 0 else " (right)")
            out.append((f"{abs(vy):.2f} m/s sideways{side}", "speed"))
        else:
            side = "" if vx == 0.0 else (" forward" if vx > 0 else " backward")
            out.append((f"{abs(vx):.2f} m/s{side}", "speed"))
    return out


def hero_hud(img: Image.Image, cmd_label: tuple[str, str], speed: float, yaw: float, up_mm: float, lo_mm: float):
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, w, h = 60, H - 60 - 206, 640, 206
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=14, fill=(18, 20, 24, 165))
    d.rectangle((x0, y0 + 14, x0 + 5, y0 + h - 14), fill=CRIMSON + (255,))
    text, measured = cmd_label
    rows = [
        ("Command", text),
        (
            "Speed" if measured == "speed" else "Yaw rate",
            f"{speed:.2f} m/s" if measured == "speed" else f"{yaw:.2f} rad/s",
        ),
        ("Leg length", f"upper {up_mm:.0f} mm · lower {lo_mm:.0f} mm"),
    ]
    lx, vx = x0 + 30, x0 + 190
    for i, (label, value) in enumerate(rows):
        y = y0 + 20 + 42 * i
        d.text((lx, y + 5), label, font=F_LABEL, fill=(175, 178, 186, 255))
        d.text((vx, y), value, font=F_VALUE, fill=(255, 255, 255, 255))
    # one bar per segment, filling with length (0 .. 100 mm scale, ticks at the 5 / 95 mm limits)
    by = y0 + 20 + 42 * 3 + 14
    for j, (label, mm) in enumerate((("upper", up_mm), ("lower", lo_mm))):
        bx = vx + j * 215
        d.text((bx, by - 5), label, font=F_SMALL, fill=(175, 178, 186, 255))
        bx0, bw, bh = bx + 62, 130, 12
        d.rounded_rectangle((bx0, by + 2, bx0 + bw, by + 2 + bh), radius=6, fill=(255, 255, 255, 45))
        fill = float(np.clip(mm / 100.0, 0.0, 1.0))
        if fill > 0.02:
            d.rounded_rectangle((bx0, by + 2, bx0 + bw * fill, by + 2 + bh), radius=6, fill=CRIMSON + (255,))
        for lim in (LEN_MIN_MM, LEN_MAX_MM):
            tx = bx0 + bw * lim / 100.0
            d.line((tx, by - 1, tx, by + bh + 5), fill=(255, 255, 255, 120), width=1)


def lineup_labels(img: Image.Image, anchors: list, morphs: np.ndarray):
    d = ImageDraw.Draw(img, "RGBA")
    for (ax, ay, az, _, _), (up, lo) in zip(anchors, morphs):
        if az <= 0 or not (0.03 < ax < 0.97) or not (0.0 < ay < 0.97):
            continue
        txt = f"upper {up:.0f} · lower {lo:.0f} mm"
        tw = d.textlength(txt, font=F_LABEL)
        x, y = ax * W - tw / 2, min(ay * H + 22, H - 48)
        d.rounded_rectangle((x - 12, y - 6, x + tw + 12, y + 32), radius=8, fill=(18, 20, 24, 165))
        d.text((x, y), txt, font=F_LABEL, fill=(255, 255, 255, 255))
    cap = "Same policy, five fixed leg lengths"
    d.rounded_rectangle(
        (60, H - 60 - 52, 60 + d.textlength(cap, font=F_VALUE) + 44, H - 60), radius=12, fill=(18, 20, 24, 165)
    )
    d.rectangle((60, H - 60 - 40, 65, H - 72), fill=CRIMSON + (255,))
    d.text((82, H - 60 - 46), cap, font=F_VALUE, fill=(255, 255, 255, 255))


def card(alpha: float, end: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 250, 251))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    if end:
        t1, t2, t3 = "NOVA", "Morphology-agnostic locomotion · one learned policy", BRAND
        f1, f2 = font(96, True), font(36)
    else:
        t1, t2, t3 = TITLE, SUBTITLE, BRAND
        f1, f2 = font(64, True), font(36)
    cy = H // 2 - 70
    for txt, f, col, y in ((t1, f1, INK, cy), (t2, f2, (90, 94, 102), cy + 110)):
        d.text(((W - d.textlength(txt, font=f)) / 2, y), txt, font=f, fill=col + (a,))
    d.rectangle((W // 2 - 60, cy + 190, W // 2 + 60, cy + 195), fill=CRIMSON + (a,))
    f3 = font(30, True)
    d.text(((W - d.textlength(t3, font=f3)) / 2, cy + 225), t3, font=f3, fill=CRIMSON + (a,))
    img.paste(layer, (0, 0), layer)
    return img


def fade_white(img: Image.Image, k: int, n: int, fade: int = 20) -> Image.Image:
    """Fade from / to white over ``fade`` frames at the start / end of a clip of ``n`` frames."""
    w = min(1.0, k / fade, (n - 1 - k) / fade)
    if w >= 1.0:
        return img
    return Image.blend(Image.new("RGB", img.size, (250, 250, 251)), img, max(w, 0.0))


# --------------------------------------------------------------------------------------------- clips


def hero_clip(root: str, first: int, last: int, fades: bool = True, hero_dir: str = "hero"):
    data = np.load(os.path.join(root, "data", "hero.npz"))
    v = np.linalg.norm(data["root_lin_vel_b"][:, 0, :2], axis=1)
    speed = moving_average(v, 25)
    yaw = moving_average(np.abs(data["root_ang_vel_b"][:, 0, 2]), 25)
    q = data["prism_pos"][:, 0] * 1e3
    up, lo = q[:, 0:2].mean(axis=1), q[:, 2:4].mean(axis=1)
    labels = command_labels(data["command"][:, 0])
    n = last - first + 1
    for k, f in enumerate(range(first, last + 1)):
        img = Image.open(os.path.join(root, "frames", hero_dir, f"{f:05d}.png")).convert("RGB")
        hero_hud(img, labels[f], speed[f], yaw[f], up[f], lo[f])
        corner_tag(img)
        yield fade_white(img, k, n) if fades else img


def lineup_clip(root: str):
    data = np.load(os.path.join(root, "data", "lineup.npz"))
    with open(os.path.join(root, "frames", "lineup", "screen.json")) as fh:
        screen = json.load(fh)
    frames = sorted(int(k) for k in screen)
    n = len(frames)
    for k, f in enumerate(frames):
        img = Image.open(os.path.join(root, "frames", "lineup", f"{f:05d}.png")).convert("RGB")
        lineup_labels(img, screen[str(f)], data["morphs_mm"])
        corner_tag(img)
        yield fade_white(img, k, n)


def card_clip(end: bool, seconds: float = 3.0):
    n = int(seconds * FPS)
    for k in range(n):
        alpha = min(1.0, k / 25.0, (n - 1 - k) / 25.0)
        yield card(max(alpha, 0.0), end)


def encode(frames, out: str):
    cmd = [
        find_ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    count = 0
    for img in frames:
        proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        count += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("[edit] ffmpeg failed")
    return count


def main():
    p = argparse.ArgumentParser(description="HUD + edit of the NOVA showcase.")
    p.add_argument("--root", default=os.path.expanduser("~/nova_showcase"))
    p.add_argument("--preview", default=None, help="first:last hero frames -> preview.mp4 (no cards)")
    p.add_argument("--hero_dir", default="hero", help="hero frame folder under <root>/frames")
    p.add_argument("--out", default="showcase_1080p.mp4", help="output file name under <root> (full edit)")
    args = p.parse_args()
    root = os.path.expanduser(args.root)
    t0 = time.time()
    if args.preview:
        first, last = (int(v) for v in args.preview.split(":"))
        out = os.path.join(root, "preview.mp4")
        n = encode(hero_clip(root, first, last, hero_dir=args.hero_dir), out)
    else:
        hero_root = os.path.join(root, "frames", args.hero_dir)
        hero_frames = sorted(int(f[:5]) for f in os.listdir(hero_root) if f.endswith(".png"))
        out = os.path.join(root, args.out)

        def all_frames():
            yield from card_clip(end=False)
            yield from hero_clip(root, hero_frames[0], hero_frames[-1], hero_dir=args.hero_dir)
            yield from lineup_clip(root)
            yield from card_clip(end=True)

        n = encode(all_frames(), out)
    print(f"[edit] wrote {out}: {n} frames ({n / FPS:.1f} s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
