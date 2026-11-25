"""Image/video output and spherical-aware annotation drawing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import sphere


def save_jpg(frame_uint8, path, quality: int = 90):
    Image.fromarray(frame_uint8).save(path, quality=quality)


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_CJK_CANDIDATES = [
    _FONT_DIR / "NotoSansSC-Bold.otf",
    _FONT_DIR / "NotoSansSC-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
_LATIN_FALLBACK = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_font(size: int = 16):
    for cand in _CJK_CANDIDATES:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(str(cand), size)
            except OSError:
                continue
    for cand in _LATIN_FALLBACK:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def write_mp4(frames, path, fps: float = 6.0):
    import imageio.v2 as imageio
    path = str(path)
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=2,
    )
    try:
        for f in frames:
            writer.append_data(np.asarray(f, dtype=np.uint8))
    finally:
        writer.close()
    return path


def write_gif(frames, path, fps: float = 6.0):
    ims = [Image.fromarray(np.asarray(f, np.uint8)) for f in frames]
    duration = int(1000 / fps)
    ims[0].save(path, save_all=True, append_images=ims[1:],
                duration=duration, loop=0, optimize=True)
    return path


def error_heatmap(diff_rgb: np.ndarray, scale: float = 6.0) -> np.ndarray:
    """Amplified absolute error -> red/yellow heat RGB."""
    mag = np.asarray(diff_rgb).mean(axis=-1) if diff_rgb.ndim == 3 \
        else np.asarray(diff_rgb)
    x = np.clip(mag * scale, 0.0, 1.0)
    r = np.clip(1.6 * x, 0, 1)
    g = np.clip(1.6 * x - 0.45, 0, 1)
    b = np.clip(1.6 * x - 1.1, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def slerp(d0, d1, t):
    omega = sphere.rotation_vector_between(d0, d1)
    return sphere.rodrigues(omega * t, d0)


def geodesic_segments(d0, d1, width: int, height: int, n: int = 14):
    """Pixel-space polylines of the shortest spherical path, split at seam.

    Splitting is delegated to the equirect projection outlet so video
    annotations and the interactive player share one seam rule.
    """
    from .projection import EquirectProjection, geodesic_polyline
    pts = geodesic_polyline(np.asarray(d0, float), np.asarray(d1, float),
                            samples=n)
    fragments = EquirectProjection(width, height).project_polyline(pts, 0)
    return [[list(p) for p in f.points] for f in fragments]


def draw_spherical_arrow(pil_img: Image.Image, d0, d1, color=(255, 255, 0),
                         width_px: int = 2, head: float = 7.0):
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    segments = geodesic_segments(np.asarray(d0, float), np.asarray(d1, float), w, h)
    for seg in segments:
        draw.line([tuple(p) for p in seg], fill=color, width=width_px,
                  joint="curve")
    if segments:
        end = segments[-1][-1]
        prev = segments[-1][-2] if len(segments[-1]) >= 2 else segments[-1][0]
        ang = np.arctan2(end[1] - prev[1], end[0] - prev[0])
        for da in (2.5, -2.5):
            p = (end[0] + head * np.cos(ang + da),
                 end[1] + head * np.sin(ang + da))
            draw.line([tuple(end), tuple(p)], fill=color, width=width_px)


def draw_track(pil_img: Image.Image, dirs, color=(0, 255, 200),
               width_px: int = 2):
    """Draw a direction trajectory with seam splitting."""
    from .projection import EquirectProjection
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    fragments = EquirectProjection(w, h).project_polyline(
        np.asarray(dirs, float), 0)
    for f in fragments:
        draw.line([tuple(p) for p in f.points], fill=color, width=width_px)


def label_bar(img: np.ndarray, text: str, bar_h: int = 26) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = np.zeros((h + bar_h, w, 3), np.uint8)
    canvas[bar_h:] = img
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, w, bar_h], fill=(16, 20, 30))
    d.text((8, 5), text, fill=(235, 240, 255), font=get_font(15))
    return np.array(pil)


def hstack_equal(images, target_h: int = 240):
    thumbs = []
    for im in images:
        pim = Image.fromarray(im)
        w = int(pim.width * target_h / pim.height)
        thumbs.append(np.array(pim.resize((w, target_h), Image.BILINEAR)))
    return np.concatenate(thumbs, axis=1)
