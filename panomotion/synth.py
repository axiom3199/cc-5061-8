"""Synthetic equirectangular sequences with exact spherical ground truth.

The world is a seamless spherical texture (patterns are evaluated from 3-D
directions, so neither the longitude seam nor the poles have painted
discontinuities).  Each frame applies a known camera orientation ``R_t``;
local objects are tangent cards driven by their own orientations ``Q_t``.

For a direction ``d`` observed in frame ``t`` the true direction in frame
``t+1`` is an exact rotation:

* background:  ``d' = R_{t+1}^T R_t d``
* object k:    ``d' = R_{t+1}^T Q_{t+1,k} Q_{t,k}^T R_t d``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from . import sphere
from .sampling import sample_equirect


def rotation_from_to(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        # 180 degree turn: pick any perpendicular axis.
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        axis = sphere.normalize(np.cross(a, ref))
        theta = np.pi
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def orientation_looking_at(center, twist: float = 0.0):
    r0 = rotation_from_to(np.array([0.0, 0.0, 1.0]), center)
    ct, st = np.cos(twist), np.sin(twist)
    rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    return r0 @ rz


# ---------------------------------------------------------------------------
# Seamless spherical world texture
# ---------------------------------------------------------------------------

def _paint_spherical_disc(img, dirs, center, radius_deg, color,
                          ring_color=None, ring_frac=0.72):
    dot = np.clip(dirs @ np.asarray(center), -1.0, 1.0)
    angle = np.arccos(dot)
    radius = np.deg2rad(radius_deg)
    edge = np.deg2rad(0.35)
    body = 1.0 - np.clip((angle - radius + edge) / (2 * edge), 0.0, 1.0)
    for c in range(3):
        img[..., c] = img[..., c] * (1 - body) + color[c] * body
    if ring_color is not None:
        ring = ((angle < radius * ring_frac + np.deg2rad(0.5))
                & (angle > radius * ring_frac - np.deg2rad(1.2)))
        ring_a = ring.astype(np.float32) * 0.9
        for c in range(3):
            img[..., c] = img[..., c] * (1 - ring_a) + ring_color[c] * ring_a


def build_world_texture(width: int = 1920, height: int = 960, seed: int = 11):
    rng = np.random.default_rng(seed)
    lon, lat = np.meshgrid(
        np.linspace(-np.pi, np.pi, width, endpoint=False) + np.pi / width,
        np.linspace(np.pi / 2, -np.pi / 2, height, endpoint=False) - np.pi / (2 * height),
    )
    dirs = sphere.lonlat_to_direction(lon, lat).astype(np.float32)

    # Latitude base gradient (ice caps, temperate, warm belt).
    t = np.clip((np.abs(lat) - np.deg2rad(25)) / np.deg2rad(55), 0, 1)
    ocean_deep = np.array([28, 78, 148], np.float32)
    ocean_light = np.array([54, 138, 196], np.float32)
    warm = np.array([96, 178, 178], np.float32)
    ice = np.array([232, 240, 246], np.float32)
    base = ocean_deep * (1 - t[..., None]) + ocean_light * t[..., None]
    belt = np.exp(-(lat / np.deg2rad(22)) ** 2)[..., None]
    base = base * (1 - 0.35 * belt) + warm * (0.35 * belt)
    cap = np.clip((np.abs(lat) - np.deg2rad(66)) / np.deg2rad(12), 0, 1)[..., None]
    base = base * (1 - cap) + ice * cap
    img = base.astype(np.float32)

    # Broad continents / colour fields.
    for _ in range(16):
        z = rng.uniform(-0.7, 0.7)
        a = rng.uniform(0, sphere.TWO_PI)
        r = np.sqrt(1 - z * z)
        center = np.array([r * np.cos(a), z, r * np.sin(a)])
        sigma = np.deg2rad(rng.uniform(12, 30))
        w = np.exp(-((np.arccos(np.clip(dirs @ center, -1, 1))) ** 2)
                   / (2 * sigma ** 2))
        land = np.array(rng.choice([
            [74, 140, 78], [122, 168, 84], [196, 170, 104], [96, 158, 96]]),
            dtype=np.float32)
        a_w = (w * rng.uniform(0.55, 0.9))[..., None]
        img = img * (1 - a_w) + land * a_w

    # Graticule (analytic -> seamless at the longitude seam and poles).
    lon_deg = (lon + np.pi) / sphere.TWO_PI * 360.0
    lat_deg = np.rad2deg(lat)
    dl = np.abs((lon_deg + 7.5) % 15.0 - 7.5)
    dm = np.abs((lat_deg + 7.5) % 15.0 - 7.5)
    line = ((dl < 0.55) | (dm < 0.55)).astype(np.float32)[..., None] * 0.22
    img = img * (1 - line) + 255 * line
    major = ((np.abs((lon_deg + 0.0) % 45.0) < 0.7)
             | (np.abs((lat_deg + 0.0) % 45.0) < 0.7)).astype(np.float32)[..., None] * 0.25
    img = img * (1 - major) + 30 * major

    # Crisp, uniquely coloured tracking landmarks.
    palette = [
        (235, 64, 52), (250, 190, 40), (240, 110, 40), (150, 70, 220),
        (40, 200, 180), (245, 120, 180), (120, 210, 60), (255, 255, 255),
    ]
    for k in range(84):
        z = rng.uniform(-0.92, 0.92)
        a = rng.uniform(0, sphere.TWO_PI)
        r = np.sqrt(1 - z * z)
        center = np.array([r * np.cos(a), z, r * np.sin(a)])
        rad = rng.uniform(1.8, 4.6)
        color = np.array(palette[k % len(palette)], np.float32)
        _paint_spherical_disc(
            img, dirs, center, rad, color,
            ring_color=(20, 20, 28) if k % 2 == 0 else (255, 255, 255))

    # Dense small high-contrast dots so most analysis nodes see 2-D texture.
    dot_palette = [
        (20, 24, 34), (255, 255, 255), (255, 94, 58), (255, 214, 32),
        (38, 90, 220), (20, 150, 110),
    ]
    for k in range(170):
        z = rng.uniform(-0.98, 0.98)
        a = rng.uniform(0, sphere.TWO_PI)
        r = np.sqrt(1 - z * z)
        center = np.array([r * np.cos(a), z, r * np.sin(a)])
        rad = rng.uniform(0.55, 1.5)
        color = np.array(dot_palette[k % len(dot_palette)], np.float32)
        _paint_spherical_disc(
            img, dirs, center, rad, color,
            ring_color=(15, 15, 20) if k % 3 == 0 else None)

    # Fine sensor-like noise for dense texture.
    img += rng.normal(0, 4.5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Object cards
# ---------------------------------------------------------------------------

@dataclass
class ObjectSpec:
    name: str
    kind: str
    half_angle_deg: float
    centers: np.ndarray
    Q: np.ndarray
    label: int

    @property
    def half_tan(self):
        return np.tan(np.deg2rad(self.half_angle_deg))

    def paint_t(self, world_dirs, t: int):
        q = self.Q[t]
        local = world_dirs @ q
        gx = local[..., 0] / np.maximum(local[..., 2], 1e-6)
        gy = local[..., 1] / np.maximum(local[..., 2], 1e-6)
        h = self.half_tan
        m = np.maximum(np.abs(gx), np.abs(gy)) / h
        inside = (local[..., 2] > 0.05) & (m < 1.0)
        # ~1.2 px anti-aliased edge in tangent space.
        alpha = np.clip((1.0 - m) / 0.06, 0.0, 1.0) * inside

        rgb = np.zeros(world_dirs.shape[:-1] + (3,), np.float32)
        if self.kind == "checker":
            n_check = 4
            cx = np.floor((gx / h * 0.5 + 0.5) * n_check).astype(np.int64)
            cy = np.floor((gy / h * 0.5 + 0.5) * n_check).astype(np.int64)
            parity = (cx + cy) % 2
            yellow = np.array([255, 214, 32], np.float32)
            black = np.array([24, 24, 30], np.float32)
            checker = np.where(parity[..., None] == 0, yellow, black)
            rgb = np.broadcast_to(checker, rgb.shape).copy()
            border = (np.abs(gx) > h * 0.9) | (np.abs(gy) > h * 0.9)
            rgb[border] = np.array([220, 40, 40])
            # Unique anchor breaks checker phase ambiguity.
            r = np.sqrt(gx ** 2 + gy ** 2)
            rgb[r < h * 0.42] = np.array([235, 235, 245])
            rgb[r < h * 0.32] = np.array([40, 90, 220])
            rgb[r < h * 0.14] = np.array([255, 214, 32])
            corner = (gx < -h * 0.55) & (gy < -h * 0.55)
            rgb[corner] = np.array([220, 60, 200])
        else:  # bullseye
            r = np.sqrt(gx ** 2 + gy ** 2) / h
            ring = (np.floor(r * 5) % 2)
            c1 = np.array([245, 245, 245], np.float32)
            c2 = np.array([225, 60, 50], np.float32)
            rgb = np.where(ring[..., None] == 0, c1, c2)
            rgb[r < 0.18] = np.array([30, 30, 40])
        return rgb, alpha


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSequence:
    width: int
    height: int
    n_frames: int
    frames: list
    labels: np.ndarray
    Rs: np.ndarray
    objects: list
    world: np.ndarray
    config: dict = field(default_factory=dict)

    # -- ground truth -------------------------------------------------------
    def background_matrix(self, t: int):
        return self.Rs[t + 1].T @ self.Rs[t]

    def object_matrix(self, t: int, obj: ObjectSpec):
        return self.Rs[t + 1].T @ obj.Q[t + 1] @ obj.Q[t].T @ self.Rs[t]

    def labels_at(self, dirs, t: int):
        u, v = sphere.direction_to_pixel(dirs, self.width, self.height)
        ui = np.mod(np.rint(u).astype(np.int64), self.width)
        vi = np.clip(np.rint(v).astype(np.int64), 0, self.height - 1)
        return self.labels[t][vi, ui]

    def gt_targets(self, src_dirs, t: int, labels=None):
        src_dirs = np.asarray(src_dirs, float)
        if labels is None:
            labels = self.labels_at(src_dirs, t)
        labels = np.asarray(labels)
        out = sphere.rotate_dirs(self.background_matrix(t), src_dirs)
        matrices = {obj.label: self.object_matrix(t, obj) for obj in self.objects}
        for lab, M in matrices.items():
            mask = labels == lab
            if np.any(mask):
                out[mask] = sphere.rotate_dirs(M, src_dirs[mask])
        return out

    def gt_matrix_for_labels(self, t: int, labels):
        """One matrix per sample label (background if label 0/unknown)."""
        mats = np.broadcast_to(self.background_matrix(t),
                               (len(labels), 3, 3)).copy()
        for obj in self.objects:
            sel = np.asarray(labels) == obj.label
            mats[sel] = self.object_matrix(t, obj)
        return mats


def camera_path(n_frames: int):
    Rs = []
    for t in range(n_frames):
        yaw = np.deg2rad(-12.0 + 2.8 * t)
        pitch = np.deg2rad(7.0 - 1.7 * t)
        roll = np.deg2rad(0.35 * (t - n_frames / 2))
        Rs.append(sphere.rotation_yxz(yaw, pitch, roll))
    return np.stack(Rs)

def polar_crossing_object(n_frames: int, label: int, Rs,
                          t_cross: float | None = None,
                          speed_deg: float = 5.2, lon0_deg: float = 24.0):
    """Great-circle path that crosses the *equirectangular* (camera) pole.

    The path is constructed in camera directions so that between two frames
    the latitude stays near +87 deg while longitude flips by ~180 deg; only
    the geodesic step stays small.  Camera orientations are then applied to
    obtain the world-space card centers.
    """
    if t_cross is None:
        t_cross = (n_frames - 1) / 2.0  # between the two middle frames
    lon0 = np.deg2rad(lon0_deg)
    p = np.array([np.cos(lon0), 0.0, np.sin(lon0)])
    yaxis = np.array([0.0, 1.0, 0.0])
    centers, Qs = [], []
    for t in range(n_frames):
        a = np.deg2rad(90.0 + speed_deg * (t - t_cross))
        d_cam = sphere.normalize(p * np.cos(a) + yaxis * np.sin(a))
        c = sphere.rotate_dirs(Rs[t], d_cam[None])[0]
        centers.append(c)
        Qs.append(orientation_looking_at(c))
    return ObjectSpec("polar-crosser", "checker", 6.5,
                      np.stack(centers), np.stack(Qs), label)


def seam_crossing_object(n_frames: int, label: int):
    centers, Qs = [], []
    for t in range(n_frames):
        lon = np.deg2rad(168.0 + 4.6 * t)  # crosses +/- pi seam
        lat = np.deg2rad(18.0 + 4.0 * np.sin(t / 2.2))
        c = sphere.lonlat_to_direction(lon, lat)
        centers.append(c)
        Qs.append(orientation_looking_at(c, np.deg2rad(3.0 * t)))
    return ObjectSpec("seam-crosser", "bullseye", 5.5,
                      np.stack(centers), np.stack(Qs), label)


def generate_synthetic(width: int = 960, height: int = 480, n_frames: int = 12,
                       seed: int = 11, verbose: bool = True):
    world = build_world_texture(seed=seed)
    Rs = camera_path(n_frames)
    objects = [polar_crossing_object(n_frames, 1, Rs),
               seam_crossing_object(n_frames, 2)]

    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    frame_dirs = sphere.pixel_to_direction(uu, vv, width, height)
    flat_dirs = frame_dirs.reshape(-1, 3)

    frames, labels = [], []
    for t in range(n_frames):
        world_dirs = sphere.rotate_dirs(Rs[t], flat_dirs).reshape(height, width, 3)
        base = sample_equirect(world, *sphere.direction_to_pixel(
            world_dirs, world.shape[1], world.shape[0]))
        frame = np.clip(base, 0, 255).astype(np.uint8).astype(np.float32)
        label_img = np.zeros((height, width), np.uint8)

        for obj in objects:
            rgb, alpha = obj.paint_t(world_dirs, t)
            a = alpha[..., None]
            frame = frame * (1 - a) + rgb * a
            label_img[alpha > 0.55] = obj.label

        frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame_u8)
        labels.append(label_img)
        if verbose:
            print(f"[synth] frame {t + 1}/{n_frames} rendered")

    seq = SyntheticSequence(
        width=width, height=height, n_frames=n_frames, frames=frames,
        labels=np.stack(labels), Rs=Rs, objects=objects, world=world,
        config={"seed": seed, "camera_yaw_step_deg": 2.8,
                "camera_pitch_step_deg": -1.7},
    )
    return seq


def save_sequence_preview_strip(seq: SyntheticSequence, path: str, count=6):
    idx = np.linspace(0, seq.n_frames - 1, count).astype(int)
    thumbs = [Image.fromarray(seq.frames[i]).resize((320, 160)) for i in idx]
    strip = Image.new("RGB", (320 * count, 160), (10, 10, 10))
    for k, im in enumerate(thumbs):
        strip.paste(im, (320 * k, 0))
    strip.save(path)
