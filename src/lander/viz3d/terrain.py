"""Procedural lunar surface textures for the 3D viewer.

The physics ground is the flat plane z = 0, so the visual terrain stays flat too:
craters exist only in the lighting (a normal map) and the albedo. With a low sun
that reads as relief, without ever putting the lander "inside" a hill. The area
around the target is kept free of large craters.

Two textures: a crater map covering +-EXTENT metres around the target, and a small
tileable detail map for close-ups. Generated with numpy (a couple of seconds) and
cached under ~/.cache/lander.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXTENT = 2000.0  # crater map covers [-EXTENT, EXTENT]^2 metres
SIZE = 2048  # crater map resolution (~2 m per pixel)
DETAIL_SIZE = 512
DETAIL_TILE = 14.0  # metres per detail tile
CACHE = Path.home() / ".cache" / "lander"
VERSION = 2


def _normals_from_height(h: np.ndarray, metres_per_px: float, periodic: bool = False) -> np.ndarray:
    if periodic:
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * metres_per_px)
        dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * metres_per_px)
    else:
        dy, dx = np.gradient(h, metres_per_px)
    n = np.stack([-dx, -dy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def _pack(normals: np.ndarray, albedo: np.ndarray) -> np.ndarray:
    """RGBA8: normal xyz in RGB (0.5 + 0.5 n), albedo in A."""
    rgba = np.empty(normals.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = np.clip((normals * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(albedo * 255, 0, 255).astype(np.uint8)
    return rgba


def crater_map(seed: int = 11) -> np.ndarray:
    path = CACHE / f"craters_v{VERSION}_{seed}_{SIZE}.npy"
    if path.exists():
        return np.load(path)
    rng = np.random.default_rng(seed)
    m_px = 2 * EXTENT / SIZE
    h = np.zeros((SIZE, SIZE), np.float32)
    albedo = np.full((SIZE, SIZE), 0.50, np.float32)
    # Power-law crater sizes: many small, few large.
    radii = np.clip(6.0 * (1 - rng.random(1800)) ** -0.9, 6.0, 260.0)
    for R in np.sort(radii)[::-1]:
        cx, cy = rng.uniform(-EXTENT, EXTENT, 2)
        if np.hypot(cx, cy) < 60 + R * 1.8:  # keep the landing site clear
            continue
        rad_px = int(R * 2.4 / m_px) + 2
        ix, iy = int((cx + EXTENT) / m_px), int((cy + EXTENT) / m_px)
        x0, x1, y0, y1 = max(ix - rad_px, 0), min(ix + rad_px, SIZE), max(iy - rad_px, 0), min(iy + rad_px, SIZE)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(xx * m_px - EXTENT - cx, yy * m_px - EXTENT - cy) / R
        depth = 0.18 * R * rng.uniform(0.6, 1.1)
        bowl = np.where(r < 1, depth * (r**2 - 1), 0.0)
        rim = 0.05 * R * np.exp(-(((r - 1.0) / 0.28) ** 2))
        h[y0:y1, x0:x1] += (bowl + rim).astype(np.float32)
        fresh = rng.random() ** 3  # a few bright young craters with ejecta
        albedo[y0:y1, x0:x1] += (fresh * 0.18 * np.exp(-(((r - 1.1) / 0.6) ** 2)) - 0.05 * (r < 1)).astype(np.float32)
    albedo += rng.normal(0, 0.02, albedo.shape).astype(np.float32)
    rgba = _pack(_normals_from_height(h, m_px), albedo)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, rgba)
    return rgba


def detail_map(seed: int = 5) -> np.ndarray:
    """Tileable fine relief (pebbles, small pits) for close-ups."""
    rng = np.random.default_rng(seed)
    n = DETAIL_SIZE
    f = np.fft.fftfreq(n)
    k = np.hypot(*np.meshgrid(f, f))
    k[0, 0] = 1
    spec = np.fft.fft2(rng.standard_normal((n, n))) / k**1.6
    h = np.real(np.fft.ifft2(spec)).astype(np.float32)  # periodic by construction
    h = h / np.abs(h).max() * 0.12
    m_px = DETAIL_TILE / n
    # a few small pits
    for _ in range(60):
        cx, cy, R = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(4, 22)
        yy, xx = np.mgrid[0:n, 0:n]
        dx = (xx - cx + n / 2) % n - n / 2
        dy = (yy - cy + n / 2) % n - n / 2
        r = np.hypot(dx, dy) / R
        h += np.where(r < 1, 0.03 * (r**2 - 1), 0.006 * np.exp(-(((r - 1) / 0.3) ** 2))).astype(np.float32) * (R / 12)
    albedo = 0.5 + 0.35 * (h - h.mean()) / (np.abs(h - h.mean()).max() + 1e-9)
    return _pack(_normals_from_height(h, m_px, periodic=True), albedo)


# --- real sites (imported LROC imagery, see import_lroc.py) -------------------------------

SITES_DIR = Path(__file__).resolve().parent.parent / "assets"


def direction(azimuth_deg_from_east: float, elevation_deg: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg_from_east), np.radians(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def load_site(name: str = "apollo11") -> dict | None:
    """Imagery and metadata for a real landing site, or None if it isn't imported.
    Images come back with row 0 = south (OpenGL's bottom-up order)."""
    import json

    from PIL import Image

    folder = SITES_DIR / f"terrain_{name}"
    if not (folder / "terrain.json").exists():
        return None
    meta = json.loads((folder / "terrain.json").read_text())

    def img(entry):
        return np.ascontiguousarray(np.asarray(Image.open(folder / entry["file"]).convert("L"))[::-1])

    sun = meta["sun"]
    return {"name": name, "meta": meta, "near": img(meta["near"]), "far": img(meta["far"]),
            "sun_dir": direction(sun["azimuth_deg_from_east_ccw"], sun["elevation_deg"]),
            "earth_dir": direction(*meta.get("earth_az_el", (180.0, 66.5)))}
