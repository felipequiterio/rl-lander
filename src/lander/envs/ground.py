"""Real lunar terrain for descent3d's uneven-ground mode.

A Heightmap is an LROC digital terrain model (DTM) of a landing site, imported by
``lander.viz3d.import_lroc`` into src/lander/assets/terrain_<site>/height.png. Heights
are bilinearly interpolated between grid posts (2-5 m apart), in the site frame:
metres, x east, y north, origin at the historic LM.

safe_targets() lists the posts where a lander could set down: the ground under the
whole landing zone (plus the legs) is gentle. The env draws one at random each
episode and moves its frame there, so the policy always sees the target at the origin.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def available_sites() -> list[str]:
    return sorted(p.name.removeprefix("terrain_") for p in ASSETS.glob("terrain_*") if (p / "height.png").exists())


class Heightmap:
    def __init__(self, z: np.ndarray, rect: tuple[float, float, float, float], name: str = ""):
        """z: heights (m) on a regular grid, row 0 = south; rect: (x0, y0, x1, y1) of
        the grid's outer edges in the site frame."""
        self.name = name
        self.z = np.ascontiguousarray(z, np.float64)
        self.rows, self.cols = self.z.shape
        self.x0, self.y0 = float(rect[0]), float(rect[1])
        self.dx = (rect[2] - rect[0]) / self.cols
        self.rect = tuple(float(v) for v in rect)
        self._flat = self.z.ravel()  # .item() on a 1-D array is the fastest scalar read

    @classmethod
    @functools.cache
    def load(cls, site: str) -> Heightmap:
        from PIL import Image

        folder = ASSETS / f"terrain_{site}"
        if not (folder / "height.png").exists():
            raise FileNotFoundError(f"no terrain for site {site!r} in {ASSETS} (have: {available_sites()}); "
                                    f"run `make import-terrain SITE={site}`")
        meta = json.loads((folder / "terrain.json").read_text())["height"]
        raw = np.asarray(Image.open(folder / meta["file"])).astype(np.float64)
        return cls((meta["z0"] + meta["dz"] * raw)[::-1], meta["rect"], name=site)

    def height(self, x: float, y: float) -> float:
        """Bilinear height at one point (clamped to the edge outside the map)."""
        fx = min(max((x - self.x0) / self.dx - 0.5, 0.0), self.cols - 1.000001)
        fy = min(max((y - self.y0) / self.dx - 0.5, 0.0), self.rows - 1.000001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        k = j * self.cols + i
        z = self._flat
        a = z.item(k) + (z.item(k + 1) - z.item(k)) * tx
        b = z.item(k + self.cols) + (z.item(k + self.cols + 1) - z.item(k + self.cols)) * tx
        return a + (b - a) * ty

    def heights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised height()."""
        fx = np.clip((np.asarray(x) - self.x0) / self.dx - 0.5, 0.0, self.cols - 1.000001)
        fy = np.clip((np.asarray(y) - self.y0) / self.dx - 0.5, 0.0, self.rows - 1.000001)
        i, j = fx.astype(int), fy.astype(int)
        tx, ty = fx - i, fy - j
        z = self.z
        a = z[j, i] + (z[j, i + 1] - z[j, i]) * tx
        b = z[j + 1, i] + (z[j + 1, i + 1] - z[j + 1, i]) * tx
        return a + (b - a) * ty

    @functools.cache  # noqa: B019 - heightmaps are loaded once and live for the process
    def safe_targets(self, max_slope_deg: float, radius: float, margin: float, max_dist: float | None = None) -> np.ndarray:
        """Grid posts (n, 2) that make a good landing target: every post within
        ``radius`` is at most ``max_slope_deg`` steep, the post is at least ``margin``
        from the map's edge (so the whole descent stays on the map) and, if given, at
        most ``max_dist`` from the site's origin."""
        gy, gx = np.gradient(self.z, self.dx)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        k = max(1, math.ceil(radius / self.dx))
        win = np.lib.stride_tricks.sliding_window_view(np.pad(slope, k, mode="edge"), (2 * k + 1, 2 * k + 1))
        ok = win.max(axis=(2, 3)) <= max_slope_deg
        m = math.ceil(margin / self.dx)
        ok[:m], ok[-m:], ok[:, :m], ok[:, -m:] = False, False, False, False
        j, i = np.nonzero(ok)
        xy = np.stack([self.x0 + (i + 0.5) * self.dx, self.y0 + (j + 0.5) * self.dx], 1)
        return xy if max_dist is None else xy[np.hypot(xy[:, 0], xy[:, 1]) <= max_dist]
