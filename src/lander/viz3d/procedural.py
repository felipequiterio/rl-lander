"""A simple Apollo-LM-like model built from primitives, used when no Bluedog Design
Bureau install is configured. Real-scale (metres), z-up, footpads at z = 0."""

from __future__ import annotations

import math

import numpy as np

from lander.viz3d.ksp_model import AssembledModel, TexturedMesh

GOLD, SILVER, DARK, FOIL_DARK = (0.86, 0.66, 0.26), (0.76, 0.78, 0.82), (0.26, 0.26, 0.28), (0.55, 0.40, 0.14)


class _Builder:
    def __init__(self, color):
        self.color, self.P, self.N, self.I = color, [], [], []

    def tri_fan(self, pts: list[np.ndarray], normal: np.ndarray) -> None:
        base = len(self.P)
        self.P += pts
        self.N += [normal] * len(pts)
        self.I += [[base, base + i, base + i + 1] for i in range(1, len(pts) - 1)]

    def quad(self, a, b, c, d) -> None:
        n = np.cross(b - a, d - a)
        n = n / (np.linalg.norm(n) + 1e-12)
        self.tri_fan([a, b, c, d], n)

    def prism(self, radii, z0, z1, sides, phase=0.0, caps=True, centre=(0.0, 0.0)) -> None:
        """Flat-shaded prism / frustum around a vertical axis; ``radii`` = (bottom, top)."""
        ang = [phase + 2 * math.pi * i / sides for i in range(sides)]
        cx, cy = centre
        ring = lambda r, z: [np.array([cx + r * math.cos(a), cy + r * math.sin(a), z]) for a in ang]
        lo, hi = ring(radii[0], z0), ring(radii[1], z1)
        for i in range(sides):
            j = (i + 1) % sides
            self.quad(lo[i], lo[j], hi[j], hi[i])
        if caps:
            self.tri_fan(hi, np.array([0, 0, 1.0]))
            self.tri_fan(lo[::-1], np.array([0, 0, -1.0]))

    def rod(self, a, b, radius, sides=8) -> None:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.linalg.norm(d)
        w = d / L
        u = np.cross(w, [0, 0, 1.0]) if abs(w[2]) < 0.9 else np.cross(w, [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        ring = lambda c: [c + radius * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * math.pi, sides, endpoint=False)]
        lo, hi = ring(a), ring(b)
        for i in range(sides):
            j = (i + 1) % sides
            self.quad(lo[i], lo[j], hi[j], hi[i])

    def mesh(self) -> TexturedMesh:
        P = np.array(self.P, np.float32)
        return TexturedMesh("", P, np.array(self.N, np.float32), np.zeros((len(P), 2), np.float32),
                            np.array(self.I, np.uint32), self.color)


def procedural_lm() -> AssembledModel:
    gold, silver, dark, foil = _Builder(GOLD), _Builder(SILVER), _Builder(DARK), _Builder(FOIL_DARK)
    gold.prism((2.05, 2.05), 1.05, 2.55, 8, phase=math.pi / 8)  # descent stage
    foil.prism((2.1, 2.1), 2.45, 2.6, 8, phase=math.pi / 8, caps=True)  # top deck rim
    dark.prism((0.95, 0.45), 0.35, 1.05, 16)  # engine bell (mouth at the bottom)
    silver.prism((1.55, 1.35), 2.6, 4.6, 6, phase=0.3)  # ascent stage cabin
    silver.prism((1.35, 0.8), 4.6, 5.2, 6, phase=0.3)
    dark.prism((0.4, 0.4), 5.2, 5.5, 12)  # docking tunnel
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        c, s = math.cos(a), math.sin(a)
        top, foot = np.array([2.0 * c, 2.0 * s, 2.0]), np.array([4.55 * c, 4.55 * s, 0.25])
        silver.rod(top, foot, 0.11)  # primary strut
        silver.rod([1.9 * c, 1.9 * s, 1.1], foot * 0.62 + np.array([0, 0, 0.5]), 0.06)  # secondary strut
        dark.prism((0.45, 0.4), 0.0, 0.18, 12, centre=(foot[0], foot[1]))  # footpad
    meshes = [b.mesh() for b in (gold, silver, dark, foil)]
    allp = np.concatenate([m.positions for m in meshes])
    return AssembledModel(meshes, None, np.array([0.0, 0.0, 0.35]), float(allp[:, 2].max()),
                          float(np.hypot(allp[:, 0], allp[:, 1]).max()))
