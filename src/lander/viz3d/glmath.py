"""Small 4x4 matrix helpers for the 3D viewer (right-handed, z-up world)."""

from __future__ import annotations

import math

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(half_w: float, half_h: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 0], m[1, 1] = 1 / half_w, 1 / half_h
    m[2, 2], m[2, 3] = -2 / (far - near), -(far + near) / (far - near)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    f = normalize(np.asarray(target, float) - eye)
    up = np.asarray(up, float)
    if abs(f @ up) > 0.999:  # looking straight up/down: pick another up
        up = np.array([0.0, 1.0, 0.0])
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def rotation_z_to(v: np.ndarray) -> np.ndarray:
    """Smallest rotation taking +z onto direction ``v`` (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    v = normalize(np.asarray(v, float))
    axis = np.cross(z, v)
    s, c = np.linalg.norm(axis), float(z @ v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


def model_matrix(position: np.ndarray, rotation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation * scale
    m[:3, 3] = position
    return m


def gl(m: np.ndarray) -> bytes:
    """numpy (row-major) -> OpenGL uniform bytes (column-major float32)."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()
