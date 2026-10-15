"""Reader for Kerbal Space Program ``.mu`` model files (the binary format KSP's
PartTools exports from Unity).

Independent implementation written from the format as documented by the community
(taniwha's io_object_mu was the reference for the layout; no code is shared). Only
what the viewer needs is kept: the transform hierarchy, render meshes (positions,
normals, UVs, triangles) and materials with their texture names. Animation clips are kept as
keyframes so a pose (e.g. deployed landing legs) can be applied. Everything else
(colliders, lights, cameras, particles, bone weights...) is parsed only to skip
over it, since the format has no length prefixes.

Coordinates are converted from Unity's left-handed, y-up frame to a right-handed,
z-up frame by swapping y and z (and reversing triangle winding to match).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 76543

# entry types
CHILD_START, CHILD_END, ANIMATION = 0, 1, 2
MESH_COLLIDER, SPHERE_COLLIDER, CAPSULE_COLLIDER, BOX_COLLIDER = 3, 4, 5, 6
MESH_FILTER, MESH_RENDERER, SKINNED_MESH_RENDERER = 7, 8, 9
MATERIALS, TEXTURES = 10, 12
MESH_START, MESH_VERTS, MESH_UV, MESH_UV2, MESH_NORMALS, MESH_TANGENTS = 13, 14, 15, 16, 17, 18
MESH_TRIANGLES, MESH_BONE_WEIGHTS, MESH_BIND_POSES, MESH_END = 19, 20, 21, 22
LIGHT, TAG_AND_LAYER = 23, 24
MESH_COLLIDER2, SPHERE_COLLIDER2, CAPSULE_COLLIDER2, BOX_COLLIDER2 = 25, 26, 27, 28
WHEEL_COLLIDER, CAMERA, PARTICLES, MESH_VERTEX_COLORS = 29, 30, 31, 32


@dataclass
class Mesh:
    positions: np.ndarray  # (n, 3) float32, z-up
    normals: np.ndarray | None
    uvs: np.ndarray | None  # (n, 2)
    submeshes: list[np.ndarray]  # one (m, 3) int32 triangle array per material slot


@dataclass
class Material:
    name: str
    shader: str
    textures: dict[str, int]  # property name (e.g. "_MainTex") -> index into MuFile.textures
    colors: dict[str, tuple]


@dataclass
class Node:
    name: str
    position: np.ndarray  # local, z-up
    rotation: np.ndarray  # local quaternion (w, x, y, z), z-up
    scale: np.ndarray
    children: list[Node] = field(default_factory=list)
    mesh: Mesh | None = None
    materials: list[int] = field(default_factory=list)  # renderer's material indices, per submesh
    parent: Node | None = field(default=None, repr=False)

    def local_matrix(self) -> np.ndarray:
        w, x, y, z = self.rotation
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        m = np.eye(4)
        m[:3, :3] = r * self.scale[None, :]
        m[:3, 3] = self.position
        return m

    def world_matrix(self) -> np.ndarray:
        m = self.local_matrix()
        return m if self.parent is None else self.parent.world_matrix() @ m

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, name: str) -> Node | None:
        return next((n for n in self.walk() if n.name == name), None)


@dataclass
class Curve:
    path: str  # transform path relative to the animated node, e.g. "leg/upperStrut"
    prop: str  # e.g. "m_LocalRotation.x"
    keys: list[tuple[float, float]]  # (time, value)


@dataclass
class Clip:
    name: str
    owner: Node  # the node holding the Animation component; curve paths start here
    curves: list[Curve]


@dataclass
class MuFile:
    name: str
    root: Node
    materials: list[Material]
    textures: list[str]  # texture names (usually without extension)
    clips: list[Clip] = field(default_factory=list)

    def pose(self, clip_name: str, at_end: bool = True) -> int:
        """Apply an animation clip's first or last keyframes to the transforms (e.g. put
        landing legs in their deployed pose). Returns the number of curves applied."""
        clip = next((c for c in self.clips if c.name == clip_name), None)
        if clip is None:
            return 0
        applied = 0
        # Unity stores rotations as separate x/y/z/w curves; collect them per node first.
        rot: dict[int, dict[str, float]] = {}
        for c in clip.curves:
            node = clip.owner
            for part in [p for p in c.path.split("/") if p]:
                node = next((ch for ch in node.children if ch.name == part), None)
                if node is None:
                    break
            if node is None or not c.keys:
                continue
            value = c.keys[-1][1] if at_end else c.keys[0][1]
            prop, _, axis = c.prop.partition(".")
            if prop == "m_LocalRotation":
                rot.setdefault(id(node), {"node": node})[axis] = value
            elif prop == "m_LocalPosition" and axis in "xyz":
                # Unity (x, y, z) -> z-up (x, z, y)
                node.position[{"x": 0, "y": 2, "z": 1}[axis]] = value
            elif prop == "m_LocalScale" and axis in "xyz":
                node.scale[{"x": 0, "y": 2, "z": 1}[axis]] = value
            else:
                continue
            applied += 1
        for entry in rot.values():
            node = entry.pop("node")
            # Missing components keep the node's current (converted back to Unity) value.
            w, x, y, z = node.rotation
            unity = {"x": -x, "y": -z, "z": -y, "w": w}
            unity.update(entry)
            q = np.array([unity["w"], -unity["x"], -unity["z"], -unity["y"]])
            node.rotation = q / np.linalg.norm(q)
        return applied


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EOFError
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def ints(self, n: int = 1):
        v = struct.unpack(f"<{n}i", self.take(4 * n))
        return v[0] if n == 1 else v

    def floats(self, n: int = 1):
        v = struct.unpack(f"<{n}f", self.take(4 * n))
        return v[0] if n == 1 else v

    def byte(self) -> int:
        return self.take(1)[0]

    def string(self) -> str:
        # length as a 7-bit variable-length integer (as written by .NET's BinaryWriter)
        n, shift = 0, 0
        while True:
            b = self.byte()
            n |= (b & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        return self.take(n).decode("utf-8", errors="replace")

    def array(self, dtype: str, count: int, width: int) -> np.ndarray:
        raw = self.take(4 * count * width)
        return np.frombuffer(raw, dtype=dtype).reshape(count, width).copy()

    def vec3(self) -> np.ndarray:
        x, y, z = self.floats(3)
        return np.array([x, z, y])  # Unity y-up LH -> z-up RH


def _to_zup(a: np.ndarray) -> np.ndarray:
    return a[:, [0, 2, 1]].astype(np.float32)


def _read_mesh(r: _Reader) -> Mesh:
    if r.ints() != MESH_START:
        raise ValueError("expected mesh start")
    n, _submesh_count = r.ints(2)
    pos = nrm = uv = None
    subs: list[np.ndarray] = []
    while True:
        t = r.ints()
        if t == MESH_END:
            break
        if t == MESH_VERTS:
            pos = _to_zup(r.array("<f4", n, 3))
        elif t == MESH_NORMALS:
            nrm = _to_zup(r.array("<f4", n, 3))
        elif t == MESH_UV:
            uv = r.array("<f4", n, 2)
        elif t == MESH_UV2:
            r.take(4 * 2 * n)
        elif t == MESH_TANGENTS:
            r.take(4 * 4 * n)
        elif t == MESH_BONE_WEIGHTS:
            r.take(4 * 8 * n)
        elif t == MESH_BIND_POSES:
            r.take(4 * 16 * r.ints())
        elif t == MESH_VERTEX_COLORS:
            r.take(4 * n)
        elif t == MESH_TRIANGLES:
            count = r.ints()
            tris = r.array("<i4", count // 3, 3)
            subs.append(tris[:, [0, 2, 1]])  # handedness flip reverses the winding
        else:
            raise ValueError(f"unknown mesh entry {t} at byte {r.pos}")
    if pos is None:
        raise ValueError("mesh without vertices")
    return Mesh(pos, nrm, uv, subs)


def _read_material(r: _Reader, version: int) -> Material:
    if version < 4:
        raise ValueError(f".mu version {version} materials are not supported")
    name, shader = r.string(), r.string()
    textures, colors = {}, {}
    for _ in range(r.ints()):
        prop, kind = r.string(), r.ints()
        if kind in (0, 1):  # color, vector
            colors[prop] = r.floats(4)
        elif kind in (2, 3):  # float, range
            r.floats()
        elif kind == 4:  # texture: index, scale (2), offset (2)
            textures[prop] = r.ints()
            r.floats(4)
        else:
            raise ValueError(f"unknown material property kind {kind}")
    return Material(name, shader, textures, colors)


def _read_animation(r: _Reader, owner: Node) -> list[Clip]:
    clips = []
    for _ in range(r.ints()):
        name = r.string()
        r.floats(6)  # local bounds centre + size
        r.ints()  # wrap mode
        curves = []
        for _ in range(r.ints()):
            path, prop = r.string(), r.string()
            kind = r.ints()
            wrap = r.ints(2)
            # Some PartTools exports omit the curve type; then the second wrap value is the key count.
            n_keys = wrap[1] if kind == 8 else r.ints()
            keys = []
            for _ in range(n_keys):
                t, v = r.floats(2)
                r.floats(2), r.ints()  # in/out tangents, tangent mode
                keys.append((t, v))
            curves.append(Curve(path, prop, keys))
        clips.append(Clip(name, owner, curves))
    r.string()  # default clip
    r.byte()  # auto play
    return clips


def _skip_collider(r: _Reader, t: int) -> None:
    if t in (MESH_COLLIDER2, SPHERE_COLLIDER2, CAPSULE_COLLIDER2, BOX_COLLIDER2):
        r.byte()  # is trigger
    if t in (MESH_COLLIDER, MESH_COLLIDER2):
        r.byte()  # convex
        _read_mesh(r)
    elif t in (SPHERE_COLLIDER, SPHERE_COLLIDER2):
        r.floats(4)
    elif t in (CAPSULE_COLLIDER, CAPSULE_COLLIDER2):
        r.floats(2), r.ints(), r.floats(3)
    elif t in (BOX_COLLIDER, BOX_COLLIDER2):
        r.floats(6)
    elif t == WHEEL_COLLIDER:
        r.floats(6 + 3 + 5 + 5)  # mass, radius, suspension, centre(3), spring(3), 2x friction(5)


def _skip_particles(r: _Reader) -> None:
    # emitter settings: a fixed layout of bytes, ints and floats
    r.byte(), r.ints(), r.floats(3 + 2 + 1 + 4), r.byte(), r.floats(2 + 2), r.ints(2)
    r.floats(3 * 3 + 3), r.byte(), r.byte(), r.floats(5 * 4 + 3 + 3 + 1 + 3 + 3 + 1)
    r.byte(), r.byte(), r.floats(3), r.ints(), r.ints(3), r.ints()


def _read_node(r: _Reader, mu: MuFile, version: int, parent: Node | None) -> Node:
    name = r.string()
    position = r.vec3()
    qx, qy, qz, qw = r.floats(4)
    rotation = np.array([qw, -qx, -qz, -qy])  # Unity LH (x, y, z, w) -> z-up RH (w, x, y, z)
    scale = r.vec3()
    node = Node(name, position, rotation, scale, parent=parent)
    while True:
        try:
            t = r.ints()
        except EOFError:
            break
        if t == CHILD_START:
            node.children.append(_read_node(r, mu, version, node))
        elif t == CHILD_END:
            break
        elif t == TAG_AND_LAYER:
            r.string(), r.ints()
        elif t in (MESH_COLLIDER, SPHERE_COLLIDER, CAPSULE_COLLIDER, BOX_COLLIDER, MESH_COLLIDER2,
                   SPHERE_COLLIDER2, CAPSULE_COLLIDER2, BOX_COLLIDER2, WHEEL_COLLIDER):
            _skip_collider(r, t)
        elif t == MESH_FILTER:
            node.mesh = _read_mesh(r)
        elif t == MESH_RENDERER:
            if version > 0:
                r.byte(), r.byte()  # cast / receive shadows
            node.materials = [r.ints() for _ in range(r.ints())]
        elif t == SKINNED_MESH_RENDERER:
            mats = [r.ints() for _ in range(r.ints())]
            r.floats(6), r.ints(), r.byte()
            for _ in range(r.ints()):
                r.string()
            node.mesh, node.materials = _read_mesh(r), mats
        elif t == ANIMATION:
            mu.clips.extend(_read_animation(r, node))
        elif t == CAMERA:
            r.ints(), r.floats(4), r.ints(), r.byte(), r.floats(4)
        elif t == PARTICLES:
            _skip_particles(r)
        elif t == LIGHT:
            r.ints(), r.floats(6), r.ints()
            if version > 1:
                r.floats()
        elif t == MATERIALS:
            mu.materials.extend(_read_material(r, version) for _ in range(r.ints()))
        elif t == TEXTURES:
            for _ in range(r.ints()):
                tex = r.string()
                r.ints()  # texture type (diffuse / normal map)
                mu.textures.append(tex)
        else:
            raise ValueError(f"unknown entry type {t} at byte {r.pos}")
    return node


def read_mu(path: str | Path) -> MuFile:
    r = _Reader(Path(path).read_bytes())
    magic, version = r.ints(2)
    if magic != MAGIC:
        raise ValueError(f"{path}: not a .mu file")
    mu = MuFile(r.string(), None, [], [])  # type: ignore[arg-type]
    mu.root = _read_node(r, mu, version, None)
    return mu
