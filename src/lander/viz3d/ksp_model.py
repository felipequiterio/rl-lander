"""Assemble the Apollo LM from Bluedog Design Bureau (BDB) KSP parts.

BDB is CC BY-NC-SA 4.0, so its files are never copied into this repo: they're read
at runtime from the user's own KSP install (``--bdb PATH`` / ``LANDER_BDB``).

KSP builds a craft from parts joined at attach nodes, and each part's ``.cfg``
says which ``.mu`` models it uses, where its nodes are, and which transforms
each variant (B9PartSwitch "subtype") shows. Here we:

1. parse the four LM part configs and their models;
2. keep each variant switch's default (first) subtype and hide the rest;
3. stack the parts: ascent stage on the descent stage's top node, engine on its
   bottom node, a leg on each of the four ``legNode_N`` transforms;
4. merge everything into one mesh per texture, scaled from KSP size to real size
   (BDB parts are ~0.64x real; 1.6x is the usual Real Solar System factor), with
   the footpads at z = 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lander.viz3d.mu import MuFile, Node, read_mu

REAL_SCALE = 1.6
PARTS_DIR = Path("Parts") / "Apollo"


# --- KSP config files ----------------------------------------------------------------


@dataclass
class CfgNode:
    name: str
    values: list[tuple[str, str]]
    children: list[CfgNode]

    def get(self, key: str) -> str | None:
        return next((v for k, v in self.values if k == key), None)

    def all(self, key: str) -> list[str]:
        return [v for k, v in self.values if k == key]

    def nodes(self, name: str) -> list[CfgNode]:
        return [c for c in self.children if c.name == name]


def parse_cfg(text: str) -> CfgNode:
    """Minimal parser for KSP's ConfigNode text format (nested `NAME { key = value }`)."""
    text = re.sub(r"//[^\n]*", "", text)
    tokens = re.findall(r"\{|\}|[^{}\n]+", text)
    root = CfgNode("root", [], [])
    stack, pending = [root], None
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok == "{":
            node = CfgNode(pending or "", [], [])
            stack[-1].children.append(node)
            stack.append(node)
            pending = None
        elif tok == "}":
            if len(stack) > 1:
                stack.pop()
        elif "=" in tok:
            k, v = tok.split("=", 1)
            stack[-1].values.append((k.strip(), v.strip()))
        else:
            pending = tok  # a node name; its "{" follows (possibly on the next line)
    return root


def _vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(",")[:3]])


def _unity_to_zup(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], v[2], v[1]])


@dataclass
class Part:
    name: str
    cfg: CfgNode
    models: list[MuFile]
    hidden: set[str]  # transform names hidden by the default variant

    def stack_node(self, key: str) -> np.ndarray:
        """Position of a `node_stack_*` attach node, z-up, part frame."""
        return _unity_to_zup(_vec(self.cfg.get(key)))


def load_part(bdb: Path, name: str) -> Part:
    cfg = parse_cfg((bdb / PARTS_DIR / f"{name}.cfg").read_text(encoding="utf-8-sig", errors="replace"))
    part = cfg.nodes("PART")[0]
    models = []
    for m in part.nodes("MODEL"):
        rel = m.get("model").removeprefix("Bluedog_DB/")
        models.append(read_mu(bdb / f"{rel}.mu"))
    hidden: set[str] = set()
    for mod in part.nodes("MODULE"):
        kind = mod.get("name")
        if kind == "ModuleB9DisableTransform":
            hidden.update(mod.all("transform"))
        elif kind == "ModuleB9PartSwitch":
            subtypes = mod.nodes("SUBTYPE")
            if subtypes:
                every = {t for st in subtypes for t in st.all("transform")}
                hidden.update(every - set(subtypes[0].all("transform")))
    return Part(name, part, models, hidden)


# --- assembly -----------------------------------------------------------------------------


def _local_world(node: Node) -> np.ndarray:
    """World matrix within the model, ignoring the root's own transform (KSP replaces it
    with the part's placement)."""
    if node.parent is None:
        return np.eye(4)
    return _local_world(node.parent) @ node.local_matrix()


def _visible_meshes(model: MuFile, hidden: set[str]):
    def walk(n: Node):
        if n.name in hidden:
            return
        if n.mesh is not None and n.materials:
            yield n
        for c in n.children:
            yield from walk(c)

    yield from walk(model.root)


@dataclass
class TexturedMesh:
    texture: str  # texture base name (no extension), "" for untextured
    positions: np.ndarray  # (n, 3) float32
    normals: np.ndarray  # (n, 3) float32
    uvs: np.ndarray  # (n, 2) float32
    indices: np.ndarray  # (m, 3) uint32
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # multiplies the texture (or is the colour if untextured)


@dataclass
class AssembledModel:
    meshes: list[TexturedMesh]
    texture_dir: Path | None
    engine_exit: np.ndarray  # nozzle exit point, model frame (z-up, metres)
    height: float
    footprint_radius: float


_SKIP_SHADERS = ("Translucent", "Additive", "Unlit", "Particles")


def assemble_lm(bdb: str | Path) -> AssembledModel:
    bdb = Path(bdb)
    ascent = load_part(bdb, "bluedog_LM_Ascent_Cockpit")
    descent = load_part(bdb, "bluedog_LM_Descent_Tanks")
    engine = load_part(bdb, "bluedog_LM_Descent_Engine")
    leg = load_part(bdb, "bluedog_LM_Descent_Leg")
    # The model's rest pose has the legs stowed; KSP deploys them with this animation.
    leg.models[0].pose("deploy", at_end=True)

    def translate(v):
        m = np.eye(4)
        m[:3, 3] = v
        return m

    placements: list[tuple[Part, np.ndarray]] = [(descent, np.eye(4))]
    # Stack attach nodes: the child's bottom node meets the parent's node.
    placements.append((ascent, translate(descent.stack_node("node_stack_top") - ascent.stack_node("node_stack_bottom"))))
    placements.append((engine, translate(descent.stack_node("node_stack_bottom") - engine.stack_node("node_stack_top"))))
    # Legs: each leg's legNode meets one of the descent stage's legNode_N. Attach nodes
    # face each other, so the leg's node is turned 180 degrees about the socket's up axis.
    leg_node = leg.models[0].root.find("legNode")
    leg_inv = np.linalg.inv(_local_world(leg_node))
    face = np.diag([-1.0, -1.0, 1.0, 1.0])
    for i in range(1, 5):
        socket = descent.models[0].root.find(f"legNode_{i}")
        placements.append((leg, _local_world(socket) @ face @ leg_inv))

    buckets: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for part, placement in placements:
        for model in part.models:
            for node in _visible_meshes(model, part.hidden):
                m = placement @ _local_world(node)
                mesh = node.mesh
                pos = (m[:3, :3] @ mesh.positions.T).T + m[:3, 3]
                nrm_m = np.linalg.inv(m[:3, :3]).T
                nrm = (nrm_m @ (mesh.normals if mesh.normals is not None else np.zeros_like(mesh.positions)).T).T
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
                if np.linalg.det(m[:3, :3]) < 0:  # mirrored transform: flip winding to keep faces outward
                    flip = True
                else:
                    flip = False
                uv = mesh.uvs if mesh.uvs is not None else np.zeros((len(pos), 2), np.float32)
                for slot, tris in enumerate(mesh.submeshes):
                    mat_index = node.materials[min(slot, len(node.materials) - 1)]
                    mat = model.materials[mat_index] if 0 <= mat_index < len(model.materials) else None
                    if mat is not None and any(s in mat.shader for s in _SKIP_SHADERS):
                        continue
                    tex_i = mat.textures.get("_MainTex", -1) if mat else -1
                    tex = Path(model.textures[tex_i]).stem if 0 <= tex_i < len(model.textures) else ""
                    t = tris[:, [0, 2, 1]] if flip else tris
                    buckets.setdefault(tex, []).append((pos, nrm, uv, t))

    # Scale to real size and put the footpads (lowest point) on z = 0.
    all_pos = np.concatenate([p for parts in buckets.values() for p, *_ in parts])
    z0 = all_pos[:, 2].min()
    meshes = []
    for tex, parts in buckets.items():
        offset, P, N, U, I = 0, [], [], [], []
        for pos, nrm, uv, tris in parts:
            p = pos.copy()
            p[:, 2] -= z0
            P.append(p * REAL_SCALE)
            N.append(nrm)
            U.append(uv)
            I.append(tris + offset)
            offset += len(pos)
        meshes.append(TexturedMesh(tex, np.concatenate(P).astype(np.float32), np.concatenate(N).astype(np.float32),
                                   np.concatenate(U).astype(np.float32), np.concatenate(I).astype(np.uint32)))
    all_pos = np.concatenate([m.positions for m in meshes])
    thrust = engine.models[0].root.find("thrustTransform")
    engine_place = placements[2][1]
    exit_pt = (engine_place @ _local_world(thrust))[:3, 3] if thrust is not None else np.array([0, 0, z0])
    exit_pt = (exit_pt - np.array([0, 0, z0])) * REAL_SCALE
    return AssembledModel(
        meshes=meshes,
        texture_dir=bdb / PARTS_DIR,
        engine_exit=exit_pt,
        height=float(all_pos[:, 2].max()),
        footprint_radius=float(np.hypot(all_pos[:, 0], all_pos[:, 1]).max()),
    )


# --- bundled copy ---------------------------------------------------------------------------

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "apollo_lm"


def save_bundle(model: AssembledModel, out: Path, texture_size: int = 1024) -> list[Path]:
    """Write an assembled model as a compact, self-contained bundle: one compressed .npz
    with the meshes and metadata, plus each texture as a JPEG in normal (top-down) row
    order. Returns the files written."""
    import json

    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {"meshes": [], "engine_exit": model.engine_exit.tolist(), "height": model.height,
                        "footprint_radius": model.footprint_radius}
    written = []
    for i, m in enumerate(model.meshes):
        if len(m.indices) == 0:
            continue
        for key in ("positions", "normals", "uvs", "indices"):
            arrays[f"m{i}_{key}"] = getattr(m, key)
        meta["meshes"].append({"index": i, "texture": m.texture, "color": list(m.color)})
        src = model.texture_dir / f"{m.texture}.dds" if model.texture_dir and m.texture else None
        dst = out / f"{m.texture}.jpg"
        if src is not None and src.exists() and not dst.exists():
            img = Image.open(src).convert("RGB")
            # KSP's DDS files are stored upside down; save in normal orientation.
            img = img.transpose(Image.FLIP_TOP_BOTTOM).resize((texture_size, texture_size), Image.LANCZOS)
            img.save(dst, quality=88)
            written.append(dst)
    np.savez_compressed(out / "model.npz", **arrays)
    (out / "model.json").write_text(json.dumps(meta, indent=1))
    return [out / "model.npz", out / "model.json", *written]


def load_bundle(folder: Path = BUNDLE_DIR) -> AssembledModel | None:
    import json

    if not (folder / "model.json").exists():
        return None
    meta = json.loads((folder / "model.json").read_text())
    data = np.load(folder / "model.npz")
    meshes = [TexturedMesh(m["texture"], *(data[f"m{m['index']}_{k}"] for k in ("positions", "normals", "uvs", "indices")),
                           tuple(m["color"])) for m in meta["meshes"]]
    return AssembledModel(meshes, folder, np.array(meta["engine_exit"]), meta["height"], meta["footprint_radius"])
