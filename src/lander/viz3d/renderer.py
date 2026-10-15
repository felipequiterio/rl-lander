"""Offscreen OpenGL renderer for the 3D viewer (moderngl).

The scene is drawn into an anti-aliased offscreen framebuffer and read back into
a pygame surface, so the existing pygame HUD simply draws on top. Under WSL2 the
d3d12 Mesa driver is selected so rendering runs on the GPU; anywhere else (or if
that fails) Mesa's software rasteriser still works, just slower.

What's drawn (all in the env's frame: metres, z up, target at the origin):
- the lander mesh, lit by a low sun with shadow mapping;
- the ground: one big flat quad whose shader adds procedural crater relief (normal
  map), fine detail for close-ups, the landing-zone markings and the lander's shadow;
- the glide-slope cone (translucent), trajectory trail and ground track (lines);
- billboards: particles (exhaust, dust, explosion), beacons, engine glow, the sun;
- stars (points) and the Earth (a textured billboard).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np
import pygame
from PIL import Image

from lander.viz3d import terrain
from lander.viz3d.glmath import gl, look_at, normalize, ortho, perspective
from lander.viz3d.ksp_model import AssembledModel

SUN_DIR = normalize(np.array([math.cos(math.radians(140)), math.sin(math.radians(140)), math.tan(math.radians(14))]))
EARTH_DIR = normalize(np.array([math.cos(math.radians(-60)), math.sin(math.radians(-60)), math.tan(math.radians(28))]))
FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])  # render upside down so the readback is already top-down


def create_context() -> moderngl.Context:
    # WSL2: route OpenGL to the GPU through Mesa's d3d12 driver (otherwise it's llvmpipe on the CPU).
    if "GALLIUM_DRIVER" not in os.environ and Path("/dev/dxg").exists():
        os.environ["GALLIUM_DRIVER"] = "d3d12"
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception:  # noqa: BLE001 - try the EGL backend before giving up
        return moderngl.create_standalone_context(require=330, backend="egl")


# --- shaders ------------------------------------------------------------------------

SHADOW_FN = """
float shadow_at(sampler2DShadow sm, vec4 lp, float bias) {
    vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) return 1.0;
    vec2 texel = 1.0 / vec2(textureSize(sm, 0));
    float s = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            s += texture(sm, vec3(p.xy + vec2(x, y) * texel, p.z - bias));
    return s / 9.0;
}
"""

MESH_VS = """
#version 330
uniform mat4 u_vp; uniform mat4 u_model; uniform mat4 u_light_vp;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
out vec3 v_nrm; out vec2 v_uv; out vec3 v_world; out vec4 v_light;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz; v_nrm = mat3(u_model) * in_nrm; v_uv = in_uv; v_light = u_light_vp * w;
    gl_Position = u_vp * w;
}
"""

MESH_FS = """
#version 330
uniform sampler2D u_tex; uniform sampler2DShadow u_shadow;
uniform vec3 u_sun; uniform vec3 u_eye; uniform vec3 u_color; uniform float u_char;
in vec3 v_nrm; in vec2 v_uv; in vec3 v_world; in vec4 v_light;
out vec4 f_color;
""" + SHADOW_FN + """
void main() {
    vec3 v = normalize(u_eye - v_world);
    vec3 n = normalize(v_nrm);
    // Double-sided: face the normal towards the viewer. (gl_FrontFacing can't be used:
    // the projection is flipped vertically, which swaps the winding.)
    if (dot(n, v) < 0.0) n = -n;
    vec3 albedo = texture(u_tex, v_uv).rgb * u_color;
    float ndl = max(dot(n, u_sun), 0.0);
    float sh = shadow_at(u_shadow, v_light, 0.002);
    vec3 h = normalize(u_sun + v);
    float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.45 * sh * step(0.0, ndl);
    float bounce = clamp(0.5 - 0.5 * n.z, 0.0, 1.0) * 0.16;  // sunlit regolith lights the underside
    vec3 col = albedo * (0.09 + bounce + 1.35 * ndl * sh) + vec3(spec);
    col *= mix(1.0, 0.22, u_char);
    f_color = vec4(col, 1.0);
}
"""

DEPTH_VS = """
#version 330
uniform mat4 u_light_vp; uniform mat4 u_model;
in vec3 in_pos;
void main() { gl_Position = u_light_vp * u_model * vec4(in_pos, 1.0); }
"""
DEPTH_FS = "#version 330\nvoid main() {}\n"

GROUND_VS = """
#version 330
uniform mat4 u_vp; uniform mat4 u_light_vp;
in vec2 in_xy;
out vec3 v_world; out vec4 v_light;
void main() {
    vec4 w = vec4(in_xy, 0.0, 1.0);
    v_world = w.xyz; v_light = u_light_vp * w;
    gl_Position = u_vp * w;
}
"""

GROUND_FS = """
#version 330
uniform sampler2D u_craters; uniform sampler2D u_detail; uniform sampler2DShadow u_shadow;
uniform vec3 u_sun; uniform vec3 u_eye; uniform float u_extent; uniform float u_detail_tile;
uniform float u_zone_r; uniform vec3 u_zone_color; uniform vec3 u_glow_pos; uniform float u_glow;
in vec3 v_world; in vec4 v_light;
out vec4 f_color;
""" + SHADOW_FN + """
void main() {
    vec2 p = v_world.xy;
    float r = length(p);
    // crater relief: the detailed map near the target, the same map scaled up farther out
    vec4 c1 = texture(u_craters, p / (2.0 * u_extent) + 0.5);
    vec4 c2 = texture(u_craters, p / (9.0 * u_extent) + 0.37);
    float near = 1.0 - smoothstep(0.75 * u_extent, 0.98 * u_extent, max(abs(p.x), abs(p.y)));
    vec4 c = mix(c2, c1, near);
    vec3 n = c.rgb * 2.0 - 1.0;
    float albedo = c.a;
    // fine detail, faded out with distance to avoid shimmering
    float de = length(u_eye - v_world);
    // fade the fine detail with distance and at grazing angles (where it aliases into streaks)
    float grazing = clamp((u_eye.z - v_world.z) / max(de, 1e-3), 0.0, 1.0);
    float dw = (1.0 - smoothstep(40.0, 350.0, de)) * smoothstep(0.03, 0.2, grazing);
    vec4 d = texture(u_detail, p / u_detail_tile);
    n = normalize(vec3(n.xy + (d.rgb * 2.0 - 1.0).xy * 1.4 * dw, n.z));
    albedo *= mix(1.0, 0.7 + 0.6 * d.a, dw);
    float ndl = max(dot(n, u_sun), 0.0);
    float sh = shadow_at(u_shadow, v_light, 0.0008);
    vec3 col = vec3(0.56, 0.55, 0.53) * albedo * 2.0 * (0.03 + 1.35 * ndl * sh);
    // landing zone: a slightly lighter pad, a ring at the zone edge, a centre cross
    float pad = 1.0 - smoothstep(u_zone_r - 0.3, u_zone_r, r);
    col = mix(col, col * 1.18 + 0.015, pad);
    float ring = 1.0 - smoothstep(0.12, 0.35, abs(r - u_zone_r));
    float cross_ = (1.0 - smoothstep(0.12, 0.3, min(abs(p.x), abs(p.y)))) * (1.0 - smoothstep(2.5, 3.0, r));
    col = mix(col, u_zone_color * (0.25 + 0.75 * ndl * sh), max(ring, cross_) * 0.9);
    // engine light spilling on the ground
    float g = length(p - u_glow_pos.xy);
    col += vec3(1.0, 0.55, 0.22) * u_glow * exp(-g * g / 60.0);
    // gentle darkening towards the horizon
    col *= 1.0 - 0.35 * smoothstep(3000.0, 15000.0, de);
    f_color = vec4(col, 1.0);
}
"""

COLOR_VS = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos; in vec4 in_color;
out vec4 v_color;
void main() { v_color = in_color; gl_Position = u_vp * vec4(in_pos, 1.0); }
"""
COLOR_FS = """
#version 330
in vec4 v_color; out vec4 f_color;
void main() { f_color = v_color; }
"""

CONE_VS = """
#version 330
uniform mat4 u_vp; uniform float u_r0; uniform float u_tan; uniform float u_height;
in vec2 in_sa;  // s in [0,1] along the height, a = angle
out float v_z;
void main() {
    float z = in_sa.x * u_height;
    float r = u_r0 + z / u_tan;
    v_z = z;
    gl_Position = u_vp * vec4(r * cos(in_sa.y), r * sin(in_sa.y), z, 1.0);
}
"""
CONE_FS = """
#version 330
uniform float u_spacing; uniform float u_height;
in float v_z; out vec4 f_color;
void main() {
    float g = abs(fract(v_z / u_spacing - 0.5) - 0.5) / fwidth(v_z / u_spacing);
    float line = 1.0 - min(g, 1.0);
    float fade = 1.0 - smoothstep(0.6, 1.0, v_z / u_height);
    f_color = vec4(1.0, 0.36, 0.28, (0.015 + 0.22 * line) * fade);
}
"""

BILLBOARD_VS = """
#version 330
uniform mat4 u_vp; uniform vec3 u_right; uniform vec3 u_up;
in vec2 in_corner; in vec3 in_center; in float in_size; in vec4 in_color;
out vec2 v_uv; out vec4 v_color;
void main() {
    vec3 p = in_center + (in_corner.x * u_right + in_corner.y * u_up) * in_size;
    v_uv = in_corner; v_color = in_color;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""
BILLBOARD_FS = """
#version 330
uniform int u_soft;  // 0: additive glow, 1: soft alpha disc (dust, smoke)
in vec2 v_uv; in vec4 v_color; out vec4 f_color;
void main() {
    float r = length(v_uv);
    if (r > 1.0) discard;
    float k = u_soft == 1 ? pow(1.0 - r, 1.5) : pow(1.0 - r, 2.0);
    f_color = u_soft == 1 ? vec4(v_color.rgb, v_color.a * k) : vec4(v_color.rgb * k * v_color.a, 1.0);
}
"""

SPRITE_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv; in vec4 v_color; out vec4 f_color;
void main() { vec4 t = texture(u_tex, v_uv * 0.5 + 0.5); f_color = vec4(t.rgb, t.a * v_color.a); }
"""

STARS_VS = """
#version 330
uniform mat4 u_vp; uniform vec3 u_eye;
in vec3 in_dir; in float in_mag;
out float v_mag;
void main() {
    v_mag = in_mag;
    vec4 p = u_vp * vec4(u_eye + in_dir * 40000.0, 1.0);
    gl_Position = vec4(p.xy, p.w * 0.99999, p.w);  // at the far plane, behind everything
    gl_PointSize = in_mag > 0.85 ? 2.0 : 1.0;
}
"""
STARS_FS = """
#version 330
in float v_mag; out vec4 f_color;
void main() { f_color = vec4(vec3(0.75, 0.8, 1.0) * v_mag, 1.0); }
"""

PLUME_VS = """
#version 330
uniform mat4 u_vp; uniform mat4 u_model;
in vec3 in_pos;  // unit plume: z from 0 (nozzle) to -1, radius grows along the length
out float v_t; out vec3 v_world;
void main() {
    v_t = -in_pos.z;
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    gl_Position = u_vp * w;
}
"""
PLUME_FS = """
#version 330
uniform float u_intensity;
in float v_t; in vec3 v_world; out vec4 f_color;
void main() {
    float k = pow(1.0 - clamp(v_t, 0.0, 1.0), 1.6) * u_intensity;
    vec3 col = mix(vec3(1.0, 0.45, 0.15), vec3(1.0, 0.92, 0.75), pow(1.0 - v_t, 3.0));
    f_color = vec4(col * k, 1.0);
}
"""


# --- per-frame inputs --------------------------------------------------------------------


@dataclass
class Frame:
    eye: np.ndarray
    look: np.ndarray
    fov: float
    lander_model: np.ndarray  # 4x4 body -> world
    charred: bool = False
    plume_model: np.ndarray | None = None  # 4x4 unit plume -> world, or None when off
    plume_intensity: float = 0.0
    glow: list = field(default_factory=list)  # additive billboards: (x, y, z, size, r, g, b, a)
    soft: list = field(default_factory=list)  # alpha billboards: same layout
    trail: np.ndarray | None = None  # (n, 3)
    cone_height: float = 0.0  # draw the glide-slope cone surface up to this height
    limit_ring: float | None = None  # altitude at which to draw the cone's ring (the lander's)
    zone_color: tuple = (0.95, 0.75, 0.2)
    ground_glow: tuple = (0.0, 0.0, 0.0, 0.0)  # x, y, z, intensity


class Renderer:
    def __init__(self, model: AssembledModel, size=(1200, 800), zone_radius=10.0, glide_slope_deg: float | None = 15.0):
        self.size = size
        self.ctx = ctx = create_context()
        self.gpu = ctx.info.get("GL_RENDERER", "?")
        self.software = "llvmpipe" in self.gpu.lower() or "softpipe" in self.gpu.lower()
        samples = 0 if self.software else 4
        self.fbo_ms = ctx.framebuffer(ctx.renderbuffer(size, 4, samples=samples), ctx.depth_renderbuffer(size, samples=samples))
        self.fbo = ctx.framebuffer(ctx.renderbuffer(size, 4)) if samples else self.fbo_ms
        self.shadow_tex = ctx.depth_texture((2048, 2048))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.zone_radius = zone_radius
        self.tan_gs = math.tan(math.radians(glide_slope_deg)) if glide_slope_deg else None

        P = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_mesh, self.p_depth, self.p_ground = P(MESH_VS, MESH_FS), P(DEPTH_VS, DEPTH_FS), P(GROUND_VS, GROUND_FS)
        self.p_color, self.p_cone, self.p_bb = P(COLOR_VS, COLOR_FS), P(CONE_VS, CONE_FS), P(BILLBOARD_VS, BILLBOARD_FS)
        self.p_sprite, self.p_stars, self.p_plume = P(BILLBOARD_VS, SPRITE_FS), P(STARS_VS, STARS_FS), P(PLUME_VS, PLUME_FS)

        self._upload_model(model)
        self._make_ground()
        self._make_sky()
        self._make_cone()
        self._make_plume()
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.bb_inst = ctx.buffer(reserve=8 * 4 * 4096, dynamic=True)
        self.vao_bb = ctx.vertex_array(self.p_bb, [(quad, "2f", "in_corner"), (self.bb_inst, "3f 1f 4f/i", "in_center", "in_size", "in_color")])
        self.sprite_inst = ctx.buffer(reserve=8 * 4, dynamic=True)
        self.vao_sprite = ctx.vertex_array(self.p_sprite, [(quad, "2f", "in_corner"), (self.sprite_inst, "3f 1f 4f/i", "in_center", "in_size", "in_color")])
        self.lines = ctx.buffer(reserve=7 * 4 * 8192, dynamic=True)
        self.vao_lines = ctx.vertex_array(self.p_color, [(self.lines, "3f 4f", "in_pos", "in_color")])

    # --- resources ----------------------------------------------------------------------

    @staticmethod
    def _set(prog, name, value) -> None:
        if name in prog:  # uniforms the compiler optimised away don't exist
            if isinstance(value, bytes):
                prog[name].write(value)
            else:
                prog[name].value = value

    def _texture(self, rgba: np.ndarray, mip=True, repeat=True) -> moderngl.Texture:
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, np.ascontiguousarray(rgba).tobytes())
        if mip:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
        tex.repeat_x = tex.repeat_y = repeat
        return tex

    def _load_texture(self, folder: Path | None, name: str) -> moderngl.Texture:
        if folder is not None and name:
            for ext in (".dds", ".png", ".jpg", ".tga", ".mbm"):
                f = folder / f"{name}{ext}"
                if f.exists():
                    try:
                        img = Image.open(f).convert("RGBA")
                        # KSP stores DDS textures pre-flipped, which is exactly OpenGL's
                        # bottom-up row order; other formats need flipping.
                        if ext != ".dds":
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        if max(img.size) > 2048:
                            img = img.resize((2048, 2048))
                        return self._texture(np.asarray(img))
                    except Exception as e:  # noqa: BLE001
                        print(f"texture {f.name}: {e}; using flat colour", flush=True)
        return self._texture(np.full((1, 1, 4), 255, np.uint8), mip=False)

    def _upload_model(self, model: AssembledModel) -> None:
        self.parts = []
        for m in model.meshes:
            if len(m.indices) == 0:
                continue
            data = np.concatenate([m.positions, m.normals, m.uvs], 1).astype("f4")
            vbo, ibo = self.ctx.buffer(data.tobytes()), self.ctx.buffer(m.indices.astype("u4").tobytes())
            vao = self.ctx.vertex_array(self.p_mesh, [(vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], index_buffer=ibo, index_element_size=4)
            dvao = self.ctx.vertex_array(self.p_depth, [(vbo, "3f 20x", "in_pos")], index_buffer=ibo, index_element_size=4)
            self.parts.append((vao, dvao, self._load_texture(model.texture_dir, m.texture), m.color))
        self.model_height = model.height
        self.model_radius = model.footprint_radius

    def _make_ground(self) -> None:
        R = 25000.0
        xy = np.array([-R, -R, R, -R, -R, R, R, R], "f4")
        self.vao_ground = self.ctx.vertex_array(self.p_ground, [(self.ctx.buffer(xy.tobytes()), "2f", "in_xy")])
        self.tex_craters = self._texture(terrain.crater_map(), repeat=True)
        self.tex_detail = self._texture(terrain.detail_map(), repeat=True)

    def _make_sky(self) -> None:
        rng = np.random.default_rng(1)
        n = 6000
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # denser band (a stand-in Milky Way) around a tilted great circle
        band = rng.normal(size=(2500, 3)) * np.array([1, 1, 0.12])
        band /= np.linalg.norm(band, axis=1, keepdims=True)
        tilt = np.array([[1, 0, 0], [0, math.cos(1.1), -math.sin(1.1)], [0, math.sin(1.1), math.cos(1.1)]])
        d = np.concatenate([d, band @ tilt.T])
        mag = np.concatenate([0.15 + 0.85 * rng.random(n) ** 4, 0.08 + 0.3 * rng.random(2500)])
        data = np.concatenate([d, mag[:, None]], 1).astype("f4")
        self.n_stars = len(data)
        self.vao_stars = self.ctx.vertex_array(self.p_stars, [(self.ctx.buffer(data.tobytes()), "3f 1f", "in_dir", "in_mag")])
        from lander.envs.scene_art import make_earth  # pygame-drawn Earth, reused as a texture

        earth = make_earth(60)
        rgba = np.dstack([pygame.surfarray.array3d(earth), pygame.surfarray.array_alpha(earth)]).transpose(1, 0, 2)
        self.tex_earth = self._texture(rgba[::-1].copy(), mip=False, repeat=False)

    def _make_cone(self) -> None:
        s, a = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 2 * math.pi, 97))
        verts = np.stack([s, a], -1).astype("f4")  # (97, 40, 2)
        idx = []
        for i in range(96):
            for j in range(39):
                k = i * 40 + j
                idx += [k, k + 40, k + 1, k + 1, k + 40, k + 41]
        self.vao_cone = self.ctx.vertex_array(self.p_cone, [(self.ctx.buffer(verts.reshape(-1, 2).tobytes()), "2f", "in_sa")],
                                              index_buffer=self.ctx.buffer(np.array(idx, "u4").tobytes()), index_element_size=4)

    def _make_plume(self) -> None:
        sides, rings = 24, 12
        verts, idx = [], []
        for j in range(rings + 1):
            t = j / rings
            r = 0.28 + 0.9 * t**0.7  # widens as it expands into vacuum
            for i in range(sides):
                a = 2 * math.pi * i / sides
                verts.append((r * math.cos(a), r * math.sin(a), -t))
        for j in range(rings):
            for i in range(sides):
                k, k2 = j * sides + i, j * sides + (i + 1) % sides
                idx += [k, k + sides, k2, k2, k + sides, k2 + sides]
        self.vao_plume = self.ctx.vertex_array(self.p_plume, [(self.ctx.buffer(np.array(verts, "f4").tobytes()), "3f", "in_pos")],
                                               index_buffer=self.ctx.buffer(np.array(idx, "u4").tobytes()), index_element_size=4)

    # --- frame ----------------------------------------------------------------------------

    def render(self, f: Frame) -> pygame.Surface:
        ctx, w, h = self.ctx, *self.size
        lander_pos = f.lander_model[:3, 3]
        dist = float(np.linalg.norm(f.eye - f.look))
        near = max(0.25, min(dist * 0.02, 5.0))
        view = look_at(f.eye, f.look)
        vp = FLIP_Y @ perspective(f.fov, w / h, near, 60000.0) @ view

        # Shadow map: an orthographic view from the sun, centred on the lander. The
        # shadow on the ground lies along the same sun ray, so it's always in frame.
        light_eye = lander_pos + SUN_DIR * 60.0
        alt = max(lander_pos[2], 0.0)
        light_vp = ortho(self.model_radius + 6, self.model_radius + 6, 1.0, 60.0 + (alt + 20) / SUN_DIR[2] + 50) @ look_at(light_eye, lander_pos)
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        self._set(self.p_depth, "u_light_vp", gl(light_vp))
        self._set(self.p_depth, "u_model", gl(f.lander_model))
        for _, dvao, _, _ in self.parts:
            dvao.render()

        fbo = self.fbo_ms
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.disable(moderngl.BLEND)

        # sky: stars (no depth write needed: they're pushed to the far plane)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self._set(self.p_stars, "u_vp", gl(vp))
        self._set(self.p_stars, "u_eye", tuple(f.eye))
        self.vao_stars.render(moderngl.POINTS, vertices=self.n_stars)

        # ground
        self.shadow_tex.use(location=2)
        self.tex_craters.use(location=0)
        self.tex_detail.use(location=1)
        g = self.p_ground
        for k, v in (("u_vp", gl(vp)), ("u_light_vp", gl(light_vp)), ("u_craters", 0), ("u_detail", 1), ("u_shadow", 2),
                     ("u_sun", tuple(SUN_DIR)), ("u_eye", tuple(f.eye)), ("u_extent", terrain.EXTENT),
                     ("u_detail_tile", terrain.DETAIL_TILE), ("u_zone_r", self.zone_radius), ("u_zone_color", f.zone_color),
                     ("u_glow_pos", tuple(f.ground_glow[:3])), ("u_glow", f.ground_glow[3])):
            self._set(g, k, v)
        self.vao_ground.render(moderngl.TRIANGLE_STRIP)

        # lander
        m = self.p_mesh
        for k, v in (("u_vp", gl(vp)), ("u_model", gl(f.lander_model)), ("u_light_vp", gl(light_vp)), ("u_shadow", 2),
                     ("u_tex", 0), ("u_sun", tuple(SUN_DIR)), ("u_eye", tuple(f.eye)), ("u_char", 1.0 if f.charred else 0.0)):
            self._set(m, k, v)
        for vao, _, tex, color in self.parts:
            tex.use(location=0)
            self._set(m, "u_color", tuple(color))
            vao.render()

        # translucent and additive overlays: depth-tested, no depth writes
        ctx.enable(moderngl.BLEND)
        fbo.depth_mask = False
        right, up = view[0, :3], view[1, :3]
        # The cone surface only makes sense seen from above it; from below (e.g. the ground
        # camera) its grid would fill the sky, so then only the altitude ring is drawn.
        eye_d = float(np.hypot(f.eye[0], f.eye[1]))
        above_cone = self.tan_gs is not None and f.eye[2] > self.tan_gs * max(eye_d - self.zone_radius, 0.0) + 2.0
        if self.tan_gs and f.cone_height > 0 and above_cone:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            for k, v in (("u_vp", gl(vp)), ("u_r0", self.zone_radius), ("u_tan", self.tan_gs), ("u_height", f.cone_height),
                         ("u_spacing", 10 ** math.floor(math.log10(max(f.cone_height, 10)) - 0.3))):
                self._set(self.p_cone, k, v)
            self.vao_cone.render()
        if self.tan_gs and f.limit_ring is not None:
            # The glide-slope limit at the lander's altitude: a horizontal ring it must stay inside.
            z = f.limit_ring
            rad = self.zone_radius + z / self.tan_gs
            a = np.linspace(0, 2 * math.pi, 181)
            ring = np.stack([rad * np.cos(a), rad * np.sin(a), np.full_like(a, z)], 1)
            data = np.concatenate([ring, np.tile([1.0, 0.42, 0.32, 0.85], (len(a), 1))], 1).astype("f4")
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._set(self.p_color, "u_vp", gl(vp))
            self.lines.write(data.tobytes())
            self.vao_lines.render(moderngl.LINE_STRIP, vertices=len(a))
        if f.trail is not None and len(f.trail) > 1:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            n = min(len(f.trail), 4000)
            pts = f.trail[-n:]
            alpha = np.linspace(0.15, 0.9, n)
            path = np.concatenate([pts, np.tile([0.47, 0.78, 1.0], (n, 1)), alpha[:, None]], 1)
            track = np.concatenate([pts * [1, 1, 0] + [0, 0, 0.08], np.tile([0.47, 0.78, 1.0], (n, 1)), alpha[:, None] * 0.4], 1)
            beam = np.array([[0, 0, 0, 0.95, 0.75, 0.2, 0.7], [0, 0, 60.0, 0.95, 0.75, 0.2, 0.0]])
            self._set(self.p_color, "u_vp", gl(vp))
            data = np.concatenate([path, track, beam]).astype("f4")
            self.lines.write(data.tobytes())
            self.vao_lines.render(moderngl.LINE_STRIP, vertices=n)
            self.vao_lines.render(moderngl.LINE_STRIP, vertices=n, first=n)
            self.vao_lines.render(moderngl.LINES, vertices=2, first=2 * n)
        ctx.blend_func = moderngl.ADDITIVE_BLENDING
        if f.plume_model is not None and f.plume_intensity > 0:
            self._set(self.p_plume, "u_vp", gl(vp))
            self._set(self.p_plume, "u_model", gl(f.plume_model))
            self._set(self.p_plume, "u_intensity", f.plume_intensity)
            self.vao_plume.render()
        for k, v in (("u_vp", gl(vp)), ("u_right", tuple(right)), ("u_up", tuple(up))):
            self._set(self.p_bb, k, v)
            self._set(self.p_sprite, k, v)
        sun = np.array([[*(f.eye + SUN_DIR * 30000.0), 900.0, 1.0, 0.95, 0.85, 1.0]], "f4")
        glow = np.concatenate([sun, np.asarray(f.glow, "f4").reshape(-1, 8)]) if f.glow else sun
        self._draw_billboards(glow, soft=False)
        if f.soft:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._draw_billboards(np.asarray(f.soft, "f4").reshape(-1, 8), soft=True)
        # Earth, low in the sky
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.sprite_inst.write(np.array([[*(f.eye + EARTH_DIR * 30000.0), 540.0, 1, 1, 1, 1]], "f4").tobytes())
        self.tex_earth.use(location=0)
        self._set(self.p_sprite, "u_tex", 0)
        self.vao_sprite.render(moderngl.TRIANGLE_STRIP, instances=1)
        fbo.depth_mask = True

        if self.fbo is not self.fbo_ms:
            ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        data = self.fbo.read(components=3, alignment=1)
        return pygame.image.frombuffer(data, self.size, "RGB")

    def _draw_billboards(self, inst: np.ndarray, soft: bool) -> None:
        inst = inst[:4096]
        self.bb_inst.write(np.ascontiguousarray(inst, "f4").tobytes())
        self._set(self.p_bb, "u_soft", 1 if soft else 0)
        self.vao_bb.render(moderngl.TRIANGLE_STRIP, instances=len(inst))
