"""3D viewer: KSP file readers, model assembly, renderer and scene smoke tests."""

import os
import struct
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from lander.viz3d.ksp_model import parse_cfg
from lander.viz3d.mu import read_mu


def test_parse_cfg_nesting_comments_and_bom():
    text = "﻿PART\n{\n  name = leg // a comment\n  MODULE\n  {\n    name = Switch\n    SUBTYPE { transform = A }\n  }\n}\n"
    part = parse_cfg(text.removeprefix("﻿")).nodes("PART")[0]
    assert part.get("name") == "leg"
    module = part.nodes("MODULE")[0]
    assert module.get("name") == "Switch" and module.nodes("SUBTYPE")[0].all("transform") == ["A"]


def _s(text: str) -> bytes:
    b = text.encode()
    return bytes([len(b)]) + b


def _synthetic_mu() -> bytes:
    """A minimal .mu: root -> child with a one-triangle mesh, a material, a texture,
    and an animation clip that moves the child."""
    p = struct.pack
    out = p("<ii", 76543, 5) + _s("model")
    out += _s("root") + p("<3f", 0, 0, 0) + p("<4f", 0, 0, 0, 1) + p("<3f", 1, 1, 1)
    # animation on the root: clip "deploy" moves child's Unity y (our z) from 2 to 5
    out += p("<i", 2) + p("<i", 1) + _s("deploy") + p("<6f", 0, 0, 0, 1, 1, 1) + p("<i", 0)
    out += p("<i", 1) + _s("child") + _s("m_LocalPosition.y") + p("<i", 0) + p("<2i", 0, 0) + p("<i", 2)
    out += p("<2f2fi", 0.0, 2.0, 0, 0, 0) + p("<2f2fi", 1.0, 5.0, 0, 0, 0)
    out += _s("deploy") + bytes([0])
    out += p("<i", 0)  # child start
    out += _s("child") + p("<3f", 1, 2, 3) + p("<4f", 0, 0, 0, 1) + p("<3f", 1, 1, 1)
    out += p("<i", 7) + p("<i", 13) + p("<2i", 3, 1)
    out += p("<i", 14) + p("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)  # Unity verts: origin, +x, +y(up)
    out += p("<i", 19) + p("<i", 3) + p("<3i", 0, 1, 2)
    out += p("<i", 22)
    out += p("<i", 8) + bytes([1, 1]) + p("<i", 1) + p("<i", 0)
    out += p("<i", 1)  # child end
    out += p("<i", 10) + p("<i", 1) + _s("mat") + _s("KSP/Diffuse") + p("<i", 1) + _s("_MainTex") + p("<i", 4) + p("<i", 0) + p("<4f", 1, 1, 0, 0)
    out += p("<i", 12) + p("<i", 1) + _s("tex.mbm") + p("<i", 0)
    return out


def test_mu_reader_converts_to_z_up(tmp_path):
    f = tmp_path / "m.mu"
    f.write_bytes(_synthetic_mu())
    mu = read_mu(f)
    child = mu.root.find("child")
    np.testing.assert_allclose(child.position, [1, 3, 2])  # Unity (x, y, z) -> (x, z, y)
    np.testing.assert_allclose(child.mesh.positions, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])  # Unity +y is up -> +z
    np.testing.assert_array_equal(child.mesh.submeshes[0], [[0, 2, 1]])  # winding reversed with the handedness
    assert child.materials == [0] and mu.materials[0].textures == {"_MainTex": 0} and mu.textures == ["tex.mbm"]
    assert mu.pose("deploy", at_end=True) == 1
    np.testing.assert_allclose(child.position, [1, 3, 5])  # animated Unity y -> our z


def test_bundled_lm_loads_at_real_scale():
    from lander.viz3d.ksp_model import BUNDLE_DIR, load_bundle

    m = load_bundle()
    assert m is not None, f"no bundled model in {BUNDLE_DIR} (make import-assets BDB=...)"
    assert 5.0 < m.height < 7.5 and 4.2 < m.footprint_radius < 5.5
    assert all((BUNDLE_DIR / f"{x.texture}.jpg").exists() for x in m.meshes if x.texture)
    assert (BUNDLE_DIR / "ATTRIBUTION.md").exists()


def test_lroc_projection_finds_tranquility_base():
    from lander.viz3d.import_lroc import DTM, LM_LAT, LM_LON, ORTHO, pixel

    s, line = pixel(ORTHO, LM_LAT, LM_LON)
    assert abs(s - 6113.7) < 1 and abs(line - 34103.9) < 1  # verified visually: the descent stage is there
    sd, ld = pixel(DTM, LM_LAT, LM_LON)
    assert abs(4 * sd - s) < 2 and abs(4 * ld - line) < 2  # the 2 m DTM and 0.5 m orthophoto line up


def test_apollo11_site_loads():
    from lander.viz3d.terrain import load_site

    site = load_site("apollo11")
    assert site is not None, "run `make import-terrain`"
    meta = site["meta"]
    x0, y0, x1, y1 = meta["near"]["rect"]
    assert x0 < 0 < x1 and y0 < 0 < y1  # the landing target is inside the 0.5 m image
    assert site["near"].shape == (2048, 2048) and site["far"].ndim == 2
    assert np.isclose(np.linalg.norm(site["sun_dir"]), 1.0) and 0 < site["sun_dir"][2] < 0.6  # a low sun
    assert site["sun_dir"][0] < -0.8  # in the west: the photo's shadows point east
    assert load_site("nowhere") is None


BDB = os.environ.get("LANDER_BDB")


@pytest.mark.skipif(not (BDB and Path(BDB).is_dir()), reason="set LANDER_BDB to a KSP Bluedog_DB folder")
def test_bdb_lm_assembles_at_real_scale():
    from lander.viz3d.ksp_model import assemble_lm

    m = assemble_lm(BDB)
    assert 5.0 < m.height < 7.5  # real LM: ~7 m
    assert 4.2 < m.footprint_radius < 5.5  # real LM legs span ~9.4 m
    assert all((m.texture_dir / f"{x.texture}.dds").exists() for x in m.meshes if x.texture)


def _gl_available() -> bool:
    try:
        from lander.viz3d.renderer import create_context

        create_context().release()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_gl = pytest.mark.skipif(not _gl_available(), reason="no OpenGL 3.3 context available")


@needs_gl
def test_renderer_draws_a_frame():
    import pygame

    from lander.viz3d.glmath import model_matrix
    from lander.viz3d.procedural import procedural_lm
    from lander.viz3d.renderer import Frame, Renderer

    pygame.init()
    r = Renderer(procedural_lm(), size=(320, 200))
    surf = r.render(Frame(eye=np.array([20.0, -20.0, 8.0]), look=np.array([0, 0, 3.0]), fov=50,
                          lander_model=model_matrix(np.zeros(3), np.eye(3)), trail=np.array([[50.0, 0, 80], [0, 0, 0]])))
    assert surf.get_size() == (320, 200)
    px = pygame.surfarray.array3d(surf)
    assert px.max() > 60 and px[:, 100:].mean() > 5  # something lit, ground below the horizon


@needs_gl
def test_renderer_draws_real_terrain():
    import pygame

    from lander.viz3d.glmath import model_matrix
    from lander.viz3d.procedural import procedural_lm
    from lander.viz3d.renderer import Frame, Renderer
    from lander.viz3d.terrain import load_site

    pygame.init()
    r = Renderer(procedural_lm(), size=(320, 200), site=load_site("apollo11"))
    surf = r.render(Frame(eye=np.array([0.0, -60.0, 20.0]), look=np.array([0.0, 20.0, 0.0]), fov=50,
                          lander_model=model_matrix(np.array([0.0, 0.0, 30.0]), np.eye(3))))
    ground = pygame.surfarray.array3d(surf)[:, 120:].astype(float)
    assert ground.mean() > 30 and ground.std() > 5  # the photo, not a flat colour


@needs_gl
def test_3d_scene_runs_an_episode(monkeypatch):
    import pygame

    monkeypatch.delenv("LANDER_BDB", raising=False)  # use the built-in model
    pygame.init()
    pygame.display.set_mode((1, 1))
    from lander.envs import get_spec, make_env

    spec = get_spec("descent3d")
    scene = spec.scenes["3d"]()
    env = make_env("descent3d")
    canvas = pygame.Surface(scene.size)
    env.reset(seed=2)
    scene.reset(env)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, 3).astype(np.float32)
        env.step(a)
        for k in range(scene.substeps):
            scene.update(env, a, 0.02, frac=(k + 1) / scene.substeps)
        scene.draw(canvas, env)
    scene.next_camera()
    scene.on_outcome(env, "crashed")
    scene.update(env, None, 0.02)
    scene.draw(canvas, env)
