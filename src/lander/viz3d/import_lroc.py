"""Import real lunar terrain imagery for the 3D viewer: the Apollo 11 landing site.

    uv run python -m lander.viz3d.import_lroc
    make import-terrain

Source: NASA's Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera
products for the Apollo 11 site, released to the Planetary Data System:
- NAC_DTM_APOLLO11: stereo digital terrain model, 2 m/px;
- NAC_DTM_APOLLO11_M150361817_50CM: orthophoto, 0.5 m/px (2011-01-22).

Both are uncompressed, so only the rows around the landing site are downloaded
(HTTP range requests, ~200 MB, cached in ~/.cache/lander/lroc). Written to
src/lander/assets/terrain_apollo11/:
- near.jpg: 0.5 m/px orthophoto, 1 km square around the target (close-ups);
- far.jpg: 2 m/px orthophoto over the whole downloaded area;
- terrain.json: where each image sits in the env's frame (metres, x east, y north,
  target at the origin), the sun direction fitted from the data, and credits.

The env's ground stays the flat plane z = 0: the real relief is carried by the
imagery, whose shading and shadows are baked in. The sun direction is estimated by
fitting the DTM's slopes to the image brightness, so the 3D lander is lit (and casts
its shadow) consistently with the photo.

LROC data are NASA products in the public domain; credit: NASA/GSFC/Arizona State University.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

BASE = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/APOLLO11"
CACHE = Path.home() / ".cache" / "lander" / "lroc"
OUT = Path(__file__).resolve().parent.parent / "assets" / "terrain_apollo11"

# Apollo 11 LM descent stage (LROC), planetocentric degrees
LM_LAT, LM_LON = 0.67416, 23.47314
# Our landing target sits a little south of the real descent stage, so the historic
# LM stays visible next to the landing zone instead of under the 3D model.
TARGET_OFFSET_M = (0.0, -25.0)  # (east, north)

# Equirectangular projection parameters from the PDS labels (centre lat 1.0, lon 180).
CENTER_LAT, CENTER_LON = 1.0, 180.0
# data_offset is 0-based: PDS byte pointers count from 1 (the DTM label says 112496;
# the orthophoto's data starts at record 2 of 16880 bytes).
DTM = {"file": "NAC_DTM_APOLLO11.TIF", "data_offset": 112495, "samples": 2111, "lines": 13978, "dtype": "<f4",
       "scale": 2.0, "res": 15161.67521207, "line_off": 18747.5, "sample_off": 2374376.5}
ORTHO = {"file": "NAC_DTM_APOLLO11_M150361817_50CM.IMG", "data_offset": 16880, "samples": 8440, "lines": 55908, "dtype": "<i2",
         "scale": 0.5, "res": 60646.7008483, "line_off": 74989.5, "sample_off": 9497505.5}

HALF_NS_M = 2600.0  # north-south half-extent to import (the east-west extent is the strip's full width)
NEAR_HALF_M = 512.0


def pixel(product: dict, lat: float, lon: float) -> tuple[float, float]:
    """(sample, line), 0-based, of a lat/lon in an equirectangular PDS product."""
    s = product["sample_off"] + (lon - CENTER_LON) * product["res"] * math.cos(math.radians(CENTER_LAT))
    line = product["line_off"] - lat * product["res"]
    return s, line


def read_rows(product: dict, r0: int, r1: int) -> np.ndarray:
    """Rows [r0, r1) of a raw fixed-record raster, fetched with an HTTP range request (cached)."""
    r0, r1 = max(r0, 0), min(r1, product["lines"])
    cache = CACHE / f"{Path(product['file']).stem}_{r0}_{r1}.npy"
    if cache.exists():
        return np.load(cache)
    row = product["samples"] * np.dtype(product["dtype"]).itemsize
    start = product["data_offset"] + r0 * row
    req = urllib.request.Request(f"{BASE}/{product['file']}", headers={"Range": f"bytes={start}-{start + (r1 - r0) * row - 1}"})
    print(f"downloading {product['file']} rows {r0}-{r1} ({(r1 - r0) * row / 1e6:.0f} MB)...", flush=True)
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = resp.read()
    arr = np.frombuffer(data, product["dtype"]).reshape(r1 - r0, product["samples"]).copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def normalise(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img - lo) / (hi - lo), 0, 1)


def fit_sun(dtm: np.ndarray, ortho2m: np.ndarray) -> tuple[float, float, float]:
    """Sun azimuth/elevation (degrees; azimuth measured from east, counter-clockwise)
    that best explains image brightness as a + b * max(0, n.sun) over the DTM.
    Returns (azimuth, elevation, correlation)."""
    dy, dx = np.gradient(dtm, 2.0)
    dy = -dy  # rows go south; we want +y = north
    n = np.stack([-dx, -dy, np.ones_like(dtm)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ok = np.isfinite(n).all(-1) & np.isfinite(ortho2m)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(ok), size=min(200_000, int(ok.sum())), replace=False)
    N, I = n.reshape(-1, 3)[idx], ortho2m.reshape(-1)[idx]
    best = (0.0, 0.0, -1.0)
    for el in np.arange(2.0, 45.0, 1.0):
        for az in np.arange(0.0, 360.0, 2.0):
            L = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                          math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
            shade = np.maximum(N @ L, 0.0)
            c = np.corrcoef(shade, I)[0, 1]
            if c > best[2]:
                best = (float(az), float(el), float(c))
    return best


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    # --- orthophoto: rows covering +-HALF_NS_M around the LM, full strip width ---
    os_, ol = pixel(ORTHO, LM_LAT, LM_LON)
    half_rows = int(HALF_NS_M / ORTHO["scale"])
    o0 = int(ol) - half_rows
    ortho = read_rows(ORTHO, o0, int(ol) + half_rows).astype(np.float32)
    ortho[ortho < -32000] = np.nan  # PDS null / saturation codes
    # --- DTM: the matching rows (4x coarser) ---
    ds, dl = pixel(DTM, LM_LAT, LM_LON)
    d0 = int(dl) - half_rows // 4
    dtm = read_rows(DTM, d0, d0 + 2 * (half_rows // 4)).astype(np.float32)
    dtm[(dtm < -1e4) | (dtm > 1e4)] = np.nan

    # Brightness normalisation from the valid pixels.
    lo, hi = np.nanpercentile(ortho, [0.5, 99.7])

    # Env frame: x east, y north, metres, target at the origin.
    tx, ty = TARGET_OFFSET_M
    def to_env(sample: float, line: float, product: dict, first_line: int) -> tuple[float, float]:
        s0, l0 = pixel(product, LM_LAT, LM_LON)
        return (sample - s0) * product["scale"] - tx, -(line + first_line - l0) * product["scale"] - ty

    # far: 2 m/px over the whole downloaded area
    h, w = ortho.shape
    far = np.nanmean(ortho[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4), axis=(1, 3))
    x0, y_top = to_env(-0.5, -0.5, ORTHO, o0)
    x1, y_bot = to_env(w // 4 * 4 - 0.5, h // 4 * 4 - 0.5, ORTHO, o0)
    far_rect = [x0, y_bot, x1, y_top]
    # near: 0.5 m/px, NEAR_HALF_M around the target
    c_s, c_l = os_ + tx / ORTHO["scale"], ol - ty / ORTHO["scale"]
    k = int(NEAR_HALF_M / ORTHO["scale"])
    ns0, nl0 = round(c_s) - k, round(c_l) - o0 - k
    near = ortho[nl0 : nl0 + 2 * k, ns0 : ns0 + 2 * k]
    nx0, ny_top = to_env(ns0 - 0.5, nl0 - 0.5, ORTHO, o0)
    near_rect = [nx0, ny_top - 2 * k * ORTHO["scale"], nx0 + 2 * k * ORTHO["scale"], ny_top]

    # Sun direction from the DTM vs the 2 m image (aligned: the ortho is 4x the DTM grid).
    s_off = round(os_ - 4 * ds)  # ortho sample offset of DTM column 0 (sub-pixel misalignment ignored)
    dh = min(dtm.shape[0], far.shape[0])
    dw = min(dtm.shape[1], far.shape[1] - max(s_off // 4, 0))
    az, el, corr = fit_sun(dtm[:dh, :dw], far[:dh, max(s_off // 4, 0) : max(s_off // 4, 0) + dw])
    print(f"fitted sun: azimuth {az:.0f} deg (from east, CCW), elevation {el:.0f} deg, correlation {corr:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, img in (("far", far), ("near", near)):
        u8 = (normalise(np.nan_to_num(img, nan=np.nanmedian(img)), lo, hi) ** 0.9 * 255).astype(np.uint8)
        Image.fromarray(u8).save(args.out / f"{name}.jpg", quality=90)
    meta = {
        "site": "Apollo 11 landing site (Tranquility Base), Mare Tranquillitatis",
        "frame": "metres; x east, y north; origin = the env's landing target",
        "far": {"file": "far.jpg", "rect": far_rect, "m_per_px": 2.0},
        "near": {"file": "near.jpg", "rect": near_rect, "m_per_px": ORTHO["scale"]},
        "apollo11_lm": [-tx, -ty],
        "sun": {"azimuth_deg_from_east_ccw": az, "elevation_deg": el, "fit_correlation": round(corr, 3)},
        "mean_brightness": float(np.nanmean(normalise(far, lo, hi) ** 0.9)),
        "sources": [f"{BASE}/{DTM['file']}", f"{BASE}/{ORTHO['file']}"],
        "credit": "NASA/GSFC/Arizona State University (LROC). Public domain.",
    }
    (args.out / "terrain.json").write_text(json.dumps(meta, indent=1))
    (args.out / "CREDITS.md").write_text(
        "# Apollo 11 landing site imagery\n\n"
        "Derived from NASA Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera products\n"
        "NAC_DTM_APOLLO11 (terrain model, 2 m/px) and NAC_DTM_APOLLO11_M150361817_50CM (orthophoto,\n"
        "0.5 m/px), from the Planetary Data System.\n\n"
        "Credit: NASA/GSFC/Arizona State University. NASA imagery is in the public domain.\n\n"
        "Changes: cropped around the Apollo 11 landing site, resampled, contrast-normalised, saved as JPEG.\n")
    total = sum(f.stat().st_size for f in args.out.iterdir())
    print(f"wrote {args.out} ({total / 1e6:.1f} MB): far {far.shape[1]}x{far.shape[0]} px, near {near.shape[1]}x{near.shape[0]} px")
    print(f"far covers x {far_rect[0]:.0f}..{far_rect[2]:.0f} m, y {far_rect[1]:.0f}..{far_rect[3]:.0f} m")


if __name__ == "__main__":
    main()
