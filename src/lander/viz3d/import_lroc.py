"""Import real lunar terrain from NASA LROC: imagery for the 3D viewer and a heightmap
for the env's uneven-terrain mode.

    uv run python -m lander.viz3d.import_lroc apollo17
    make import-terrain SITE=apollo17

Sources: Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera digital
terrain models (DTMs) and orthophotos, released to the Planetary Data System:
- apollo11 (Tranquility Base, mare): NAC_DTM_APOLLO11, 2 m/px, and the
  0.5 m/px orthophoto NAC_DTM_APOLLO11_M150361817_50CM;
- apollo17 (Taurus-Littrow valley, highlands): NAC_DTM_APOLLO17, 5 m/px, and the
  orthomosaics NAC_DTM_APOLLO17_MOSAIC_5M and _MOSAIC_120CM.

The products are uncompressed rasters, so only the rows around the landing site are
downloaded (HTTP range requests, a few hundred MB, cached in ~/.cache/lander/lroc).
Written to src/lander/assets/terrain_<site>/:
- near.jpg: high-resolution orthophoto around the historic LM (close-ups);
- far.jpg: lower-resolution orthophoto over the whole imported area;
- height.png: the DTM, 16-bit (z0 + dz * value), used by the env (descent3d's
  ``terrain`` option) and drawn as real 3D relief by the viewer;
- terrain.json: where each raster sits in the site frame (metres, x east, y north,
  origin at the historic LM), the sun direction fitted from the data, the Earth's
  direction, and credits.

The sun direction is estimated by fitting the DTM's slopes to the image brightness,
so the 3D lander is lit (and casts its shadow) consistently with the photo.

LROC data are NASA products in the public domain; credit: NASA/GSFC/Arizona State University.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

PDS = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM"
CACHE = Path.home() / ".cache" / "lander" / "lroc"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
CREDIT = "NASA/GSFC/Arizona State University (LROC). Public domain."

# Raw rasters in an equirectangular projection (centre longitude 180), described by
# their PDS labels. data_offset is 0-based: PDS3 byte pointers count from 1 (the
# Apollo 11 DTM label says 112496); PDS4 offsets are already 0-based.
# res is pixels per degree; line_off / sample_off are the projection offsets.
SITES = {
    "apollo11": {
        "title": "Apollo 11 landing site (Tranquility Base), Mare Tranquillitatis",
        "landmark": "Apollo 11 LM · 20 July 1969",
        "lm": (0.67416, 23.47314),  # LROC position of the descent stage (planetocentric lat, lon)
        # The old flat-ground viewer's target: a little south of the LM, so the historic
        # descent stage stays visible next to the landing zone.
        "default_target": (0.0, -25.0),
        "center_lat": 1.0,
        "dtm": {"dir": "APOLLO11", "file": "NAC_DTM_APOLLO11.TIF", "data_offset": 112495, "samples": 2111, "lines": 13978,
                "dtype": "<f4", "scale": 2.0, "res": 15161.67521207, "line_off": 18747.5, "sample_off": 2374376.5},
        "near": {"dir": "APOLLO11", "file": "NAC_DTM_APOLLO11_M150361817_50CM.IMG", "data_offset": 16880, "samples": 8440,
                 "lines": 55908, "dtype": "<i2", "scale": 0.5, "res": 60646.7008483, "line_off": 74989.5, "sample_off": 9497505.5},
        "far": None,  # derived from the near orthophoto (4x averaged), which covers the DTM strip
        "half_ns_m": 2600.0,  # north-south half-extent; the strips are only ~4 km wide east-west
        "half_ew_m": None,  # None = the DTM strip's full width
        "near_half_m": 512.0,
        "height_step": 2,  # store the 2 m DTM at 4 m
    },
    "apollo17": {
        "title": "Apollo 17 landing site, Taurus-Littrow valley",
        "landmark": "Apollo 17 LM Challenger · 11 Dec 1972",
        "lm": (20.19080, 30.77168),
        "default_target": (0.0, -25.0),
        "center_lat": 20.0,
        "dtm": {"dir": "APOLLO17", "file": "NAC_DTM_APOLLO17.TIF", "data_offset": 93471, "samples": 9985, "lines": 11600,
                "dtype": "<f4", "scale": 5.0, "res": 6064.67008483, "line_off": 129197.5, "sample_off": 855375.5},
        "near": {"dir": "APOLLO17", "file": "NAC_DTM_APOLLO17_MOSAIC_120CM.IMG", "data_offset": 83198, "samples": 41599,
                 "lines": 48330, "dtype": "<i2", "scale": 1.2, "res": 25269.4615201, "line_off": 538322.5, "sample_off": 3564066.5},
        "far": {"dir": "APOLLO17", "file": "NAC_DTM_APOLLO17_MOSAIC_5M.IMG", "data_offset": 19970, "samples": 9985,
                "lines": 11600, "dtype": "<i2", "scale": 5.0, "res": 6064.67008483, "line_off": 129197.5, "sample_off": 855375.5},
        "half_ns_m": 5000.0,  # the valley floor and the foot of the North and South Massifs
        "half_ew_m": 5000.0,
        "near_half_m": 2457.6,  # 4096 px at 1.2 m: sharp ground under every target the env draws here
        "height_step": 1,
    },
}


def pixel(product: dict, lat: float, lon: float, center_lat: float) -> tuple[float, float]:
    """(sample, line), 0-based, of a lat/lon in an equirectangular PDS product."""
    s = product["sample_off"] + (lon - 180.0) * product["res"] * math.cos(math.radians(center_lat))
    line = product["line_off"] - lat * product["res"]
    return s, line


def read_rows(product: dict, r0: int, r1: int) -> np.ndarray:
    """Rows [r0, r1) of a raw fixed-record raster, fetched with an HTTP range request (cached)."""
    r0, r1 = max(r0, 0), min(r1, product["lines"])
    stem = Path(product["file"]).stem
    for f in CACHE.glob(f"{stem}_*_*.npy"):  # any cached download that covers these rows
        m = re.fullmatch(rf"{stem}_(\d+)_(\d+)", f.stem)
        if m and int(m[1]) <= r0 and r1 <= int(m[2]):
            a = int(m[1])
            return np.load(f, mmap_mode="r")[r0 - a : r1 - a].copy()
    cache = CACHE / f"{stem}_{r0}_{r1}.npy"
    row = product["samples"] * np.dtype(product["dtype"]).itemsize
    start = product["data_offset"] + r0 * row
    url = f"{PDS}/{product['dir']}/{product['file']}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + (r1 - r0) * row - 1}"})
    print(f"downloading {product['file']} rows {r0}-{r1} ({(r1 - r0) * row / 1e6:.0f} MB)...", flush=True)
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = resp.read()
    arr = np.frombuffer(data, product["dtype"]).reshape(r1 - r0, product["samples"]).copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def crop(product: dict, site: dict, half_ew: float | None, half_ns: float) -> tuple[np.ndarray, list[float]]:
    """The raster around the site's LM (rows north-first, as stored) and its rect
    [x0, y0, x1, y1] in the site frame (metres from the LM, x east, y north).
    half_ew=None keeps the full width of the strip."""
    s0, l0 = pixel(product, *site["lm"], site["center_lat"])
    k = product["scale"]
    r0, r1 = int(l0 - half_ns / k), int(l0 + half_ns / k)
    c0, c1 = (0, product["samples"]) if half_ew is None else (int(s0 - half_ew / k), int(s0 + half_ew / k))
    rows = read_rows(product, r0, r1)
    r0 = max(r0, 0)  # read_rows clamps at the raster edges
    img = rows[:, max(c0, 0) : c1]
    c0 = max(c0, 0)
    rect = [(c0 - 0.5 - s0) * k, -(r0 + img.shape[0] - 0.5 - l0) * k, (c0 + img.shape[1] - 0.5 - s0) * k, -(r0 - 0.5 - l0) * k]
    return img, rect


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks stay NaN
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def shrink_rect(rect: list[float], shape: tuple[int, int], f: int, k: float) -> list[float]:
    """The rect of block_mean(a, f): trailing rows/columns that didn't fill a block are dropped."""
    h, w = shape[0] // f * f, shape[1] // f * f
    return [rect[0], rect[3] - h * k, rect[0] + w * k, rect[3]]


def fill_gaps(a: np.ndarray) -> np.ndarray:
    """Fill missing DTM cells (strip edges, failed correlation) by interpolating along rows."""
    a = a.copy()
    cols = np.arange(a.shape[1])
    for row in a:
        ok = np.isfinite(row)
        if ok.any() and not ok.all():
            row[~ok] = np.interp(cols[~ok], cols[ok], row[ok])
    return np.where(np.isfinite(a), a, np.nanmedian(a))


def resample(img: np.ndarray, rect: list[float], grid_rect: list[float], grid_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour sample of a north-first raster at the cell centres of another
    grid (NaN where the raster doesn't reach)."""
    gh, gw = grid_shape
    xs = grid_rect[0] + (np.arange(gw) + 0.5) * (grid_rect[2] - grid_rect[0]) / gw
    ys = grid_rect[3] - (np.arange(gh) + 0.5) * (grid_rect[3] - grid_rect[1]) / gh
    h, w = img.shape
    ci = np.floor((xs - rect[0]) / (rect[2] - rect[0]) * w).astype(int)
    ri = np.floor((rect[3] - ys) / (rect[3] - rect[1]) * h).astype(int)
    out = img[np.ix_(np.clip(ri, 0, h - 1), np.clip(ci, 0, w - 1))].astype(np.float32)
    out[(ri < 0) | (ri >= h)] = np.nan
    out[:, (ci < 0) | (ci >= w)] = np.nan
    return out


def fit_sun(dtm: np.ndarray, m_per_px: float, image: np.ndarray) -> tuple[float, float, float]:
    """Sun azimuth/elevation (degrees; azimuth measured from east, counter-clockwise)
    that best explains image brightness as a + b * max(0, n.sun) over the DTM
    (both north-first, on the same grid). Returns (azimuth, elevation, correlation)."""
    dy, dx = np.gradient(dtm, m_per_px)
    dy = -dy  # rows go south; we want +y = north
    n = np.stack([-dx, -dy, np.ones_like(dtm)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ok = np.isfinite(n).all(-1) & np.isfinite(image)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(ok), size=min(200_000, int(ok.sum())), replace=False)
    N, I = n.reshape(-1, 3)[idx], image.reshape(-1)[idx]
    best = (0.0, 0.0, -1.0)
    for el in np.arange(1.0, 60.0, 1.0):
        for az in np.arange(0.0, 360.0, 2.0):
            L = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                          math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
            c = np.corrcoef(np.maximum(N @ L, 0.0), I)[0, 1]
            if c > best[2]:
                best = (float(az), float(el), float(c))
    return best


def earth_az_el(lat: float, lon: float) -> tuple[float, float]:
    """Where the Earth stands in the sky from a near-side site (azimuth from east, CCW;
    elevation), ignoring libration: it hangs over the sub-Earth point (0, 0)."""
    p, q = math.radians(lat), math.radians(lon)
    east, north, up = -math.sin(q), -math.sin(p) * math.cos(q), math.cos(p) * math.cos(q)
    return round(math.degrees(math.atan2(north, east)) % 360, 1), round(math.degrees(math.asin(up)), 1)


def to_u8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    return (np.clip((img - lo) / (hi - lo), 0, 1) ** 0.9 * 255).astype(np.uint8)


def import_site(name: str, out: Path) -> None:
    site = SITES[name]
    # --- heights ---
    dtm, dtm_rect = crop(site["dtm"], site, site["half_ew_m"], site["half_ns_m"])
    dtm = dtm.astype(np.float32)
    dtm[(dtm < -1e4) | (dtm > 1e4)] = np.nan
    print(f"DTM {dtm.shape[1]}x{dtm.shape[0]} px, {np.isnan(dtm).mean():.1%} missing, "
          f"elevation {np.nanmin(dtm):.0f}..{np.nanmax(dtm):.0f} m", flush=True)
    # --- imagery ---
    near_src, near_src_rect = crop(site["near"], site, site["near_half_m"] if site["far"] else site["half_ew_m"],
                                   site["near_half_m"] if site["far"] else site["half_ns_m"])
    near_src = near_src.astype(np.float32)
    near_src[near_src < -32000] = np.nan  # PDS null / saturation codes
    if site["far"] is None:  # far = the near orthophoto 4x averaged; near = its central crop
        k = site["near"]["scale"]
        far = block_mean(near_src, 4)
        far_rect = shrink_rect(near_src_rect, near_src.shape, 4, k)
        n = round(site["near_half_m"] / k)
        s0 = round((0.0 - near_src_rect[0]) / k)
        l0 = round((near_src_rect[3] - 0.0) / k)
        near = near_src[l0 - n : l0 + n, s0 - n : s0 + n]
        near_rect = [(s0 - n) * k + near_src_rect[0], near_src_rect[3] - (l0 + n) * k, (s0 + n) * k + near_src_rect[0], near_src_rect[3] - (l0 - n) * k]
    else:
        far, far_rect = crop(site["far"], site, site["half_ew_m"], site["half_ns_m"])
        far = far.astype(np.float32)
        far[far < -32000] = np.nan
        near, near_rect = near_src, near_src_rect
    lo, hi = np.nanpercentile(far, [0.5, 99.7])

    # Fit on the near image: mosaics stitch photos taken under different suns, and the
    # one around the LM is what the viewer shows close up.
    az, el, corr = fit_sun(dtm, site["dtm"]["scale"], resample(near, near_rect, dtm_rect, dtm.shape))
    print(f"fitted sun: azimuth {az:.0f} deg (from east, CCW), elevation {el:.0f} deg, correlation {corr:.2f}", flush=True)

    # --- write ---
    out.mkdir(parents=True, exist_ok=True)
    step = site["height_step"]
    hm = block_mean(dtm, step)
    hm_rect = shrink_rect(dtm_rect, dtm.shape, step, site["dtm"]["scale"])
    hm = fill_gaps(hm)
    z0 = float(np.floor(hm.min()))
    dz = max(0.1, float(hm.max() - z0) / 65000)
    Image.fromarray(np.round((hm - z0) / dz).astype(np.uint16)).save(out / "height.png", optimize=True)
    Image.fromarray(to_u8(far, lo, hi)).save(out / "far.jpg", quality=90)
    Image.fromarray(to_u8(near, lo, hi)).save(out / "near.jpg", quality=90)
    lm_lat, lm_lon = site["lm"]
    meta = {
        "site": site["title"],
        "frame": "metres; x east, y north; origin = the historic LM; heights = DTM elevation (m, relative to 1737.4 km)",
        "landmark": {"xy": [0.0, 0.0], "label": site["landmark"]},
        "default_target": list(site["default_target"]),
        "far": {"file": "far.jpg", "rect": far_rect, "m_per_px": (far_rect[2] - far_rect[0]) / far.shape[1]},
        "near": {"file": "near.jpg", "rect": near_rect, "m_per_px": (near_rect[2] - near_rect[0]) / near.shape[1]},
        "height": {"file": "height.png", "rect": hm_rect, "m_per_px": site["dtm"]["scale"] * step, "z0": z0, "dz": dz},
        "sun": {"azimuth_deg_from_east_ccw": az, "elevation_deg": el, "fit_correlation": round(corr, 3)},
        "earth_az_el": list(earth_az_el(lm_lat, lm_lon)),
        "mean_brightness": float(np.mean(to_u8(far, lo, hi)) / 255),
        "sources": [f"{PDS}/{site[k]['dir']}/{site[k]['file']}" for k in ("dtm", "near", "far") if site[k]],
        "credit": CREDIT,
    }
    (out / "terrain.json").write_text(json.dumps(meta, indent=1))
    names = ", ".join(Path(s).stem for s in meta["sources"])
    (out / "CREDITS.md").write_text(
        f"# {site['title']}\n\n"
        f"Derived from NASA Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera products\n"
        f"{names}, from the Planetary Data System.\n\n"
        "Credit: NASA/GSFC/Arizona State University. NASA imagery is in the public domain.\n\n"
        "Changes: cropped around the landing site, resampled, contrast-normalised, saved as JPEG (imagery)\n"
        "and 16-bit PNG (terrain heights).\n")
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"wrote {out} ({total / 1e6:.1f} MB): heights {hm.shape[1]}x{hm.shape[0]} @ {meta['height']['m_per_px']:g} m, "
          f"far {far.shape[1]}x{far.shape[0]}, near {near.shape[1]}x{near.shape[0]} px", flush=True)
    print(f"covers x {hm_rect[0]:.0f}..{hm_rect[2]:.0f} m, y {hm_rect[1]:.0f}..{hm_rect[3]:.0f} m around the LM; "
          f"relief {hm.max() - hm.min():.0f} m", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sites", nargs="*", default=list(SITES), choices=sorted(SITES), help="default: all")
    p.add_argument("--out", type=Path, help="output folder (default: src/lander/assets/terrain_<site>)")
    args = p.parse_args(argv)
    for name in args.sites:
        print(f"== {name}", flush=True)
        import_site(name, args.out or ASSETS / f"terrain_{name}")


if __name__ == "__main__":
    main()
