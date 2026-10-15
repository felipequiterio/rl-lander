"""Import the Apollo LM from a Bluedog Design Bureau (KSP mod) install into the repo.

    uv run python -m lander.viz3d.import_bdb "/path/to/GameData/Bluedog_DB"
    make import-assets BDB="/path/to/GameData/Bluedog_DB"

Assembles the LM from the KSP parts (see ksp_model.py) and writes a compact bundle
to src/lander/assets/apollo_lm/: the merged meshes (.npz), metadata (.json) and
1024px JPEG textures. The viewer then needs neither KSP nor the .mu parser.

The bundle is a derivative of Bluedog Design Bureau's assets, which are licensed
CC BY-NC-SA 4.0; an ATTRIBUTION.md with the required notices is written alongside.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lander.viz3d.ksp_model import BUNDLE_DIR, assemble_lm, save_bundle

ATTRIBUTION = """# Apollo LM model: attribution and license

The files in this folder are derived from **Bluedog Design Bureau**, a Kerbal Space
Program mod by CobaltWolf (Matthew Mlodzienski) and the BDB team (Zorg, Jso,
Invaderchaos and contributors): https://github.com/CobaltWolf/Bluedog-Design-Bureau

Original work licensed under **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International** (CC BY-NC-SA 4.0): https://creativecommons.org/licenses/by-nc-sa/4.0/

Changes made: the LM parts (ascent stage cockpit, descent stage, descent engine,
landing legs) were assembled at their attach nodes with default variants and legs
in their deployed pose, scaled 1.6x, merged into one mesh per texture and converted
to NumPy arrays; textures were converted from DDS to 1024x1024 JPEG.

These derived files are distributed under the same license, CC BY-NC-SA 4.0: they
may not be used for commercial purposes. This license applies only to the files in
this folder, not to the rest of the repository.
"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bdb", type=Path, help="path to GameData/Bluedog_DB")
    p.add_argument("--out", type=Path, default=BUNDLE_DIR)
    p.add_argument("--texture-size", type=int, default=1024)
    args = p.parse_args(argv)
    model = assemble_lm(args.bdb)
    files = save_bundle(model, args.out, args.texture_size)
    (args.out / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    total = sum(f.stat().st_size for f in args.out.iterdir())
    print(f"wrote {len(files) + 1} files to {args.out} ({total / 1e6:.1f} MB): "
          f"{sum(len(m.indices) for m in model.meshes):,} triangles, {sum(1 for m in model.meshes if m.texture)} textures")


if __name__ == "__main__":
    main()
