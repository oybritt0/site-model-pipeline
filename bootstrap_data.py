r"""
bootstrap_data.py -- copy the derived data this pipeline needs out of an
existing checkout of the parent project.

Copies the catalogue manifest, the protocol registry, every 2D layout and feature
matrix, the per-work signals, and the co-segmentation labels. About 118 MB.

Does NOT copy the source images. They are ~4 GB of third-party artwork, they are
not redistributable, and only step 2 needs them -- see DATA.md.

    python bootstrap_data.py --from "C:/path/to/Histories"
    python bootstrap_data.py --from ../Histories --assemblages ../Histories/data/gap_space/site_tile/assemblages
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FILES = [
    "data/catalog/manifest.json",
    "data/embeddings/protocols.json",
    "data/embeddings/index.json",
    "data/embeddings/affordance_vocab.json",
    "data/embeddings/misclosure_stability.json",
    "data/embeddings/projection_stress.json",
    "data/embeddings/verbal_profile.npy",
]
DIRS = ["data/dino_dense/coseg/labels"]


def copy(src: Path, dst: Path, label: str, seen: dict):
    if not src.exists():
        print(f"  MISSING  {label}")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        n = 0
        for f in src.rglob("*"):
            if f.is_file():
                t = dst / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
                n += f.stat().st_size
    else:
        shutil.copy2(src, dst)
        n = src.stat().st_size
    seen[label] = n
    print(f"  {n/1e6:8.1f} MB  {label}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="root of the parent project (the one holding data/catalog)")
    ap.add_argument("--to", default=".")
    ap.add_argument("--assemblages", default=None,
                    help="optional: also copy one tile's assemblages, which makes "
                         "steps 3-5 runnable without the corpus")
    a = ap.parse_args()

    src, dst = Path(a.src).resolve(), Path(a.to).resolve()
    if not (src / "data" / "catalog" / "manifest.json").exists():
        raise SystemExit(f"no data/catalog/manifest.json under {src}")

    total, seen = 0, {}
    for rel in FILES:
        total += copy(src / rel, dst / rel, rel, seen)

    # every layout and feature matrix the registry actually points at
    reg = json.loads((src / "data/embeddings/protocols.json").read_text(encoding="utf-8"))
    for p in reg["protocols"]:
        for key in ("latent_2d_file", "features_file"):
            f = p.get(key)
            if f and (src / "data/embeddings" / f).exists():
                total += copy(src / "data/embeddings" / f,
                              dst / "data/embeddings" / f,
                              f"data/embeddings/{f}", seen)

    for rel in DIRS:
        total += copy(src / rel, dst / rel, rel, seen)

    if a.assemblages:
        ad = Path(a.assemblages).resolve()
        out = dst / "data" / "gap_space" / "handover_assemblages"
        n = copy(ad, out, "one tile's assemblages", seen)
        total += n

    (dst / "data" / "gap_space").mkdir(parents=True, exist_ok=True)
    print(f"\n  {total/1e6:8.1f} MB total")

    missing = [f for f in FILES if f not in seen]
    if missing:
        print("\nnot found, so some steps will fail:")
        for m in missing:
            print("  ", m)

    imgs = json.loads((dst / "data/catalog/manifest.json").read_text(encoding="utf-8"))
    have = sum(1 for m in imgs if m.get("path") and Path(m["path"]).exists())
    print(f"\nsource images resolving on this machine: {have} of {len(imgs)}")
    if have == 0:
        print("  -> tile_assemblages.py will not run. Everything else will. "
              "See DATA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
