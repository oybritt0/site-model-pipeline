r"""
tile_contact.py -- a labelled contact sheet of every assemblage in a tile.

One cell per composition, captioned with its point mode, its coordinate and the
two periods that hold the ground there, so the sheet can be read against
points.json rather than just looked at.

    python tile_contact.py --assemblages data/gap_space/site_tile/assemblages
    python tile_contact.py --assemblages .../site_tile_flux/assemblages --cols 9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gap_space_core as G

OUT = G.OUT / "site_tile"
SHORT = {"painting_sculpture": "p/s", "drawing_print": "d/p", "architecture": "arch"}


def font(sz):
    for n in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemblages", default=str(OUT / "assemblages"))
    ap.add_argument("--protocol", default="drift")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--vignette", action="store_true",
                    help="use the finished PNG on its grey vignette instead of "
                         "the matte composited onto white")
    a = ap.parse_args()

    src = Path(a.assemblages)
    dirs = sorted(d for d in src.iterdir()
                  if d.is_dir() and (d / f"{a.protocol}.png").exists())
    if not dirs:
        raise SystemExit(f"no {a.protocol}.png under {src}")

    man = {m["id"]: m for m in
           json.load(open(G.CAT / "manifest.json", encoding="utf-8"))}

    cell, cols = a.cell, a.cols
    cap = 34
    rows = (len(dirs) + cols - 1) // cols
    head = 46
    sheet = Image.new("RGB", (cols * cell, head + rows * (cell + cap)),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f_lab, f_ttl = font(13), font(20)

    title = a.title or f"{src.parent.name} — {len(dirs)} assemblages, {a.protocol}"
    d.text((8, 14), title, fill=(0, 0, 0), font=f_ttl)

    for i, p in enumerate(dirs):
        side = json.loads((p / f"{a.protocol}.json").read_text(encoding="utf-8"))
        # the finished PNG sits on a soft grey vignette; the matte is the same
        # composition with the ground removed, so it drops onto white cleanly
        matte = p / f"{a.protocol}_matte.png"
        if matte.exists() and not a.vignette:
            rgba = np.asarray(Image.open(matte).convert("RGBA"), np.float32) / 255.0
            al = rgba[:, :, 3:4]
            rgb = rgba[:, :, :3] * al + (1.0 - al)          # over white
            im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        else:
            im = Image.open(p / f"{a.protocol}.png").convert("RGB")
        im = im.resize((cell, cell), Image.LANCZOS)
        x = (i % cols) * cell
        y = head + (i // cols) * (cell + cap)
        sheet.paste(im, (x, y))

        per = ", ".join(f"{k.split('_', 1)[-1][:11]} {w:.2f}"
                        for k, w in side["periods"][:2])
        cats = {}
        for w in side.get("sources", []):
            c = man.get(w, {}).get("category", "?")
            cats[c] = cats.get(c, 0) + 1
        tot = sum(cats.values()) or 1
        mix = " ".join(f"{SHORT.get(k, k[:4])}{round(100*v/tot):d}"
                       for k, v in sorted(cats.items(), key=lambda t: -t[1]))
        d.text((x + 4, y + cell + 2), f"{i:02d} {p.name.split('__')[0]}",
               fill=(0, 0, 0), font=f_lab)
        d.text((x + 4, y + cell + 16), per, fill=(110, 110, 110), font=f_lab)
        d.text((x + cell - 4, y + cell + 16), mix, fill=(150, 90, 40), font=f_lab,
               anchor="ra")

    out = Path(a.out) if a.out else src.parent / "contact_sheet.png"
    sheet.save(out)
    print(f"{len(dirs)} cells -> {out}  ({sheet.size[0]}x{sheet.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
