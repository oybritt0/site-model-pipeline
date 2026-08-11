r"""
gap_space_core.py — painting an unoccupied coordinate, CPU only.

Engine for the gap-space prototype. No diffusion, no LoRA, no network, no GPU.
Everything here runs off files already on disk (see GAP_SPACE.md section 9).

The move (GAP_SPACE.md 9a): a 2D map position is not a coordinate, it is an
interface. A click is lifted to a real coordinate by inverse-distance weighting
over its k nearest works, and every painter below consumes that same weight
vector. So the three painters are three answers to one question, not three
different questions.

Painters
    ae        decode the corpus's own trained autoencoder at the lifted code.
              Learned, corpus-only, 128px/32-dim (category) or 64px/16-dim
              (period). At that bottleneck it can only carry composition, tone
              and colour mass -- which is what makes it the compositional read
              rather than a picture.
    quilt     Efros-Freeman image quilting from the neighbourhood, patches drawn
              with probability proportional to the lift weights. Novel image,
              made literally out of the material of its influences. Reports seam
              energy, which rises as the point moves away from the data.
    composite the honest baseline: weighted mean of the neighbours. Mush, by
              design. It is here so the other two have something to beat.

Nothing re-embeds anything. That is deliberate and deferred.

CLI (no browser needed):
    python gap_space_core.py --selftest
    python gap_space_core.py --list-gaps --protocol dinov2_base -n 12
    python gap_space_core.py --at 4.2 -1.8 --method quilt --out out.png
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field as _dcfield
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None       # giant scans exist in this corpus


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

def find_root(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / "data" / "catalog" / "manifest.json").exists():
            return c
    return start


ROOT = find_root(Path(__file__).resolve().parent)
E = ROOT / "data" / "embeddings"
CAT = ROOT / "data" / "catalog"
AE_CATEGORY = ROOT / "data" / "autoencoders_category"     # 128px / 32-dim conv AE
AE_VAE = ROOT / "data" / "vae_archetypes"                 # 256px / 256-dim perceptual VAE
AE_PERIOD = ROOT / "data" / "autoencoders"                # 64px / 16-dim, per period
OUT = ROOT / "data" / "gap_space"


def longpath(p) -> str:
    """Windows refuses >260 char paths without the \\?\ prefix. This corpus has them."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def load_rgb(path, max_edge=512) -> Image.Image:
    im = Image.open(longpath(path))
    try:
        im.draft("RGB", (max_edge, max_edge))       # cheap JPEG downscale-on-decode
    except Exception:
        pass
    im = im.convert("RGB")
    if max(im.size) > max_edge:
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return im


# ----------------------------------------------------------------------------
# Field: one protocol's 2D layout plus the per-work signals
# ----------------------------------------------------------------------------

@dataclass
class Field:
    protocol: str
    display_name: str
    ids: list                       # (N,)  work ids, layout order
    xy: np.ndarray                  # (N,2) 2D layout
    period: list
    category: list
    path: list
    stability: np.ndarray           # (N,) cross-protocol agreement, nan where absent
    tear: np.ndarray                # (N,) projection tear (CLIP layout), nan where absent
    tear_is_native: bool            # True only when protocol == the one stress was computed on
    _tree: object = _dcfield(default=None, repr=False)

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.xy)
        return self._tree

    def __len__(self):
        return len(self.ids)


def list_protocols(enabled_only=True) -> list:
    """Protocols from the registry that actually have a 2D layout on disk."""
    reg = json.load(open(E / "protocols.json", encoding="utf-8"))["protocols"]
    out = []
    for p in reg:
        if enabled_only and not p.get("enabled", False):
            continue
        if (E / p.get("latent_2d_file", "___")).exists():
            out.append(p)
    out.sort(key=lambda p: p.get("sort_order", 999))
    return out


_FIELD_CACHE: dict = {}


def load_field(protocol_id: str) -> Field:
    if protocol_id in _FIELD_CACHE:
        return _FIELD_CACHE[protocol_id]

    reg = json.load(open(E / "protocols.json", encoding="utf-8"))["protocols"]
    proto = next((p for p in reg if p["id"] == protocol_id), None)
    if proto is None:
        raise SystemExit(f"protocol '{protocol_id}' not in protocols.json")

    l2 = json.load(open(E / proto["latent_2d_file"], encoding="utf-8"))
    entries = l2["entries"]
    ids = [e["id"] for e in entries]
    xy = np.array([[e["umap_x"], e["umap_y"]] for e in entries], dtype=np.float64)

    manifest = json.load(open(CAT / "manifest.json", encoding="utf-8"))
    by_id = {m["id"]: m for m in manifest}
    period = [by_id.get(i, {}).get("period", "unknown") for i in ids]
    category = [by_id.get(i, {}).get("category", "unknown") for i in ids]
    path = [by_id.get(i, {}).get("path", "") for i in ids]

    stability = np.full(len(ids), np.nan)
    f = E / "misclosure_stability.json"
    if f.exists():
        s = json.load(open(f, encoding="utf-8"))
        d = dict(zip(s["ids"], s["stability"]))
        stability = np.array([d.get(i, np.nan) for i in ids], dtype=np.float64)

    tear = np.full(len(ids), np.nan)
    native = False
    f = E / "projection_stress.json"
    if f.exists():
        s = json.load(open(f, encoding="utf-8"))
        d = dict(zip(s["ids"], s["tear"]))
        tear = np.array([d.get(i, np.nan) for i in ids], dtype=np.float64)
        # projection_stress records which layout it was computed on ("clip")
        native = str(s.get("space", "")).lower() in protocol_id.lower()

    fld = Field(protocol_id, proto.get("display_name", protocol_id), ids, xy,
                period, category, path, stability, tear, native)
    _FIELD_CACHE[protocol_id] = fld
    return fld


# ----------------------------------------------------------------------------
# The lift (GAP_SPACE.md 9a)
# ----------------------------------------------------------------------------

@dataclass
class Lift:
    x: float
    y: float
    idx: np.ndarray                 # (k,) indices into the field
    w: np.ndarray                   # (k,) weights, sum to 1
    d: np.ndarray                   # (k,) 2D distances
    d_nearest: float                # distance to the single nearest work
    step_unit: float                # median nearest-neighbour spacing in this layout
    subset: str = "all"             # which restriction produced these neighbours

    @property
    def steps_out(self) -> float:
        """d* expressed in units of one work's worth of distance."""
        return float(self.d_nearest / self.step_unit) if self.step_unit > 0 else 0.0


_STEP_UNIT: dict = {}


def step_unit(fld: Field) -> float:
    """Median nearest-neighbour distance in the layout. One work's worth of space."""
    if fld.protocol in _STEP_UNIT:
        return _STEP_UNIT[fld.protocol]
    d, _ = fld.tree().query(fld.xy, k=2)
    u = float(np.median(d[:, 1]))
    _STEP_UNIT[fld.protocol] = u
    return u


def lift(fld: Field, px: float, py: float, k: int = 12, power: float = 2.0,
         mask: np.ndarray | None = None) -> Lift:
    """2D position -> weights over k nearest works. `mask` restricts the pool
    (used to keep an autoencoder lift inside one model's own basis)."""
    if mask is None:
        pool = np.arange(len(fld))
        xy = fld.xy
        subset = "all"
    else:
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            raise ValueError("empty neighbour pool after masking")
        xy = fld.xy[pool]
        subset = "masked"

    d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
    kk = int(min(k, len(pool)))
    nn = np.argpartition(d, kk - 1)[:kk] if kk < len(pool) else np.arange(len(pool))
    nn = nn[np.argsort(d[nn])]
    dd = d[nn]

    w = 1.0 / (dd ** power + 1e-9)
    w = w / w.sum()

    # d_nearest is always measured against the whole field, not the masked pool:
    # "how far out am I" must not depend on which decoder was picked.
    dn, _ = fld.tree().query([px, py], k=1)
    return Lift(float(px), float(py), pool[nn], w, dd, float(dn),
                step_unit(fld), subset)


# ----------------------------------------------------------------------------
# Painter 1 — the corpus's own autoencoder
# ----------------------------------------------------------------------------

class ConvAE:
    """Architecture of vibe_autoencoder.py / category_vibe.py. Rebuilt here so
    the tool has no import dependency on the training scripts."""

    def __init__(self, img=128, bottleneck=32):
        import torch.nn as nn
        self.spatial = img // 8
        self.flat = 128 * self.spatial * self.spatial
        self.net = nn.Module()
        self.net.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
        )
        self.net.fc_enc = nn.Linear(self.flat, bottleneck)
        self.net.fc_dec = nn.Linear(bottleneck, self.flat)
        self.net.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def load(self, state_dict):
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()
        return self

    def decode(self, z):
        import torch
        with torch.no_grad():
            h = self.net.fc_dec(z).view(-1, 128, self.spatial, self.spatial)
            return self.net.dec(h)


class PerceptualVAE:
    """Architecture of perceptual_vae.py. 256px, 256-dim, VGG-perceptual, and a
    real (if low-KL) variational prior, so unlike ConvAE its latent is close
    enough to a density that averaging codes stays on the manifold."""

    def __init__(self, img=256, latent=256, decoder_norm=False):
        import torch.nn as nn

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1),
                                 nn.GroupNorm(8, co), nn.LeakyReLU(0.2, True))

        def up(ci, co):
            # decoder_norm reproduces the original collapsed architecture; see
            # perceptual_vae.py VAE._up. Detected from the checkpoint, not guessed.
            layers = [nn.ConvTranspose2d(ci, co, 4, 2, 1)]
            if decoder_norm:
                layers.append(nn.GroupNorm(8, co))
            layers.append(nn.ReLU(True))
            return nn.Sequential(*layers)

        self.spatial = img // 32                  # 5 stride-2 blocks
        self.c = 256
        self.flat = self.c * self.spatial * self.spatial
        self.net = nn.Module()
        self.net.enc = nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                                     block(128, 256), block(256, 256))
        self.net.fc_mu = nn.Linear(self.flat, latent)
        self.net.fc_logvar = nn.Linear(self.flat, latent)
        self.net.fc_dec = nn.Linear(latent, self.flat)
        self.net.dec = nn.Sequential(
            up(256, 256), up(256, 128), up(128, 64), up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def load(self, state_dict):
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()
        return self

    def decode(self, z):
        import torch
        with torch.no_grad():
            h = self.net.fc_dec(z).view(-1, self.c, self.spatial, self.spatial)
            return self.net.dec(h)


def build_decoder(ck: dict, fallback_dim: int | None = None):
    """Construct the right decoder for a checkpoint and load it.

    Single source of truth for which architecture a model.pt holds, so callers
    never have to guess. Returns (model, family).

      perceptual_vae  config has "latent", or the state dict has fc_mu.
                      dec.N.1.* exists only when the upsampling blocks carry
                      GroupNorm, which is the collapsed variant.
      conv_ae         config has "bottleneck", no fc_mu.
    """
    sd = ck["state_dict"]
    cfg = ck.get("config", {})
    if "fc_mu.weight" in sd or "latent" in cfg:
        dim = int(cfg.get("latent", fallback_dim or sd["fc_mu.weight"].shape[0]))
        model = PerceptualVAE(img=int(cfg.get("img", 256)), latent=dim,
                              decoder_norm=("dec.0.1.weight" in sd))
        return model.load(sd), "perceptual_vae", dim
    dim = int(cfg.get("bottleneck", fallback_dim or sd["fc_enc.weight"].shape[0]))
    model = ConvAE(img=int(cfg.get("img", 128)), bottleneck=dim)
    return model.load(sd), "conv_ae", dim


@dataclass
class AEBasis:
    """One trained autoencoder plus the corpus codes that live in its latent."""
    kind: str                       # "category" | "period"
    name: str
    model: object
    codes: np.ndarray               # (M, dim)
    ids: list                       # (M,) work ids aligned to codes
    img: int
    dim: int
    family: str = "conv_ae"         # "conv_ae" | "perceptual_vae"

    @property
    def sigma(self) -> np.ndarray:
        return self.codes.std(axis=0) + 1e-6

    @property
    def centre(self) -> np.ndarray:
        return self.codes.mean(axis=0)


_AE_CACHE: dict = {}


def category_dir(name: str) -> Path:
    """Prefer the perceptual VAE (256px/256-dim) when it has been trained,
    fall back to the original 128px/32-dim conv autoencoder."""
    d = AE_VAE / name
    if (d / "model.pt").exists() and (d / "codes.npy").exists():
        return d
    return AE_CATEGORY / name


def available_bases() -> dict:
    """What decoders exist on disk, as {'category': [...], 'period': [...]}."""
    out = {"category": [], "period": []}
    seen = set()
    for root in (AE_VAE, AE_CATEGORY):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.name in seen or not d.is_dir():
                continue
            if (d / "model.pt").exists() and (d / "codes.npy").exists():
                out["category"].append(d.name)
                seen.add(d.name)
    if AE_PERIOD.is_dir() and (AE_PERIOD / "ae_codes.npy").exists():
        for d in sorted(AE_PERIOD.iterdir()):
            if d.is_dir() and (d / "model.pt").exists():
                out["period"].append(d.name)
    return out


def load_basis(kind: str, name: str) -> AEBasis:
    key = (kind, name)
    if key in _AE_CACHE:
        return _AE_CACHE[key]
    import torch

    model = None
    family = "conv_ae"
    if kind == "category":
        d = category_dir(name)
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        codes = np.load(d / "codes.npy").astype(np.float32)
        ids = json.load(open(d / "index.json", encoding="utf-8"))["ids"]
        model, family, dim = build_decoder(ck, codes.shape[1])
        img = int(ck.get("config", {}).get("img", 128))
    elif kind == "period":
        d = AE_PERIOD / name
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        img = int(cfg.get("img", 64))
        dim = int(cfg.get("bottleneck", 16))
        # ae_codes.npy is global and index-aligned; only this period's rows are valid
        all_codes = np.load(AE_PERIOD / "ae_codes.npy").astype(np.float32)
        ae_idx = json.load(open(AE_PERIOD / "ae_index.json", encoding="utf-8"))
        manifest = json.load(open(CAT / "manifest.json", encoding="utf-8"))
        per = {m["id"]: m.get("period") for m in manifest}
        keep = [i for i, wid in enumerate(ae_idx["ids"]) if per.get(wid) == name]
        codes = all_codes[keep]
        ids = [ae_idx["ids"][i] for i in keep]
    else:
        raise ValueError(f"unknown basis kind {kind!r}")

    if model is None:                      # period bases are always ConvAE
        model, family, dim = build_decoder(ck, dim)
    b = AEBasis(kind, name, model, codes, ids, img, dim, family)
    _AE_CACHE[key] = b
    return b


def pick_basis(fld: Field, lf: Lift, kind: str = "category") -> str:
    """Whichever basis carries the most weight in this neighbourhood."""
    labels = fld.category if kind == "category" else fld.period
    have = set(available_bases()[kind])
    acc: dict = {}
    for i, w in zip(lf.idx, lf.w):
        lab = labels[i]
        if lab in have:
            acc[lab] = acc.get(lab, 0.0) + float(w)
    if not acc:
        raise ValueError(f"no trained {kind} decoder covers this neighbourhood")
    return max(acc.items(), key=lambda kv: kv[1])[0]


def paint_ae(fld: Field, px: float, py: float, *, kind="category", name=None,
             k=6, power=3.0, alpha=0.0, direction="outward", seed=0,
             size=512, renorm=None, stretch=2.0) -> tuple:
    """Decode the lifted code. Returns (PIL.Image, info dict).

    Two things had to be measured before this worked, and both are worth knowing:

    1. These are plain autoencoders, not VAEs. Their latent is not a density, so
       a weighted mean of codes lands off the manifold, *toward the centroid*,
       and the decoder reads a short radius as "the average image". Measured on
       the architecture basis: the mean of 12 distant codes decodes to luminance
       std 0.054 against 0.20 for a single real code -- near-total collapse.
       `renorm` keeps the direction the lift found and restores the radius that
       real codes have there (the weighted mean of the neighbours' own radii).
       That recovers a good part of it, 0.054 -> 0.089 wide, 0.135 -> 0.163 local.
       `renorm=None` applies it only where it is warranted: the perceptual VAE
       from perceptual_vae.py has an actual variational prior, so its latent is
       close enough to a density that averaging stays on the manifold and the
       correction would distort rather than repair. Off there, on for ConvAE.
    2. A tight neighbourhood matters more than the correction does. Hence the
       defaults k=6 / power=3.0 rather than the k=12 / power=2.0 used elsewhere.

    On the 128px/32-dim ConvAE this is a tone and mass field, not a picture --
    32 dimensions and an MSE objective cannot carry more. Read it as the
    compositional skeleton and use `paint_guided` to put material on it. On the
    256px/256-dim perceptual VAE it carries considerably more.

    alpha steps away from the neighbourhood in units of the basis's own
    per-dimension spread, the same line perceptual_vae.py:339 uses for its
    'latent sample (novel work)' column. 0 = the neighbourhood's implied image.
    Stepping outward raises contrast, because it moves away from the mean.
    """
    import torch

    labels = fld.category if kind == "category" else fld.period
    if name in (None, "", "auto"):
        name = pick_basis(fld, lift(fld, px, py, k=k, power=power), kind)

    basis = load_basis(kind, name)
    code_of = {wid: r for r, wid in enumerate(basis.ids)}

    # Restrict the lift to works this decoder actually knows, so the weighted
    # mean happens inside one latent basis. Codes from two different models are
    # not comparable and averaging them would be meaningless.
    mask = np.array([labels[i] == name and fld.ids[i] in code_of
                     for i in range(len(fld))], dtype=bool)
    lf = lift(fld, px, py, k=k, power=power, mask=mask)

    rows = np.array([code_of[fld.ids[i]] for i in lf.idx])
    nb = basis.codes[rows]
    z = (nb * lf.w[:, None]).sum(0).astype(np.float32)

    if renorm is None:
        # On for both families. It was briefly off for perceptual_vae, on the
        # reasoning that a variational prior keeps averages on the manifold --
        # but LOGVAR_MAX=-6 makes that model near-deterministic, so the prior
        # does no such work. Measured on the retrained drawing_print VAE:
        # 0.021 -> 0.038 and 0.035 -> 0.059 with it on.
        renorm = True
    r_raw = float(np.linalg.norm(z - basis.centre))
    r_target = float((np.linalg.norm(nb - basis.centre, axis=1) * lf.w).sum())
    if renorm and r_raw > 1e-9:
        z = (basis.centre + (z - basis.centre) * (r_target / r_raw)).astype(np.float32)

    z_out = z.copy()
    if alpha != 0.0:
        if direction == "outward":
            v = z - basis.centre
            n = np.linalg.norm(v)
            v = v / n if n > 1e-9 else np.ones_like(v) / np.sqrt(len(v))
            z_out = z + alpha * basis.sigma * v
        elif direction == "random":
            rng = np.random.default_rng(seed)
            z_out = z + alpha * basis.sigma * rng.standard_normal(basis.dim).astype(np.float32)
        elif direction == "inward":
            v = basis.centre - z
            n = np.linalg.norm(v)
            v = v / n if n > 1e-9 else np.zeros_like(v)
            z_out = z + alpha * basis.sigma * v

    t0 = time.time()
    arr = basis.model.decode(torch.tensor(z_out[None], dtype=torch.float32))
    arr = np.clip(arr[0].numpy().transpose(1, 2, 0), 0, 1)
    lum_raw = float((arr @ np.array([0.2126, 0.7152, 0.0722])).std())

    # A 32-dim MSE decode uses only part of the range -- at a typical point it
    # spans about 117..251 of 0..255, which reads as fog even though the layout
    # is in there. Rescaling the percentile range makes what was decoded legible
    # without inventing anything; it is a display transform, recorded as one.
    # paint_guided turns this off, since the guide is consumed numerically.
    if stretch and stretch > 0:
        lo, hi = np.percentile(arr, stretch), np.percentile(arr, 100.0 - stretch)
        if hi - lo > 1e-6:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    if size and size != basis.img:
        img = img.resize((size, size), Image.LANCZOS)

    lum = float((arr @ np.array([0.2126, 0.7152, 0.0722])).std())
    info = {
        "method": "ae",
        "basis_kind": kind, "basis": name, "basis_family": basis.family,
        "native_px": basis.img, "latent_dim": basis.dim,
        "k": int(len(lf.idx)), "power": power, "renorm": bool(renorm),
        "alpha": alpha, "direction": direction if alpha else "none",
        "d_nearest": lf.d_nearest, "steps_out": lf.steps_out,
        "step_unit": lf.step_unit,
        "radius_raw": round(r_raw, 3), "radius_restored": round(r_target, 3),
        "z_shift": round(float(np.linalg.norm(z_out - z)), 3),
        # structure retained. ~0.20 is a single real code; below ~0.07 is collapse.
        # luminance_std_raw is the honest measure of the decode; luminance_std is
        # after the display stretch.
        "luminance_std": round(lum, 4),
        "luminance_std_raw": round(lum_raw, 4),
        "stretch_pct": float(stretch or 0.0),
        "seconds": round(time.time() - t0, 3),
        "neighbours": _neighbour_rows(fld, lf),
    }
    return img, info


# ----------------------------------------------------------------------------
# Painter 2 — quilting the point out of its own neighbourhood
# ----------------------------------------------------------------------------

def _mincut_mask(err: np.ndarray) -> np.ndarray:
    """Efros-Freeman minimum-error boundary cut down a vertical overlap strip.

    err is (h, ov). Returns a bool mask, True where the NEW patch wins.
    Column-wise DP with the inner loop vectorised.
    """
    h, ov = err.shape
    if ov < 2:
        return np.ones_like(err, dtype=bool)
    cum = err.astype(np.float64).copy()
    back = np.zeros((h, ov), dtype=np.int32)
    ar = np.arange(ov)
    for i in range(1, h):
        prev = cum[i - 1]
        left = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        stack = np.stack([left, prev, right])          # (3, ov)
        choice = np.argmin(stack, axis=0)              # 0=j-1, 1=j, 2=j+1
        cum[i] += np.min(stack, axis=0)
        back[i] = np.clip(ar + (choice - 1), 0, ov - 1)

    mask = np.zeros((h, ov), dtype=bool)
    j = int(np.argmin(cum[-1]))
    for i in range(h - 1, -1, -1):
        mask[i, :j] = False        # keep what is already on the canvas
        mask[i, j:] = True         # take the new patch
        j = int(back[i, j])
    return mask


def _neighbour_rows(fld: Field, lf: Lift) -> list:
    return [{"id": fld.ids[i], "period": fld.period[i], "category": fld.category[i],
             "w": round(float(w), 4), "d": round(float(d), 4)}
            for i, w, d in zip(lf.idx, lf.w, lf.d)]


def paint_quilt(fld: Field, px: float, py: float, *, k=10, power=2.0,
                size=512, patch=112, overlap=None, pool_per_source=48,
                tolerance=0.12, seed=0, source_edge=384,
                guide=None, guide_weight=0.0, guide_grid=4) -> tuple:
    """Image quilting from the weighted neighbourhood.

    Patches are drawn from the k nearest works with probability proportional to
    the lift weights, so the picture is made out of the material of its own
    influences. Mean seam energy is reported: it rises as the requested point
    moves away from the data, because the patches being forced together come
    from works with less in common.

    `guide` is an optional (H,W) luminance field in [0,1]. When given, patch
    selection also matches the guide's tonal arrangement inside each tile at
    `guide_grid` x `guide_grid` resolution. That is what `paint_guided` uses to
    put the autoencoder's composition under the quilt's material.
    """
    rng = np.random.default_rng(seed)
    lf = lift(fld, px, py, k=k, power=power)
    t0 = time.time()

    if overlap is None:
        overlap = max(8, patch // 5)
    if guide is not None and patch % guide_grid:
        patch -= patch % guide_grid                   # keep the block mean exact
        overlap = max(8, patch // 5)
    step = patch - overlap

    # ---- patch pool, sampled per source in proportion to its weight ----------
    pool, pool_src, loaded, failed = [], [], [], []
    total = max(1, int(round(pool_per_source * len(lf.idx))))
    quota = np.maximum(1, np.round(lf.w * total).astype(int))
    for slot, (i, q) in enumerate(zip(lf.idx, quota)):
        p = fld.path[i]
        if not p:
            failed.append(fld.ids[i]); continue
        try:
            im = load_rgb(p, max_edge=source_edge)
        except Exception:
            failed.append(fld.ids[i]); continue
        a = np.asarray(im, dtype=np.float32) / 255.0
        if a.shape[0] < patch or a.shape[1] < patch:
            sc = patch / min(a.shape[0], a.shape[1])
            im = im.resize((max(patch, int(im.width * sc) + 1),
                            max(patch, int(im.height * sc) + 1)), Image.LANCZOS)
            a = np.asarray(im, dtype=np.float32) / 255.0
        loaded.append(fld.ids[i])
        H, W = a.shape[:2]
        for _ in range(int(q)):
            r = rng.integers(0, H - patch + 1)
            c = rng.integers(0, W - patch + 1)
            pool.append(a[r:r + patch, c:c + patch])
            pool_src.append(slot)
    if not pool:
        raise RuntimeError("no source images could be loaded for this neighbourhood")

    P = np.stack(pool)                                  # (n, patch, patch, 3)
    src = np.array(pool_src)
    src_w = lf.w[src]
    src_w = src_w / src_w.sum()

    # ---- raster quilt --------------------------------------------------------
    n_tiles = max(1, int(np.ceil((size - overlap) / step)))
    canvas_px = overlap + n_tiles * step
    canvas = np.zeros((canvas_px, canvas_px, 3), dtype=np.float32)
    filled = np.zeros((canvas_px, canvas_px), dtype=bool)
    seam_costs = []

    # ---- guide: block-mean luminance of every pool patch, once ---------------
    G = None
    if guide is not None and guide_weight > 0:
        g = np.asarray(guide, dtype=np.float32)
        if g.ndim == 3:
            g = g @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        if g.shape != (canvas_px, canvas_px):
            g = np.asarray(Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
                           .resize((canvas_px, canvas_px), Image.LANCZOS),
                           dtype=np.float32) / 255.0
        G = g
        b = patch // guide_grid
        plum = P @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)   # (n,p,p)
        P_g = plum.reshape(len(P), guide_grid, b, guide_grid, b).mean(axis=(2, 4))
        P_g = P_g.reshape(len(P), -1)                                     # (n, grid^2)

    for ty in range(n_tiles):
        for tx in range(n_tiles):
            y0, x0 = ty * step, tx * step
            has_left, has_top = tx > 0, ty > 0

            err_ov = np.zeros(len(P), dtype=np.float64)
            if has_left:
                ref = canvas[y0:y0 + patch, x0:x0 + overlap]
                err_ov += ((P[:, :, :overlap] - ref) ** 2).sum(axis=(1, 2, 3))
            if has_top:
                ref = canvas[y0:y0 + overlap, x0:x0 + patch]
                err_ov += ((P[:, :overlap, :] - ref) ** 2).sum(axis=(1, 2, 3))

            npix = (patch * overlap * 3) * (int(has_left) + int(has_top))
            if has_left or has_top:
                seam_costs.append(float(err_ov.min()) / max(1, npix))

            # combine the seam term and the guide term after normalising each by
            # its own median, so guide_weight is scale-free
            err = np.zeros(len(P), dtype=np.float64)
            if has_left or has_top:
                m = np.median(err_ov)
                err += err_ov / m if m > 1e-12 else err_ov
            if G is not None:
                b = patch // guide_grid
                gt = G[y0:y0 + patch, x0:x0 + patch]
                gt = gt.reshape(guide_grid, b, guide_grid, b).mean(axis=(1, 3)).ravel()
                err_g = ((P_g - gt) ** 2).sum(axis=1)
                m = np.median(err_g)
                err += guide_weight * (err_g / m if m > 1e-12 else err_g)

            if not err.any():
                pick = int(rng.choice(len(P), p=src_w))
            else:
                lo = float(err.min())
                # Efros-Freeman: choose randomly among patches within tolerance
                # of the best, biased by how much the source is weighted here.
                ok = np.flatnonzero(err <= lo * (1.0 + tolerance) + 1e-9)
                w_ok = src_w[ok] / src_w[ok].sum()
                pick = int(rng.choice(ok, p=w_ok))

            tile = P[pick]
            take = np.ones((patch, patch), dtype=bool)
            if has_left:
                e = ((tile[:, :overlap] - canvas[y0:y0 + patch, x0:x0 + overlap]) ** 2).sum(-1)
                take[:, :overlap] &= _mincut_mask(e)
            if has_top:
                e = ((tile[:overlap, :] - canvas[y0:y0 + overlap, x0:x0 + patch]) ** 2).sum(-1)
                take[:overlap, :] &= _mincut_mask(e.T).T
            take |= ~filled[y0:y0 + patch, x0:x0 + patch]

            region = canvas[y0:y0 + patch, x0:x0 + patch]
            region[take] = tile[take]
            filled[y0:y0 + patch, x0:x0 + patch] = True

    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    if canvas_px != size:
        img = img.resize((size, size), Image.LANCZOS)

    info = {
        "method": "quilt",
        "k": int(len(lf.idx)), "power": power,
        "patch": patch, "overlap": overlap, "pool": int(len(P)),
        "guide_weight": float(guide_weight) if G is not None else 0.0,
        "sources_loaded": len(loaded), "sources_failed": failed,
        # rises as the point moves away from the data: the patches being forced
        # together come from works with less in common. An escape metric that
        # needs no re-encoding step.
        "seam_energy": round(float(np.mean(seam_costs)), 6) if seam_costs else 0.0,
        "seam_energy_p90": round(float(np.percentile(seam_costs, 90)), 6) if seam_costs else 0.0,
        "d_nearest": lf.d_nearest, "steps_out": lf.steps_out,
        "step_unit": lf.step_unit,
        "seconds": round(time.time() - t0, 3),
        "neighbours": _neighbour_rows(fld, lf),
    }
    return img, info


def paint_guided(fld: Field, px: float, py: float, *, k=10, power=2.0, size=512,
                 patch=96, guide_weight=1.5, ae_kind="category", ae_name=None,
                 ae_k=6, ae_power=3.0, alpha=0.0, direction="outward",
                 seed=0, **quilt_kw) -> tuple:
    """Composition from the corpus's own latent, material from its own patches.

    The autoencoder decode is too low-frequency to be a picture but it is a
    real reading of where mass, void and tone sit at this coordinate. The quilt
    has material and no composition -- it tiles whatever matches at the seams.
    Using the first as a tonal guide for the second gives each what the other
    lacks, and both halves come only from this corpus. No outside generator
    touches it.

    guide_weight trades the two off: 0 is a plain quilt, 3+ follows the
    autoencoder's tonal arrangement closely at the cost of seam quality.
    """
    g_img, g_info = paint_ae(fld, px, py, kind=ae_kind, name=ae_name, k=ae_k,
                             power=ae_power, alpha=alpha, direction=direction,
                             seed=seed, size=size, stretch=0.0)
    guide = np.asarray(g_img.convert("L"), dtype=np.float32) / 255.0
    img, info = paint_quilt(fld, px, py, k=k, power=power, size=size,
                            patch=patch, seed=seed, guide=guide,
                            guide_weight=guide_weight, **quilt_kw)
    info["method"] = "guided"
    info["guide"] = {kk: g_info[kk] for kk in
                     ("basis_kind", "basis", "basis_family", "native_px",
                      "latent_dim", "luminance_std", "alpha", "direction",
                      "renorm")}
    return img, info


# ----------------------------------------------------------------------------
# Painter 3 — the baseline
# ----------------------------------------------------------------------------

def paint_composite(fld: Field, px: float, py: float, *, k=12, power=2.0,
                    size=512, tone=False, source_edge=384) -> tuple:
    """Weighted mean of the neighbours. Mush by construction -- it is the control
    the other two painters have to beat. `tone` averages luminance only, which
    is the most legible reading of where mass and void sit at this coordinate."""
    lf = lift(fld, px, py, k=k, power=power)
    t0 = time.time()
    acc = np.zeros((size, size, 3), dtype=np.float64)
    wsum, loaded, failed = 0.0, [], []
    for i, w in zip(lf.idx, lf.w):
        p = fld.path[i]
        if not p:
            failed.append(fld.ids[i]); continue
        try:
            im = load_rgb(p, max_edge=source_edge).resize((size, size), Image.LANCZOS)
        except Exception:
            failed.append(fld.ids[i]); continue
        a = np.asarray(im, dtype=np.float64) / 255.0
        if tone:
            g = a @ np.array([0.2126, 0.7152, 0.0722])
            a = np.repeat(g[:, :, None], 3, axis=2)
        acc += w * a
        wsum += float(w)
        loaded.append(fld.ids[i])
    if wsum <= 0:
        raise RuntimeError("no source images could be loaded for this neighbourhood")
    acc /= wsum
    img = Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))

    info = {
        "method": "composite", "tone": bool(tone),
        "k": int(len(lf.idx)), "power": power,
        "sources_loaded": len(loaded), "sources_failed": failed,
        "d_nearest": lf.d_nearest, "steps_out": lf.steps_out,
        "step_unit": lf.step_unit,
        "seconds": round(time.time() - t0, 3),
        "neighbours": _neighbour_rows(fld, lf),
    }
    return img, info


# ----------------------------------------------------------------------------
# Painter 4 — assemblage: cut real forms out, place them by where they belong
# ----------------------------------------------------------------------------
# Quilting can only ever look quilted: it moves square tiles, so the output
# reads as stitched rectangles no matter how good the seams are. This painter
# uses the co-segmentation already on disk (data/dino_dense/coseg, 4449 part
# maps on a 52x52 grid, 24 parts with 8 dropped) to cut semantically coherent
# forms away from their backgrounds and recompose them.
#
# The placement logic, which is the part that decides whether the result means
# anything:
#   position  each form keeps the normalised centroid it had in its source, so
#             a pediment stays high and a plinth stays low. The composition
#             inherits the compositional grammar of its neighbourhood rather
#             than scattering parts at random.
#   scale     driven by the source work's lift weight. How strongly a work
#             claims this coordinate becomes how large it is in the picture, so
#             influence is legible as size.
#   rotation  the form's own principal axis in its source, turned toward a
#             single shared axis by `coherence` -- 0 keeps every form's original
#             orientation, 1 aligns them all.
#   order     largest first, so big forms sit behind small ones.

COSEG = ROOT / "data" / "dino_dense" / "coseg"
_COSEG_META: dict = {}


def coseg_parts() -> dict:
    """Kept part ids and the period x part histogram, from the coseg metadata."""
    if _COSEG_META:
        return _COSEG_META
    m = json.load(open(COSEG / "period_part_matrix.json", encoding="utf-8"))
    dropped = set(m.get("dropped", []))
    _COSEG_META.update(
        k=int(m["k"]),
        kept=[i for i in range(int(m["k"])) if i not in dropped],
        periods=m["periods"],
        hist=np.array(m["hist"], dtype=float),
    )
    return _COSEG_META


def _load_square(path: str, s: int) -> np.ndarray:
    """dino_dense.py resizes to a square with no crop before gridding, so the
    label map only lines up with the image if it is squashed the same way."""
    im = load_rgb(path, max_edge=max(s, 512)).resize((s, s), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def _labels_square(work_id: str, s: int):
    f = COSEG / "labels" / f"{work_id}.npy"
    if not f.exists():
        return None
    lab = np.load(f)
    return np.asarray(Image.fromarray(lab.astype(np.int16), "I;16")
                      .resize((s, s), Image.NEAREST), dtype=np.int32)


def _forms_from_work(work_id: str, path: str, weight: float, s: int,
                     min_frac: float, max_frac: float, max_per_work: int,
                     part_w: np.ndarray | None, rng,
                     min_contrast: float = 0.055,
                     edge: str = "soft", n_superpixels: int = 900) -> list:
    """Cut every reasonably sized connected part out of one work."""
    from scipy import ndimage as ndi

    f = COSEG / "labels" / f"{work_id}.npy"
    if not f.exists():
        return []
    lab_native = np.load(f).astype(np.int32)
    try:
        img = _load_square(path, s)
    except Exception:
        return []

    sp = sp_ids = None
    if edge == "hard":
        from skimage.segmentation import slic
        # one segmentation per work, shared by all its parts
        sp = slic(img, n_segments=n_superpixels, compactness=9.0,
                  sigma=1.0, start_label=0, channel_axis=2)
        sp_ids = np.unique(sp)

    kept = coseg_parts()["kept"]
    g = lab_native.shape[0]                 # 52; work at grid resolution
    total = float(g * g)
    out = []
    for p in kept:
        m = lab_native == p
        if not m.any():
            continue
        # Morphology belongs at grid resolution, where a 2x2 element means one
        # patch. Doing it after upsampling just blurs stair-steps around.
        m = ndi.binary_closing(m, structure=np.ones((2, 2)))
        m = ndi.binary_opening(m, structure=np.ones((2, 2)))
        cc, n = ndi.label(m)
        if n == 0:
            continue
        for ci in range(1, n + 1):
            comp = cc == ci
            frac = float(comp.sum()) / total
            if frac < min_frac or frac > max_frac:
                continue
            ys, xs = np.nonzero(comp)
            # principal axis and elongation from the mask's second moments
            pts = np.stack([xs.astype(float), ys.astype(float)])
            cov = np.cov(pts) if pts.shape[1] > 2 else np.eye(2)
            ev, evec = np.linalg.eigh(cov)
            angle = float(np.degrees(np.arctan2(evec[1, -1], evec[0, -1])))
            elong = float(np.sqrt(max(ev[-1], 1e-9) / max(ev[0], 1e-9)))
            cxg, cyg = float(xs.mean() / g), float(ys.mean() / g)

            if edge == "hard":
                # Snap the coarse 52x52 mask to the image's own edges: SLIC
                # superpixels follow real gradients, so voting whole superpixels
                # in or out gives a boundary that lands on actual contours and
                # stays hard. Thresholding the upsampled grid instead would just
                # give a 52x52 staircase.
                up = np.asarray(
                    Image.fromarray((comp.astype(np.float32) * 255).astype(np.uint8), "L")
                    .resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
                cov = ndi.mean(up, labels=sp, index=sp_ids)
                keep = np.zeros(int(sp_ids.max()) + 1, dtype=bool)
                keep[sp_ids] = cov >= 0.45
                hard = keep[sp]
                hard = ndi.binary_closing(hard, structure=np.ones((3, 3)))
                hard = ndi.binary_fill_holes(hard)
                cc2, n2 = ndi.label(hard)
                if n2 > 1:                      # keep only the main body
                    sizes = ndi.sum(hard, cc2, range(1, n2 + 1))
                    hard = cc2 == (int(np.argmax(sizes)) + 1)
                # sigma 0.6 antialiases the staircase without softening the edge
                soft = ndi.gaussian_filter(hard.astype(np.float32), sigma=0.6)
            else:
                # Upsample the component mask BILINEARLY and keep the soft values
                # as alpha. Thresholding, or NEAREST, is what makes cut-outs look
                # like 52x52 lego; the smooth ramp gives a real edge for free.
                soft = np.asarray(
                    Image.fromarray((comp.astype(np.float32) * 255).astype(np.uint8), "L")
                    .resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
                soft = ndi.gaussian_filter(soft, sigma=max(1.0, s / (g * 1.6)))
                soft = np.clip((soft - 0.30) / 0.45, 0.0, 1.0)   # tighten the ramp
            ay, ax = np.nonzero(soft > 0.02)
            if len(ay) < 16:
                continue
            y0, y1, x0, x1 = ay.min(), ay.max() + 1, ax.min(), ax.max() + 1
            alpha = soft[y0:y1, x0:x1]
            crop = img[y0:y1, x0:x1]
            # A region of blank paper or flat sky is a part, but it is not a
            # form: scaled up it is just a pale blob. Score contrast inside the
            # mask and drop the featureless ones.
            inside = alpha > 0.5
            if inside.sum() < 24:
                continue
            g_ = crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            contrast = float(g_[inside].std())
            if contrast < min_contrast:
                continue
            rgba = np.dstack([crop, alpha])
            pw = 1.0 if part_w is None else float(part_w[p])
            out.append({
                "rgba": rgba, "part": int(p), "w": weight * pw,
                "cx": cxg, "cy": cyg,
                "area_frac": frac, "angle": angle, "elong": elong,
                "contrast": contrast,
                "id": work_id,
            })
    if len(out) > max_per_work:
        pr = np.array([o["area_frac"] * (0.25 + o["contrast"]) for o in out], dtype=float)
        pr = pr / pr.sum()
        sel = rng.choice(len(out), size=max_per_work, replace=False, p=pr)
        out = [out[i] for i in sel]
    return out


def paint_assemblage(fld: Field, px: float, py: float, *, k=10, power=2.0,
                     size=512, n_forms=20, scale=1.7, influence=0.7,
                     coherence=0.0, spin=0.0, jitter=0.07, min_frac=0.012,
                     max_frac=0.45, max_per_work=10, period_bias=True, min_contrast=0.055,
                     edge="soft", category=None,
                     background="tone", seed=0, source_edge=768) -> tuple:
    """Recompose the neighbourhood out of cut-out forms rather than square tiles.

    `category` restricts the neighbourhood to one medium, so the same coordinate
    can be answered by painting, by architecture, or by drawing alone. Note that
    d_nearest is still measured against the whole field: how far out the point
    is should not change with which medium is allowed to speak for it.
    """
    rng = np.random.default_rng(seed)
    mask = None
    if category:
        mask = np.array([c == category for c in fld.category], dtype=bool)
        if not mask.any():
            raise ValueError(f"no works in category {category!r}")
    lf = lift(fld, px, py, k=k, power=power, mask=mask)
    t0 = time.time()
    meta = coseg_parts()

    # bias part choice toward parts characteristic of this neighbourhood's periods
    part_w = None
    if period_bias:
        acc = np.zeros(meta["k"], dtype=float)
        pidx = {p: i for i, p in enumerate(meta["periods"])}
        for i, w in zip(lf.idx, lf.w):
            r = pidx.get(fld.period[i])
            if r is not None:
                acc += w * meta["hist"][r]
        if acc.sum() > 0:
            acc = acc / acc.max()
            part_w = 0.35 + 0.65 * acc          # never fully exclude a part

    pool, failed = [], []
    for i, w in zip(lf.idx, lf.w):
        if not fld.path[i]:
            continue
        try:
            fs = _forms_from_work(fld.ids[i], fld.path[i], float(w), size,
                                  min_frac, max_frac, max_per_work, part_w, rng,
                                  min_contrast, edge)
            pool.extend(fs)
            if not fs:
                failed.append(fld.ids[i])
        except Exception:
            failed.append(fld.ids[i])
    if not pool:
        raise RuntimeError("no forms could be cut from this neighbourhood; "
                           "try a larger k or a smaller minimum form size")

    # ---- background ---------------------------------------------------------
    if background == "none":
        canvas = np.ones((size, size, 3), dtype=np.float32)
    else:
        base, wsum = np.zeros((size, size, 3), dtype=np.float32), 0.0
        for i, w in zip(lf.idx, lf.w):
            if not fld.path[i]:
                continue
            try:
                a = _load_square(fld.path[i], size)
            except Exception:
                continue
            base += float(w) * a
            wsum += float(w)
        canvas = base / max(wsum, 1e-6)
        from scipy import ndimage as ndi
        canvas = ndi.gaussian_filter(canvas, sigma=(size / 24.0, size / 24.0, 0))
        if background == "tone":
            g = canvas @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            canvas = np.repeat(g[:, :, None], 3, axis=2)
        canvas = 0.55 + 0.45 * canvas          # lift it so forms read against it

    # ---- choose and place ---------------------------------------------------
    pr = np.array([f["w"] for f in pool], dtype=float)
    pr = pr / pr.sum()
    n = int(min(n_forms, len(pool)))
    chosen = [pool[i] for i in rng.choice(len(pool), size=n, replace=False, p=pr)]
    chosen.sort(key=lambda f: -f["area_frac"])          # big behind, small in front

    wmean = float(np.mean([f["w"] for f in chosen])) or 1.0
    placed = 0
    for f in chosen:
        h0, w0 = f["rgba"].shape[:2]
        if h0 < 2 or w0 < 2:
            continue
        # scale: influence -> size, on top of the form's own footprint
        gain = (f["w"] / wmean) ** influence
        target = f["area_frac"] * (size * size) * (scale ** 2) * gain
        cur = max(1.0, float((f["rgba"][:, :, 3] > 0.3).sum()))
        sc = float(np.clip(np.sqrt(target / cur), 0.15, 3.0))
        if max(h0, w0) * sc > 1.5 * size:      # no single form dominates
            sc = 1.5 * size / max(h0, w0)
        nh, nw = max(2, int(h0 * sc)), max(2, int(w0 * sc))
        if nh > 4 * size or nw > 4 * size:
            continue

        tile = Image.fromarray((np.clip(f["rgba"], 0, 1) * 255).astype(np.uint8),
                               "RGBA").resize((nw, nh), Image.LANCZOS)
        ang = f["angle"] * (1.0 - coherence)
        if spin:
            ang += float(rng.uniform(-spin, spin))
        if abs(ang) > 0.5:
            tile = tile.rotate(ang, expand=True, resample=Image.BICUBIC)

        ta = np.asarray(tile, dtype=np.float32) / 255.0
        th, tw = ta.shape[:2]
        cx = f["cx"] + float(rng.normal(0, jitter))
        cy = f["cy"] + float(rng.normal(0, jitter))
        x0 = int(round(cx * size - tw / 2))
        y0 = int(round(cy * size - th / 2))

        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(size, x0 + tw), min(size, y0 + th)
        if xs1 <= xs0 or ys1 <= ys0:
            continue
        sub = ta[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        a = sub[:, :, 3:4]
        canvas[ys0:ys1, xs0:xs1] = sub[:, :, :3] * a + canvas[ys0:ys1, xs0:xs1] * (1 - a)
        placed += 1

    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    parts_used = sorted({f["part"] for f in chosen})
    info = {
        "method": "assemblage",
        "k": int(len(lf.idx)), "power": power,
        "forms_available": len(pool), "forms_placed": placed,
        "parts_used": parts_used, "sources_failed": failed,
        "scale": scale, "influence": influence, "coherence": coherence,
        "spin": spin, "jitter": jitter, "background": background,
        "period_bias": bool(period_bias), "min_contrast": min_contrast,
        "edge": edge, "category": category or "all",
        "d_nearest": lf.d_nearest, "steps_out": lf.steps_out,
        "step_unit": lf.step_unit,
        "seconds": round(time.time() - t0, 3),
        "neighbours": _neighbour_rows(fld, lf),
    }
    return img, info


PAINTERS = {"ae": paint_ae, "quilt": paint_quilt, "guided": paint_guided,
            "assemblage": paint_assemblage, "composite": paint_composite}


# ----------------------------------------------------------------------------
# Gap finding (GAP_SPACE.md 9b)
# ----------------------------------------------------------------------------

SORTS = {
    "empty": "furthest from any existing work",
    "mixed": "most period-mixed neighbourhood (interstitial, between clusters)",
    "contested": "protocols agree least about the surrounding works",
    "settled": "protocols agree most about the surrounding works",
}


def gap_candidates(fld: Field, n=24, samples=6000, seed=0, min_steps=1.0,
                   max_tear=None, inside_hull=True, k_ctx=12,
                   sort="empty", min_separation=None) -> list:
    """Sample the layout, keep points that are genuinely empty, and report the
    context signals for each so a candidate can be judged before it is painted.

    min_steps is in units of one work's worth of distance (the median nearest-
    neighbour spacing), so "one step outside" is literal. Note that UMAP packs
    clusters tightly, so that unit is small and real voids read as many tens of
    steps; the percentile is the more legible number and is reported alongside.

    `sort` picks what kind of gap is wanted. "empty" finds the big vacancies,
    which tend to sit outside a single cluster. "mixed" finds the interstitial
    points whose neighbours span several periods -- usually the more interesting
    ones to paint, because the question there is what lies *between* two
    settled regions rather than merely beyond one.
    """
    rng = np.random.default_rng(seed)
    lo, hi = fld.xy.min(0), fld.xy.max(0)
    pts = rng.uniform(lo, hi, size=(samples, 2))

    if inside_hull:
        try:
            from scipy.spatial import Delaunay
            hull = Delaunay(fld.xy)
            pts = pts[hull.find_simplex(pts) >= 0]
        except Exception:
            pass
    if len(pts) == 0:
        return []

    u = step_unit(fld)
    d_all, _ = fld.tree().query(pts, k=1)
    keep = d_all >= min_steps * u
    pts, d = pts[keep], d_all[keep]
    if len(pts) == 0:
        return []

    # context: the signals of the works that surround each candidate
    kk = int(min(k_ctx, len(fld)))
    _, nn = fld.tree().query(pts, k=kk)
    with np.errstate(invalid="ignore"):
        stab = np.nanmean(fld.stability[nn], axis=1)
        tear = np.nanmean(fld.tear[nn], axis=1)

    if max_tear is not None:
        m = ~np.isfinite(tear) | (tear <= max_tear)
        pts, d, stab, tear, nn = pts[m], d[m], stab[m], tear[m], nn[m]
        if len(pts) == 0:
            return []

    # Period entropy of the neighbourhood. Normalised by a FIXED denominator --
    # how many distinct periods k neighbours could possibly show -- so that a
    # point mixing five periods outranks one mixing two. Normalising by the
    # observed label count instead makes any even split score 1.0, which makes
    # the whole ranking useless.
    per_arr = np.array(fld.period, dtype=object)
    denom = np.log(max(2, min(kk, len(set(fld.period)))))
    div = np.zeros(len(pts))
    for j in range(len(pts)):
        _, cnt = np.unique(per_arr[nn[j]], return_counts=True)
        if len(cnt) > 1:
            p = cnt / cnt.sum()
            div[j] = float(-(p * np.log(p)).sum() / denom)

    if sort == "mixed":
        score = -div
    elif sort == "contested":
        score = np.where(np.isfinite(stab), stab, np.inf)
    elif sort == "settled":
        score = -np.where(np.isfinite(stab), stab, -np.inf)
    else:
        score = -d
    order = np.argsort(score, kind="stable")

    # Spatially thin the result. Without this the top n are all the same void
    # seen n times, because the score varies smoothly across a large vacancy.
    if min_separation is None:
        extent = float(np.hypot(*(fld.xy.max(0) - fld.xy.min(0))))
        min_separation = max(min_steps * u * 3.0, 0.035 * extent)

    chosen: list = []
    for j in order:
        if len(chosen) >= n:
            break
        if any(np.hypot(*(pts[j] - pts[c])) < min_separation for c in chosen):
            continue
        chosen.append(int(j))
    if not chosen:                       # separation too aggressive; fall back
        chosen = [int(j) for j in order[:n]]

    out = []
    for j in chosen:
        idx = nn[j]
        per: dict = {}
        cat: dict = {}
        for i in idx:
            per[fld.period[i]] = per.get(fld.period[i], 0) + 1
            cat[fld.category[i]] = cat.get(fld.category[i], 0) + 1
        out.append({
            "x": float(pts[j][0]), "y": float(pts[j][1]),
            "d_nearest": float(d[j]),
            "steps_out": float(d[j] / u),
            # how empty this is relative to everywhere else inside the hull
            "emptiness_pct": float((d_all < d[j]).mean()),
            "diversity": float(div[j]),
            "stability": None if not np.isfinite(stab[j]) else float(stab[j]),
            "tear": None if not np.isfinite(tear[j]) else float(tear[j]),
            "periods": sorted(per.items(), key=lambda kv: -kv[1])[:3],
            "categories": sorted(cat.items(), key=lambda kv: -kv[1]),
        })
    return out


# ----------------------------------------------------------------------------
# Saving, with provenance (matching the project's sidecar habit)
# ----------------------------------------------------------------------------

def save(img: Image.Image, info: dict, fld: Field, tag="") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{fld.protocol}_{info['method']}" + (f"_{tag}" if tag else "")
    png = OUT / f"{base}.png"
    img.save(png)
    rec = dict(info)
    rec["protocol"] = fld.protocol
    rec["protocol_display"] = fld.display_name
    rec["written"] = stamp
    rec["tool"] = "gap_space_core.py"
    (OUT / f"{base}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return png


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _selftest() -> int:
    print(f"root: {ROOT}")
    protos = list_protocols()
    print(f"protocols with a 2D layout: {[p['id'] for p in protos]}")
    if not protos:
        print("FAIL: no protocol has a usable 2D layout"); return 1

    bases = available_bases()
    print(f"decoders on disk: {len(bases['category'])} category "
          f"{bases['category']}, {len(bases['period'])} period")

    ok = True
    for proto in protos:
        pid = proto["id"]
        try:
            fld = load_field(pid)
        except Exception as ex:
            print(f"  {pid:<24} FAIL load: {ex}"); ok = False; continue
        u = step_unit(fld)
        cands = gap_candidates(fld, n=3, samples=3000, min_steps=1.0)
        print(f"  {pid:<24} n={len(fld):<5} step_unit={u:.4f}  "
              f"gaps found={len(cands)}"
              + (f"  best={cands[0]['steps_out']:.1f} steps out" if cands else ""))
        if not cands:
            ok = False

    # paint an interstitial candidate every way, on the default protocol
    pid = protos[0]["id"]
    fld = load_field(pid)
    cands = gap_candidates(fld, n=1, samples=4000, min_steps=1.5, sort="mixed")
    if not cands:
        print("FAIL: no gap candidate to paint"); return 1
    c = cands[0]
    print(f"\npainting ({c['x']:.3f}, {c['y']:.3f}) on {pid} — "
          f"{c['steps_out']:.0f} steps out ({c['emptiness_pct']:.1%} emptier than "
          f"the field), diversity={c['diversity']:.2f}, "
          f"periods={[p for p, _ in c['periods']]}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn, kwargs in [
        ("composite", paint_composite, {}),
        ("ae", paint_ae, {}),
        ("ae_step2", paint_ae, {"alpha": 2.0}),
        ("quilt", paint_quilt, {}),
        ("guided", paint_guided, {}),
        ("guided_hard", paint_guided, {"guide_weight": 4.0}),
    ]:
        try:
            img, info = fn(fld, c["x"], c["y"], **kwargs)
            p = save(img, info, fld, tag="selftest_" + name)
            extra = ""
            if "seam_energy" in info:
                extra = f"  seam={info['seam_energy']:.5f}"
            if info["method"] == "ae":
                extra = (f"  {info['basis']} {info['native_px']}px/"
                         f"{info['latent_dim']}d  lum_std={info['luminance_std']:.3f}")
            print(f"  {name:<12} {info['seconds']:>6.2f}s  {img.size}{extra}  -> {p.name}")
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  {name:<12} FAIL: {ex}"); ok = False

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list-gaps", action="store_true")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--method", choices=list(PAINTERS), default="quilt")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("-k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    protos = list_protocols()
    if not protos:
        print("no protocol has a usable 2D layout"); return 1
    pid = args.protocol or protos[0]["id"]
    fld = load_field(pid)

    if args.list_gaps:
        for i, c in enumerate(gap_candidates(fld, n=args.n)):
            st = "  -  " if c["stability"] is None else f"{c['stability']:.3f}"
            tr = "  -  " if c["tear"] is None else f"{c['tear']:.3f}"
            print(f"{i:>3}  ({c['x']:8.3f},{c['y']:8.3f})  "
                  f"{c['steps_out']:5.2f} steps  stab={st}  tear={tr}  "
                  f"{[p for p, _ in c['periods']]}")
        return 0

    if args.at:
        kw = {"k": args.k, "size": args.size}
        if args.method == "ae":
            kw.update(alpha=args.alpha, seed=args.seed)
        elif args.method == "quilt":
            kw.update(seed=args.seed)
        img, info = PAINTERS[args.method](fld, args.at[0], args.at[1], **kw)
        p = Path(args.out) if args.out else save(img, info, fld)
        if args.out:
            img.save(p)
        print(json.dumps({kk: vv for kk, vv in info.items() if kk != "neighbours"},
                         indent=2))
        print(f"-> {p}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
