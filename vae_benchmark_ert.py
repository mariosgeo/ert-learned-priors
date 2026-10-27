"""
vae_benchmark_ert.py
====================
Benchmark of four ERT inversion methods on 100 synthetic models of increasing
complexity. For every model ONE noisy data set is simulated, saved to disk and
inverted by all methods:

  pygimli     pyGIMLi ERTManager (zWeight 0.3, blocky, largest lambda with chi2 <= 1.05)
  sparse      sparse biharmonic PyTorch Gauss-Newton (sparse_torch_inversion_ert.py)
  vae_random  VAE prior trained on 200 000 RANDOM models (general family)
  vae_pg      the same VAE fine-tuned on 20 000 models built around the pyGIMLi
              inversion of this data set (pyGIMLi-guided prior)
  vae_pg_ar   as vae_pg, but the pyGIMLi analysis ignores anomalies at the line ends
              (edge filter, 2 m) and the contrast increase adapts to the resolution
              (small where the data resolve well, large where they do not)
  pygimli_fine / vae_pg_fine
              pyGIMLi and the pyGIMLi-guided VAE (same network) on a FINE inversion
              mesh (cells <= 0.25 m2 instead of 1 m2), same data; models are averaged
              onto the standard mesh cells (area fraction) for the error metrics
  vae_pg_dip  as vae_pg, but the pyGIMLi layering is read ALONG its dominant dip,
              each interface gets its own fitted dip, and the training models vary
              the dip around it (instead of around horizontal)
  pygimli_robust  pyGIMLi with robust data weighting (robustData=True: L1-type data
              misfit by iterative reweighting, down-weights data the model cannot
              explain), otherwise as pygimli (zWeight 0.3, blocky, same lambda rule)
  pygimli_dipsteer  pyGIMLi with geostatistical regularization along the dominant dip
              of its own model (dip-aware analysis; correlation lengths 10 m / 2 m)
  hybrid_vae_dip    pyGIMLi with the dip-aware VAE result as starting AND reference
              model: the VAE gives the structure, pyGIMLi adds cell-wise corrections
  layered_param     explicit dipping-layer inversion (layered_param.py): interface
              depths, dips and layer resistivities, evaluated on all cells of the
              larger mesh; started from the dip-aware analysis
  vae_pg_layer      pyGIMLi-guided VAE whose training models are built around the
              explicit-layer result (its depths, dips, layer values; contrast x0.9-1.1),
              started also from the encoded layered model
  layers_vae        sharp explicit dipping layers + a body-only VAE for everything the
              layering does not explain, inverted jointly (layers_vae.py)
  ttl_cnn           test-time learning (Xu & Heagy 2024): the VAE decoder architecture,
              UNTRAINED, weights fitted to this data set only; no training set, no
              explicit regularisation, early stop at chi2 <= 1.05 (ttl_inversion.py)
  ttl_field         same with a neural field (coordinate MLP, Fourier features;
              Xu & Heagy 2025)
  gan         GAN generator (WGAN-GP, gan_prior_ert.py) trained on the same 200 000
              random models; same latent inversion, start = projected pyGIMLi model

Both VAE inversions start from z = 0, from the ENCODED pyGIMLi model and from a
random point. Everything is saved for re-use (see ert_ml_reuse.py):

  outputs/vae_benchmark/
      data/<name>.dat          noisy data set (pyGIMLi DataContainerERT)
      models/<name>.npz        test-model spec, true model, all recovered models, metrics
      vae_pg/<name>.pt         fine-tuned pyGIMLi-guided VAE of this data set
      figures/<name>.png       true model, four recovered models, recovered / true ratios
      benchmark_results.csv    one row per model and method
      benchmark_summary.txt    statistics by group
      benchmark_summary.png    statistics figure
      suite_overview.png       thumbnails of the 100 test models

Test suite (100 models, make_suite):
  A  2 horizontal layers (8)       D1 1 dipping interface (8)
  B  3 horizontal layers (12)      D2 2 dipping interfaces (8)
  C  4 horizontal layers (5)       D3 complex interfaces (14): sinusoidal (various
                                      amplitudes / wavelengths, in / out of phase,
                                      on a dip), multi-sine, basin, anticline,
                                      fault, truncated, pinch-out
  E  1 body (20)   F  2 bodies (15)   G  3-5 bodies on complex backgrounds (10)

Usage
-----
    conda activate pg2
    python vae_benchmark_ert.py                 # all 100 models (resumes; 6 workers)
    python vae_benchmark_ert.py --summary       # statistics from saved results
    python vae_benchmark_ert.py --only D3,G01   # a subset (name prefixes)
    python vae_benchmark_ert.py --overview      # thumbnails of the test models
"""

import os
import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LARGE = os.environ.get("ERT_DOMAIN", "standard") == "large"     # larger inversion domain
HETERO = os.environ.get("ERT_HETERO")          # e.g. "30": every layer/body varies +-30 % (geostatistical)
TEXTURE = os.environ.get("ERT_TEXTURE") == "1"  # use the VAE trained on textured models
_het = f"_het{HETERO}" if HETERO else ""
OUTB = os.path.join(HERE, "outputs", "vae_benchmark" + ("_large" if LARGE else "") + _het + ("_tex" if TEXTURE else ""))
DATA_DIR = os.path.join(HERE, "outputs", "vae_benchmark" + _het, "data")   # the SAME noisy data for every variant
RANDOM_CKPT = os.path.join(HERE, "outputs", "vae_prior" + ("_large" if LARGE else "") + ("_tex" if TEXTURE else ""),
                           "general", "vae_N200000.pt")
LN10 = np.log(10.0)
XM = 23.5
METHODS = ["pygimli", "sparse", "vae_random", "vae_pg", "vae_pg_ar", "gan", "pygimli_fine", "vae_pg_fine",
           "vae_pg_dip", "pygimli_robust", "pygimli_dipsteer", "hybrid_vae_dip", "layered_param",
           "vae_pg_layer", "layers_vae", "ttl_cnn", "ttl_field"]
if LARGE:                                                  # methods re-run on the larger domain
    METHODS = ["pygimli", "sparse", "vae_random", "vae_pg", "vae_pg_dip"]
CORE = ["pygimli", "sparse", "vae_random", "vae_pg"]       # methods every saved result has
if HETERO:                                                 # heterogeneous suites: the informative methods
    METHODS = ["pygimli", "vae_random", "vae_pg_dip", "pygimli_dipsteer", "layered_param", "layers_vae",
               "ttl_cnn", "ttl_field", "hybrid_vae_dip"]
    if TEXTURE:                                            # retrained (textured) VAEs only
        METHODS = ["pygimli", "vae_random", "vae_pg_dip"]
    CORE = ["pygimli", "vae_random"]
FINE_CELL = 0.25                                           # max cell area (m2) of the fine inversion mesh
LABELS = {"pygimli": "pyGIMLi", "sparse": "Sparse biharmonic (GN)",
          "vae_random": "VAE, random prior", "vae_pg": "VAE, pyGIMLi-guided prior",
          "vae_pg_ar": "VAE, pyGIMLi-guided, anomaly removal", "gan": "GAN, random prior",
          "pygimli_fine": "pyGIMLi, fine mesh", "vae_pg_fine": "VAE, pyGIMLi-guided, fine mesh",
          "vae_pg_dip": "VAE, pyGIMLi-guided, dip-aware", "pygimli_robust": "pyGIMLi, robust (L1 data)",
          "pygimli_dipsteer": "pyGIMLi, dip-steered regularization",
          "hybrid_vae_dip": "Hybrid: pyGIMLi with VAE dip-aware reference",
          "layered_param": "Explicit dipping-layer inversion",
          "vae_pg_layer": "VAE, layer-guided prior",
          "layers_vae": "Sharp layers + body VAE",
          "ttl_cnn": "Test-time CNN (untrained decoder)",
          "ttl_field": "Test-time neural field"}
SHORT = {"pygimli": "pyGIMLi", "sparse": "sparse GN", "vae_random": "VAE random",
         "vae_pg": "VAE pyGIMLi-guided", "vae_pg_ar": "VAE pg anomaly-removal", "gan": "GAN random",
         "pygimli_fine": "pyGIMLi fine mesh", "vae_pg_fine": "VAE pg fine mesh", "vae_pg_dip": "VAE pg dip-aware",
         "pygimli_robust": "pyGIMLi robust", "pygimli_dipsteer": "pyGIMLi dip-steered",
         "hybrid_vae_dip": "Hybrid pyGIMLi+VAE", "layered_param": "Dipping layers",
         "vae_pg_layer": "VAE layer-guided", "layers_vae": "Layers + body VAE",
         "ttl_cnn": "Test-time CNN", "ttl_field": "Test-time field"}
COLORS = {"pygimli": "#2a78d6", "sparse": "#e87ba4", "vae_random": "#eb6834", "vae_pg": "#1baf7a",
          "vae_pg_ar": "#eda100", "gan": "#7c5cd6", "pygimli_fine": "#104281", "vae_pg_fine": "#0b6e4f",
          "vae_pg_dip": "#c2185b", "pygimli_robust": "#5f5e5a",
          "pygimli_dipsteer": "#00838f", "hybrid_vae_dip": "#6d4c41", "layered_param": "#9e9d24",
          "vae_pg_layer": "#ad1457", "layers_vae": "#283593",
          "ttl_cnn": "#8d6e00", "ttl_field": "#4e342e"}
GROUPS = ["A", "B", "C", "D1", "D2", "D3", "E", "F", "G"]
GROUP_NAMES = {"A": "2 layers", "B": "3 layers", "C": "4 layers", "D1": "1 dipping interface",
               "D2": "2 dipping interfaces", "D3": "complex interfaces", "E": "1 body",
               "F": "2 bodies", "G": "3-5 bodies, complex"}
DEFAULTS = dict(threads=4, n_pg=20000, ft_steps=3000, batch=256, sigma=0.1, pg_mix=0.15,
                pg_extend=1.0, restarts=3, lams="10,3,1,0.3,0.1,0.03,0.01",
                random_ckpt=RANDOM_CKPT,
                gan_ckpt=os.path.join(HERE, "outputs", "gan_prior", "gan_general.pt"))


# ── geometry of the test models ────────────────────────────────────────────
def body_mask(x, z, b):
    th = np.deg2rad(b.get("ang", 0.0))
    u = (x - b["xc"]) * np.cos(th) + (z - b["zc"]) * np.sin(th)
    v = -(x - b["xc"]) * np.sin(th) + (z - b["zc"]) * np.cos(th)
    s = b["shape"]
    if s in ("rect", "dyke"):
        return (np.abs(u) <= b["w"] / 2) & (np.abs(v) <= b["h"] / 2)
    if s == "ellipse":
        return (u / (b["w"] / 2)) ** 2 + (v / (b["h"] / 2)) ** 2 <= 1
    if s == "channel":
        v0 = b["amp"] * np.sin(2 * np.pi * u / b["wl"])
        return (np.abs(u) <= b["w"] / 2) & (np.abs(v - v0) <= b["h"] / 2)
    if s == "polygon":
        from matplotlib.path import Path
        return Path(np.asarray(b["verts"])).contains_points(np.column_stack([x, z]))
    raise ValueError(f"unknown body shape {s}")


def interface_depth(it, x):
    """Depth (m) of one interface at x. Keys: z (depth at x = 23.5 m), dip (deg),
    amp/wl (one sinusoid), sines [[amp, wl, phase], ...], bump [A, xc, w]
    (Gaussian; A > 0 = basin, A < 0 = anticline), xmin/xmax (truncation)."""
    d = it["z"] + np.tan(np.deg2rad(it.get("dip", 0.0))) * (x - XM)
    if it.get("amp"):
        d = d + it["amp"] * np.sin(2 * np.pi * x / it.get("wl", 25.0) + it.get("phase", 0.0))
    for amp, wl, ph in it.get("sines", []):
        d = d + amp * np.sin(2 * np.pi * x / wl + ph)
    if it.get("bump"):
        A, xc, w = it["bump"]
        d = d + A * np.exp(-((x - xc) / w) ** 2)
    if "xmax" in it:
        d = np.where(x <= it["xmax"], d, np.inf)
    if "xmin" in it:
        d = np.where(x >= it["xmin"], d, np.inf)
    return d


def eval_model(spec, cc):
    """log(rho) of a test model at points cc (x, y = -depth). A spec is a dict
    with layers (interfaces, see interface_depth), rhos (one per layer, top
    down), bodies (shape rect | dyke | ellipse | channel | polygon) and an
    optional fault {x0, dip, throw}."""
    x, z = cc[:, 0], -cc[:, 1]
    zq = z
    f = spec.get("fault")
    if f:
        hang = x > f["x0"] + z / np.tan(np.deg2rad(f["dip"]))
        zq = np.where(hang, z - f["throw"], z)
    layer = np.zeros(len(x), dtype=int)
    for it in spec["layers"]:
        layer += zq > interface_depth(it, x)
    rho = np.asarray(spec["rhos"], dtype=float)[np.minimum(layer, len(spec["rhos"]) - 1)]
    for b in spec.get("bodies", []):
        rho = np.where(body_mask(x, z, b), b["rho"], rho)
    out = np.log(rho)
    h = spec.get("hetero")
    if h:                                                  # geostatistical variation inside layers/bodies
        out = out + h["sigma"] * hetero_field(x, z, h["seed"], h["Lx"], h["Lz"])
    return out


def hetero_field(x, z, seed, Lx=10.0, Lz=2.0, n_modes=128):
    """Smooth Gaussian random field, unit variance, correlation lengths Lx
    (horizontal) and Lz (vertical), via random Fourier features - defined at
    any point, so the same field is used on every mesh."""
    rng = np.random.default_rng(seed)
    kx = rng.normal(0, 1.0 / Lx, n_modes)
    kz = rng.normal(0, 1.0 / Lz, n_modes)
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    return np.sqrt(2.0 / n_modes) * np.cos(np.multiply.outer(x, kx) + np.multiply.outer(z, kz) + ph).sum(-1)


def bodies_mask(spec, cc):
    x, z = cc[:, 0], -cc[:, 1]
    m = np.zeros(len(x), dtype=bool)
    for b in spec.get("bodies", []):
        m |= body_mask(x, z, b)
    return m


def jsonable(o):
    """Spec -> plain JSON types (for saving / editing test models)."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _contrast(*r):
    return "-".join("H" if v >= max(r) / 1.5 and v > min(r) * 1.5 else
                    ("L" if v <= min(r) * 1.5 else "M") for v in r)


def make_suite(seed=7):
    """The 100 test models, simple to complex."""
    rng = np.random.default_rng(seed)
    M = []

    def add(group, label, layers, rhos, bodies=(), **kw):
        n = sum(m["group"] == group for m in M) + 1
        M.append(dict(name=f"{group}_{n:02d}_{label}", group=group, layers=list(layers),
                      rhos=list(rhos), bodies=list(bodies), **kw))

    # A: 2 horizontal layers
    for z, r1, r2 in [(1.5, 300, 30), (1.5, 30, 300), (3, 200, 50), (3, 50, 500),
                      (6, 500, 100), (6, 80, 800), (10, 300, 40), (10, 40, 400)]:
        add("A", f"z{z:g}_{_contrast(r1, r2)}", [dict(z=z)], [r1, r2])
    # B: 3 horizontal layers
    for z1, z2, r in [(2, 6, (400, 40, 400)), (3, 9, (300, 30, 1000)), (1.5, 4, (200, 20, 200)),
                      (2, 6, (40, 400, 40)), (3, 9, (30, 300, 10)), (1.5, 5, (50, 500, 50)),
                      (2, 7, (1000, 200, 20)), (2, 7, (20, 100, 500)), (4, 5, (100, 10, 100)),
                      (4, 5, (100, 1000, 100)), (5, 14, (200, 50, 500)), (6, 15, (50, 500, 50))]:
        add("B", f"z{z1:g}-{z2:g}_{_contrast(*r)}", [dict(z=z1), dict(z=z2)], r)
    # C: 4 horizontal layers
    for zs, r in [((1.5, 4, 9), (300, 50, 500, 20)), ((2, 5, 10), (50, 500, 30, 300)),
                  ((1, 3, 8), (1000, 200, 50, 10)), ((2, 6, 12), (20, 100, 500, 2000)),
                  ((3, 6, 11), (100, 20, 300, 50))]:
        add("C", "4lay_" + _contrast(*r), [dict(z=z) for z in zs], r)
    # D1: one dipping interface
    for dip in (5, 10, 20, 30):
        for r in ((300, 40), (40, 300)):
            add("D1", f"dip{dip}_{_contrast(*r)}", [dict(z=6.0, dip=dip if r[0] > r[1] else -dip)], r)
    # D2: two dipping interfaces
    for lay, r, lab in [((dict(z=3, dip=5), dict(z=9, dip=5)), (300, 40, 300), "parallel5"),
                        ((dict(z=3, dip=10), dict(z=9, dip=10)), (40, 400, 40), "parallel10"),
                        ((dict(z=4, dip=20), dict(z=10, dip=20)), (30, 200, 1000), "parallel20"),
                        ((dict(z=3, dip=-8), dict(z=10, dip=8)), (200, 30, 200), "opposite8"),
                        ((dict(z=3, dip=12), dict(z=10, dip=-12)), (50, 500, 50), "opposite12"),
                        ((dict(z=4, dip=0), dict(z=8, dip=12)), (300, 50, 300), "wedge"),
                        ((dict(z=2, dip=-5), dict(z=7, dip=10)), (100, 1000, 30), "diverging"),
                        ((dict(z=5, dip=25), dict(z=11, dip=25)), (200, 20, 200), "steep25")]:
        add("D2", lab, lay, r)
    # D3: complex interfaces
    add("D3", "sine_a1_wl25", [dict(z=6, amp=1.0, wl=25)], (150, 40))
    add("D3", "sine_a2.5_wl20", [dict(z=6, amp=2.5, wl=20)], (40, 300))
    add("D3", "sine_short_wl12", [dict(z=5, amp=1.5, wl=12)], (200, 50))
    add("D3", "sine_long_wl45", [dict(z=7, amp=3.0, wl=45)], (300, 30))
    add("D3", "two_sines_inphase", [dict(z=3, amp=1.0, wl=25), dict(z=9, amp=1.5, wl=25)], (300, 50, 300))
    add("D3", "two_sines_antiphase", [dict(z=3, amp=1.0, wl=20),
                                      dict(z=9, amp=1.5, wl=20, phase=np.pi)], (50, 400, 50))
    add("D3", "sine_on_dip10", [dict(z=6, dip=10, amp=1.5, wl=20)], (200, 40))
    add("D3", "multisine", [dict(z=6, sines=[[1.5, 40, 0.3], [0.8, 17, 1.7], [0.4, 9, 4.0]])], (100, 500))
    add("D3", "basin", [dict(z=3, bump=[5.0, 24, 8])], (300, 30))
    add("D3", "anticline", [dict(z=4), dict(z=11, bump=[-4.0, 22, 9])], (50, 500, 50))
    add("D3", "fault", [dict(z=3), dict(z=9)], (200, 60, 25), fault=dict(x0=25, dip=70, throw=3))
    add("D3", "truncated", [dict(z=3), dict(z=9, xmax=25)], (250, 80, 30))
    add("D3", "pinchout", [dict(z=3, dip=-6), dict(z=6, dip=6)], (300, 40, 300))
    add("D3", "4dipping_sine", [dict(z=2, dip=6), dict(z=5, dip=8), dict(z=9, dip=10, amp=1.5, wl=18)],
        (500, 100, 400, 20))

    # bodies
    bgs = [dict(layers=[], rhos=[100], tag="homog"),
           dict(layers=[dict(z=3)], rhos=[300, 50], tag="2lay"),
           dict(layers=[dict(z=2), dict(z=8)], rhos=[300, 50, 300], tag="3lay"),
           dict(layers=[dict(z=6, dip=10)], rhos=[200, 40], tag="dipping")]

    def host_rho(bg, xc, zc):
        return float(np.exp(eval_model(dict(bg, bodies=[]), np.array([[xc, -zc]]))[0]))

    def rand_body(shape, xc, top, resistive, bg):
        if shape == "rect":
            w, h = rng.uniform(4, 10), rng.uniform(2, 4)
            b = dict(shape="rect", xc=xc, zc=top + h / 2, w=w, h=h, ang=0.0)
        elif shape == "ellipse":
            w, h = rng.uniform(4, 10), rng.uniform(2, 5)
            b = dict(shape="ellipse", xc=xc, zc=top + h / 2, w=w, h=h, ang=rng.uniform(-20, 20))
        elif shape == "dyke":
            L, ang = rng.uniform(8, 12), rng.choice([-1, 1]) * rng.uniform(45, 70)
            b = dict(shape="dyke", w=L, h=rng.uniform(1.5, 2.5), ang=ang, xc=xc,
                     zc=top + L / 2 * np.sin(np.deg2rad(abs(ang))))
        elif shape == "channel":
            b = dict(shape="channel", xc=xc, zc=top + 1.5, w=rng.uniform(12, 20), h=rng.uniform(1.5, 2.5),
                     amp=rng.uniform(0.8, 1.5), wl=rng.uniform(10, 15), ang=0.0)
        else:
            n = rng.integers(5, 8)
            a = np.sort(rng.uniform(0, 2 * np.pi, n))
            r = rng.uniform(2, 4, n)
            zc = top + 3.0
            b = dict(shape="polygon", xc=xc, zc=zc,
                     verts=np.column_stack([xc + 1.6 * r * np.cos(a), zc + r * np.sin(a)]).tolist())
        host = host_rho(bg, b["xc"], b["zc"])
        b["rho"] = float(np.clip(host * rng.uniform(5, 15) if resistive else host / rng.uniform(5, 15), 2, 8000))
        return b

    shapes = ["rect", "ellipse", "dyke", "channel", "polygon"]
    for i in range(20):                                      # E: one body, gradually deeper
        bg, shape = bgs[(i // 5) % 4], shapes[i % 5]
        top = 1.0 + 9.0 * i / 19
        b = rand_body(shape, rng.uniform(14, 33), top, i % 2 == 0, bg)
        add("E", f"{shape}_{'res' if i % 2 == 0 else 'con'}_top{top:.0f}_{bg['tag']}",
            bg["layers"], bg["rhos"], [b])
    for i in range(15):                                      # F: two bodies
        bg = bgs[i % 4]
        s1, s2 = shapes[i % 5], shapes[(i + 2) % 5]
        r1, r2 = (True, False) if i % 3 == 0 else ((True, True) if i % 3 == 1 else (False, False))
        b1 = rand_body(s1, rng.uniform(8, 18), rng.uniform(1, 7), r1, bg)
        b2 = rand_body(s2, rng.uniform(29, 39), rng.uniform(1, 7), r2, bg)
        add("F", f"{s1}+{s2}_{bg['tag']}", bg["layers"], bg["rhos"], [b1, b2])
    cbgs = [dict(layers=[dict(z=3, dip=8), dict(z=9, dip=8)], rhos=[300, 60, 200], tag="dip2"),
            dict(layers=[dict(z=6, amp=2.0, wl=25)], rhos=[150, 40], tag="sine"),
            dict(layers=[dict(z=3), dict(z=9)], rhos=[200, 60, 25], tag="fault",
                 fault=dict(x0=25, dip=70, throw=3)),
            dict(layers=[dict(z=3, amp=1.0, wl=20), dict(z=10, amp=1.5, wl=30, phase=1.0)],
                 rhos=[300, 50, 300], tag="2sines"),
            dict(layers=[dict(z=3, bump=[4.0, 24, 8])], rhos=[250, 40], tag="basin")]
    for i in range(10):                                      # G: 3-5 bodies, complex backgrounds
        bg = cbgs[i % 5]
        nb = 3 + i % 3
        xs = np.linspace(6, 41, nb) + rng.uniform(-2, 2, nb)
        bodies = [rand_body(shapes[(i + k) % 5], xs[k], rng.uniform(1, 8), rng.random() < 0.5, bg)
                  for k in range(nb)]
        extra = {k: v for k, v in bg.items() if k == "fault"}
        add("G", f"{nb}bodies_{bg['tag']}", bg["layers"], bg["rhos"], bodies, **extra)
    assert len(M) == 100, len(M)
    M = [jsonable(m) for m in M]
    if HETERO:                                             # +-HETERO % (one standard deviation)
        for i, m in enumerate(M):
            m["hetero"] = dict(sigma=float(np.log(1 + int(HETERO) / 100)), seed=500 + i, Lx=10.0, Lz=2.0)
    return M


# ── shared inversion context ───────────────────────────────────────────────
G = {}


def setup(**opts):
    """Mesh, image grid, sampling matrix, sensitivity and options; stored in G."""
    o = dict(DEFAULTS, **{k: v for k, v in opts.items() if v is not None})
    os.environ["OMP_NUM_THREADS"] = str(o["threads"])
    import torch
    import sparse_torch_inversion_ert as sp
    import vae_prior_inversion_ert as v
    from pygimli.physics import ert
    sp._threads(o["threads"])
    scheme = ert.createData(elecs=sp.ELEC_X, schemeName="dd")
    mesh, fwd0, cc, area = sp.build_inversion_mesh(scheme)
    J0, _ = fwd0.jacobian_loglog(np.full(len(cc), np.log(100.0)))
    cov = np.sqrt(np.sum((J0 / sp.NOISE) ** 2, axis=0)) / np.sqrt(area)
    xg, zg = v.make_grid(cc)
    G.update(sp=sp, v=v, torch=torch, mesh=mesh, cc=cc, tri=sp._triangles(fwd0.fop.paraDomain),
             xg=xg, zg=zg, S=v.bilinear_matrix(cc, xg, zg), cov=cov,
             covg=v.grid_from_cells(cc, cov / cov.max(), xg, zg), opts=o,
             device="cuda" if torch.cuda.is_available() else "cpu")
    return G


def _init_worker(opts):
    setup(**opts)


def load_vae(path):
    torch, v = G["torch"], G["v"]
    ck = torch.load(path, weights_only=False)
    model = v.make_vae(ck["nz"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck["mean"], ck["std"]


def encode(model, img, mean, std):
    torch = G["torch"]
    with torch.no_grad():
        mu, _ = model.float().encode(torch.from_numpy((img[None, None] - mean) / std).float())
    return mu.numpy()[0].astype(float)


class _GPULock:
    """Machine-wide lock so only ONE worker process trains on the GPU at a time
    (several processes sharing the 12 GB card made Windows page GPU memory and
    slowed everything by ~100x). A lock older than 30 min is treated as stale."""
    path = os.path.join(HERE, "outputs", ".gpu.lock")         # one lock for every run on this machine

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > 1800:
                        os.remove(self.path)
                except OSError:
                    pass
                time.sleep(3)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def finetune(X, seed=0):
    """Continue training the random-prior VAE on pyGIMLi-guided models (same
    normalization, so the latent space keeps its meaning). GPU use is serialized
    across processes and the GPU memory is released afterwards."""
    torch, o = G["torch"], G["opts"]
    model, mean, std = load_vae(o["random_ckpt"])
    torch.manual_seed(seed)
    dev = G["device"]
    import contextlib
    with (_GPULock() if dev == "cuda" else contextlib.nullcontext()):   # the CPU needs no GPU lock
        model = model.float().to(dev).train()
        Xt = torch.from_numpy((X[:, None] - mean) / std).float().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, o["ft_steps"])
        s2 = 2 * (o["sigma"] / std) ** 2
        for _ in range(o["ft_steps"]):
            x = Xt[torch.randint(0, len(Xt), (o["batch"],), device=dev)]
            mu, logvar = model.encode(x)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            rec = ((model.decode(z) - x) ** 2).sum((1, 2, 3)) / s2
            kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
            loss = (rec + kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model = model.cpu().eval()
        del Xt, opt, sched, x, mu, logvar, z, rec, kl, loss
        if dev == "cuda":
            torch.cuda.empty_cache()                    # give the GPU memory back
    return model, mean, std


# ── data: simulated ONCE per model, saved, shared by all methods ───────────
def get_dataset(spec, data_file, seed):
    import pygimli as pg
    if os.path.exists(data_file):
        return pg.DataContainerERT(data_file)
    data = G["v"].simulate(lambda c: eval_model(spec, c), seed=seed)
    data.save(data_file)
    return pg.DataContainerERT(data_file)                 # identical object for every method


def _cell_samples(k=8):
    """36 interior points of every inversion-mesh triangle (barycentric grid)."""
    if "P" not in G:
        nodes, tris = G["tri"]
        bc = np.array([(i + 1 / 3, j + 1 / 3) for i in range(k) for j in range(k - i)]) / k
        w = np.column_stack([bc[:, 0], bc[:, 1], 1 - bc.sum(1)])
        G["P"] = np.einsum("pk,ckd->cpd", w, nodes[tris])
    return G["P"]


def _fine():
    """Fine inversion mesh, its sampling matrix and the averaging matrix that maps
    fine-mesh models onto the standard cells (area fraction, 36 points per cell)."""
    if "fine" not in G:
        import pygimli.meshtools as mt
        from pygimli.physics import ert
        from ert_common import ERTForward
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        sp, v = G["sp"], G["v"]
        scheme = ert.createData(elecs=sp.ELEC_X, schemeName="dd")
        mesh = mt.createParaMesh(scheme, paraDepth=sp.PARA_DEPTH, quality=33, paraMaxCellSize=FINE_CELL,
                                 boundaryMaxCellSize=50)
        pdf = ERTForward(scheme, mesh).fop.paraDomain
        ccf = np.array([[c.center().x(), c.center().y()] for c in pdf.cells()])
        P = _cell_samples()
        nc, ns = P.shape[:2]
        idx = cKDTree(ccf).query(P.reshape(-1, 2))[1]
        M = csr_matrix((np.full(nc * ns, 1.0 / ns), (np.repeat(np.arange(nc), ns), idx)), shape=(nc, len(ccf)))
        G["fine"] = dict(mesh=mesh, cc=ccf, S=v.bilinear_matrix(ccf, G["xg"], G["zg"]), M=M)
    return G["fine"]


def true_on_mesh(spec):
    """The true model on the inversion mesh BY AREA FRACTION: each cell gets the
    mean log-resistivity of 36 points inside it (geometric mean resistivity),
    so cells cut by a boundary get an intermediate value instead of the value
    at their centre."""
    P = _cell_samples()
    return eval_model(spec, P.reshape(-1, 2)).reshape(P.shape[:2]).mean(1)


def metrics(spec, cc, m, chi2):
    out = dict(chi2=float(chi2))
    if spec is None:
        return out
    true = true_on_mesh(spec)
    e = (m - true) / LN10
    roi = G["sp"].roi_mask(cc)
    out["rms"] = float(np.sqrt(np.mean(e[roi] ** 2)))
    bm = bodies_mask(spec, cc)
    out["body_err"] = float(np.median(np.abs(e[bm]))) if bm.any() else np.nan
    return out


def invert_all(name, outdir, spec=None, data_file=None, seed=0, methods=METHODS, verbose=False):
    """Run (or reuse) every method on one data set. spec = test model (None for
    field data, then data_file must exist). Results go to outdir/models/<name>.npz;
    methods already stored there are not re-run."""
    sp, v, o = G["sp"], G["v"], G["opts"]
    from ert_common import ERTForward
    from scipy.spatial import cKDTree
    for d in ("data", "models", "figures", "vae_pg", "vae_pg_ar", "vae_pg_dip", "vae_pg_layer"):
        os.makedirs(os.path.join(outdir, d), exist_ok=True)
    fn = os.path.join(outdir, "models", name + ".npz")
    res = np.load(fn, allow_pickle=True)["res"].item() if os.path.exists(fn) else {}
    data_file = data_file or os.path.join(outdir, "data", name + ".dat")
    data = get_dataset(spec, data_file, seed)
    dobs = np.array(data["rhoa"])
    fwd = ERTForward(data, G["mesh"])
    cc, xg, zg, S = G["cc"], G["xg"], G["zg"], G["S"]
    lams = [float(x) for x in o["lams"].split(",")]

    def save():
        np.savez(fn, spec=np.array(spec, dtype=object), res=np.array(res, dtype=object),
                 m_true=true_on_mesh(spec) if spec else np.array([]), data_file=data_file,
                 opts=np.array(o, dtype=object))

    if "pygimli" in methods and "pygimli" not in res:
        t = time.time()
        pgr, _ = sp.invert_pygimli(data, G["mesh"], [300, 100, 50, 20, 10, 5, 2], 1.05, 0.3, True,
                                   verbose=verbose)
        m = pgr["m"][cKDTree(pgr["cc"]).query(cc)[1]]
        res["pygimli"] = dict(m=m, lam=pgr["lam"], time=time.time() - t, **metrics(spec, cc, m, pgr["chi2"]))
        save()
    img = v.grid_from_cells(cc, res["pygimli"]["m"] / LN10, xg, zg)   # pyGIMLi model as an image

    if "sparse" in methods and "sparse" not in res:
        t = time.time()
        r = v.run_sparse(fwd, cc, dobs, G["cov"])
        res["sparse"] = dict(m=r["m"], lam=r["lam"], n_par=r["n_par"], time=time.time() - t,
                             **metrics(spec, cc, r["m"], r["chi2"]))
        save()

    if "vae_random" in methods and "vae_random" not in res:
        t = time.time()
        model, mean, std = load_vae(o["random_ckpt"])
        r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"],
                            verbose=verbose, z_starts=[encode(model, img, mean, std)])
        res["vae_random"] = dict(m=r["m"], lam=r["lam"], z=r["z"].numpy(), calls=r["calls"],
                                 time=time.time() - t, **metrics(spec, cc, r["m"], r["chi2"]))
        save()

    for key, edge, adaptive, dipaw in (("vae_pg", None, False, False), ("vae_pg_ar", 2.0, True, False),
                                       ("vae_pg_dip", None, False, True)):
        if key not in methods or key in res:
            continue
        t = time.time()
        torch = G["torch"]
        info = v.analyse_pg(img, G["covg"], xg, zg, edge_margin=edge, dip_aware=dipaw)
        ck = os.path.join(outdir, key, name + ".pt")
        if os.path.exists(ck):
            model, mean, std = load_vae(ck)
        else:
            Xpg = v.random_models_pg(o["n_pg"], xg, zg, info, seed, o["pg_mix"], o["pg_extend"],
                                     adaptive=adaptive)
            model, mean, std = finetune(Xpg, seed=seed)
            torch.save(dict(state=model.state_dict(), nz=model.nz, mean=mean, std=std,
                            base=o["random_ckpt"], n_pg=o["n_pg"], ft_steps=o["ft_steps"],
                            interfaces=info["z_if"].tolist(), edge_margin=edge, adaptive=adaptive,
                            slopes=[float(q) for q in info.get("slopes", [])],
                            bodies=[{k: b[k] for k in ("sign", "xc", "zc", "L", "W", "theta")}
                                    for b in info["bodies"]]), ck)
        r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"],
                            verbose=verbose, z_starts=[encode(model, img, mean, std)])
        res[key] = dict(m=r["m"], lam=r["lam"], z=r["z"].numpy(), calls=r["calls"],
                             time=time.time() - t, **metrics(spec, cc, r["m"], r["chi2"]))
        save()
    if "gan" in methods and "gan" not in res:
        if not os.path.exists(o["gan_ckpt"]):
            print(f"  (GAN prior not found: {o['gan_ckpt']} - skipping the gan method)")
        else:
            import gan_prior_ert as gp
            t = time.time()
            gen, mean, std = gp.load_gan(o["gan_ckpt"])
            z0 = gp.project(gen, img, mean, std)
            r = v.invert_latent(fwd, gen, mean, std, S, dobs, lams, restarts=o["restarts"],
                                verbose=verbose, z_starts=[z0])
            res["gan"] = dict(m=r["m"], lam=r["lam"], z=r["z"].numpy(), calls=r["calls"],
                              time=time.time() - t,
                              path=[dict(lam=p["lam"], chi2=p["chi2"], m=np.asarray(p["m"], np.float32))
                                    for p in r["path"]],
                              **metrics(spec, cc, r["m"], r["chi2"]))
            save()
    if "pygimli_robust" in methods and "pygimli_robust" not in res:
        t = time.time()
        pgr, _ = sp.invert_pygimli(data, G["mesh"], [300, 100, 50, 20, 10, 5, 2], 1.05, 0.3, True,
                                   verbose=verbose, robust=True)
        m = pgr["m"][cKDTree(pgr["cc"]).query(cc)[1]]
        res["pygimli_robust"] = dict(m=m, lam=pgr["lam"], time=time.time() - t,
                                     **metrics(spec, cc, m, pgr["chi2"]))
        save()
    if ("pygimli_dipsteer" in methods and "pygimli_dipsteer" not in res) or \
            ("layered_param" in methods and "layered_param" not in res):
        dinfo = v.analyse_pg(img, G["covg"], xg, zg, dip_aware=True)          # dip-aware reading of pyGIMLi
    if "pygimli_dipsteer" in methods and "pygimli_dipsteer" not in res:
        t = time.time()
        dip = -float(np.degrees(np.arctan(dinfo["shear"])))
        pgr, _ = sp.invert_pygimli(data, G["mesh"], [300, 100, 50, 20, 10, 5, 2], 1.05, 0.3, False,
                                   verbose=verbose, dip=dip, corr=[10.0, 2.0])
        res["pygimli_dipsteer"] = dict(m=pgr["m"], lam=pgr["lam"], dip=dip, time=time.time() - t,
                                       **metrics(spec, cc, pgr["m"], pgr["chi2"]))
        save()
    if "hybrid_vae_dip" in methods and "hybrid_vae_dip" not in res and "vae_pg_dip" in res:
        t = time.time()
        pgr, _ = sp.invert_pygimli(data, G["mesh"], [300, 100, 50, 20, 10, 5, 2], 1.05, 0.3, False,
                                   verbose=verbose, start=res["vae_pg_dip"]["m"])
        res["hybrid_vae_dip"] = dict(m=pgr["m"], lam=pgr["lam"], time=time.time() - t,
                                     **metrics(spec, cc, pgr["m"], pgr["chi2"]))
        save()
    if "layered_param" in methods and "layered_param" not in res:
        import layered_param as lp
        t = time.time()
        best = lp.invert_layered(data, dict(dinfo, zg=zg), noise=sp.NOISE, verbose=verbose)
        P = _cell_samples()
        m = lp.evaluate(best, P.reshape(-1, 2)).reshape(P.shape[:2]).mean(1)   # area fraction on the cells
        res["layered_param"] = dict(m=m, lam=0.0, K=best["K"], theta=best["theta"], time=time.time() - t,
                                    **metrics(spec, cc, m, best["chi2"]))
        save()
    need_lay = ("vae_pg_layer" in methods and "vae_pg_layer" not in res) or \
               ("layers_vae" in methods and "layers_vae" not in res)
    if need_lay and "layered_param" in res:
        import layered_param as lp
        L = res["layered_param"]
        Kl, thl = L["K"], np.asarray(L["theta"], float)
        Xg_, Zg_ = np.meshgrid(xg, zg)
        lay_img = lp.evaluate(dict(theta=thl, K=Kl), np.column_stack([Xg_.ravel(), -Zg_.ravel()])
                              ).reshape(Xg_.shape) / LN10          # sharp layered model as an image (log10)
    if "vae_pg_layer" in methods and "vae_pg_layer" not in res and "layered_param" in res:
        t = time.time()
        torch = G["torch"]
        info = v.analyse_pg(img, G["covg"], xg, zg, dip_aware=True)          # bodies / DOI from pyGIMLi
        info.update(z_if=thl[:Kl], slopes=thl[Kl:2 * Kl], shear=float(np.median(thl[Kl:2 * Kl])),
                    trend=lay_img[:, int(np.argmin(np.abs(xg - XM)))])        # layering from the layer inversion
        ck = os.path.join(outdir, "vae_pg_layer", name + ".pt")
        if os.path.exists(ck):
            model, mean, std = load_vae(ck)
        else:
            Xpl = v.random_models_pg(o["n_pg"], xg, zg, info, seed, o["pg_mix"], o["pg_extend"],
                                     contrast=(0.9, 1.1))
            model, mean, std = finetune(Xpl, seed=seed)
            torch.save(dict(state=model.state_dict(), nz=model.nz, mean=mean, std=std, base=o["random_ckpt"],
                            guide="layered_param"), ck)
        z_starts = [encode(model, img, mean, std), encode(model, lay_img, mean, std)]
        r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"],
                            verbose=verbose, z_starts=z_starts)
        res["vae_pg_layer"] = dict(m=r["m"], lam=r["lam"], z=r["z"].numpy(), calls=r["calls"],
                                   time=time.time() - t, **metrics(spec, cc, r["m"], r["chi2"]))
        save()
    if "layers_vae" in methods and "layers_vae" not in res and "layered_param" in res:
        import layers_vae as lv
        if os.path.exists(lv.BODY_CKPT):
            t = time.time()
            bmodel, bmean, bstd = load_vae(lv.BODY_CKPT)
            z_res = encode(bmodel, img - lay_img, bmean, bstd)               # pyGIMLi minus layers
            r = lv.invert_layers_vae(data, dict(theta=thl, K=Kl), [np.zeros(bmodel.nz), z_res], xg, zg,
                                     verbose=verbose)
            m = lv.evaluate(r, _cell_samples(), S)
            res["layers_vae"] = dict(m=m, lam=r["lam"], K=r["K"], theta=r["theta"], z=r["z"],
                                     time=time.time() - t, **metrics(spec, cc, m, r["chi2"]))
            save()
    for key, kind in (("ttl_cnn", "cnn"), ("ttl_field", "field")):
        if key in methods and key not in res:
            import ttl_inversion as tt
            t = time.time()
            r = tt.invert_ttl(fwd, dobs, cc, S, kind=kind, seed=seed, verbose=verbose)
            res[key] = dict(m=r["m"], lam=0.0, calls=r["calls"], time=time.time() - t,
                            **metrics(spec, cc, r["m"], r["chi2"]))
            save()
    if "pygimli_fine" in methods and "pygimli_fine" not in res:
        F = _fine()
        t = time.time()
        pgr, _ = sp.invert_pygimli(data, F["mesh"], [300, 100, 50, 20, 10, 5, 2], 1.05, 0.3, True,
                                   verbose=verbose)
        mf = pgr["m"][cKDTree(pgr["cc"]).query(F["cc"])[1]]
        m = F["M"] @ mf
        res["pygimli_fine"] = dict(m=m, m_fine=mf.astype(np.float32), lam=pgr["lam"], time=time.time() - t,
                                   **metrics(spec, cc, m, pgr["chi2"]))
        save()
    ck = os.path.join(outdir, "vae_pg", name + ".pt")
    if ("vae_pg_fine" in methods and "vae_pg_fine" not in res and "pygimli_fine" in res
            and os.path.exists(ck)):
        F = _fine()
        t = time.time()
        fwd_f = ERTForward(data, F["mesh"])
        model, mean, std = load_vae(ck)                    # same network as vae_pg (mesh-independent)
        img_f = v.grid_from_cells(F["cc"], res["pygimli_fine"]["m_fine"].astype(float) / LN10, xg, zg)
        r = v.invert_latent(fwd_f, model, mean, std, F["S"], dobs, lams, restarts=o["restarts"],
                            verbose=verbose, z_starts=[encode(model, img_f, mean, std)])
        m = F["M"] @ r["m"]
        res["vae_pg_fine"] = dict(m=m, m_fine=np.asarray(r["m"], np.float32), lam=r["lam"], z=r["z"].numpy(),
                                  calls=r["calls"], time=time.time() - t, **metrics(spec, cc, m, r["chi2"]))
        save()
    plot_model(name, spec, res, os.path.join(outdir, "figures", name + ".png"))
    return res


def recompute_metrics(outdir=OUTB, figures=True):
    """Recompute the metrics (and figures) of all saved results with the current
    definition of the true model on the mesh - no inversion is repeated."""
    mdir = os.path.join(outdir, "models")
    for f in sorted(os.listdir(mdir)):
        fn = os.path.join(mdir, f)
        z = np.load(fn, allow_pickle=True)
        spec, res = z["spec"].item(), z["res"].item()
        if spec is None:
            continue
        for k, r in res.items():
            r.update(metrics(spec, G["cc"], r["m"], r["chi2"]))
        np.savez(fn, spec=np.array(spec, dtype=object), res=np.array(res, dtype=object),
                 m_true=true_on_mesh(spec), data_file=str(z["data_file"]), opts=z["opts"])
        if figures:
            plot_model(f[:-4], spec, res, os.path.join(outdir, "figures", f[:-4] + ".png"))
    print("metrics recomputed for", len(os.listdir(mdir)), "saved data sets")


def run_benchmark_model(spec):
    """Pool task: one test model."""
    try:
        seed = 1000 + make_suite_index(spec["name"])
        t0 = time.time()
        res = invert_all(spec["name"], OUTB, spec=spec, seed=seed,
                         data_file=os.path.join(DATA_DIR, spec["name"] + ".dat"))
        return spec["name"], f"{time.time() - t0:.0f}s  RMS " + "  ".join(
            f"{k}={res[k]['rms']:.3f}" for k in METHODS if k in res)
    except Exception as e:
        import traceback
        with open(os.path.join(OUTB, "errors.log"), "a") as fh:
            fh.write(f"=== {spec['name']}\n{traceback.format_exc()}\n")
        return spec["name"], f"ERROR {e}"


def make_suite_index(name):
    return [m["name"] for m in make_suite()].index(name)


# ── plotting ───────────────────────────────────────────────────────────────
def plot_model(name, spec, res, fn):
    """Top: true model and the four recovered models. Bottom: metrics and the
    ratio recovered / true for every method (field data: models only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    sp, cc, tri = G["sp"], G["cc"], G["tri"]
    have = [k for k in METHODS if k in res]
    true = true_on_mesh(spec) if spec else None
    ref = true if true is not None else np.concatenate([res[k]["m"] for k in have])
    norm = LogNorm(np.exp(ref.min()) / 1.5, np.exp(ref.max()) * 1.5)
    rows = 2 if spec else 1
    nc = 1 + len(have)
    fig, axs = plt.subplots(rows, nc, figsize=(4.4 * nc, 3.0 * rows + 0.6), constrained_layout=True, squeeze=False)
    if spec:
        pc = sp._cellplot(axs[0, 0], tri, np.exp(true), norm=norm, cmap="turbo")
        axs[0, 0].set_title(f"{name}\nTRUE model (area fraction on the mesh)", fontsize=9)
    else:
        axs[0, 0].axis("off")
        axs[0, 0].text(0.02, 0.5, f"{name}\n(field / user data:\nno true model)", fontsize=10,
                       transform=axs[0, 0].transAxes)
    for j, k in enumerate(have, start=1):
        r = res[k]
        pc = sp._cellplot(axs[0, j], tri, np.exp(r["m"]), norm=norm, cmap="turbo")
        extra = f", RMS={r['rms']:.3f}" if "rms" in r else ""
        axs[0, j].set_title(f"{LABELS[k]}\nchi2={r['chi2']:.2f}{extra}", fontsize=9)
        if spec:
            pr = sp._cellplot(axs[1, j], tri, np.exp(r["m"] - true), norm=LogNorm(0.1, 10), cmap="turbo")
            axs[1, j].set_title(f"ratio recovered / true: {SHORT[k]}", fontsize=9)
    if spec:
        ax = axs[1, 0]
        ax.axis("off")
        txt = ["metrics (covered zone)", f"{'method':26s}{'chi2':>6s}{'RMS':>7s}{'body':>7s}"]
        for k in have:
            r = res[k]
            be = "" if not np.isfinite(r.get("body_err", np.nan)) else f"{r['body_err']:.3f}"
            txt.append(f"{LABELS[k][:26]:26s}{r['chi2']:6.2f}{r['rms']:7.3f}{be:>7s}")
        txt += ["", "RMS = rms log10(rec/true)", "body = median |log10 err| in bodies"]
        ax.text(0.0, 1.0, "\n".join(txt), family="monospace", fontsize=7.5, va="top", transform=ax.transAxes)
    for ax in axs.ravel():
        if ax.axison:
            ax.set_xlim(*sp.XLIM); ax.set_ylim(*sp.YLIM); ax.set_aspect("equal")
            ax.plot(sp.ELEC_X, 0 * sp.ELEC_X, "v", color="k", ms=2)
            ax.tick_params(labelsize=7)
    fig.colorbar(pc, ax=axs[0].tolist(), shrink=0.85, label="resistivity (Ohm.m)")
    if spec:
        cb = fig.colorbar(pr, ax=axs[1].tolist(), shrink=0.85, label="rho_rec / rho_true")
        cb.set_ticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
        cb.set_ticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    G["sp"]._save(fig, fn, dpi=85)                      # tolerant of a PNG open in a viewer
    plt.close(fig)


def plot_overview(suite, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    x, z = np.linspace(-2, 49, 160), np.linspace(0, 20, 64)
    X, Z = np.meshgrid(x, z)
    pts = np.column_stack([X.ravel(), -Z.ravel()])
    fig, axs = plt.subplots(10, 10, figsize=(20, 9.5), constrained_layout=True)
    for ax, spec in zip(axs.ravel(), suite):
        im = ax.imshow(np.exp(eval_model(spec, pts)).reshape(X.shape), extent=(x[0], x[-1], -z[-1], 0),
                       cmap="turbo", norm=LogNorm(5, 3000), aspect="auto")
        ax.set_title(spec["name"], fontsize=5.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5, label="resistivity (Ohm.m)")
    fig.savefig(fn, dpi=110)
    plt.close(fig)


def summarize(outdir=OUTB, methods=None):
    """CSV, statistics table and figure from all saved results in outdir."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, mdir = [], os.path.join(outdir, "models")
    for f in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
        z = np.load(os.path.join(mdir, f), allow_pickle=True)
        spec, res = z["spec"].item(), z["res"].item()
        if spec is None or not all(k in res for k in CORE):
            continue
        for k in [q for q in METHODS if q in res]:
            r = res[k]
            rows.append(dict(model=spec["name"], group=spec["group"], method=k, rms=r["rms"],
                             chi2=r["chi2"], body_err=r["body_err"], lam=r["lam"], time=r["time"]))
    if not rows:
        print("no complete results yet")
        return
    with open(os.path.join(outdir, "benchmark_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {(r["model"], r["method"]): r for r in rows}
    models = sorted({r["model"] for r in rows})
    MS = [k for k in METHODS if all((m, k) in by for m in models)]    # methods done for every model
    tag = ""
    if methods:                                         # statistics restricted to a subset of methods
        MS = [k for k in MS if k in methods]
        tag = "_" + "_vs_".join(MS)
    gof = {m: by[(m, "pygimli")]["group"] for m in models}
    best = {m: min(MS, key=lambda k: by[(m, k)]["rms"]) for m in models}
    rank = {k: np.mean([sorted(MS, key=lambda q: by[(m, q)]["rms"]).index(k) + 1 for m in models])
            for k in MS}

    def line(name, ms):
        med = [np.median([by[(m, k)]["rms"] for m in ms]) for k in MS]
        nb = [sum(best[m] == k for m in ms) for k in MS]
        return f"{name:26s}{len(ms):4d}" + "".join(f"{x:10.3f}" for x in med) + "   " + "".join(f"{x:8d}" for x in nb)

    ab = dict(pygimli="pyGIMLi", sparse="sparse", vae_random="VAE-rnd", vae_pg="VAE-pg",
              vae_pg_ar="VAE-pgAR", gan="GAN", pygimli_fine="pyG-fine", vae_pg_fine="VAEpg-fine",
              vae_pg_dip="VAEpg-dip", pygimli_robust="pyG-rob",
              pygimli_dipsteer="pyG-dip", hybrid_vae_dip="Hybrid", layered_param="Layers",
              vae_pg_layer="VAE-lay", layers_vae="Lay+VAE", ttl_cnn="TTL-CNN", ttl_field="TTL-NF")
    hdr = "".join(f"{ab[k]:>10s}" for k in MS)
    lines = [f"BENCHMARK  {len(models)} models; identical noisy data for every method",
             "", "Median RMS log10(rec/true) in the covered zone (lower = better) | number of models where "
             "the method is best", f"{'group':26s}{'n':>4s}{hdr}   " + "".join(f"{ab[k][:7]:>8s}" for k in MS)]
    for g in GROUPS:
        ms = [m for m in models if gof[m] == g]
        if ms:
            lines.append(line(f"{g:3s}{GROUP_NAMES[g]}", ms))
    lines.append(line("ALL", models))
    lines += ["", f"Mean rank (1 = best of {len(MS)}): " + ", ".join(f"{LABELS[k]} {rank[k]:.2f}" for k in MS),
              "Beats pyGIMLi (share of models): " + ", ".join(
                  f"{LABELS[k]} {100 * np.mean([by[(m, k)]['rms'] < by[(m, 'pygimli')]['rms'] for m in models]):.0f}%"
                  for k in MS[1:]),
              "Fits the data (chi2 <= 1.10): " + ", ".join(
                  f"{LABELS[k]} {100 * np.mean([by[(m, k)]['chi2'] <= 1.10 for m in models]):.0f}%" for k in MS)]
    bm = [m for m in models if np.isfinite(by[(m, "pygimli")]["body_err"])]
    if bm:
        lines.append("Median |log10 error| inside the bodies (groups E-G): " + ", ".join(
            f"{LABELS[k]} {np.median([by[(m, k)]['body_err'] for m in bm]):.3f}" for k in MS))
    txt = "\n".join(lines)
    open(os.path.join(outdir, f"benchmark_summary{tag}.txt"), "w").write(txt)
    print(txt)

    fig, axs = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    gs = [g for g in GROUPS if any(gof[m] == g for m in models)]
    ax = axs[0, 0]
    for j, k in enumerate(MS):
        data = [[by[(m, k)]["rms"] for m in models if gof[m] == g] for g in gs]
        bp = ax.boxplot(data, positions=np.arange(len(gs)) + (j - (len(MS) - 1) / 2) * 0.8 / len(MS), widths=0.7 / len(MS), patch_artist=True,
                        medianprops=dict(color="black"), flierprops=dict(markersize=3))
        for p in bp["boxes"]:
            p.set_facecolor(COLORS[k])
        ax.plot([], [], "s", color=COLORS[k], label=LABELS[k])
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([f"{g}\n" + GROUP_NAMES[g].replace(" ", "\n", 1) for g in gs], fontsize=7.5)
    ax.set_ylabel("RMS log10(rec / true)"); ax.set_title("(a) Model error by test group (lower is better)")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", color="#d9d8d2")
    ax = axs[0, 1]
    bottom = np.zeros(len(gs))
    for k in MS:
        cnt = np.array([sum(best[m] == k for m in models if gof[m] == g) for g in gs])
        ax.bar(range(len(gs)), cnt, bottom=bottom, color=COLORS[k], label=LABELS[k], edgecolor="white")
        bottom += cnt
    ax.set_xticks(range(len(gs))); ax.set_xticklabels(gs)
    ax.set_ylabel("number of models"); ax.set_title("(b) Which method gives the lowest error, per group")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 0]
    p = np.array([by[(m, "pygimli")]["rms"] for m in models])
    for k in MS[1:]:
        ax.plot(p, [by[(m, k)]["rms"] for m in models], "o", ms=4, color=COLORS[k], label=LABELS[k], alpha=0.8)
    lim = [0, float(max(r["rms"] for r in rows)) * 1.05]
    ax.plot(lim, lim, "--", color="#5f5e5a", lw=1, label="equal error")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("pyGIMLi RMS"); ax.set_ylabel("method RMS")
    ax.set_title("(c) Per model: below the line = better than pyGIMLi")
    ax.legend(fontsize=8, frameon=False); ax.grid(color="#d9d8d2")
    ax = axs[1, 1]
    for k in MS:
        v_ = np.sort([by[(m, k)]["rms"] for m in models])
        ax.plot(v_, np.arange(1, len(v_) + 1) / len(v_), color=COLORS[k], lw=2, label=LABELS[k])
    ax.set_xlabel("RMS log10(rec / true)"); ax.set_ylabel("share of models with error below x")
    ax.set_title("(d) Cumulative distribution (further left / higher is better)")
    ax.legend(fontsize=8, frameon=False); ax.grid(color="#d9d8d2")
    fig.savefig(os.path.join(outdir, f"benchmark_summary{tag}.png"), dpi=120)
    plt.close(fig)
    plot_rms_per_model(by, models, MS, os.path.join(outdir, f"benchmark_rms_per_model{tag}.png"))
    plot_best_counts(by, models, MS, os.path.join(outdir, f"best_method_counts{tag}.png"))
    plot_best_counts(by, models, MS, os.path.join(outdir, f"worst_method_counts{tag}.png"), worst=True)


def plot_best_counts(by, models, MS, fn, worst=False):
    """Table: how many times each method gives the lowest (worst=True: highest)
    error, per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grp = {m: by[(m, "pygimli")]["group"] for m in models}
    pick = max if worst else min
    best = {m: pick(MS, key=lambda k: by[(m, k)]["rms"]) for m in models}
    cells, rowlab = [], []
    for g in [g for g in GROUPS if any(grp[m] == g for m in models)] + ["All"]:
        ms = [m for m in models if g == "All" or grp[m] == g]
        cells.append([str(len(ms))] + [str(sum(best[m] == k for m in ms)) for k in MS])
        rowlab.append("All models" if g == "All" else f"{g}: {GROUP_NAMES[g]}")
    cols = ["Models"] + [SHORT[k] for k in MS]
    fig, ax = plt.subplots(figsize=(2.2 + 1.7 * len(cols), 0.42 * len(cells) + 1.3))
    ax.axis("off")
    tb = ax.table(cellText=cells, rowLabels=rowlab, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d9d8d2")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eef3fb")
        if r == len(cells):
            cell.set_text_props(weight="bold")
        if r >= 1 and c >= 1:                               # winner(s) of each row
            vals = [int(x) for x in cells[r - 1][1:]]
            if int(cells[r - 1][c]) == max(vals) and max(vals) > 0:
                cell.set_facecolor("#f7d9d4" if worst else "#d6f0e4")
                cell.set_text_props(weight="bold")
    ax.set_title(f"Number of test models where each method gives the {'HIGHEST' if worst else 'lowest'} error "
                 f"(RMS log10, {len(models)} models)", fontsize=11, pad=12)
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rms_per_model(by, models, MS, fn):
    """1-D view: RMS of every method for every model (suite order, simple ->
    complex), and the difference to pyGIMLi per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = {m["name"]: i for i, m in enumerate(make_suite())}
    ms = sorted(models, key=lambda m: order.get(m, 1e9))
    x = np.arange(1, len(ms) + 1)
    grp = [by[(m, "pygimli")]["group"] for m in ms]
    fig, axs = plt.subplots(2, 1, figsize=(18, 8.5), sharex=True, constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[3, 2]))
    ref = np.array([by[(m, "pygimli")]["rms"] for m in ms])
    for k in MS:
        y = np.array([by[(m, k)]["rms"] for m in ms])
        axs[0].plot(x, y, "o-", color=COLORS[k], lw=1.4, ms=3.5, label=f"{LABELS[k]} (median {np.median(y):.3f})")
        if k != "pygimli":
            axs[1].plot(x, y - ref, "o-", color=COLORS[k], lw=1.2, ms=3, label=LABELS[k])
    axs[1].axhline(0, color="#5f5e5a", lw=1)
    # best method per model: larger marker with a dark outline
    best = [min(MS, key=lambda k: by[(m, k)]["rms"]) for m in ms]
    for k in MS:
        idx = [i for i, b in enumerate(best) if b == k]
        if not idx:
            continue
        y = np.array([by[(ms[i], k)]["rms"] for i in idx])
        axs[0].plot(x[idx], y, "o", ms=10, mfc=COLORS[k], mec="black", mew=1.2, zorder=5)
        axs[1].plot(x[idx], y - ref[idx], "o", ms=9, mfc=COLORS[k], mec="black", mew=1.2, zorder=5)
    axs[0].plot([], [], "o", ms=10, mfc="white", mec="black", mew=1.2,
                label="large marker = best method for that model")
    counts = ", ".join(f"{SHORT[k]} {sum(b == k for b in best)}" for k in MS)
    axs[0].text(0.995, 0.97, f"best per model: {counts}", ha="right", va="top", fontsize=8.5,
                transform=axs[0].transAxes, bbox=dict(fc="white", ec="#d9d8d2"))
    axs[0].set_ylabel("RMS log10(rec / true)")
    axs[0].set_title("Model error of every method for each of the 100 test models (lower is better)")
    axs[1].set_ylabel("RMS(method) - RMS(pyGIMLi)")
    axs[1].set_title("Difference to pyGIMLi per model (below 0 = better than pyGIMLi)")
    axs[1].set_xlabel("test model (suite order, simple -> complex)")
    # group bands and labels
    starts = [i for i in range(len(ms)) if i == 0 or grp[i] != grp[i - 1]]
    for j, s0 in enumerate(starts):
        s1 = starts[j + 1] if j + 1 < len(starts) else len(ms)
        for ax in axs:
            if j % 2 == 0:
                ax.axvspan(s0 + 0.5, s1 + 0.5, color="#f1f0ec", lw=0, zorder=0)
        axs[0].text((s0 + s1) / 2 + 0.5, 1.0, f"{grp[s0]}\n{GROUP_NAMES[grp[s0]]}", ha="center", va="bottom",
                    fontsize=7.5, color="#5f5e5a", transform=axs[0].get_xaxis_transform())
    for ax in axs:
        ax.grid(axis="y", color="#d9d8d2")
        ax.set_xlim(0.5, len(ms) + 0.5)
        ax.legend(fontsize=8, frameon=False, ncol=len(MS), loc="upper left")
    axs[0].set_title("Model error of every method for each of the 100 test models (lower is better)", pad=28)
    fig.savefig(fn, dpi=120)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--random-ckpt", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=None, help="threads per worker")
    ap.add_argument("--n-pg", type=int, default=None)
    ap.add_argument("--ft-steps", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--lams", default=None)
    ap.add_argument("--only", default=None, help="comma list of model-name prefixes, e.g. A_01,D3")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--methods", default=None,
                    help="with --summary: statistics for these methods only, e.g. pygimli,sparse")
    ap.add_argument("--recompute-metrics", action="store_true",
                    help="recompute metrics + figures of saved results (true model by area fraction)")
    ap.add_argument("--overview", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTB, exist_ok=True)
    suite = make_suite()
    with open(os.path.join(OUTB, "suite.json"), "w") as fh:
        json.dump(suite, fh, indent=1)
    if a.overview or not os.path.exists(os.path.join(OUTB, "suite_overview.png")):
        plot_overview(suite, os.path.join(OUTB, "suite_overview.png"))
        if a.overview:
            return
    if a.recompute_metrics:
        setup(threads=4)
        recompute_metrics()
        summarize()
        return
    if a.summary:
        summarize(methods=a.methods.split(",") if a.methods else None)
        return
    if a.only:
        suite = [s for s in suite if any(s["name"].startswith(p) for p in a.only.split(","))]
    opts = dict(random_ckpt=a.random_ckpt, threads=a.threads, n_pg=a.n_pg, ft_steps=a.ft_steps,
                restarts=a.restarts, lams=a.lams)
    ck = a.random_ckpt or RANDOM_CKPT
    if not os.path.exists(ck):
        sys.exit(f"random-prior VAE not found: {ck}  (train: vae_prior_inversion_ert.py --family general)")
    done = lambda s: os.path.exists(os.path.join(OUTB, "models", s["name"] + ".npz")) and all(
        k in np.load(os.path.join(OUTB, "models", s["name"] + ".npz"), allow_pickle=True)["res"].item()
        for k in METHODS)
    todo = [s for s in suite if not done(s)]
    print(f"{len(suite)} models selected, {len(todo)} to run, {a.workers} workers", flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers, initializer=_init_worker, initargs=(opts,)) as pool:
        for k, (name, msg) in enumerate(pool.imap_unordered(run_benchmark_model, todo), start=1):
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {name}: {msg}   ({el / 60:.0f} min, "
                  f"~{el / k * (len(todo) - k) / 60:.0f} min left)", flush=True)
    summarize()


if __name__ == "__main__":
    main()
