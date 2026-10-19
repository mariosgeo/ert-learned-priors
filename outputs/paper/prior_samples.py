"""How prior information changes the training models.

  prior_samples/random_prior_100.png      first 100 of the 200 000 random-prior training models (seed 1)
  prior_samples/guided_<model>.png        for every benchmark test: true model, pyGIMLi model with the
                                          dip-aware analysis, and the first 100 of its 20 000 guided
                                          training models (same seed as in the benchmark: 1000 + index)
  fig_prior_comparison.png / .pdf         compact version for the paper (random vs guided, 5 tests)
Training models are log10 images on the 32 x 96 grid of the inversion domain (x 0-47 m plus 2 m
margins, depth 0-20 m); exactly the ones the networks were trained on, since generation is sequential.
"""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os
import sys
import numpy as np
sys.path.insert(0, _ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
import vae_benchmark_ert as vb

OUT = os.path.join(_ROOT, "outputs")
P = os.path.join(OUT, "paper", "prior_samples")
NORM = LogNorm(5, 3000)
LN10 = np.log(10.0)
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def setup():
    vb.setup(threads=2)
    G = vb.G
    return G, G["v"], G["xg"], G["zg"]


def show(ax, img, ext, title=None, fs=6):
    im = ax.imshow(10 ** img, extent=ext, cmap="turbo", norm=NORM, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)
    return im


def grid100(fig, gs, X, ext):
    """10 x 10 thumbnails into a gridspec slot."""
    sub = gs.subgridspec(10, 10, wspace=0.04, hspace=0.06)
    for k in range(min(100, len(X))):
        im = show(fig.add_subplot(sub[k // 10, k % 10]), X[k], ext)
    return im


def overlay(ax, info, xg, zg):
    """What the dip-aware analysis read from the pyGIMLi image."""
    xm = 0.5 * (xg[0] + xg[-1])
    for i, d in enumerate(info["z_if"]):
        s = info["slopes"][i] if info.get("slopes") is not None else 0.0
        ax.plot(xg, -(d + s * (xg - xm)), "w--", lw=1.1)
    ax.plot(xg, -info["doi"], "k-", lw=0.9)
    for b in info["bodies"]:
        c, s_ = np.cos(b["theta"]), np.sin(b["theta"])
        corners = [(-b["L"] / 2, -b["W"] / 2), (b["L"] / 2, -b["W"] / 2), (b["L"] / 2, b["W"] / 2), (-b["L"] / 2, b["W"] / 2)]
        ax.add_patch(Polygon([(b["xc"] + u * c - v * s_, -(b["zc"] + u * s_ + v * c)) for u, v in corners],
                             fill=False, ec="white", lw=1.0))
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(-zg[-1], 0)


def guided_set(G, v, xg, zg, spec, n=100):
    """pyGIMLi image, dip-aware analysis and the first n guided training models of one test."""
    o = G["opts"]
    name = spec["name"]
    r = np.load(os.path.join(OUT, "vae_benchmark", "models", name + ".npz"), allow_pickle=True)
    img = v.grid_from_cells(G["cc"], r["res"].item()["pygimli"]["m"] / LN10, xg, zg)
    info = v.analyse_pg(img, G["covg"], xg, zg, dip_aware=True)
    X = v.random_models_pg(n, xg, zg, info, 1000 + vb.make_suite_index(name), o["pg_mix"], o["pg_extend"])
    Xg, Zg = np.meshgrid(xg, zg)
    true = vb.eval_model(spec, np.column_stack([Xg.ravel(), -Zg.ravel()])).reshape(Xg.shape) / LN10
    return img, info, X, true


def main():
    os.makedirs(P, exist_ok=True)
    G, v, xg, zg = setup()
    ext = (xg[0], xg[-1], -zg[-1], 0)
    # 1. random prior
    Xr = v.random_models_general(100, xg, zg, seed=1)
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(1, 1, left=0.02, right=0.9, top=0.92, bottom=0.02)
    im = grid100(fig, gs[0], Xr, ext)
    cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=cax, label="resistivity (Ω·m)")
    fig.suptitle("Random prior: the first 100 of the 200 000 training models (no information from the data)", fontsize=11)
    fig.savefig(os.path.join(P, "random_prior_100.png"), dpi=130)
    plt.close(fig)
    print("random prior done", flush=True)
    # 2. guided prior for every benchmark test
    suite = vb.make_suite()
    keep = {}
    only = ["B_05_z3-9_M-H-L", "D2_03_parallel20", "D3_09_basin", "E_07_ellipse_res_top4_2lay", "G_03_5bodies_fault"]
    for i, spec in enumerate(suite):
        if "--compare-only" in sys.argv:
            if spec["name"] in only:
                img, info, X, true = guided_set(G, v, xg, zg, spec)
                keep[spec["name"]] = (true, img, info, X)
            continue
        img, info, X, true = guided_set(G, v, xg, zg, spec)
        fig = plt.figure(figsize=(14, 9.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.1], left=0.02, right=0.9, top=0.93, bottom=0.02, hspace=0.12)
        top = gs[0].subgridspec(1, 2, wspace=0.05)
        show(fig.add_subplot(top[0]), true, ext, "true model (not used)", fs=9)
        ax = fig.add_subplot(top[1])
        show(ax, img, ext, "pyGIMLi model and its dip-aware analysis: interfaces (dashed), anomalies (boxes), "
             "depth of investigation (black)", fs=8)
        overlay(ax, info, xg, zg)
        im = grid100(fig, gs[1], X, ext)
        cax = fig.add_axes([0.92, 0.15, 0.012, 0.6])
        fig.colorbar(im, cax=cax, label="resistivity (Ω·m)")
        fig.suptitle(f"{spec['name']}: the first 100 of its 20 000 guided training models "
                     f"({100 * G['opts']['pg_mix']:.0f} % of the set are general random models)", fontsize=11)
        fig.savefig(os.path.join(P, f"guided_{i + 1:03d}_{spec['name']}.png"), dpi=110)
        plt.close(fig)
        keep[spec["name"]] = (true, img, info, X)
        print(f"{i + 1}/100 {spec['name']}", flush=True)
    # 3. compact comparison for the paper
    names = ["B_05_z3-9_M-H-L", "D2_03_parallel20", "D3_09_basin", "E_07_ellipse_res_top4_2lay", "G_03_5bodies_fault"]
    ns = 6
    fig, axs = plt.subplots(1 + len(names), 2 + ns, figsize=(17, 0.95 * (1 + len(names)) + 0.6),
                            gridspec_kw=dict(wspace=0.05, hspace=0.42))
    for j in range(2 + ns):
        if j < 2:
            axs[0, j].set_axis_off()
        else:
            show(axs[0, j], Xr[j - 2], ext)
    axs[0, 2].set_title("(a) random prior: training models (no information from the data)", loc="left", fontsize=9)
    for i, n in enumerate(names, 1):
        true, img, info, X = keep[n]
        show(axs[i, 0], true, ext, "_".join(n.split("_")[:2]) + ": true model", fs=7)
        show(axs[i, 1], img, ext, "pyGIMLi + analysis", fs=7)
        overlay(axs[i, 1], info, xg, zg)
        for j in range(ns):
            show(axs[i, 2 + j], X[j], ext)
        if i == 1:
            axs[i, 2].set_title("(b) guided prior: training models built around the pyGIMLi model of each test",
                                loc="left", fontsize=9)
    fig.colorbar(axs[1, 2].images[0], ax=axs, shrink=0.5, pad=0.01, label="resistivity (Ω·m)")
    for ext_ in ("png", "pdf"):                       # Fig. 4 of the manuscript
        fig.savefig(os.path.join(OUT, "paper", f"fig04_training_models.{ext_}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("comparison done", flush=True)


if __name__ == "__main__":
    main()
