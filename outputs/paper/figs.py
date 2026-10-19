"""Figures for the manuscript (outputs/paper/fig*.png)."""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os
import sys
import json
import numpy as np
sys.path.insert(0, _ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
from matplotlib.patches import FancyBboxPatch
import vae_benchmark_ert as vb

OUT = os.path.join(_ROOT, "outputs")
P = os.path.join(OUT, "paper")
LN10 = np.log(10.0)
COL = {"pygimli": "#2a78d6", "vae_random": "#eb6834", "vae_pg_dip": "#c2185b", "hybrid_vae_dip": "#00897b",
       "tex": "#6d4c41"}
LAB = {"pygimli": "pyGIMLi (L1, zW 0.3)", "vae_random": "VAE, random prior",
       "vae_pg_dip": "VAE, pyGIMLi-guided (dip-aware)", "hybrid_vae_dip": "Hybrid (pyGIMLi + guided VAE)"}
SHORTLAB = {"pygimli": "pyGIMLi", "vae_random": "VAE random", "vae_pg_dip": "VAE guided", "hybrid_vae_dip": "Hybrid"}
NORM = LogNorm(5, 3000)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType in PDFs (no Type 3 fonts)
S = json.load(open(os.path.join(P, "paper_stats.json")))
GR = ["A", "B", "C", "D1", "D2", "D3", "E", "F", "G"]
M3 = ["pygimli", "vae_random", "vae_pg_dip"]
M4 = M3 + ["hybrid_vae_dip"]


def save(fig, name, dpi=170, tight=False):
    """PNG (for the Word draft) and vector PDF (for the journal) of the same figure."""
    kw = dict(bbox_inches="tight") if tight else {}
    fig.savefig(os.path.join(P, name + ".png"), dpi=dpi, **kw)
    fig.savefig(os.path.join(P, name + ".pdf"), dpi=300, **kw)       # dpi only affects embedded images


def fig_overview():
    """All 100 test models (as outputs/vae_benchmark/suite_overview.png)."""
    x, z = np.linspace(-2, 49, 160), np.linspace(0, 20, 64)
    X, Z = np.meshgrid(x, z)
    pts = np.column_stack([X.ravel(), -Z.ravel()])
    fig, axs = plt.subplots(10, 10, figsize=(20, 9.5), constrained_layout=True)
    for ax, spec in zip(axs.ravel(), vb.make_suite()):
        im = ax.imshow(np.exp(vb.eval_model(spec, pts)).reshape(X.shape), extent=(x[0], x[-1], -z[-1], 0),
                       cmap="turbo", norm=NORM, aspect="auto", interpolation="nearest")
        ax.set_title(spec["name"], fontsize=5.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5, label="resistivity (Ω·m)")
    save(fig, "fig02_suite_overview", dpi=110)
    plt.close(fig)


def fig_hetero_models():
    """Six models at 0, 10, 30, 50 % within-layer variation (same field, growing amplitude)."""
    base = vb.make_suite()
    names = ["B_05_z3-9_M-H-L", "C_01_4lay_M-M-H-L", "D2_03_parallel20", "D3_08_multisine",
             "E_15_polygon_res_top8_3lay", "G_03_5bodies_fault"]
    x, z = np.linspace(-2, 49, 160), np.linspace(0, 20, 64)
    X, Z = np.meshgrid(x, z)
    pts = np.column_stack([X.ravel(), -Z.ravel()])
    cols = [("constant", None), ("±10 %", 10), ("±30 %", 30), ("±50 %", 50)]
    fig, axs = plt.subplots(len(names), 4, figsize=(12, 1.75 * len(names) + 0.5), constrained_layout=True)
    for i, n in enumerate(names):
        k = [j for j, s_ in enumerate(base) if s_["name"] == n][0]
        for c, (lab, H) in enumerate(cols):
            spec = dict(base[k])
            if H:
                spec["hetero"] = dict(sigma=float(np.log(1 + H / 100)), seed=500 + k, Lx=10.0, Lz=2.0)
            im = axs[i, c].imshow(np.exp(vb.eval_model(spec, pts)).reshape(X.shape), extent=(x[0], x[-1], -z[-1], 0),
                                  cmap="turbo", norm=NORM, aspect="auto", interpolation="nearest")
            axs[i, c].set_xticks([] if i < len(names) - 1 else [0, 12, 24, 36, 48])
            axs[i, c].set_yticks([-20, -10, 0] if c == 0 else [])
            if i == 0:
                axs[i, c].set_title(lab, fontsize=11)
            if c == 0:
                axs[i, c].set_ylabel(n, fontsize=7)
    axs[-1, 0].set_xlabel("x (m)")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="resistivity (Ω·m)")
    save(fig, "fig03_heterogeneity_models", dpi=150)
    plt.close(fig)


def fig_workflow():
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)

    def box(x, y, w, h, t, c, bold=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=c, ec="0.3", lw=1, zorder=2))
        if bold:
            ax.text(x + w / 2, y + h - 0.17, bold, ha="center", va="top", fontsize=8.5, fontweight="bold", zorder=3)
            ax.text(x + w / 2, y + (h - 0.3) / 2, t, ha="center", va="center", fontsize=7.8, zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.2, zorder=3)

    def arr(x0, y0, x1, y1, c="0.25", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.3, ls=ls, shrinkA=0, shrinkB=0))
    # per-data-set background
    ax.add_patch(Rectangle((0.05, 0.1), 11.9, 3.85, fc="#f6f6f6", ec="0.75", lw=0.8, zorder=0))
    ax.text(0.18, 3.78, "PER DATA SET", fontsize=8, color="0.4", fontweight="bold", va="top")
    # offline
    box(0.2, 4.2, 4.6, 0.85, "200 000 random model images (no forward modelling)\n→ train the VAE: 64 or 128 latent dimensions, ~40 min",
        "#fbe3d6", bold="OFFLINE, once per survey geometry")
    box(5.2, 4.2, 3.0, 0.85, "decoder = learned prior\n(random prior)", "#fbe3d6", bold="General VAE")
    arr(4.8, 4.62, 5.2, 4.62)
    # conventional inversion and guided prior
    box(0.2, 2.55, 1.55, 0.85, "apparent\nresistivities", "#e8eef7", bold="ERT data")
    box(2.1, 2.55, 2.7, 0.85, "blocky, zWeight 0.3, χ² ≤ 1.05\n~1 min", "#d6e4f5", bold="pyGIMLi inversion")
    box(5.2, 2.55, 2.6, 0.85, "dip, interfaces, anomalies,\ndepth of investigation", "#f6d5e2", bold="Analyse the image")
    box(5.2, 1.0, 2.75, 0.95, "20 000 guided training models\n→ fine-tune VAE (~2.5 min, GPU)", "#f6d5e2",
        bold="Guided prior")
    arr(1.75, 2.97, 2.1, 2.97)
    arr(4.8, 2.97, 5.2, 2.97)
    arr(6.5, 2.55, 6.5, 1.95)
    ax.plot([8.2, 8.2], [4.2, 1.72], color="0.55", ls="--", lw=1.2, zorder=1)
    arr(8.2, 1.72, 7.99, 1.72, c="0.55")
    ax.text(8.27, 3.55, "fine-tuned\nfrom", fontsize=7, color="0.45", va="center")
    # the three inversions
    box(8.75, 2.75, 3.1, 0.95, "latent Gauss–Newton search,\nno training per data set (~2 min)", "#e9e3f3",
        bold="1  Random prior")
    box(8.75, 1.45, 3.1, 0.95, "latent Gauss–Newton search\n(~5 min incl. fine-tuning)", "#e9e3f3",
        bold="2  Guided prior")
    box(8.75, 0.18, 3.1, 1.12, "pyGIMLi with the guided model\nas starting and reference model\n(fits to noise level; +10 s)",
        "#d7ecd9", bold="3  Hybrid")
    arr(7.4, 4.2, 9.2, 3.7)
    arr(7.95, 1.5, 8.75, 1.85)
    arr(10.3, 1.45, 10.3, 1.3)
    ax.annotate("", xy=(8.75, 0.45), xytext=(3.45, 2.55), zorder=1,
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.1, ls=":", shrinkA=0, shrinkB=0,
                                connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
    ax.text(3.55, 0.55, "same data, error model and mesh", fontsize=7, color="0.45")
    save(fig, "fig01_workflow", dpi=200, tight=True)
    plt.close(fig)


def _setup():
    if "tri" not in vb.G:
        vb.setup(threads=4)
    return vb.G


def fig_training():
    G = _setup()
    v, xg, zg = G["v"], G["xg"], G["zg"]
    Xr = v.random_models_general(8, xg, zg, seed=5)
    name = "D2_03_parallel20"
    r = np.load(os.path.join(OUT, "vae_benchmark", "models", name + ".npz"), allow_pickle=True)
    img = v.grid_from_cells(G["cc"], r["res"].item()["pygimli"]["m"] / LN10, xg, zg)
    info = v.analyse_pg(img, G["covg"], xg, zg, dip_aware=True)
    Xg = v.random_models_pg(6, xg, zg, info, seed=3)
    ext = (xg[0], xg[-1], -zg[-1], 0)
    fig, axs = plt.subplots(3, 4, figsize=(12, 4.8), constrained_layout=True)
    for ax, x in zip(axs[0], Xr[:4]):
        im = ax.imshow(10 ** x, extent=ext, cmap="turbo", norm=NORM, aspect="auto")
    axs[0, 0].set_title("(a) random-prior training models (4 of 200 000)", loc="left")
    tm = vb.true_on_mesh([s for s in vb.make_suite() if s["name"] == name][0])
    axs[1, 0].imshow(10 ** v.grid_from_cells(G["cc"], tm / LN10, xg, zg), extent=ext, cmap="turbo", norm=NORM,
                     aspect="auto")
    axs[1, 0].set_title("(b) true model D2_03 (not used)", loc="left")
    ax = axs[1, 1]
    ax.imshow(10 ** img, extent=ext, cmap="turbo", norm=NORM, aspect="auto")
    xm = 0.5 * (xg[0] + xg[-1])
    for d, s in zip(info["z_if"], info["slopes"]):
        ax.plot(xg, -(d + s * (xg - xm)), "w--", lw=1.2)
    ax.plot(xg, -info["doi"], "k-", lw=1)
    ax.set_title("(c) its pyGIMLi model + analysis", loc="left")
    for ax, x in zip(list(axs[1, 2:]) + list(axs[2]), Xg[:6]):
        ax.imshow(10 ** x, extent=ext, cmap="turbo", norm=NORM, aspect="auto")
    axs[1, 2].set_title("(d) guided training models", loc="left")
    for ax in axs.ravel():
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([])
        ax.set_yticks([])
    axs[2, 0].set_xticks([0, 24, 47])
    axs[2, 0].set_yticks([-20, -10, 0])
    axs[2, 0].set_xlabel("x (m)")
    axs[2, 0].set_ylabel("elevation (m)")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="resistivity (Ω·m)")
    save(fig, "fig04_training_models", dpi=170)
    plt.close(fig)


def fig_vae_demo():
    """How the general (random-prior) VAE works, shown on benchmark test models it was never trained on:
    (a) encode -> 64 numbers -> decode; (b) walking through the latent space between two test models;
    (c) random latent vectors decoded. Also writes the reconstruction error of all 100 test models."""
    import torch
    G = _setup()
    v, xg, zg = G["v"], G["xg"], G["zg"]
    model, mean, std = vb.load_vae(vb.RANDOM_CKPT)
    model = model.float().eval()
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), -Z.ravel()])
    suite = {s["name"]: s for s in vb.make_suite()}

    def image(name):                                   # true model as a 32 x 96 log10 image
        return (vb.eval_model(suite[name], pts) / LN10).reshape(X.shape)

    def decode(z):
        with torch.no_grad():
            return model.decode(torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)).numpy()[:, 0] * std + mean

    def rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    recon = {n: rms(image(n), decode(vb.encode(model, image(n), mean, std))[0]) for n in suite}
    json.dump(dict(median=float(np.median(list(recon.values()))), p90=float(np.percentile(list(recon.values()), 90)),
                   per_model=recon), open(os.path.join(P, "vae_reconstruction.json"), "w"), indent=1)
    ext = (xg[0], xg[-1], -zg[-1], 0)
    names = ["B_04_z2-6_L-H-L", "D1_05_dip20_H-L", "D3_08_multisine", "E_07_ellipse_res_top4_2lay", "G_03_5bodies_fault"]
    fig = plt.figure(figsize=(12.5, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 5)

    def show(ax, img, title):
        im = ax.imshow(10 ** img, extent=ext, cmap="turbo", norm=NORM, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        return im
    for j, n in enumerate(names):
        im = show(fig.add_subplot(gs[0, j]), image(n), ("(a) " if j == 0 else "") + f"test model {n.split('_')[0]}_{n.split('_')[1]}")
        z = vb.encode(model, image(n), mean, std)
        show(fig.add_subplot(gs[1, j]), decode(z)[0], f"encoded (64 numbers) → decoded\nRMS {recon[n]:.3f}")
    a_, b_ = "D1_05_dip20_H-L", "E_07_ellipse_res_top4_2lay"
    za, zb = vb.encode(model, image(a_), mean, std), vb.encode(model, image(b_), mean, std)
    for j, t in enumerate(np.linspace(0, 1, 5)):
        show(fig.add_subplot(gs[2, j]), decode((1 - t) * za + t * zb)[0],
             ("(b) " if j == 0 else "") + f"z = {1 - t:.2f}·z(D1_05) + {t:.2f}·z(E_07)")
    rng = np.random.default_rng(4)
    u = rng.standard_normal(model.nz)
    u /= np.linalg.norm(u)                              # one random direction in the latent space
    radii = [0, 2, 4, 6, 8]
    for j, x in enumerate(decode(np.outer(radii, u))):
        show(fig.add_subplot(gs[3, j]), x, ("(c) " if j == 0 else "") + f"random direction, |z| = {radii[j]}")
    fig.colorbar(im, ax=fig.axes, shrink=0.5, label="resistivity (Ω·m)")
    save(fig, "fig05_vae_demo", dpi=170)
    plt.close(fig)
    print("reconstruction RMS of the 100 test models: median", np.median(list(recon.values())))


def _tri():
    G = _setup()
    nodes, tris = G["tri"]
    return Triangulation(nodes[:, 0], nodes[:, 1], tris)


def _panel(ax, tri, vals, title):
    im = ax.tripcolor(tri, np.exp(vals), cmap="turbo", norm=NORM, shading="flat")
    ax.set_xlim(-1, 48)
    ax.set_ylim(-14, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_examples():
    tri = _tri()
    names = ["B_05_z3-9_M-H-L", "D1_05_dip20_H-L", "D3_09_basin", "E_07_ellipse_res_top4_2lay",
             "F_05_polygon+ellipse_homog", "G_03_5bodies_fault"]
    fig, axs = plt.subplots(len(names), 5, figsize=(14.5, 1.0 * len(names) + 0.5), constrained_layout=True)
    for i, n in enumerate(names):
        z = np.load(os.path.join(OUT, "vae_benchmark", "models", n + ".npz"), allow_pickle=True)
        r = z["res"].item()
        im = _panel(axs[i, 0], tri, z["m_true"], f"{n}: true model")
        for j, k in enumerate(M4, 1):
            _panel(axs[i, j], tri, r[k]["m"], f"{SHORTLAB[k]}: RMS {r[k]['rms']:.3f}, χ² {r[k]['chi2']:.2f}")
    axs[-1, 0].set_xticks([0, 24, 47])
    axs[-1, 0].set_yticks([-12, -6, 0])
    axs[-1, 0].set_xlabel("x (m)")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.4, label="resistivity (Ω·m)")
    save(fig, "fig06_examples", dpi=170)
    plt.close(fig)


def fig_stats():
    d = S["vae_benchmark"]
    by = {}
    for f in os.listdir(os.path.join(OUT, "vae_benchmark", "models")):
        r = np.load(os.path.join(OUT, "vae_benchmark", "models", f), allow_pickle=True)["res"].item()
        by[f[:-4]] = {k: r[k]["rms"] for k in M4}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 1.4, 1.1]))
    ax = axs[0]
    cm = plt.get_cmap("tab10")
    for gi, g in enumerate(GR):
        ms = [m for m in by if m.split("_")[0] == g]
        ax.scatter([by[m]["pygimli"] for m in ms], [by[m]["vae_pg_dip"] for m in ms], s=18, color=cm(gi),
                   label=g, edgecolor="white", lw=0.4)
    ax.plot([0.03, 0.7], [0.03, 0.7], color="0.4", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 0.7)
    ax.set_ylim(0.03, 0.7)
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4]))
        a_.set_minor_locator(NullLocator())
        a_.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel("pyGIMLi RMS (log10 units)")
    ax.set_ylabel("guided VAE RMS (log10 units)")
    ax.set_title(f"(a) per model; below the line: VAE better ({100 * d['vae_pg_dip']['wins']:.0f} %)", loc="left")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper left")
    ax = axs[1]
    w = 0.2
    for j, k in enumerate(M4):
        ax.bar(np.arange(len(GR)) + (j - 1.5) * w, [d[k]["groups"][g] for g in GR], w * 0.92, color=COL[k], label=LAB[k])
    ax.set_xticks(range(len(GR)))
    ax.set_xticklabels(GR)
    ax.set_ylabel("median RMS (log10 units)")
    ax.set_title("(b) median error per model group", loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax = axs[2]
    for j, k in enumerate(["vae_pg_dip", "hybrid_vae_dip"]):
        wins = [d[k]["group_wins"][g] for g in GR]
        ax.barh(np.arange(len(GR)) + (j - 0.5) * 0.4, [100 * x for x in wins], 0.38, color=COL[k], label=SHORTLAB[k])
    ax.axvline(50, color="0.4", lw=1, ls="--")
    ax.set_yticks(range(len(GR)))
    ax.set_yticklabels(GR)
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlim(0, 105)
    ax.set_xlabel("models better than pyGIMLi (%)")
    ax.set_title("(c) where the learned prior helps", loc="left")
    save(fig, "fig07_benchmark_statistics", dpi=170)
    plt.close(fig)


def fig_structure():
    import csv
    rows = list(csv.DictReader(open(os.path.join(OUT, "structure_metrics", "vae_benchmark_structure.csv"))))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    spec = [("rms", "(a) RMS, log10 (lower is better)"), ("ssim", "(b) SSIM (higher is better)"),
            ("if_err", "(c) interface depth error, m (lower)"), ("body_iou", "(d) body IoU (higher is better)")]
    for ax, (q, lab) in zip(axs, spec):
        data = []
        for k in M4:
            x = [float(r[q]) for r in rows if r["method"] == k and r[q] not in ("", "nan")]
            data.append([v for v in x if np.isfinite(v)])
        b = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
        for p_, k in zip(b["boxes"], M4):
            p_.set_facecolor(COL[k])
            p_.set_alpha(0.8)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["pyGIMLi", "VAE\nrandom", "VAE\nguided", "Hybrid"])
        ax.set_title(lab, loc="left")
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    save(fig, "fig08_structure_metrics", dpi=170)
    plt.close(fig)


def fig_hetero(tuned=None):
    lv = ["0", "10", "30", "50"]
    su = {"0": "vae_benchmark", "10": "vae_benchmark_het10", "30": "vae_benchmark_het30", "50": "vae_benchmark_het50"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    x = np.arange(4)
    for k in M4:
        if not all(k in S[su[l]] for l in lv):              # e.g. hybrid before its heterogeneous runs
            continue
        axs[0].plot(x, [S[su[l]][k]["median"] for l in lv], "o-", color=COL[k], lw=2, label=LAB[k])
        axs[1].plot(x, [S[su[l]][k]["ssim"] for l in lv], "o-", color=COL[k], lw=2)
        if k != "pygimli":
            axs[2].plot(x, [100 * S[su[l]][k]["wins"] for l in lv], "o-", color=COL[k], lw=2)
    xt = [1, 2, 3]
    axs[0].plot(xt, [S[f"vae_benchmark_het{l}_tex"]["vae_pg_dip"]["median"] for l in lv[1:]], "s--",
                color=COL["tex"], label="VAE guided, textured training models")
    axs[1].plot(xt, [S[f"vae_benchmark_het{l}_tex"]["vae_pg_dip"]["ssim"] for l in lv[1:]], "s--", color=COL["tex"])
    axs[2].plot(xt, [100 * S[f"vae_benchmark_het{l}_tex"]["vae_pg_dip"]["wins"] for l in lv[1:]], "s--",
                color=COL["tex"])
    if tuned:
        axs[0].plot(tuned["x"], tuned["rms"], "D:", color="0.15", label=tuned["label"])
        axs[2].plot(tuned["x"], tuned["wins"], "D:", color="0.15")
    axs[2].axhline(50, color="0.5", ls="--", lw=1)
    for ax, t in zip(axs, ["(a) median RMS (log10 units)", "(b) median SSIM", "(c) models better than pyGIMLi (%)"]):
        ax.set_xticks(x)
        ax.set_xticklabels(["0 %", "±10 %", "±30 %", "±50 %"])
        ax.set_title(t, loc="left")
        ax.set_xlabel("variation inside layers and bodies")
        ax.grid(color="0.92")
    axs[0].legend(fontsize=7, frameon=False)
    save(fig, "fig09_heterogeneity", dpi=170)
    plt.close(fig)


def fig_hetero_example(name="F_01_rect+dyke_homog"):
    tri = _tri()
    levels = [("0 %", "vae_benchmark"), ("±10 %", "vae_benchmark_het10"), ("±30 %", "vae_benchmark_het30"),
              ("±50 %", "vae_benchmark_het50")]
    res = [np.load(os.path.join(OUT, su, "models", name + ".npz"), allow_pickle=True) for _, su in levels]
    rows = ["pygimli", "vae_pg_dip"] + (["hybrid_vae_dip"] if all("hybrid_vae_dip" in z["res"].item() for z in res) else [])
    fig, axs = plt.subplots(1 + len(rows), 4, figsize=(12, 1.25 * (1 + len(rows)) + 0.3), constrained_layout=True)
    for j, ((l, _), z) in enumerate(zip(levels, res)):
        r = z["res"].item()
        im = _panel(axs[0, j], tri, z["m_true"], f"true model, {l}")
        for i, k in enumerate(rows, 1):
            _panel(axs[i, j], tri, r[k]["m"], f"{SHORTLAB[k]}: RMS {r[k]['rms']:.3f}")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="resistivity (Ω·m)")
    save(fig, "fig10_heterogeneity_example", dpi=170)
    plt.close(fig)


def fig_where():
    su = [("0 %", "vae_benchmark"), ("±10 %", "vae_benchmark_het10"), ("±30 %", "vae_benchmark_het30"),
          ("±50 %", "vae_benchmark_het50")]
    keys = ["vae_pg_dip"] + (["hybrid_vae_dip"] if all("hybrid_vae_dip" in S[s] for _, s in su) else [])
    fig, axs = plt.subplots(1, len(keys), figsize=(4.0 * len(keys) + 1.6, 4.2), constrained_layout=True, squeeze=False)
    cnt = {g: sum(s["group"] == g for s in vb.make_suite()) for g in GR}
    for ax, k in zip(axs[0], keys):
        M = np.array([[S[s][k]["group_wins"][g] for _, s in su] for g in GR]) * 100
        im = ax.imshow(M, cmap="RdBu", vmin=0, vmax=100, aspect="auto")
        for i in range(len(GR)):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j] - 50) > 30 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels([a for a, _ in su])
        ax.set_yticks(range(len(GR)))
        ax.set_yticklabels([f"{g}: {vb.GROUP_NAMES[g]} (n={cnt[g]})" for g in GR] if k == keys[0] else [], fontsize=8)
        ax.set_xlabel("variation inside layers and bodies")
        ax.set_title(f"({'ab'[keys.index(k)]}) {SHORTLAB[k]}", loc="left")
    fig.colorbar(im, ax=axs[0].tolist(), label="models better than pyGIMLi (%)")
    save(fig, "fig11_where_it_helps", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig_workflow", "fig_overview", "fig_hetero_models", "fig_examples", "fig_stats", "fig_structure",
                              "fig_hetero", "fig_hetero_example", "fig_where"]:
        globals()[f]()
        print("done", f, flush=True)
