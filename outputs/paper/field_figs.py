"""Field-data figures of the manuscript (estimated data errors, outputs/real_data/auto_error).

  fig14_field_models   both lines: pyGIMLi, guided VAE and hybrid on the 64- and the 128-dimensional (plain) prior
  fig15_field_repeats  both lines: pyGIMLi, mean and standard deviation of the hybrid over five repeats
                       (strict guided prior, new training models and fine-tuning per seed)
Opacity: pyGIMLi coverage() as in pyGIMLi's showModel; dashed: standardizedCoverage() (coverage > 0.01).
"""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os
import sys
import glob
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable

ROOT = _ROOT
sys.path.insert(0, ROOT)
from real_data_inversion import coverage_alpha, COV_CUT          # noqa: E402

R = os.path.join(ROOT, "outputs", "real_data", "auto_error")
P = os.path.join(ROOT, "outputs", "paper")
LN10 = np.log(10.0)
LINES = ["g4_02", "n2dd"]
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def load(fn):
    z = np.load(fn, allow_pickle=True)
    info = z["info"].item()
    f = info["f"]
    cc = z["cc"] / f
    verts = (z["tri_nodes"] / f)[z["tri_cells"]]
    cv = z["res"].item()["coverage"]
    idx = cKDTree(cv["cc"]).query(cc)[1]
    cov = cv["cov"][idx]
    alpha = coverage_alpha(cv["cov"])[idx]
    alpha = np.where(cov > COV_CUT, alpha, np.minimum(alpha, 0.3))
    return z["res"].item(), info, cc, verts, cov, alpha


def panel(ax, verts, cc, cov, alpha, vals, cmap, nrm, info, title):
    cm = matplotlib.colormaps[cmap]
    col = cm(nrm(vals))
    col[:, :3] = alpha[:, None] * col[:, :3] + (1 - alpha[:, None])
    ax.add_collection(PolyCollection(verts, facecolors=col, edgecolors=col, linewidths=0.25))
    ax.tricontour(cc[:, 0], cc[:, 1], cov, levels=[COV_CUT], colors="k", linewidths=0.6, linestyles="--")
    L = info["length"]
    e = np.linspace(0, L, info["n_elec"])
    ax.plot(e, np.zeros_like(e), "kv", ms=1.8)
    ax.set_xlim(-0.03 * L, 1.03 * L)
    ax.set_ylim(-0.42 * L, 0)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    return ScalarMappable(norm=nrm, cmap=cm)


def norm_for(res, keys, cov):
    allm = np.concatenate([res[k]["m"][cov > COV_CUT] for k in keys]) / LN10
    return LogNorm(10 ** np.percentile(allm, 1), 10 ** np.percentile(allm, 99))


def fig_models():
    rows = [("pygimli", "pyGIMLi"), ("vae_pg_dip", "guided VAE, 64"), ("hybrid_vae_dip", "hybrid, 64"),
            ("vae_pg_dip_p128", "guided VAE, 128"), ("hybrid_vae_dip_p128", "hybrid, 128")]
    fig, axs = plt.subplots(len(rows), 2, figsize=(10, 9.2), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[13.8, 23.5]))
    for j, name in enumerate(LINES):
        res, info, cc, verts, cov, alpha = load(os.path.join(R, f"{name}_results.npz"))
        nrm = norm_for(res, [k for k, _ in rows], cov)
        for i, (k, lab) in enumerate(rows):
            sm = panel(axs[i, j], verts, cc, cov, alpha, np.exp(res[k]["m"]), "turbo", nrm, info,
                       f"({'abcdefghij'[2 * i + j]}) {name}, {lab}: χ² = {res[k]['chi2']:.2f}")
            if j == 0:
                axs[i, j].set_ylabel("depth (m)")
        axs[-1, j].set_xlabel("x (m)")
        fig.colorbar(sm, ax=axs[:, j].tolist(), shrink=0.35, aspect=25, location="bottom", pad=0.02,
                     label=f"{name}: resistivity (Ω·m)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(P, f"fig14_field_models.{ext}"), dpi=170)
    plt.close(fig)


def fig_repeats():
    key = "hybrid_vae_dip_strict"
    fig, axs = plt.subplots(2, 3, figsize=(11, 4.6), constrained_layout=True)
    for i, name in enumerate(LINES):
        runs = sorted(glob.glob(os.path.join(R, "repeats", "seed*", f"{name}_results.npz")))
        res, info, cc, verts, cov, alpha = load(runs[0])
        M = np.array([np.load(f, allow_pickle=True)["res"].item()[key]["m"] for f in runs]) / LN10
        chi = [np.load(f, allow_pickle=True)["res"].item()[key]["chi2"] for f in runs]
        nrm = norm_for(res, ["pygimli", key], cov)
        sm = panel(axs[i, 0], verts, cc, cov, alpha, np.exp(res["pygimli"]["m"]), "turbo", nrm, info,
                   f"({'ad'[i]}) {name}, pyGIMLi")
        panel(axs[i, 1], verts, cc, cov, alpha, 10 ** M.mean(0), "turbo", nrm, info,
              f"({'be'[i]}) hybrid, mean of {len(runs)} repeats (χ² {min(chi):.2f}–{max(chi):.2f})")
        sd = M.std(0, ddof=1)
        s2 = panel(axs[i, 2], verts, cc, cov, alpha, sd, "magma_r", Normalize(0, 0.5), info,
                   f"({'cf'[i]}) hybrid, standard deviation")
        fig.colorbar(sm, ax=axs[i, :2].tolist(), shrink=0.8, pad=0.01, label="Ω·m")
        fig.colorbar(s2, ax=axs[i, 2], shrink=0.8, pad=0.01, label="σ of log10 ρ")
        axs[i, 0].set_ylabel("depth (m)")
        ok = cov > COV_CUT
        print(f"{name}: hybrid sigma inside DOI {np.sqrt(np.mean(sd[ok] ** 2)):.3f}, below "
              f"{np.sqrt(np.mean(sd[~ok] ** 2)):.3f}; chi2 {min(chi):.2f}-{max(chi):.2f}")
    for ax in axs[-1]:
        ax.set_xlabel("x (m)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(P, f"fig15_field_repeats.{ext}"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_models()
    fig_repeats()
    print("done")
