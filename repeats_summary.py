"""Run-to-run variability of the guided VAE and the hybrid on the field data.

Reads outputs/real_data/<run>/repeats/seed*/ (real_data_inversion.py --seed N: new guided training
models and fine-tuning per seed, same pyGIMLi model and data error) and writes to <run>/repeats/:
  <name>_repeats.png/.pdf    pyGIMLi; mean and standard deviation of log10 resistivity over the seeds
                             for the guided VAE and the hybrid; every seed of the hybrid
  repeats_table.md           chi2, RMS (%) and lambda of every seed; spread inside / below the DOI
Opacity and dashed line: pyGIMLi coverage() and standardizedCoverage() (coverage > 0.01).
"""
import os
import glob
import sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from real_data_inversion import FILES, LAB, COV_CUT, coverage_alpha

LN10 = np.log(10.0)
RUN = sys.argv[1] if len(sys.argv) > 1 else "auto_error"
R = os.path.join("outputs", "real_data", RUN, "repeats")
KEYS = ["vae_pg_dip_strict", "hybrid_vae_dip_strict"]
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def load(name):
    runs = []
    for d in sorted(glob.glob(os.path.join(R, "seed*")), key=lambda p: int(p.rsplit("seed", 1)[1])):
        fn = os.path.join(d, f"{name}_results.npz")
        if os.path.exists(fn):
            z = np.load(fn, allow_pickle=True)
            if all(k in z["res"].item() for k in KEYS):
                runs.append((int(d.rsplit("seed", 1)[1]), z))
    return runs


def main():
    rows = []
    for name in FILES:
        runs = load(name)
        if len(runs) < 2:
            print(f"{name}: {len(runs)} finished repeats, skipped")
            continue
        z0 = runs[0][1]
        info, f, err, dobs = z0["info"].item(), z0["info"].item()["f"], z0["info"].item()["err"], z0["dobs"]
        L = info["length"]
        cc, verts = z0["cc"] / f, (z0["tri_nodes"] / f)[z0["tri_cells"]]
        r0 = z0["res"].item()
        cv = r0["coverage"]
        idx = cKDTree(cv["cc"]).query(cc)[1]
        cov = cv["cov"][idx]
        ok = cov > COV_CUT
        alpha = np.where(ok, coverage_alpha(cv["cov"])[idx], np.minimum(coverage_alpha(cv["cov"])[idx], 0.3))
        M = {k: np.array([z["res"].item()[k]["m"] for _, z in runs]) / LN10 for k in KEYS}
        for s, z in runs:
            for k in KEYS:
                q = z["res"].item()[k]
                rows.append((name, s, LAB[k], q["chi2"], 100 * np.sqrt(np.mean((q["resp"] / dobs - 1) ** 2)), q["lam"]))
        sd = {k: M[k].std(0, ddof=1) for k in KEYS}
        for k in KEYS:
            rows.append((name, "spread", LAB[k], np.sqrt(np.mean(sd[k][ok] ** 2)), np.sqrt(np.mean(sd[k][~ok] ** 2)), None))
        # figure
        pg = r0["pygimli"]["m"] / LN10
        allm = np.concatenate([pg[ok]] + [M[k][:, ok].ravel() for k in KEYS])
        norm = LogNorm(10 ** np.percentile(allm, 1), 10 ** np.percentile(allm, 99))
        snorm = Normalize(0, 0.5)
        panels = [("pyGIMLi", pg, "turbo", norm),
                  (f"guided VAE 64 (strict prior): mean of {len(runs)} repeats", M["vae_pg_dip_strict"].mean(0), "turbo", norm),
                  ("guided VAE 64 (strict prior): standard deviation", sd["vae_pg_dip_strict"], "magma_r", snorm),
                  (f"hybrid: mean of {len(runs)} repeats", M["hybrid_vae_dip_strict"].mean(0), "turbo", norm),
                  ("hybrid: standard deviation", sd["hybrid_vae_dip_strict"], "magma_r", snorm)]
        panels += [(f"hybrid, seed {s}: χ² = {z['res'].item()['hybrid_vae_dip_strict']['chi2']:.2f}",
                    z["res"].item()["hybrid_vae_dip_strict"]["m"] / LN10, "turbo", norm) for s, z in runs]
        ncol = 2
        nrow = (len(panels) + 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(11, 1.75 * nrow + 0.8), constrained_layout=True)
        axs = axs.ravel()
        elec = np.linspace(0, L, info["n_elec"])
        for ax, (title, val, cmap, nrm) in zip(axs, panels):
            cm = matplotlib.colormaps[cmap]
            col = cm(nrm(10 ** val if cmap == "turbo" else val))
            col[:, :3] = alpha[:, None] * col[:, :3] + (1 - alpha[:, None])
            ax.add_collection(PolyCollection(verts, facecolors=col, edgecolors=col, linewidths=0.25))
            ax.tricontour(cc[:, 0], cc[:, 1], cov, levels=[COV_CUT], colors="k", linewidths=0.6, linestyles="--")
            ax.plot(elec, np.zeros_like(elec), "kv", ms=2)
            ax.set_xlim(-0.05 * L, 1.05 * L)
            ax.set_ylim(-0.45 * L, 0)
            ax.set_aspect("equal")
            ax.set_title(title, loc="left", fontsize=8)
            fig.colorbar(ScalarMappable(nrm, cm), ax=ax, shrink=0.85, pad=0.01,
                         label="Ω·m" if cmap == "turbo" else "σ log10 ρ")
        for ax in axs[len(panels):]:
            ax.set_axis_off()
        fig.suptitle(f"{name}: {len(runs)} repeats of the guided VAE and the hybrid (new training models and "
                     f"fine-tuning per seed), data error {err:.1%}\nopacity: pyGIMLi coverage();  dashed: "
                     f"pyGIMLi standardizedCoverage() (coverage > {COV_CUT:g})", fontsize=8.5)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(R, f"{name}_repeats.{ext}"), dpi=160)
        plt.close(fig)
    md = ["| line | seed | method | χ² | RMS (%) | λ |", "|---|---|---|---|---|---|"]
    for n, s, k, a, b, c in rows:
        if s == "spread":
            md.append(f"| {n} | **spread** | {k} | σ covered {a:.3f} | σ below DOI {b:.3f} | |")
        else:
            md.append(f"| {n} | {s} | {k} | {a:.2f} | {b:.2f} | {c:g} |")
    open(os.path.join(R, "repeats_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
