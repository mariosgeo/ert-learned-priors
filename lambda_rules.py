"""
lambda_rules.py
===============
Data-only rules for choosing the regularization strength lambda, tested on the
saved lambda paths of the tuning study (outputs/vae_benchmark/tuning/*.npz) for
all 100 models - no inversion is run. Every rule uses only the data misfit and
the recovered models (never the truth); the truth is used only to score them.

Methods at their tuned settings: pyGIMLi (zWeight 0.3, blocky), sparse GN
(L1, aniso 2, zWeight 0.3, lambda path without refinement), VAE random prior,
VAE pyGIMLi-guided (extension 1).

Rules
  discrepancy_t   first lambda (large -> small) with chi2 <= t, else the stage
                  before chi2 stops improving by >= 5 % (t = 1.05 is the
                  current rule; 1.5, 2, 3 allow for modelling error)
  plateau_p       first lambda after which a further decrease improves chi2 by
                  less than p (10 %, 20 %)
  lcurve          corner (maximum curvature) of log chi2 versus log model
                  roughness along the lambda path
  quasi_opt       lambda where the model changes least between successive
                  lambdas (quasi-optimality criterion)
  oracle          best lambda per model (needs the truth; upper bound)

    python lambda_rules.py   -> outputs/vae_benchmark/lambda_rules.txt / .csv / .png
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vae_benchmark_ert as vb                     # noqa: E402

TUNED = {"pygimli": "zW0.3_blocky", "sparse": "l1_aniso2_zW0.3",
         "vae_random": "general200k", "vae_pg": "extend1"}
OUT = vb.OUTB


def roughness_pairs(cc):
    """Neighbouring cell pairs (centres closer than 1.6 m) for a model-roughness measure."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(cc).query_pairs(1.6, output_type="ndarray")
    d = np.hypot(*(cc[pairs[:, 0]] - cc[pairs[:, 1]]).T)
    return pairs, d


def rules(path, pairs, dist, roi):
    """Index chosen by every rule along one lambda path (ordered large -> small lambda)."""
    chi = np.array([p["chi2"] for p in path])
    out = {}
    for t in (1.05, 1.5, 2.0, 3.0):
        k = len(path) - 1
        for i in range(len(path)):
            if chi[i] <= t:
                k = i
                break
            if i > 0 and chi[i] > 0.95 * chi[i - 1]:
                k = i - 1
                break
        out[f"discrepancy_{t:g}"] = k
    for p_ in (0.10, 0.20):
        k = len(path) - 1
        for i in range(len(path) - 1):
            if (chi[i] - chi[i + 1]) / chi[i] < p_:
                k = i
                break
        out[f"plateau_{int(p_ * 100)}"] = k
    m = [np.asarray(p["m"], float) for p in path]
    rough = np.array([np.mean(np.abs(q[pairs[:, 0]] - q[pairs[:, 1]]) / dist) for q in m])
    if len(path) >= 3:
        x, y = np.log(rough + 1e-12), np.log(chi)
        # discrete curvature of the (x, y) polyline at interior points (Menger)
        curv = []
        for i in range(1, len(path) - 1):
            a, b, c = np.array([x[i - 1], y[i - 1]]), np.array([x[i], y[i]]), np.array([x[i + 1], y[i + 1]])
            area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            den = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
            curv.append(2 * area2 / den if den > 0 else 0.0)
        out["lcurve"] = 1 + int(np.argmax(curv))
        change = [np.sqrt(np.mean((m[i + 1][roi] - m[i][roi]) ** 2)) for i in range(len(path) - 1)]
        out["quasi_opt"] = int(np.argmin(change))
    else:
        out["lcurve"] = out["quasi_opt"] = out["discrepancy_1.05"]
    out["oracle"] = int(np.argmin([p["rms"] for p in path]))
    return out


def main():
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vb.setup(threads=2)
    cc = vb.G["cc"]
    pairs, dist = roughness_pairs(cc)
    roi = vb.G["sp"].roi_mask(cc)
    tdir = os.path.join(OUT, "tuning")
    rows, err = [], {}
    names = sorted(f[:-4] for f in os.listdir(tdir) if f.endswith(".npz"))
    for n in names:
        z = np.load(os.path.join(tdir, n + ".npz"), allow_pickle=True)
        R, spec = z["R"].item(), z["spec"].item()
        for meth, cfg in TUNED.items():
            path = R[meth][cfg]
            for e in path:                                   # score with the current truth definition
                e.update(vb.metrics(spec, cc, np.asarray(e["m"], float), e["chi2"]))
            for rule, k in rules(path, pairs, dist, roi).items():
                err.setdefault((meth, rule), []).append(path[k]["rms"])
                rows.append(dict(model=n, group=spec["group"], method=meth, rule=rule,
                                 lam=path[k]["lam"], chi2=path[k]["chi2"], rms=path[k]["rms"]))
    with open(os.path.join(OUT, "lambda_rules.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    R_ = sorted({r["rule"] for r in rows}, key=lambda s: (s == "oracle", s))
    L = [f"LAMBDA-SELECTION RULES on the saved lambda paths ({len(names)} models; data-only rules, truth only for scoring)",
         "", "Median RMS log10(rec/true) | in brackets: share of the gap current rule -> oracle that the rule closes",
         f"{'rule':18s}" + "".join(f"{k:>20s}" for k in TUNED)]
    for rule in R_:
        line = f"{rule:18s}"
        for meth in TUNED:
            cur, ora = np.median(err[(meth, "discrepancy_1.05")]), np.median(err[(meth, "oracle")])
            v = np.median(err[(meth, rule)])
            closed = 100 * (cur - v) / (cur - ora) if cur > ora else 0
            line += f"{v:12.3f} ({closed:+4.0f}%)"
        L.append(line)
    L += ["", "Paired against the current rule, same method (models better / 100, Wilcoxon p):"]
    for rule in R_:
        if rule in ("discrepancy_1.05", "oracle"):
            continue
        L.append(f"  {rule:16s}" + "".join(
            f"  {meth}: {np.sum(np.array(err[(meth, rule)]) < np.array(err[(meth, 'discrepancy_1.05')])):3d} "
            f"p={stats.wilcoxon(err[(meth, rule)], err[(meth, 'discrepancy_1.05')]).pvalue:.2g}" for meth in TUNED))
    # best data-only rule per method, and the VAE vs pyGIMLi comparison under each rule
    L += ["", "VAE pyGIMLi-guided vs pyGIMLi under the SAME rule (models better / 100, Wilcoxon p):"]
    for rule in R_:
        a, b = np.array(err[("vae_pg", rule)]), np.array(err[("pygimli", rule)])
        L.append(f"  {rule:16s} {np.median(a):.3f} vs {np.median(b):.3f}   {np.sum(a < b):3d}/100   p={stats.wilcoxon(a, b).pvalue:.2g}")
    txt = "\n".join(L)
    open(os.path.join(OUT, "lambda_rules.txt"), "w").write(txt)
    print(txt)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for j, meth in enumerate(TUNED):
        ax.bar(np.arange(len(R_)) + (j - 1.5) * 0.2, [np.median(err[(meth, r)]) for r in R_], 0.18,
               color=vb.COLORS[meth], label=vb.LABELS[meth])
    ax.set_xticks(range(len(R_)))
    ax.set_xticklabels(R_, rotation=25, ha="right")
    ax.set_ylim(0.12, None)
    ax.set_ylabel("median RMS log10(rec / true)")
    ax.set_title("Lambda-selection rules (data only) vs the oracle (needs the truth); lower is better")
    ax.grid(axis="y", color="#d9d8d2"); ax.legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(OUT, "lambda_rules.png"), dpi=120)
    plt.close(fig)


SHOW = ["A_06", "B_04", "C_02", "D1_05", "D2_02", "D3_02", "E_06", "G_02"]


def plot_rule_models(prefixes=None):
    """True model and the models recovered at the lambda chosen by the current
    rule, the L-curve and the oracle, for pyGIMLi and the pyGIMLi-guided VAE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    vb.setup(threads=2)
    G, sp = vb.G, vb.G["sp"]
    cc, tri = G["cc"], G["tri"]
    pairs, dist = roughness_pairs(cc)
    roi = sp.roi_mask(cc)
    odir = os.path.join(OUT, "lambda_rule_models")
    os.makedirs(odir, exist_ok=True)
    tdir = os.path.join(OUT, "tuning")
    names = sorted(f[:-4] for f in os.listdir(tdir) if f.endswith(".npz"))
    if prefixes:
        names = [n for n in names if any(n.startswith(p) for p in prefixes)]
    cols = [("discrepancy_1.05", "current rule (chi2 <= 1.05)"), ("lcurve", "L-curve corner"),
            ("oracle", "oracle (needs the truth)")]
    for n in names:
        z = np.load(os.path.join(tdir, n + ".npz"), allow_pickle=True)
        R, spec = z["R"].item(), z["spec"].item()
        true = vb.true_on_mesh(spec)
        norm = LogNorm(np.exp(true.min()) / 1.5, np.exp(true.max()) * 1.5)
        fig, axs = plt.subplots(2, 4, figsize=(19, 5.6), constrained_layout=True)
        for i, meth in enumerate(("pygimli", "vae_pg")):
            path = R[meth][TUNED[meth]]
            for e in path:
                e.update(vb.metrics(spec, cc, np.asarray(e["m"], float), e["chi2"]))
            idx = rules(path, pairs, dist, roi)
            pc = sp._cellplot(axs[i, 0], tri, np.exp(true), norm=norm, cmap="turbo")
            axs[i, 0].set_title(f"{n}\nTRUE model", fontsize=9)
            axs[i, 0].set_ylabel(vb.LABELS[meth], fontsize=10)
            for j, (rule, lab) in enumerate(cols, start=1):
                e = path[idx[rule]]
                sp._cellplot(axs[i, j], tri, np.exp(np.asarray(e["m"], float)), norm=norm, cmap="turbo")
                axs[i, j].set_title(f"{vb.SHORT[meth]}: {lab}\nlambda = {e['lam']:g}, chi2 = {e['chi2']:.2f}, "
                                    f"RMS = {e['rms']:.3f}", fontsize=9)
        for ax in axs.ravel():
            ax.set_xlim(*sp.XLIM); ax.set_ylim(*sp.YLIM); ax.set_aspect("equal")
            ax.plot(sp.ELEC_X, 0 * sp.ELEC_X, "v", color="k", ms=2)
            ax.tick_params(labelsize=7)
        fig.colorbar(pc, ax=axs.ravel().tolist(), shrink=0.8, label="resistivity (Ohm.m)")
        fig.savefig(os.path.join(odir, n + ".png"), dpi=90)
        plt.close(fig)
    print(f"{len(names)} figures in {odir}")


if __name__ == "__main__":
    if "--models" in sys.argv:
        plot_rule_models(None if "--all" in sys.argv else SHOW)
    else:
        main()
