"""
vae_benchmark_tuning.py
=======================
"Each method at its best setting": re-inverts the 100 SAVED benchmark data sets
(outputs/vae_benchmark/data, identical data for every method) over a grid of
settings and the full lambda path of each, and compares the methods at three
levels:

  default  the settings of vae_benchmark_ert.py
  tuned    lambda chosen from the data only (discrepancy principle: largest
           lambda with chi2 <= 1.05, or the last stage before chi2 stalls), and
           ONE fixed setting per method - the one with the lowest median error
           over the 100 models (what you get after tuning a method once for
           this kind of survey)
  oracle   per model, the setting and lambda closest to the true model
           (an upper bound; needs the truth, impossible on field data)

Settings grids:
  pygimli     zWeight {1, 0.3} x {smooth, blocky}; lambda 300, 100, 30, 10, 3
  sparse      (norm, aniso, zweight) in {(l1,2,0.3), (l2,1,1), (l1,1,1), (l2,2,0.3)};
              lambda 100 ... 0.01; plus the default run with refinement
  vae_random  lambda 10 ... 0.01 (prior strength)
  vae_pg      deeper continuation of anomalies --pg-extend {0, 1, 2}
              (0 = no extension); lambda 10 ... 0.01

Results: outputs/vae_benchmark/tuning/<name>.npz (resumable, per method),
tuning_summary.txt, tuning_summary.png, tuning_results.csv.

Usage
-----
    python vae_benchmark_tuning.py              # all 100 models, 6 workers
    python vae_benchmark_tuning.py --summary    # statistics from saved results
    python vae_benchmark_tuning.py --quick      # smoke test on 1 model
"""

import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vae_benchmark_ert as vb                      # noqa: E402

OUTT = os.path.join(vb.OUTB, "tuning")
CKDIR = os.path.join(vb.OUTB, "vae_pg")                 # fine-tuned pyGIMLi-guided networks
LAMS = dict(pygimli=[300, 100, 30, 10, 3], sparse=[100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01],
            vae=[10, 3, 1, 0.3, 0.1, 0.03, 0.01])
PG_CONFIGS = [(1.0, False), (1.0, True), (0.3, False), (0.3, True)]
SP_CONFIGS = [("l1", 2.0, 0.3), ("l2", 1.0, 1.0), ("l1", 1.0, 1.0), ("l2", 2.0, 0.3)]
EXTENDS = [0.0, 1.0, 2.0]
TM = ["pygimli", "sparse", "vae_random", "vae_pg"]      # methods of the tuning study


def discrepancy_pick(path, target=1.05, stall=0.95):
    """Index of the stage chosen from the data only (same rule as the inversions)."""
    for k, s in enumerate(path):
        if s["chi2"] <= target:
            return k
        if k > 0 and s["chi2"] > stall * path[k - 1]["chi2"]:
            return k - 1
    return len(path) - 1


def _entry(spec, lam, chi2, m):
    cc = vb.G["cc"]
    return dict(lam=float(lam), m=np.asarray(m, dtype=np.float32), **vb.metrics(spec, cc, m, chi2))


def _init(opts, quick):
    """Worker initializer (Windows spawn: module globals must be set here)."""
    global OUTT, CKDIR
    if quick:
        LAMS.update(pygimli=[100, 10], sparse=[10, 1, 0.1], vae=[1, 0.1])
        OUTT = CKDIR = os.path.join(vb.OUTB, "tuning_quick")
    vb.setup(**opts)


def run_tuning(name):
    try:
        return name, _run_tuning(name)
    except Exception as e:
        import traceback
        with open(os.path.join(OUTT, "errors.log"), "a") as fh:
            fh.write(f"=== {name}\n{traceback.format_exc()}\n")
        return name, f"ERROR {e}"


def _run_tuning(name):
    import pygimli as pg
    from ert_common import ERTForward
    from scipy.spatial import cKDTree
    G, sp, v, o = vb.G, vb.G["sp"], vb.G["v"], vb.G["opts"]
    t0 = time.time()
    fn = os.path.join(OUTT, name + ".npz")
    R = np.load(fn, allow_pickle=True)["R"].item() if os.path.exists(fn) else {}
    zb = np.load(os.path.join(vb.OUTB, "models", name + ".npz"), allow_pickle=True)
    spec, base = zb["spec"].item(), zb["res"].item()
    data = pg.DataContainerERT(os.path.join(vb.OUTB, "data", name + ".dat"))   # the SAME data
    dobs = np.array(data["rhoa"])
    fwd = ERTForward(data, G["mesh"])
    cc, xg, zg, S = G["cc"], G["xg"], G["zg"], G["S"]
    seed = 1000 + vb.make_suite_index(name)

    def save():
        np.savez(fn, R=np.array(R, dtype=object), spec=np.array(spec, dtype=object))

    if "pygimli" not in R:
        R["pygimli"] = {}
        for zw, blocky in PG_CONFIGS:
            _, runs = sp.invert_pygimli(data, G["mesh"], LAMS["pygimli"], 1.05, zw, blocky,
                                        run_all=True, verbose=False)
            R["pygimli"][f"zW{zw:g}_{'blocky' if blocky else 'smooth'}"] = [
                _entry(spec, r["lam"], r["chi2"], r["m"][cKDTree(r["cc"]).query(cc)[1]]) for r in runs]
        save()

    if "sparse" not in R:
        R["sparse"] = {}
        p0 = None
        for norm, aniso, zw in SP_CONFIGS:
            ctrl = sp.place_controls(cc)
            W, D = sp.weights_tps(cc, ctrl, aniso), sp.regularization_operator(cc[ctrl], zw)
            p0 = np.full(len(ctrl), np.log(np.median(dobs)))
            _, stages, _ = sp.invert_gn(fwd, W, D, dobs, p0, LAMS["sparse"], chi2_target=-1.0,
                                        stall=1e9, norm=norm, verbose=False)
            R["sparse"][f"{norm}_aniso{aniso:g}_zW{zw:g}"] = [_entry(spec, s["lam"], s["chi2"], s["m"])
                                                              for s in stages]
        b = base["sparse"]
        R["sparse"]["default_l1_aniso2_zW0.3_refined"] = [_entry(spec, b["lam"], b["chi2"], b["m"])]
        save()

    img = v.grid_from_cells(cc, base["pygimli"]["m"] / vb.LN10, xg, zg)       # guide = default pyGIMLi
    lams = LAMS["vae"]
    if "vae_random" not in R:
        model, mean, std = vb.load_vae(o["random_ckpt"])
        r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"], chi2_target=-1.0,
                            stall=1e9, verbose=False, z_starts=[vb.encode(model, img, mean, std)])
        R["vae_random"] = {"general200k": [_entry(spec, s["lam"], s["chi2"], s["m"]) for s in r["path"]]}
        save()

    if "vae_pg" not in R or len(R["vae_pg"]) < len(EXTENDS):
        R.setdefault("vae_pg", {})
        info = v.analyse_pg(img, G["covg"], xg, zg)
        for ext in EXTENDS:
            key = f"extend{ext:g}"
            if key in R["vae_pg"]:
                continue
            ck = os.path.join(CKDIR, name + f"_ext{ext:g}.pt")
            if ext == 1.0 and os.path.exists(os.path.join(vb.OUTB, "vae_pg", name + ".pt")):
                ck = os.path.join(vb.OUTB, "vae_pg", name + ".pt")        # the benchmark network (extend 1)
            if os.path.exists(ck):
                model, mean, std = vb.load_vae(ck)
            else:
                X = v.random_models_pg(o["n_pg"], xg, zg, info, seed, o["pg_mix"], ext)
                model, mean, std = vb.finetune(X, seed=seed)
                G["torch"].save(dict(state=model.state_dict(), nz=model.nz, mean=mean, std=std,
                                     base=o["random_ckpt"], n_pg=o["n_pg"], ft_steps=o["ft_steps"],
                                     pg_extend=ext), ck)
            r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"],
                                chi2_target=-1.0, stall=1e9, verbose=False,
                                z_starts=[vb.encode(model, img, mean, std)])
            R["vae_pg"][key] = [_entry(spec, s["lam"], s["chi2"], s["m"]) for s in r["path"]]
            save()
    return f"{time.time() - t0:.0f}s"


# ── statistics ─────────────────────────────────────────────────────────────
def summarize():
    import csv
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(f for f in os.listdir(OUTT) if f.endswith(".npz"))
    data = {}
    for f in files:
        z = np.load(os.path.join(OUTT, f), allow_pickle=True)
        R, spec = z["R"].item(), z["spec"].item()
        if all(k in R for k in TM) and len(R["vae_pg"]) == len(EXTENDS):
            base = np.load(os.path.join(vb.OUTB, "models", f), allow_pickle=True)["res"].item()
            data[f[:-4]] = (spec, R, base)
    names = sorted(data)
    if not names:
        print("no complete results yet")
        return
    if "cc" not in vb.G:
        vb.setup(threads=2)
    for n in names:                    # metrics with the current true-model definition (area fraction)
        spec, R, base = data[n]
        for k in R:
            for path in R[k].values():
                for e in path:
                    e.update(vb.metrics(spec, vb.G["cc"], e["m"].astype(float), e["chi2"]))
        for k in base:
            base[k].update(vb.metrics(spec, vb.G["cc"], base[k]["m"], base[k]["chi2"]))
    group = {n: data[n][0]["group"] for n in names}

    # per method and config: data-chosen error per model; oracle per model
    picked, oracle, default = {}, {}, {}
    rows = []
    for k in TM:
        picked[k], oracle[k], default[k] = {}, [], []
        for n in names:
            spec, R, base = data[n]
            default[k].append(base[k]["rms"])
            oracle[k].append(min(e["rms"] for path in R[k].values() for e in path))
            for cfg, path in R[k].items():
                e = path[discrepancy_pick(path)]
                picked[k].setdefault(cfg, []).append(e["rms"])
                for s in path:
                    rows.append(dict(model=n, group=group[n], method=k, config=cfg, lam=s["lam"],
                                     chi2=s["chi2"], rms=s["rms"], body_err=s.get("body_err", np.nan),
                                     data_choice=s is e))
    with open(os.path.join(vb.OUTB, "tuning_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tuned_cfg = {k: min(picked[k], key=lambda c: np.median(picked[k][c])) for k in TM}
    tuned = {k: np.array(picked[k][tuned_cfg[k]]) for k in TM}
    oracle = {k: np.array(v_) for k, v_ in oracle.items()}
    default = {k: np.array(v_) for k, v_ in default.items()}

    def test(a, b):
        d = a - b
        return d.mean(), int((d < 0).sum()), stats.wilcoxon(a, b, alternative="less").pvalue

    L = [f"EACH METHOD AT ITS BEST SETTING  ({len(names)} models, identical data for every method)", "",
         "Median RMS log10(rec/true) per setting (lambda chosen from the data):"]
    for k in TM:
        for cfg in picked[k]:
            mark = "  <- tuned" if cfg == tuned_cfg[k] else ""
            L.append(f"  {vb.LABELS[k]:28s} {cfg:34s} {np.median(picked[k][cfg]):.3f}{mark}")
    L += ["", f"{'level':10s}" + "".join(f"{vb.SHORT[k]:>21s}" for k in TM) + "   (median RMS)"]
    for lab, d_ in (("default", default), ("tuned", tuned), ("oracle", oracle)):
        L.append(f"{lab:10s}" + "".join(f"{np.median(d_[k]):21.3f}" for k in TM))
    L += ["", "Paired tests against pyGIMLi AT THE SAME LEVEL (mean RMS change, models better, Wilcoxon p):"]
    for lab, d_ in (("default", default), ("tuned", tuned), ("oracle", oracle)):
        for k in TM[1:]:
            m_, w_, p_ = test(d_[k], d_["pygimli"])
            L.append(f"  {lab:8s} {vb.LABELS[k]:28s} {m_:+.4f}  {w_:3d}/{len(names)}  p={p_:.2g}")
    L += ["", "Tuned, by group (median RMS):",
          f"  {'group':24s}" + "".join(f"{vb.SHORT[k]:>21s}" for k in TM)]
    for g in vb.GROUPS:
        idx = [i for i, n in enumerate(names) if group[n] == g]
        if idx:
            L.append(f"  {g + ' ' + vb.GROUP_NAMES[g]:24s}" + "".join(f"{np.median(tuned[k][idx]):21.3f}"
                                                                    for k in TM))
    # effect of the deeper continuation in the pyGIMLi-guided prior
    L += ["", "pyGIMLi-guided prior: deeper continuation of anomalies (lambda from the data):"]
    body = [i for i, n in enumerate(names) if group[n] in ("E", "F", "G")]
    e0 = np.array(picked["vae_pg"]["extend0"])
    for ext in ("extend1", "extend2"):
        e1 = np.array(picked["vae_pg"][ext])
        for lab, idx in (("all models", list(range(len(names)))), ("bodies (E-G)", body)):
            m_, w_, p_ = test(e1[idx], e0[idx])
            L.append(f"  {ext} vs extend0, {lab:13s}: mean {m_:+.4f}, better {w_}/{len(idx)}, p={p_:.2g}")
    txt = "\n".join(L)
    open(os.path.join(vb.OUTB, "tuning_summary.txt"), "w").write(txt)
    print(txt)

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    ax = axs[0]
    for j, k in enumerate(TM):
        vals = [np.median(d_[k]) for d_ in (default, tuned, oracle)]
        ax.bar(np.arange(3) + (j - 1.5) * 0.2, vals, 0.18, color=vb.COLORS[k], label=vb.LABELS[k])
    ax.set_xticks(range(3)); ax.set_xticklabels(["default", "tuned\n(data-chosen lambda)", "oracle\n(needs the truth)"])
    ax.set_ylabel("median RMS log10(rec / true)"); ax.set_title("(a) Median error at three tuning levels")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", color="#d9d8d2")
    ax = axs[1]
    for k in TM:
        v_ = np.sort(tuned[k])
        ax.plot(v_, np.arange(1, len(v_) + 1) / len(v_), color=vb.COLORS[k], lw=2, label=vb.LABELS[k])
    ax.set_xlabel("RMS log10(rec / true)"); ax.set_ylabel("share of models with error below x")
    ax.set_title("(b) Tuned methods: cumulative distribution"); ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#d9d8d2")
    ax = axs[2]
    gs = [g for g in vb.GROUPS if any(group[n] == g for n in names)]
    for j, k in enumerate(TM):
        vals = [np.median(tuned[k][[i for i, n in enumerate(names) if group[n] == g]]) for g in gs]
        ax.bar(np.arange(len(gs)) + (j - 1.5) * 0.2, vals, 0.18, color=vb.COLORS[k], label=vb.LABELS[k])
    ax.set_xticks(range(len(gs))); ax.set_xticklabels(gs)
    ax.set_ylabel("median RMS log10(rec / true)"); ax.set_title("(c) Tuned methods by test group")
    ax.grid(axis="y", color="#d9d8d2")
    fig.savefig(os.path.join(vb.OUTB, "tuning_summary.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 model, short lambda paths")
    a = ap.parse_args()
    global OUTT
    if a.quick:
        OUTT = os.path.join(vb.OUTB, "tuning_quick")
    os.makedirs(OUTT, exist_ok=True)
    if a.summary:
        summarize()
        return
    names = [m["name"] for m in vb.make_suite()]
    names = [n for n in names if os.path.exists(os.path.join(vb.OUTB, "data", n + ".dat"))]
    if a.quick:
        a.only, a.workers = a.only or "A_01", 1
    if a.only:
        names = [n for n in names if any(n.startswith(p) for p in a.only.split(","))]
    opts = dict(threads=a.threads)
    if a.quick:
        opts.update(n_pg=1000, ft_steps=100, restarts=1)
    from multiprocessing import Pool
    t0 = time.time()
    print(f"{len(names)} models, {a.workers} workers", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(opts, a.quick)) as pool:
        for k, (n, msg) in enumerate(pool.imap_unordered(run_tuning, names), start=1):
            el = time.time() - t0
            print(f"[{k}/{len(names)}] {n}: {msg}   ({el / 60:.0f} min, ~{el / k * (len(names) - k) / 60:.0f} min left)",
                  flush=True)
    if not a.quick:
        summarize()


if __name__ == "__main__":
    main()
