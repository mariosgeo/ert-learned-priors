"""
hetero_tuning.py
================
Were the heterogeneity results limited by untuned settings? A check on 20 of the
100 models (every 5th: all groups) at +-50 % within-layer variation, with the
SAME data as vae_benchmark_het50.

pyGIMLi (benchmark setting: blocky L1 model norm, zWeight 0.3) against
    pg_L2_zw03   smooth L2, zWeight 0.3
    pg_L2_zw1    smooth L2, zWeight 1.0
    pg_L1_zw1    blocky L1, zWeight 1.0
textured VAE (benchmark setting: nz 64, likelihood sigma 0.1 log10) against
    s05   nz 64,  sigma 0.05  (sharper reconstructions: texture is not "noise")
    nz128 nz 128, sigma 0.1   (more latent dimensions for the texture)
each as vae_random_<tag> (general prior) and vae_pg_dip_<tag> (dip-aware
pyGIMLi-guided fine-tuning, same sigma as the base).

    python hetero_tuning.py --stage train      # two textured VAEs (GPU, ~1 h each)
    python hetero_tuning.py --stage run --workers 6
    python hetero_tuning.py --summary
Full follow-up (all 100 models, VAE settings only; the pyGIMLi settings were
checked on the 20-model subset):
    python hetero_tuning.py --level 30 --all --vae-only --stage run
    python hetero_tuning.py --level 50 --all --vae-only --stage run   # adds the other 80
"""
import os
import sys
import time
import argparse
import numpy as np

LEVEL = sys.argv[sys.argv.index("--level") + 1] if "--level" in sys.argv else "50"
if LEVEL == "0":                                         # constant layers: the original benchmark data
    os.environ.pop("ERT_HETERO", None)
else:
    os.environ["ERT_HETERO"] = LEVEL                     # the +-LEVEL % suite and its data
os.environ["ERT_TEXTURE"] = "1"                          # textured training models
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vae_benchmark_ert as vb                             # noqa: E402

CKPT_DIR = os.path.join(HERE, "outputs", "hetero_tuning")          # textured VAE variants (shared)
OUTT = os.path.join(HERE, "outputs", "hetero_tuning" + ("" if LEVEL == "50" else f"_het{LEVEL}"))
_suite = "vae_benchmark" + ("" if LEVEL == "0" else f"_het{LEVEL}")
REF = os.path.join(HERE, "outputs", _suite, "models")
REF_TEX = os.path.join(HERE, "outputs", _suite + "_tex", "models")      # absent for 0 %
VARIANTS = {"s05": dict(nz=64, sigma=0.05), "nz128": dict(nz=128, sigma=0.1)}
# control: same 128-latent network trained on the same random models WITHOUT texture
# (homogeneous layers and bodies) - separates the effect of textured training from network size
ALLV = dict(VARIANTS, plain128=dict(nz=128, sigma=0.1, texture=False))
PG_CONFIGS = {"pg_L2_zw03": (0.3, False), "pg_L2_zw1": (1.0, False), "pg_L1_zw1": (1.0, True)}
# options read at import so that the pool workers (which re-import this module) see them too
if "--vae-only" in sys.argv:
    PG_CONFIGS.clear()
if "--variants" in sys.argv:
    VARIANTS = {k: ALLV[k] for k in sys.argv[sys.argv.index("--variants") + 1].split(",")}
LAMS_PG = [300, 100, 50, 20, 10, 5, 2]
# hybrid (as hybrid_vae_dip of the benchmark): pyGIMLi, smooth, zWeight 0.3, with the guided-VAE model of a
# variant as starting AND reference model -> hybrid_<tag>. --hybrid: after the VAE runs; --hybrid-only: only
# the hybrid, on saved guided models (no VAE inversions, no extra pyGIMLi settings)
HYBRID = "--hybrid" in sys.argv or "--hybrid-only" in sys.argv
HYBRID_ONLY = "--hybrid-only" in sys.argv
if HYBRID_ONLY:
    PG_CONFIGS.clear()


def ckpt(tag):
    return os.path.join(CKPT_DIR, f"vae_tex_{tag}.pt" if ALLV[tag].get("texture", True) else f"vae_{tag}.pt")


def subset(all_models=False):
    return [s for i, s in enumerate(vb.make_suite()) if all_models or i % 5 == 0]


def train():
    import torch
    vb.setup(threads=8)
    v, G = vb.G["v"], vb.G
    os.makedirs(OUTT, exist_ok=True)
    sets = {}
    for tag, c in VARIANTS.items():
        if os.path.exists(ckpt(tag)):
            continue
        tex = c.get("texture", True)
        v.TEXTURE = tex                                 # textured or homogeneous training models
        if tex not in sets:
            t = time.time()
            sets[tex] = (v.random_models_general(200000, G["xg"], G["zg"], seed=1),
                         v.random_models_general(2000, G["xg"], G["zg"], seed=2))
            print(f"generated 200000 {'textured' if tex else 'homogeneous'} models in {time.time() - t:.0f}s", flush=True)
        X, Xv = sets[tex]
        t = time.time()
        with vb._GPULock():
            model, mean, std = v.train_vae(X, c["nz"], 40000, 256, c["sigma"], G["device"])
        torch.save(dict(state=model.state_dict(), nz=c["nz"], mean=mean, std=std, sigma=c["sigma"],
                        n_train=len(X), steps=40000, texture=c.get("texture", True)), ckpt(tag))
        print(f"{tag}: trained in {(time.time() - t) / 60:.0f} min; held-out reconstruction RMS "
              f"{v.recon_rms(model, Xv, mean, std):.3f} log10", flush=True)


def run_model(spec):
    from ert_common import ERTForward
    from scipy.spatial import cKDTree
    try:
        G, sp, v, o = vb.G, vb.G["sp"], vb.G["v"], vb.G["opts"]
        name = spec["name"]
        seed = 1000 + vb.make_suite_index(name)
        fn = os.path.join(OUTT, "models", name + ".npz")
        res = np.load(fn, allow_pickle=True)["res"].item() if os.path.exists(fn) else {}
        ref = np.load(os.path.join(REF, name + ".npz"), allow_pickle=True)["res"].item()
        fn_t = os.path.join(REF_TEX, name + ".npz")
        reft = np.load(fn_t, allow_pickle=True)["res"].item() if os.path.exists(fn_t) else None
        for k in ("pygimli", "vae_random", "vae_pg_dip", "hybrid_vae_dip"):   # benchmark settings, for reference
            if k not in ref:
                continue
            res.setdefault(k, {q: ref[k][q] for q in ("rms", "chi2", "lam")})
            if reft is not None and k in reft:                          # textured benchmark: no hybrid
                res.setdefault(k + "_tex", {q: reft[k][q] for q in ("rms", "chi2", "lam")})
        data = vb.get_dataset(spec, os.path.join(vb.DATA_DIR, name + ".dat"), seed)
        dobs, cc, xg, zg, S = np.array(data["rhoa"]), G["cc"], G["xg"], G["zg"], G["S"]
        fwd = ERTForward(data, G["mesh"])
        lams = [float(x) for x in o["lams"].split(",")]

        def save():
            np.savez(fn, res=np.array(res, dtype=object), spec=np.array(spec, dtype=object))
        for key, (zw, blocky) in PG_CONFIGS.items():
            if key in res:
                continue
            t = time.time()
            pgr, _ = sp.invert_pygimli(data, G["mesh"], LAMS_PG, 1.05, zw, blocky, verbose=False)
            m = pgr["m"][cKDTree(pgr["cc"]).query(cc)[1]]
            res[key] = dict(m=m.astype(np.float32), lam=pgr["lam"], time=time.time() - t, **vb.metrics(spec, cc, m, pgr["chi2"]))
            save()
        img = v.grid_from_cells(cc, ref["pygimli"]["m"] / vb.LN10, xg, zg)   # benchmark pyGIMLi image
        for tag, c in ({} if HYBRID_ONLY else VARIANTS).items():
            o["random_ckpt"], o["sigma"] = ckpt(tag), c["sigma"]
            v.TEXTURE = c.get("texture", True)          # guided training models with / without texture
            base = vb.load_vae(ckpt(tag))
            if f"vae_random_{tag}" not in res:
                t = time.time()
                model, mean, std = base
                r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"],
                                    verbose=False, z_starts=[vb.encode(model, img, mean, std)])
                res[f"vae_random_{tag}"] = dict(m=np.asarray(r["m"], np.float32), lam=r["lam"], time=time.time() - t,
                                                **vb.metrics(spec, cc, r["m"], r["chi2"]))
                save()
            if f"vae_pg_dip_{tag}" not in res:
                t = time.time()
                info = v.analyse_pg(img, G["covg"], xg, zg, dip_aware=True)
                Xpg = v.random_models_pg(o["n_pg"], xg, zg, info, seed, o["pg_mix"], o["pg_extend"])
                model, mean, std = vb.finetune(Xpg, seed=seed)
                r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"],
                                    verbose=False, z_starts=[vb.encode(model, img, mean, std)])
                res[f"vae_pg_dip_{tag}"] = dict(m=np.asarray(r["m"], np.float32), lam=r["lam"], time=time.time() - t,
                                                **vb.metrics(spec, cc, r["m"], r["chi2"]))
                save()
        for tag in (VARIANTS if HYBRID else {}):
            g = res.get(f"vae_pg_dip_{tag}")
            if g is None or "m" not in g or f"hybrid_{tag}" in res:
                continue
            t = time.time()
            pgr, _ = sp.invert_pygimli(data, G["mesh"], LAMS_PG, 1.05, 0.3, False, verbose=False,
                                       start=np.asarray(g["m"], float))
            m = pgr["m"][cKDTree(pgr["cc"]).query(cc)[1]]
            res[f"hybrid_{tag}"] = dict(m=m.astype(np.float32), lam=pgr["lam"], time=time.time() - t,
                                        **vb.metrics(spec, cc, m, pgr["chi2"]))
            save()
        return name, "  ".join(f"{k}={r['rms']:.3f}" for k, r in res.items())
    except Exception as e:
        import traceback
        with open(os.path.join(OUTT, "errors.log"), "a") as fh:
            fh.write(f"=== {spec['name']}\n{traceback.format_exc()}\n")
        return spec["name"], f"ERROR {e}"


def summary():
    d = os.path.join(OUTT, "models")
    by = {f[:-4]: np.load(os.path.join(d, f), allow_pickle=True)["res"].item() for f in sorted(os.listdir(d))}
    keys = ["pygimli"] + list(PG_CONFIGS) + ["vae_random", "vae_random_tex"] + \
           [f"vae_random_{t}" for t in VARIANTS] + ["vae_pg_dip", "vae_pg_dip_tex"] + [f"vae_pg_dip_{t}" for t in VARIANTS] + \
           ["hybrid_vae_dip"] + [f"hybrid_{t}" for t in ALLV]
    lab = {"pygimli": "pyGIMLi L1 zW0.3 (benchmark)", "vae_random": "VAE random (constant-layer prior)",
           "vae_random_tex": "VAE random textured s0.1 nz64 (benchmark)",
           "vae_pg_dip": "VAE dip-aware (constant-layer prior)",
           "vae_pg_dip_tex": "VAE dip-aware textured s0.1 nz64 (benchmark)",
           "hybrid_vae_dip": "Hybrid on VAE dip-aware nz64 (benchmark)"}
    from scipy.stats import wilcoxon
    L = [f"HETEROGENEITY +-{LEVEL} %: TUNING CHECK on {len(by)} models", "",
         "beats best pyG: against the best of the pyGIMLi settings run on that model (all four only on the",
         "20-model subset); p: paired Wilcoxon test against pyGIMLi (benchmark setting)", "",
         f"{'setting':46s}{'median RMS':>11s}{'beats pyG(bench)':>18s}{'beats best pyG':>16s}{'p':>8s}"]
    best_pg = {m: min(r[k]["rms"] for k in ["pygimli"] + list(PG_CONFIGS) if k in r) for m, r in by.items()}
    for k in keys:
        ms = [m for m in by if k in by[m]]
        if not ms:
            continue
        x = np.array([by[m][k]["rms"] for m in ms])
        w1 = np.mean([by[m][k]["rms"] < by[m]["pygimli"]["rms"] for m in ms])
        w2 = np.mean([by[m][k]["rms"] < best_pg[m] for m in ms])
        p0 = np.array([by[m]["pygimli"]["rms"] for m in ms])
        pv = wilcoxon(x, p0).pvalue if k != "pygimli" and np.any(x != p0) else np.nan
        L.append(f"{lab.get(k, k):46s}{np.median(x):11.3f}{100 * w1:17.0f}%{100 * w2:15.0f}%{pv:8.3f}   (n={len(ms)})")
    txt = "\n".join(L)
    open(os.path.join(OUTT, "hetero_tuning.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "run"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--level", default="50")
    ap.add_argument("--all", action="store_true", help="all 100 models (default: every 5th)")
    ap.add_argument("--vae-only", action="store_true", help="skip the extra pyGIMLi settings")
    ap.add_argument("--variants", default=None, help="comma list of textured-VAE variants, e.g. nz128")
    ap.add_argument("--hybrid", action="store_true", help="also the hybrid on each variant's guided model (read at import)")
    ap.add_argument("--hybrid-only", action="store_true", help="only the hybrid, on saved guided models (read at import)")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUTT, "models"), exist_ok=True)
    if a.summary:
        return summary()
    if a.stage == "train":
        return train()
    todo = subset(a.all)
    print(f"{len(todo)} models, {a.workers} workers", flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers, initializer=vb._init_worker, initargs=(dict(threads=a.threads),)) as pool:
        for k, (name, msg) in enumerate(pool.imap_unordered(run_model, todo), 1):
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {name}: {msg}   ({el / 60:.0f} min, ~{el / k * (len(todo) - k) / 60:.0f} min left)",
                  flush=True)
    summary()


if __name__ == "__main__":
    main()
