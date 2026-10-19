"""Figure: does textured training help? Four heterogeneous (+-50 %) benchmark models inverted with the guided VAE
trained on homogeneous ('plain') and on textured training models, with 64 and 128 latent dimensions.

    python outputs/paper/texture_examples.py compute   # the textured 128-dimensional guided VAE for the four models
    python outputs/paper/texture_examples.py plot      # fig12_texture_training.png / .pdf

The textured 128-dimensional runs of the +-50 % suite were made before the models were stored, so they are
repeated here for the four models with the same seed and settings as hetero_tuning.py (written to
outputs/paper/texture_examples/, the benchmark statistics are not touched); their RMS is compared with the stored one.
"""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os
import sys
import numpy as np

ROOT = _ROOT
OUT = os.path.join(ROOT, "outputs")
D = os.path.join(OUT, "paper", "texture_examples")
NAMES = ["B_01_z2-6_H-L-H", "D3_11_fault", "E_08_dyke_con_top4_2lay", "G_05_4bodies_basin"]
sys.path.insert(0, ROOT)


def compute():
    sys.argv = ["hetero_tuning.py", "--level", "50", "--variants", "nz128"]
    import hetero_tuning as ht                      # sets the +-50 % suite and the textured training models
    import vae_benchmark_ert as vb
    from ert_common import ERTForward
    vb.setup(threads=8)
    G, v, o = vb.G, vb.G["v"], vb.G["opts"]
    suite = {s["name"]: s for s in vb.make_suite()}
    os.makedirs(D, exist_ok=True)
    for name in NAMES:
        fn = os.path.join(D, name + ".npz")
        if os.path.exists(fn):
            continue
        spec, seed = suite[name], 1000 + vb.make_suite_index(name)
        ref = np.load(os.path.join(ht.REF, name + ".npz"), allow_pickle=True)["res"].item()
        data = vb.get_dataset(spec, os.path.join(vb.DATA_DIR, name + ".dat"), seed)
        dobs, cc, xg, zg, S = np.array(data["rhoa"]), G["cc"], G["xg"], G["zg"], G["S"]
        fwd = ERTForward(data, G["mesh"])
        lams = [float(x) for x in o["lams"].split(",")]
        c = ht.ALLV["nz128"]
        o["random_ckpt"], o["sigma"] = ht.ckpt("nz128"), c["sigma"]
        v.TEXTURE = True
        img = v.grid_from_cells(cc, ref["pygimli"]["m"] / vb.LN10, xg, zg)
        info = v.analyse_pg(img, G["covg"], xg, zg, dip_aware=True)
        Xpg = v.random_models_pg(o["n_pg"], xg, zg, info, seed, o["pg_mix"], o["pg_extend"])
        model, mean, std = vb.finetune(Xpg, seed=seed)
        r = v.invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=o["restarts"], verbose=False,
                            z_starts=[vb.encode(model, img, mean, std)])
        met = vb.metrics(spec, cc, r["m"], r["chi2"])
        stored = np.load(os.path.join(ht.OUTT, "models", name + ".npz"), allow_pickle=True)["res"].item()["vae_pg_dip_nz128"]
        np.savez(fn, m=np.asarray(r["m"], np.float32), rms=met["rms"], chi2=r["chi2"], rms_stored=stored["rms"])
        print(f"{name}: RMS {met['rms']:.3f} (stored benchmark run {stored['rms']:.3f})", flush=True)


def plot():
    os.chdir(os.path.join(OUT, "paper"))
    sys.path.insert(0, os.path.join(OUT, "paper"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figs
    tri = figs._tri()
    cols = [("true model (±50 %)", None), ("pyGIMLi", None), ("64, plain training", None), ("64, textured training", None),
            ("128, plain training", None), ("128, textured training", None)]
    fig, axs = plt.subplots(len(NAMES), len(cols), figsize=(15, 1.0 * len(NAMES) + 0.7), constrained_layout=True)
    for i, name in enumerate(NAMES):
        a = np.load(os.path.join(OUT, "vae_benchmark_het50", "models", name + ".npz"), allow_pickle=True)
        b = np.load(os.path.join(OUT, "vae_benchmark_het50_tex", "models", name + ".npz"), allow_pickle=True)["res"].item()
        h = np.load(os.path.join(OUT, "hetero_tuning", "models", name + ".npz"), allow_pickle=True)["res"].item()
        t = np.load(os.path.join(D, name + ".npz"))
        ra = a["res"].item()
        panels = [(a["m_true"], name.split("_")[0] + "_" + name.split("_")[1]),
                  (ra["pygimli"]["m"], f"RMS {ra['pygimli']['rms']:.3f}"),
                  (ra["vae_pg_dip"]["m"], f"RMS {ra['vae_pg_dip']['rms']:.3f}"),
                  (b["vae_pg_dip"]["m"], f"RMS {b['vae_pg_dip']['rms']:.3f}"),
                  (h["vae_pg_dip_plain128"]["m"], f"RMS {h['vae_pg_dip_plain128']['rms']:.3f}"),
                  (t["m"], f"RMS {float(t['rms']):.3f}")]
        for j, (m, title) in enumerate(panels):
            im = figs._panel(axs[i, j], tri, m, (cols[j][0] + "\n" if i == 0 else "") + title)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="resistivity (Ω·m)")
    figs.save(fig, "fig12_texture_training", dpi=170)
    plt.close(fig)
    print("saved fig12_texture_training", flush=True)


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]]()
