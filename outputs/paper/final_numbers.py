"""All numbers still missing from the manuscript, computed once every run has finished:
Table 5 (training models x latent size, guided and random VAE, 0/10/30/50 %), comparison with pyGIMLi,
and the timing benchmark. Writes outputs/paper/final_numbers.json and final_numbers.txt."""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os
import json
import numpy as np
from scipy.stats import wilcoxon

OUT = os.path.join(_ROOT, "outputs")
LEVELS = {"0": ("vae_benchmark", "hetero_tuning_het0", None),
          "10": ("vae_benchmark_het10", "hetero_tuning_het10", "vae_benchmark_het10_tex"),
          "30": ("vae_benchmark_het30", "hetero_tuning_het30", "vae_benchmark_het30_tex"),
          "50": ("vae_benchmark_het50", "hetero_tuning", "vae_benchmark_het50_tex")}
CELLS = {"homog64": None, "hetero64": "tex", "homog128": "plain128", "hetero128": "nz128", "hetero64_s05": "s05"}


def load(d):
    p = os.path.join(OUT, d, "models")
    if not os.path.isdir(p):
        return {}
    return {f[:-4]: np.load(os.path.join(p, f), allow_pickle=True)["res"].item() for f in os.listdir(p)}


res, lines = {}, ["FINAL NUMBERS (median RMS log10; 'vs homog64' = share better than the homogeneously trained "
                  "64-latent prior + Wilcoxon p; 'vs pyGIMLi' likewise)", ""]
for lv, (bench, tune, tex) in LEVELS.items():
    B, T, X = load(bench), load(tune), load(tex) if tex else {}
    res[lv] = {}
    for vae in ("vae_pg_dip", "vae_random"):
        base = {m: B[m][vae]["rms"] for m in B if vae in B[m]}
        pyg = {m: B[m]["pygimli"]["rms"] for m in B}
        for cell, tag in CELLS.items():
            if tag is None:
                vals = base
            elif tag == "tex":
                vals = {m: X[m][vae]["rms"] for m in X if vae in X[m]}
            else:
                vals = {m: T[m][f"{vae}_{tag}"]["rms"] for m in T if f"{vae}_{tag}" in T[m]}
            ms = sorted(set(vals) & set(base) & set(pyg))
            if not ms:
                continue
            x, b, p = (np.array([d[m] for m in ms]) for d in (vals, base, pyg))
            e = dict(n=len(ms), median=float(np.median(x)), mean=float(np.mean(x)),
                     better_than_pyg=float(np.mean(x < p)), p_pyg=float(wilcoxon(x, p).pvalue),
                     pyg_median=float(np.median(p)))
            if cell != "homog64":
                e.update(better_than_homog64=float(np.mean(x < b)), p_homog64=float(wilcoxon(x, b).pvalue))
            res[lv][f"{vae}:{cell}"] = e
            lines.append(f"{lv:>3s} %  {vae:11s} {cell:13s} n={len(ms):3d}  median {e['median']:.3f}  "
                         + (f"vs homog64 {100 * e['better_than_homog64']:3.0f}% p={e['p_homog64']:.2g}  " if cell != "homog64" else " " * 30)
                         + f"vs pyGIMLi {100 * e['better_than_pyg']:3.0f}% p={e['p_pyg']:.2g} (pyG {e['pyg_median']:.3f})")
    lines.append("")
tf = os.path.join(OUT, "timing", "timing_summary.txt")
if os.path.exists(tf):
    lines += ["", open(tf, encoding="utf-8").read()]
json.dump(res, open(os.path.join(OUT, "paper", "final_numbers.json"), "w"), indent=1)
open(os.path.join(OUT, "paper", "final_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
