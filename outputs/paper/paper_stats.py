"""Numbers for the manuscript: pyGIMLi vs VAE random prior vs dip-aware pyGIMLi-guided VAE."""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os, json, csv, numpy as np
from scipy.stats import wilcoxon
OUT = os.path.join(_ROOT, "outputs")
M3 = ["pygimli", "vae_random", "vae_pg_dip", "hybrid_vae_dip"]
GR = ["A", "B", "C", "D1", "D2", "D3", "E", "F", "G"]
def load(suite):
    by = {}
    for f in os.listdir(os.path.join(OUT, suite, "models")):
        r = np.load(os.path.join(OUT, suite, "models", f), allow_pickle=True)["res"].item()
        by[f[:-4]] = {k: dict(rms=v["rms"], chi2=v.get("chi2", np.nan)) for k, v in r.items() if "rms" in v}
    return by
def struct(suite):
    S = {}
    fn = os.path.join(OUT, "structure_metrics", f"{suite}_structure.csv")
    for r in csv.DictReader(open(fn)):
        S.setdefault(r["model"], {})[r["method"]] = {k: float(r[k]) if r[k] not in ("", "nan") else np.nan
                                                     for k in ("ssim", "if_err", "body_iou", "body_contr", "body_shift")}
    return S
res = {}
for suite in ["vae_benchmark", "vae_benchmark_het10", "vae_benchmark_het30", "vae_benchmark_het50",
              "vae_benchmark_het10_tex", "vae_benchmark_het30_tex", "vae_benchmark_het50_tex"]:
    by, S = load(suite), struct(suite)
    d = {}
    for k in M3:
        ms = [m for m in by if k in by[m]]
        if not ms:                                       # method not run for this suite (yet)
            continue
        x = np.array([by[m][k]["rms"] for m in ms]); p = np.array([by[m]["pygimli"]["rms"] for m in ms])
        e = dict(n=len(ms), median=float(np.median(x)), mean=float(np.mean(x)),
                 chi2_median=float(np.nanmedian([by[m][k]["chi2"] for m in ms])),
                 chi2_le_105=float(np.mean([by[m][k]["chi2"] <= 1.05 for m in ms])))
        if k != "pygimli":
            e.update(wins=float(np.mean(x < p)), p=float(wilcoxon(x, p).pvalue), mean_diff=float(np.mean(x - p)),
                     rel_median=float(np.median(x) / np.median(p) - 1))
            ms_r = [m for m in by if "vae_random" in by[m] and k in by[m]]
            if k == "vae_pg_dip":
                xr = np.array([by[m]["vae_random"]["rms"] for m in ms_r]); xd = np.array([by[m][k]["rms"] for m in ms_r])
                e.update(wins_vs_random=float(np.mean(xd < xr)), p_vs_random=float(wilcoxon(xd, xr).pvalue))
        for q in ("ssim", "if_err", "body_iou", "body_contr", "body_shift"):
            v = [S[m][k][q] for m in S if k in S[m] and np.isfinite(S[m][k][q])]
            e[q] = float(np.median(v)) if v else np.nan
            if k != "pygimli":
                pairs = [(S[m][k][q], S[m]["pygimli"][q]) for m in S if k in S[m] and "pygimli" in S[m]
                         and np.isfinite(S[m][k][q]) and np.isfinite(S[m]["pygimli"][q])]
                hi = q in ("ssim", "body_iou", "body_contr")
                e[q + "_wins"] = float(np.mean([(a > b) if hi else (a < b) for a, b in pairs])) if pairs else np.nan
                e[q + "_n"] = len(pairs)
        e["groups"] = {g: float(np.median([by[m][k]["rms"] for m in ms if m.split("_")[0] == g])) for g in GR}
        if k != "pygimli":
            e["group_wins"] = {g: float(np.mean([by[m][k]["rms"] < by[m]["pygimli"]["rms"] for m in ms if m.split("_")[0] == g])) for g in GR}
        d[k] = e
    KK = [k for k in M3 if k in d]
    best = [min(KK, key=lambda k: by[m][k]["rms"]) for m in by if all(k in by[m] for k in KK)]
    d["best_counts"] = {k: best.count(k) for k in KK}
    res[suite] = d
json.dump(res, open(os.path.join(OUT, "paper", "paper_stats.json"), "w"), indent=1)
for s, d in res.items():
    print(s, {k: (round(d[k]["median"], 3), round(d[k].get("wins", 0), 2), f"{d[k].get('p', 1):.1g}", round(d[k]["ssim"], 3), round(d[k]["if_err"], 2), round(d[k]["body_iou"], 2)) for k in M3 if k in d}, d["best_counts"])
print(json.dumps(res["vae_benchmark"]["vae_pg_dip"]["groups"]), json.dumps(res["vae_benchmark"]["pygimli"]["groups"]))
print(json.dumps(res["vae_benchmark"]["vae_pg_dip"]["group_wins"]))
print({k: (round(res["vae_benchmark"][k]["chi2_median"], 2), res["vae_benchmark"][k]["chi2_le_105"]) for k in M3})
print({k: res["vae_benchmark"]["vae_pg_dip"].get(k) for k in ("wins_vs_random", "p_vs_random", "rel_median")})
