"""Data misfit of the field-data inversions, as chi-squared and as RMS.

  chi2          mean of ((log f(m) - log d) / err)^2      (target 1)
  RMS (err.w.)  sqrt(chi2)                                 (target 1)
  RMS (%)       sqrt(mean(((f(m) - d) / d)^2)) * 100       (target about the data error)
Written to outputs/real_data/misfit_table.csv and .md for every run (err3pct, auto_error) that exists.
"""
import os
import numpy as np
from real_data_inversion import METHODS, LAB, FILES

OUT = os.path.join("outputs", "real_data")


def main():
    rows = []
    for run in ("err3pct", "auto_error"):
        for name in FILES:
            fn = os.path.join(OUT, run, f"{name}_results.npz")
            if not os.path.exists(fn):
                continue
            z = np.load(fn, allow_pickle=True)
            res, err, dobs = z["res"].item(), z["info"].item()["err"], z["dobs"]
            for k in [k for k in METHODS if k in res]:
                resp = res[k]["resp"]
                chi2 = float(np.mean(((np.log(resp) - np.log(dobs)) / err) ** 2))
                rel = 100 * float(np.sqrt(np.mean((resp / dobs - 1) ** 2)))
                rows.append((name, f"{100 * err:.1f}", LAB[k], chi2, np.sqrt(chi2), rel))
    hdr = ["line", "data error (%)", "method", "chi2", "RMS (error-weighted)", "RMS (%)"]
    with open(os.path.join(OUT, "misfit_table.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(hdr) + "\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},\"{r[2]}\",{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}\n")
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} |" for r in rows]
    open(os.path.join(OUT, "misfit_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
