"""
timing_realdata.py
==================
Time and memory per inversion method on the FIELD data (g4_02: 24 electrodes, 365 data;
n2dd: 48 electrodes, 917 data), with the estimated data errors of outputs/real_data/auto_error.

Every method runs alone in a fresh folder (outputs/timing_realdata/runs/<method>/), with only
what it builds on copied in from auto_error: the data error, pyGIMLi's coverage and, except for
pyGIMLi itself, the pyGIMLi model; for a hybrid also its guided-VAE model. So the time of a hybrid
is the hybrid step alone, and a guided VAE includes analysis + training models + fine-tuning +
latent inversion. Not timed: the error estimate, the pyGIMLi result folders in metres, plotting.
Same phases, forward/Jacobian counts and memory monitor as timing_benchmark.py (peak RAM of the
process sampled every 0.2 s; peak PyTorch GPU memory). Run on an idle machine.

    python timing_realdata.py                 # both lines, all methods
    python timing_realdata.py --summary
"""
import os
import sys
import csv
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vae_benchmark_ert as vb                                  # noqa: E402
import real_data_inversion as rd                                # noqa: E402
import timing_benchmark as tb                                   # noqa: E402

OUTT = os.path.join(HERE, "outputs", "timing_realdata")
BASE = os.path.join(HERE, "outputs", "real_data", "auto_error")
METHODS = ["pygimli", "vae_random", "vae_pg_dip", "vae_pg_dip_strict", "vae_random_128", "vae_pg_dip_128",
           "vae_random_p128", "vae_pg_dip_p128", "hybrid_vae_dip", "hybrid_vae_dip_strict",
           "hybrid_vae_dip_128", "hybrid_vae_dip_p128"]


def prepare(name, meth, d):
    """Fresh results file holding only what the method builds on."""
    z = dict(np.load(os.path.join(BASE, f"{name}_results.npz"), allow_pickle=True))
    r = z["res"].item()
    keep = ["coverage"] + [k for k in ("pygimli_topo",) if k in r]
    if meth != "pygimli":
        keep.append("pygimli")
    if meth.startswith("hybrid"):
        keep.append(meth.replace("hybrid_vae_dip", "vae_pg_dip"))
    z["res"] = np.array({k: r[k] for k in keep}, dtype=object)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.startswith(name):
            os.remove(os.path.join(d, f))
    np.savez(os.path.join(d, f"{name}_results.npz"), **z)


def run(threads):
    import torch
    vb.setup(threads=threads)
    vb.setup = lambda **k: vb.G                                  # the benchmark grid is built once, not per run
    rd.save_pygimli = lambda *a, **k: None                       # result folders in metres: not part of the timing
    tb.instrument()
    if vb.G["device"] == "cuda":
        torch.zeros(1, device="cuda").sum().item()               # CUDA start-up outside the timings
    os.makedirs(OUTT, exist_ok=True)
    rows = []
    for name, fn in rd.FILES.items():
        for meth in METHODS:
            d = os.path.join(OUTT, "runs", meth)
            prepare(name, meth, d)
            tb._LOG.clear()
            for k in tb._FWD:
                tb._FWD[k] = 0
            with tb.MemoryMonitor() as mon:
                t0 = time.perf_counter()
                rd.invert_file(name, fn, d, error="auto", methods=[meth], verbose=False, threads=threads)
                wall = time.perf_counter() - t0
            res = np.load(os.path.join(d, f"{name}_results.npz"), allow_pickle=True)["res"].item()
            ph = {p: sum(s for q, s in tb._LOG if q == p) for p in tb.PHASES}
            ph["other"] = max(wall - sum(ph.values()), 0.0)
            row = dict(line=name, method=meth, total=wall, chi2=res[meth]["chi2"], **ph,
                       n_fwd=tb._FWD["n_resp"], n_jac=tb._FWD["n_jac"], t_fwd=tb._FWD["t_resp"],
                       t_jac=tb._FWD["t_jac"], **mon.stats())
            rows.append(row)
            print(f"  {name:6s} {meth:22s} {wall:7.1f}s  " +
                  "  ".join(f"{p} {ph[p]:.0f}" for p in tb.PHASES if ph[p] > 0.5) +
                  f"  | {row['n_jac']} Jacobians, {row['n_fwd']} forwards | RAM peak {row['ram_peak_mb']:.0f} MB "
                  f"(+{row['ram_increase_mb']:.0f}), GPU {row['gpu_alloc_mb']:.0f} MB | chi2 {row['chi2']:.2f}", flush=True)
            with open(os.path.join(OUTT, "timing_realdata.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, list(rows[0]))
                w.writeheader()
                w.writerows(rows)


def summary():
    R = list(csv.DictReader(open(os.path.join(OUTT, "timing_realdata.csv"))))
    import platform
    L = ["TIMING AND MEMORY ON THE FIELD DATA", f"machine: {platform.processor()}  |  {platform.platform()}", "",
         "seconds; memory in MB (RAM pk = peak RAM of the process incl. what is already loaded, +RAM = increase",
         "during the method, GPU = peak PyTorch GPU allocation, 0 = CPU only)", ""]
    for name in rd.FILES:
        L += [f"{name}", f"{'method':42s}{'total':>7s}{'analys':>7s}{'tr.mod':>7s}{'f-tune':>7s}{'invers':>7s}"
              f"{'other':>6s}{'Jac':>5s}{'fwd':>5s}{'RAM pk':>8s}{'+RAM':>6s}{'GPU':>6s}{'chi2':>7s}"]
        for r in [r for r in R if r["line"] == name]:
            f = lambda q: float(r[q])
            L.append(f"{rd.LAB[r['method']]:42s}{f('total'):7.0f}{f('analysis'):7.0f}{f('training models'):7.0f}"
                     f"{f('fine-tuning'):7.0f}{f('inversion'):7.0f}{f('other'):6.0f}{f('n_jac'):5.0f}{f('n_fwd'):5.0f}"
                     f"{f('ram_peak_mb'):8.0f}{f('ram_increase_mb'):6.0f}{f('gpu_alloc_mb'):6.0f}{f('chi2'):7.2f}")
        L.append("")
    L += ["Hybrid = the hybrid step alone (its guided VAE is listed separately; add both for the full cost).",
          "Guided VAE = dip-aware analysis + 20 000 training models + fine-tuning (GPU) + latent inversion.",
          "pyGIMLi's own inversions use their own operators, so their Jacobian counts show as 0."]
    txt = "\n".join(L)
    open(os.path.join(OUTT, "timing_realdata_summary.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if not a.summary:
        run(a.threads)
    summary()


if __name__ == "__main__":
    main()
