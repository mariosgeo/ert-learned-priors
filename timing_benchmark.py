"""
timing_benchmark.py
===================
How long does each inversion method take, how much memory (CPU RAM and GPU) does it
need, and how long does the training take?
Run on an IDLE machine: one process, a fixed number of threads, methods one at
a time, so the numbers are not distorted by parallel workers or by waiting for
the GPU (the benchmark's own stored times are, see the note in the summary).

1. Per method, on 6 representative benchmark models (same data as the benchmark),
   the time split into phases:
     analysis          dip-aware / standard reading of the pyGIMLi image
     training models   generating the 20 000 pyGIMLi-guided training models
     fine-tuning       fine-tuning the general VAE on them (GPU)
     inversion         the inversion itself (latent Gauss-Newton, pyGIMLi, layers, ...)
   plus the number and cost of forward solutions (pyGIMLi response / Jacobian) and the
   memory: peak RAM of the process (sampled every 0.2 s, incl. child processes; also as
   the increase over the RAM in use before the method) and peak GPU memory allocated /
   reserved by PyTorch (torch.cuda.max_memory_*; pyGIMLi does not use the GPU).
   Each method runs alone; the methods it builds on (always pyGIMLi; see MANUAL.md)
   were computed in an earlier call and are not counted.
2. Training (one-off, before any inversion): data generation and GPU training
   speed measured on short runs and scaled to the full set-up (200 000 models /
   40 000 steps for the general VAE, 100 000 / 20 000 for the body VAE), plus the
   logged times of the trainings actually done.

    python timing_benchmark.py                   # everything (~2 h)
    python timing_benchmark.py --threads 8 --only A_03,E_07
    python timing_benchmark.py --training-only
    python timing_benchmark.py --summary
"""
import os
import sys
import csv
import time
import argparse
import platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vae_benchmark_ert as vb                                  # noqa: E402

OUTT = os.path.join(HERE, "outputs", "timing")
MODELS = ["A_03_z3_H-L", "B_05_z3-9_M-H-L", "C_01_4lay_M-M-H-L", "D2_02_parallel10",
          "E_07_ellipse_res_top4_2lay", "G_03_5bodies_fault"]
ORDER = ["pygimli", "sparse", "vae_random", "vae_pg", "vae_pg_ar", "vae_pg_dip", "gan",
         "pygimli_robust", "pygimli_dipsteer", "hybrid_vae_dip", "layered_param", "vae_pg_layer",
         "layers_vae", "ttl_cnn", "ttl_field", "pygimli_fine", "vae_pg_fine"]
PHASES = ["analysis", "training models", "fine-tuning", "inversion", "other"]
_LOG = []                                                         # (phase, seconds) of the current call
_FWD = dict(n_resp=0, t_resp=0.0, n_jac=0, t_jac=0.0)
_IN_JAC = [False]


class MemoryMonitor:
    """Peak RAM (RSS of this process and its children) sampled in a thread, and peak GPU
    memory from PyTorch, over a with-block. Values in MB."""
    def __init__(self, dt=0.2):
        import psutil
        self.p, self.dt = psutil.Process(), dt

    def rss(self):
        r = self.p.memory_info().rss
        for c in self.p.children(recursive=True):
            try:
                r += c.memory_info().rss
            except Exception:
                pass
        return r / 2 ** 20

    def __enter__(self):
        import threading
        import torch
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.base = self.peak = self.rss()
        self.stop = threading.Event()

        def run():
            while not self.stop.wait(self.dt):
                self.peak = max(self.peak, self.rss())
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        import torch
        self.stop.set()
        self.th.join()
        self.peak = max(self.peak, self.rss())
        self.gpu_alloc = self.gpu_res = 0.0
        if self.cuda:
            torch.cuda.synchronize()
            self.gpu_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
            self.gpu_res = torch.cuda.max_memory_reserved() / 2 ** 20
        return False

    def stats(self):
        return dict(ram_peak_mb=self.peak, ram_increase_mb=self.peak - self.base,
                    gpu_alloc_mb=self.gpu_alloc, gpu_reserved_mb=self.gpu_res)


def _wrap(mod, attr, phase):
    f = getattr(mod, attr)
    if getattr(f, "_timed", False):
        return

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            _LOG.append((phase, time.perf_counter() - t))
    g._timed = True
    setattr(mod, attr, g)


def instrument():
    import ert_common
    import layered_param as lp
    import layers_vae as lv
    import ttl_inversion as tt
    import gan_prior_ert as gp
    v, sp = vb.G["v"], vb.G["sp"]
    for mod, attr, ph in [(v, "analyse_pg", "analysis"), (v, "random_models_pg", "training models"),
                          (vb, "finetune", "fine-tuning"), (v, "invert_latent", "inversion"),
                          (sp, "invert_pygimli", "inversion"), (v, "run_sparse", "inversion"),
                          (lp, "invert_layered", "inversion"), (lv, "invert_layers_vae", "inversion"),
                          (tt, "invert_ttl", "inversion"), (gp, "project", "analysis")]:
        _wrap(mod, attr, ph)
    E = ert_common.ERTForward
    if not getattr(E.response, "_timed", False):
        r0, j0 = E.response, E.jacobian_loglog

        def resp(self, m):                                       # stand-alone forward solutions only
            t = time.perf_counter()
            out = r0(self, m)
            if not _IN_JAC[0]:
                _FWD["n_resp"] += 1
                _FWD["t_resp"] += time.perf_counter() - t
            return out

        def jac(self, m):
            t = time.perf_counter()
            _IN_JAC[0] = True
            try:
                out = j0(self, m)                                 # response + Jacobian
            finally:
                _IN_JAC[0] = False
            _FWD["n_jac"] += 1
            _FWD["t_jac"] += time.perf_counter() - t
            return out
        resp._timed = jac._timed = True
        E.response, E.jacobian_loglog = resp, jac


def time_methods(models, threads):
    import torch
    vb.setup(threads=threads)
    instrument()
    if vb.G["device"] == "cuda":                                  # CUDA start-up outside the timings
        torch.zeros(1, device="cuda").sum().item()
    rows = []
    suite = {s["name"]: s for s in vb.make_suite()}
    for name in models:
        spec = suite[name]
        out = os.path.join(OUTT, "runs")
        for d in ("models", "vae_pg", "vae_pg_ar", "vae_pg_dip", "vae_pg_layer"):  # start fresh
            p = os.path.join(out, d, name + (".npz" if d == "models" else ".pt"))
            if os.path.exists(p):
                os.remove(p)
        for i, meth in enumerate(ORDER):
            _LOG.clear()
            for k in _FWD:
                _FWD[k] = 0
            with MemoryMonitor() as mon:
                t0 = time.perf_counter()
                res = vb.invert_all(name, out, spec=spec, seed=1000 + vb.make_suite_index(name),
                                    data_file=os.path.join(vb.DATA_DIR, name + ".dat"), methods=ORDER[:i + 1])
                wall = time.perf_counter() - t0
            if meth not in res:
                print(f"  {name} {meth}: not run (prerequisite missing)", flush=True)
                continue
            ph = {p: sum(s for q, s in _LOG if q == p) for p in PHASES}
            ph["other"] = max(wall - sum(ph.values()), 0.0)
            row = dict(model=name, method=meth, total=wall, rms=res[meth]["rms"], **ph,
                       n_fwd=_FWD["n_resp"], n_jac=_FWD["n_jac"],
                       t_fwd=_FWD["t_resp"], t_jac=_FWD["t_jac"], **mon.stats())
            rows.append(row)
            print(f"  {name:28s} {meth:18s} {wall:7.1f}s  " +
                  "  ".join(f"{p} {ph[p]:.0f}" for p in PHASES if ph[p] > 0.5) +
                  f"  | {row['n_jac']} Jacobians, {row['n_fwd']} forwards"
                  f"  | RAM peak {row['ram_peak_mb']:.0f} MB (+{row['ram_increase_mb']:.0f}),"
                  f" GPU {row['gpu_alloc_mb']:.0f} MB", flush=True)
            with open(os.path.join(OUTT, "timing_methods.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return rows


def time_training(threads):
    import torch
    import layers_vae as lv
    vb.setup(threads=threads)
    v, G = vb.G["v"], vb.G
    dev = G["device"]
    rows = []

    def probe(label, gen, n_gen, n_full, nz, steps_probe, steps_full, batch=256, sigma=0.1):
        with MemoryMonitor() as mg:
            t = time.perf_counter()
            X = gen(n_gen)
            tg = (time.perf_counter() - t) * n_full / n_gen
        if dev == "cuda":
            torch.cuda.synchronize()
        with MemoryMonitor() as mt:
            t = time.perf_counter()
            with vb._GPULock():
                v.train_vae(X, nz, steps_probe, batch, sigma, dev, verbose=False)
            if dev == "cuda":
                torch.cuda.synchronize()
            tt = (time.perf_counter() - t) * steps_full / steps_probe
        rows.append(dict(network=label, n_models=n_full, steps=steps_full, latent=nz,
                         generate_s=tg, train_s=tt, total_s=tg + tt, device=dev,
                         ram_generate_mb=mg.stats()["ram_increase_mb"], ram_train_mb=mt.stats()["ram_peak_mb"],
                         gpu_train_mb=mt.stats()["gpu_alloc_mb"]))
        print(f"  {label:34s} generate {tg / 60:5.1f} min + train {tt / 60:5.1f} min "
              f"= {(tg + tt) / 60:5.1f} min", flush=True)

    xg, zg = G["xg"], G["zg"]
    probe("general VAE (random prior)", lambda n: v.random_models_general(n, xg, zg, seed=1), 20000, 200000,
          64, 2000, 40000)
    v.TEXTURE = True
    probe("general VAE, textured", lambda n: v.random_models_general(n, xg, zg, seed=1), 20000, 200000,
          64, 2000, 40000)
    probe("general VAE, textured, 128 latent", lambda n: v.random_models_general(n, xg, zg, seed=1), 20000,
          200000, 128, 2000, 40000)
    v.TEXTURE = False
    probe("body-only VAE (option B)", lambda n: lv.body_models(n, xg, zg, seed=11), 20000, 100000,
          32, 2000, 20000)
    with open(os.path.join(OUTT, "timing_training.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


LOGGED = [("general VAE (random prior), incl. its test inversions", "outputs/vae_prior/general", "total 10264s"),
          ("GAN (WGAN-GP, 20 000 generator steps)", "outputs/gan_prior/train.log", "83 min"),
          ("body-only VAE", "outputs/layers_vae/train.log", "19 min"),
          ("textured VAE, sigma 0.05", "outputs/hetero_tuning/train.log", "32 min"),
          ("textured VAE, 128 latent", "outputs/hetero_tuning/train.log", "32 min")]


def summary():
    L = ["TIMING BENCHMARK", f"machine: {platform.processor()}  |  {platform.platform()}", ""]
    fn = os.path.join(OUTT, "timing_training.csv")
    if os.path.exists(fn):
        R = list(csv.DictReader(open(fn)))
        L += ["1. TRAINING (once; measured on short runs, scaled to the full set-up)", "",
              f"{'network':36s}{'models':>8s}{'steps':>7s}{'generate':>10s}{'train':>8s}{'total':>8s}"
              f"{'RAM set':>9s}{'GPU':>8s}"]
        for r in R:
            L.append(f"{r['network']:36s}{int(r['n_models']):8d}{int(r['steps']):7d}"
                     f"{float(r['generate_s']) / 60:8.1f} m{float(r['train_s']) / 60:6.1f} m"
                     f"{float(r['total_s']) / 60:6.1f} m"
                     + (f"{float(r['ram_generate_mb']) * int(r['n_models']) / 20000 / 1024:7.1f} GB"
                        f"{float(r['gpu_train_mb']):6.0f} MB" if 'gpu_train_mb' in r else ""))
        L += ["", "   RAM set: memory of the training set, measured on 20 000 models and scaled to the full",
              "   set. GPU: peak PyTorch allocation during training."]
        L += ["", "   logged full trainings: " + "; ".join(f"{a}: {c}" for a, _, c in LOGGED), ""]
    fn = os.path.join(OUTT, "timing_methods.csv")
    if os.path.exists(fn):
        R = list(csv.DictReader(open(fn)))
        n = len({r["model"] for r in R})
        L += [f"2. INVERSION per data set (median over {n} models; one process, methods run alone; seconds)", "",
              f"{'method':18s}{'total':>8s}{'analysis':>10s}{'train.mod':>10s}{'fine-tune':>10s}"
              f"{'inversion':>10s}{'other':>7s}{'Jacob.':>8s}{'fwd':>6s}{'s/Jac':>7s}{'s/fwd':>7s}"
              f"{'RAM pk':>8s}{'+RAM':>7s}{'GPU':>7s}"]
        for m in ORDER:
            Rm = [r for r in R if r["method"] == m]
            if not Rm:
                continue
            med = lambda q: np.median([float(r[q]) for r in Rm])
            sj = np.median([float(r["t_jac"]) / float(r["n_jac"]) for r in Rm if float(r["n_jac"]) > 0] or [np.nan])
            sf = np.median([float(r["t_fwd"]) / float(r["n_fwd"]) for r in Rm if float(r["n_fwd"]) > 0] or [np.nan])
            L.append(f"{m:18s}{med('total'):8.0f}{med('analysis'):10.0f}{med('training models'):10.0f}"
                     f"{med('fine-tuning'):10.0f}{med('inversion'):10.0f}{med('other'):7.0f}"
                     f"{med('n_jac'):8.0f}{med('n_fwd'):6.0f}{sj:7.2f}{sf:7.2f}"
                     + (f"{max(float(r['ram_peak_mb']) for r in Rm):8.0f}{med('ram_increase_mb'):7.0f}"
                        f"{max(float(r['gpu_alloc_mb']) for r in Rm):7.0f}" if 'ram_peak_mb' in Rm[0] else ""))
        L += ["", "   'other' = loading the data / networks, writing results, true-model metrics.",
              "   Memory in MB: RAM pk = highest peak RAM of the process over the models (includes what is",
              "   already loaded: Python, PyTorch, the networks, pyGIMLi meshes), +RAM = median increase",
              "   during the method, GPU = highest peak PyTorch GPU allocation (0 = CPU only).",
              "   VAE variants: the network is trained ONCE (section 1); per data set the pyGIMLi-guided",
              "   variants add generating 20 000 training models + fine-tuning; 'inversion' is the latent",
              "   Gauss-Newton search (3 restarts + lambda stages), dominated by pyGIMLi Jacobians.",
              "   Times assume pyGIMLi has already been run (it is needed by most methods).",
              "   Jacobian / forward counts cover the ERTForward wrapper (VAE, GAN, sparse, test-time methods);",
              "   pyGIMLi's own inversions and the explicit-layer methods use their own operators (shown as 0)."]
    txt = "\n".join(L)
    open(os.path.join(OUTT, "timing_summary.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default=None, help="model-name prefixes")
    ap.add_argument("--training-only", action="store_true")
    ap.add_argument("--methods-only", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTT, exist_ok=True)
    if not a.summary:
        if not a.methods_only:
            print("training speed:", flush=True)
            time_training(a.threads)
        if not a.training_only:
            models = [m for m in MODELS if not a.only or any(m.startswith(p) for p in a.only.split(","))]
            print(f"methods on {len(models)} models, {a.threads} threads:", flush=True)
            time_methods(models, a.threads)
    summary()


if __name__ == "__main__":
    main()
