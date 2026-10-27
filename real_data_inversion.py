"""
real_data_inversion.py
======================
Invert field data with pyGIMLi, the random-prior VAE and the dip-aware pyGIMLi-guided VAE,
re-using the networks trained for the synthetic benchmark (no retraining).

Re-using the trained VAE for another electrode spacing
------------------------------------------------------
DC resistivity is scale invariant: multiplying every distance (electrodes and model) by the
same factor f leaves the APPARENT resistivities unchanged. Each line is therefore rescaled so
that its length matches the benchmark line (47 units, electrodes from 0 to 47), the
inversion is done in these units with the benchmark mesh settings and the benchmark VAE, and
the results are scaled back to metres (x / f, depth / f) for the plots.
  * resistances (V/I) are NOT scale invariant: they are converted to apparent resistivity with
    the geometric factors of the TRUE geometry before rescaling;
  * the surface is taken as flat (the VAE images and meshes are flat). For a file with
    topography in its .A2D companion, pyGIMLi is also run with the topography as a check.

Data error
----------
--error 0.03   a fixed relative error for all data (the benchmark value)
--error auto   (default) estimated per line from the data: pyGIMLi is run down a long lambda
               path with 3 % error, every lambda to convergence (no stop at chi2 = 1, up to 40
               iterations, so that the 3 % does not limit the misfit); the relative RMS misfit 3 % * sqrt(chi2) stops decreasing
               at the noise floor - the first lambda whose misfit is less than 10 % below the
               previous one gives the error (at least 1 %). All methods then use this error.

Methods: pygimli; vae_random, vae_pg_dip (64-latent VAE of the benchmark);
         vae_random_128, vae_pg_dip_128 (128-latent VAE trained on textured models),
         vae_random_p128, vae_pg_dip_p128 (128-latent VAE trained on plain models: the control).

    python real_data_inversion.py --cpu                   # both files, estimated error
    python real_data_inversion.py --error 0.03 --only n2dd
    python real_data_inversion.py --plot-only
Outputs: outputs/real_data/<auto_error | err3pct>/ : <name>_models / _differences / _datafit /
         _error_estimate / _topography_check (.png, .pdf), <name>_results.npz,
         pygimli_<name>_<flat|topography>/ERTManager (pyGIMLi saveResult, in metres)
"""
import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTR = os.path.join(HERE, "outputs", "real_data")
LN10 = np.log(10.0)
L_BENCH = 47.0                                   # benchmark line length (units)
NOISE0 = 0.03                                    # relative data error of the benchmark
FILES = {"g4_02": "real_data/g4_02.pyg", "n2dd": "real_data/n2dd.pyg", "N4DD": "real_data/N4DD.pyg"}
TOPO = {"n2dd": "real_data/n2dd.A2D", "N4DD": "real_data/N4DD.A2D"} # electrode elevations (column 3)
LAMS_PG = [300, 100, 50, 20, 10, 5, 2]
LAMS_EST = [300, 100, 50, 20, 10, 5, 2, 1, 0.5]  # lambda path for the error estimate
VAE128 = os.path.join(HERE, "outputs", "hetero_tuning", "vae_tex_nz128.pt")
VAEP128 = os.path.join(HERE, "outputs", "hetero_tuning", "vae_plain128.pt")   # control: 128 latent, no texture
METHODS = ["pygimli", "vae_random", "vae_pg_dip", "vae_random_128", "vae_pg_dip_128",
           "vae_random_p128", "vae_pg_dip_p128", "vae_pg_dip_strict", "hybrid_vae_dip", "hybrid_vae_dip_128",
           "hybrid_vae_dip_p128", "hybrid_vae_dip_strict"]
STRICT = dict(min_amp=0.2, drop_below_doi=True, edge_margin=3.0)   # stricter reading of the pyGIMLi image
STRICT_EXTRAS = (0.9, 0.1, 0.0)                    # fewer random extra bodies in the guided training set
LAB = {"pygimli": "pyGIMLi (blocky, zWeight 0.3)", "vae_random": "VAE, random prior (64)",
       "vae_pg_dip": "VAE, pyGIMLi-guided (64)", "vae_random_128": "VAE, random prior (128, textured)",
       "vae_pg_dip_128": "VAE, pyGIMLi-guided (128, textured)",
       "hybrid_vae_dip": "Hybrid (pyGIMLi + guided VAE 64)",
       "hybrid_vae_dip_128": "Hybrid (pyGIMLi + guided VAE 128)",
       "vae_random_p128": "VAE, random prior (128, plain)", "vae_pg_dip_p128": "VAE, pyGIMLi-guided (128, plain)",
       "hybrid_vae_dip_p128": "Hybrid (pyGIMLi + guided VAE 128 plain)",
       "vae_pg_dip_strict": "VAE, pyGIMLi-guided (64, strict prior)",
       "hybrid_vae_dip_strict": "Hybrid (pyGIMLi + guided VAE 64, strict prior)"}


# ── data ─────────────────────────────────────────────────────────────────
def read_data(fn, err=NOISE0):
    """Field data -> (data in benchmark units with rhoa, err, k; info incl. scale factor f)."""
    import pygimli as pg
    from pygimli.physics import ert
    d = pg.DataContainerERT(os.path.join(HERE, fn))
    x = np.array([p.x() for p in d.sensors()])
    if d.haveData("rhoa") and np.any(np.array(d["rhoa"]) != 0):
        rhoa = np.array(d["rhoa"])
        kind = "apparent resistivity"
    else:                                         # resistance: k of the TRUE spacing on a flat surface
        dk = pg.DataContainerERT(d)               # elevations in the file (z column) would be read as burial
        for i in range(dk.sensorCount()):         # depths by the analytical k (up to 2x too large), so x only
            dk.setSensorPosition(i, [x[i], 0.0, 0.0])
        rhoa = np.array(ert.createGeometricFactors(dk, numerical=False)) * np.array(d["r"])
        kind = "resistance -> apparent resistivity (true spacing, flat surface)"
    x0 = x.min()
    f = L_BENCH / (x.max() - x0)
    ds = pg.DataContainerERT(d)
    for i in range(ds.sensorCount()):
        ds.setSensorPosition(i, [(x[i] - x0) * f, 0.0, 0.0])
    ds["k"] = ert.createGeometricFactors(ds, numerical=False)
    ds["rhoa"] = rhoa
    ds["err"] = np.full(ds.size(), err)
    bad = np.where(~(rhoa > 0))[0]
    if len(bad):
        ds.markInvalid(bad)
        ds.removeInvalid()
    info = dict(file=fn, kind=kind, n_elec=d.sensorCount(), n_data=ds.size(), removed=len(bad),
                spacing=float(np.median(np.diff(np.sort(x)))), length=float(x.max() - x0), f=f, x0=x0, err=err)
    return ds, info


def read_topography(fn):
    """Electrode x and elevation from an .A2D file (index, x, z, ...)."""
    lines = open(os.path.join(HERE, fn)).read().split("\n")
    n = int(lines[3].split()[0])
    xz = np.array([[float(v) for v in ln.split()[1:3]] for ln in lines[4:4 + n]])
    return xz[:, 0], xz[:, 1]


def set_error(ds, info, err):
    """Use relative error err everywhere (data weights, chi2 of every method)."""
    import sparse_torch_inversion_ert as sp
    ds["err"] = np.full(ds.size(), err)
    info["err"] = float(err)
    sp.NOISE = float(err)                         # chi2 / residuals in invert_pygimli and invert_latent


def estimate_error(ds, mesh, verbose=True):
    """Noise floor of the data from a pyGIMLi lambda path (see the module docstring)."""
    import sparse_torch_inversion_ert as sp
    sp.NOISE = NOISE0
    ds["err"] = np.full(ds.size(), NOISE0)
    _, runs = sp.invert_pygimli(ds, mesh, LAMS_EST, 1e-9, 0.3, True, run_all=True, verbose=verbose,
                                inv_kw=dict(stopAtChi1=False, maxIter=40))   # converge at every lambda
    lams = np.array([r["lam"] for r in runs])
    rms = NOISE0 * np.sqrt(np.array([r["chi2"] for r in runs]))
    k = next((i for i in range(1, len(rms)) if rms[i] > 0.9 * rms[i - 1]), len(rms) - 1)
    return max(float(rms[k]), 0.01), dict(lams=lams, rms=rms, k=k)


# ── inversions ───────────────────────────────────────────────────────────
def context(ds):
    """Mesh, image grid, sampling and sensitivity for this survey (benchmark settings)."""
    import sparse_torch_inversion_ert as sp
    import vae_prior_inversion_ert as v
    from ert_common import ERTForward
    mesh, fwd0, cc, area = sp.build_inversion_mesh(ds)
    m_hom = np.full(len(cc), np.log(np.median(np.array(ds["rhoa"]))))
    J0, _ = fwd0.jacobian_loglog(m_hom)
    cov = np.sqrt(np.sum(J0 ** 2, axis=0)) / np.sqrt(area)        # relative: scale-free
    xg, zg = v.make_grid(cc)
    return dict(mesh=mesh, cc=cc, tri=sp._triangles(fwd0.fop.paraDomain), xg=xg, zg=zg,
                S=v.bilinear_matrix(cc, xg, zg), cov=cov, covg=v.grid_from_cells(cc, cov / cov.max(), xg, zg),
                fwd=ERTForward(ds, mesh))


def invert_file(name, fn, out, error="auto", methods=METHODS, verbose=True, threads=4, cpu=False, seed=1):
    import torch
    from scipy.spatial import cKDTree
    import vae_benchmark_ert as vb
    vb.setup(threads=threads)                             # options and fine-tuning helpers
    if cpu:                                               # never touch the GPU (or its lock), so that
        vb.G["device"] = "cpu"                            # a parallel timing benchmark stays clean

        class _NoLock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        vb._GPULock = _NoLock
    sp, v, o = vb.G["sp"], vb.G["v"], vb.G["opts"]
    os.makedirs(out, exist_ok=True)
    fn_out = os.path.join(out, f"{name}_results.npz")
    old = np.load(fn_out, allow_pickle=True) if os.path.exists(fn_out) else None
    res = old["res"].item() if old is not None else {}
    ds, info = read_data(fn)
    C = context(ds)
    if error == "auto":
        if old is not None and "err_path" in old["info"].item():
            info.update({k: old["info"].item()[k] for k in ("err", "err_path")})
        else:
            err, path = estimate_error(ds, C["mesh"], verbose=verbose)
            info.update(err=err, err_path=path)
    else:
        info["err"] = float(error)
    set_error(ds, info, info["err"])
    print(f"{name}: {info}", flush=True)
    ds.save(os.path.join(out, f"{name}_scaled.dat"))
    dobs = np.array(ds["rhoa"])
    lams = [float(x) for x in o["lams"].split(",")]

    def save():
        np.savez(fn_out, res=np.array(res, dtype=object), info=np.array(info, dtype=object),
                 cc=C["cc"], tri_nodes=C["tri"][0], tri_cells=C["tri"][1], cov=C["cov"], dobs=dobs,
                 xg=C["xg"], zg=C["zg"])

    def response(m):
        return np.array(C["fwd"].response(m))
    if "pygimli" not in res:
        t = time.time()
        pgr, _ = sp.invert_pygimli(ds, C["mesh"], LAMS_PG, 1.05, 0.3, True, verbose=verbose)
        m = pgr["m"][cKDTree(pgr["cc"]).query(C["cc"])[1]]
        res["pygimli"] = dict(m=m, lam=pgr["lam"], chi2=pgr["chi2"], resp=response(m), time=time.time() - t)
        save()
    if "coverage" not in res:                            # pyGIMLi coverage() of the pyGIMLi inversion in metres
        r = invert_real_geometry(name, ds, info, res["pygimli"]["lam"])
        res["coverage"] = dict(cov=r["coverage"], cc=r["cc"])
        save()
    img = v.grid_from_cells(C["cc"], res["pygimli"]["m"] / LN10, C["xg"], C["zg"])
    info_pg = None
    for key in [k for k in methods if k.startswith("vae")]:
        if key in res:
            continue
        t = time.time()
        ckpt = VAE128 if key.endswith("_128") else VAEP128 if key.endswith("_p128") else vb.RANDOM_CKPT
        if "pg_dip" in key:                                     # guided: fine-tune around pyGIMLi
            strict = key.endswith("_strict")
            info_pg = v.analyse_pg(img, C["covg"], C["xg"], C["zg"], dip_aware=True, **(STRICT if strict else {}))
            if verbose:
                print(f"  {key}: {len(info_pg['bodies'])} anomalies kept, {len(info_pg['dropped'])} dropped", flush=True)
            v.TEXTURE = key.endswith("_128")                    # as the base VAE was trained
            X = v.random_models_pg(o["n_pg"], C["xg"], C["zg"], info_pg, seed, o["pg_mix"], o["pg_extend"],
                                   extras=STRICT_EXTRAS if strict else (0.6, 0.3, 0.1))
            v.TEXTURE = False
            o["random_ckpt"] = ckpt
            model, mean, std = vb.finetune(X, seed=seed)
            o["random_ckpt"] = vb.RANDOM_CKPT
            torch.save(dict(state=model.state_dict(), nz=model.nz, mean=mean, std=std),
                       os.path.join(out, f"{name}_{key}.pt"))
        else:
            model, mean, std = vb.load_vae(ckpt)
        r = v.invert_latent(C["fwd"], model, mean, std, C["S"], dobs, lams, restarts=3, verbose=verbose,
                            z_starts=[vb.encode(model, img, mean, std)])
        res[key] = dict(m=r["m"], lam=r["lam"], chi2=r["chi2"], z=r["z"].numpy(), resp=response(r["m"]),
                        time=time.time() - t)
        save()
    for key in [k for k in methods if k.startswith("hybrid") and k not in res]:
        base = key.replace("hybrid_vae_dip", "vae_pg_dip")   # hybrid: pyGIMLi (smooth, zWeight 0.3) with the
        if base not in res:                                  # guided-VAE model as starting AND reference model,
            continue                                         # as the benchmark method hybrid_vae_dip
        t = time.time()
        pgr, _ = sp.invert_pygimli(ds, C["mesh"], LAMS_PG, 1.05, 0.3, False, verbose=verbose, start=res[base]["m"])
        m = pgr["m"][cKDTree(pgr["cc"]).query(C["cc"])[1]]
        res[key] = dict(m=m, lam=pgr["lam"], chi2=pgr["chi2"], resp=response(m), time=time.time() - t)
        save()
    if name in TOPO and "pygimli_topo" not in res:      # check: pyGIMLi with the real topography
        res["pygimli_topo"] = invert_real_geometry(name, ds, info, res["pygimli"]["lam"], topography=True)
        save()
    save_pygimli(name, fn, out)
    return fn_out


COV_CUT = 0.01                                   # pyGIMLi standardizedCoverage(): coverage() > 0.01
COV_SCALE_FREE = set(FILES)                      # coverage per cell area in benchmark units (see doi_coverage)


def doi_coverage(name, cv, f):
    """pyGIMLi coverage() for the DOI threshold. coverage() divides scale-free log sensitivities by the cell
    area in m^2, so the fixed 0.01 is stricter on long lines (16x for the 94 m N4DD vs the 23.5 m n2dd). For
    the lines in COV_SCALE_FREE (all field lines) the areas are taken in benchmark units (line = 47), as in the
    synthetic benchmark, so that the same threshold means the same for every line."""
    return cv["cov"] - 2 * np.log10(f) if name in COV_SCALE_FREE else cv["cov"]


RHO_LIM = {"N4DD": (40, 300)}                    # fixed resistivity colour scale (ohm m); others: 1-99 % of the cells
RHO_TICKS = [20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000]


def rho_norm(name, allm):
    """Log colour scale: RHO_LIM for this line, else the 1-99 % percentiles of log10(rho) values allm."""
    from matplotlib.colors import LogNorm
    if name in RHO_LIM:
        return LogNorm(*RHO_LIM[name])
    return LogNorm(10 ** np.percentile(allm, 1), 10 ** np.percentile(allm, 99))


def rho_colorbar(fig, mappable, name, **kw):
    """Resistivity colour bar; with RHO_LIM labelled ticks across the range, arrows for clipped values."""
    if name not in RHO_LIM:
        return fig.colorbar(mappable, **kw)
    lo, hi = RHO_LIM[name]
    cb = fig.colorbar(mappable, extend="both", **kw)
    t = [v for v in RHO_TICKS if lo <= v <= hi]
    cb.set_ticks(t, labels=[f"{v:g}" for v in t])
    cb.ax.minorticks_off()
    return cb


def invert_real_geometry(name, ds, info, lam, topography=False, save_dir=None):
    """pyGIMLi on the TRUE geometry in metres (flat, or with the .A2D topography), with the
    benchmark settings and the lambda chosen in the scaled inversion; the mesh settings are the
    benchmark ones scaled back by 1/f, so the result is the scaled inversion in metres.
    save_dir: also write pyGIMLi's own result folder (ERTManager.saveResult)."""
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    import sparse_torch_inversion_ert as sp
    f, x0, err = info["f"], info["x0"], info["err"]
    if topography:
        xe, ze = read_topography(TOPO[name])
    else:
        xe = np.array([p.x() for p in ds.sensors()]) / f + x0
        ze = np.zeros(len(xe))
    d = pg.DataContainerERT(ds)
    for i in range(d.sensorCount()):
        d.setSensorPosition(i, [xe[i] - x0, ze[i], 0.0])           # x along the line, elevation as y
    d["k"] = ert.createGeometricFactors(d, numerical=bool(topography))
    d["rhoa"] = ds["rhoa"]
    d["err"] = np.full(d.size(), err)
    mesh = mt.createParaMesh(d, paraDepth=sp.PARA_DEPTH / f, paraBoundary=sp.PARA_BOUNDARY / f, quality=33,
                             paraMaxCellSize=sp.INV_CELL / f ** 2, boundaryMaxCellSize=50 / f ** 2)
    mgr = ert.ERTManager(d, verbose=False)
    mgr.invert(mesh=mesh, lam=lam, zWeight=0.3, blockyModel=True, verbose=False)
    pd = mgr.paraDomain
    resp = np.array(mgr.inv.response)
    chi2 = float(np.mean(((np.log(resp) - np.log(np.array(d["rhoa"]))) / err) ** 2))
    if save_dir:
        out = mgr.saveResult(save_dir)
        path = out[0] if isinstance(out, tuple) else out
        import matplotlib.pyplot as plt
        plt.close("all")
        print(f"  pyGIMLi results ({'topography' if topography else 'flat'}) saved to {path}", flush=True)
    return dict(m=np.log(np.array(mgr.paraModel())), chi2=chi2, lam=lam,
                coverage=np.array(mgr.coverage()),        # pyGIMLi: log10(sum |dlog rhoa/dlog rho| / area)
                cc=np.array([[c.center().x(), c.center().y()] for c in pd.cells()]),
                tri_nodes=np.array([[n.pos().x(), n.pos().y()] for n in pd.nodes()]),
                tri_cells=np.array([[n.id() for n in c.nodes()] for c in pd.cells()]),
                elev=np.column_stack([xe - x0, ze]))


def save_pygimli(name, fn, out):
    """pyGIMLi result folders (ERTManager.saveResult) in metres, with the error and lambda of the
    saved run: <out>/pygimli_<name>_flat/ERTManager/ and ..._topography/ERTManager/."""
    fn_out = os.path.join(out, f"{name}_results.npz")
    z = np.load(fn_out, allow_pickle=True)
    res, info = z["res"].item(), z["info"].item()
    ds, _ = read_data(fn, err=info["err"])
    for topo in ([False, True] if name in TOPO else [False]):
        tag = "topography" if topo else "flat"
        r = invert_real_geometry(name, ds, info, res["pygimli"]["lam"], topography=topo,
                                 save_dir=os.path.join(out, f"pygimli_{name}_{tag}"))
        print(f"  {name} {tag}: error {info['err']:.2%}, lambda {res['pygimli']['lam']:g}, chi2 {r['chi2']:.2f}",
              flush=True)


# ── plots ────────────────────────────────────────────────────────────────
def _save(fig, path, **kw):
    """Save; if the file is open in a viewer (locked on Windows), save next to it as <name>_new.<ext>."""
    try:
        fig.savefig(path, **kw)
    except OSError:
        root, ext = os.path.splitext(path)
        fig.savefig(root + "_new" + ext, **kw)
        print(f"  {os.path.basename(path)} is open elsewhere - saved as {os.path.basename(root)}_new{ext}", flush=True)


def coverage_alpha(C, drop=0.4):
    """Opacity from pyGIMLi coverage values exactly as pygimli.viewer.mpl.addCoverageAlpha: the 2 %
    quantile is transparent, values above the 40 % quantile are opaque, linear in between."""
    C = np.asarray(C, float)
    nn, hh = np.histogram(C, 50)
    nnn = nn.cumsum() / float(len(C))
    mi = hh[np.min(np.where(nnn > 0.02)[0])]
    ma = np.max(C) if np.min(nnn) > drop else hh[np.max(np.where(nnn < drop)[0])]
    A = (C - mi) / (ma - mi)
    A[A < 0] = 0.0
    A[A > 0.95] = 1.0
    return np.clip(A, 0, 1)


def plot_file(name, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.tri import Triangulation
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    z = np.load(os.path.join(out, f"{name}_results.npz"), allow_pickle=True)
    res, info = z["res"].item(), z["info"].item()
    f, err, L = info["f"], info["err"], info["length"]
    nodes = z["tri_nodes"] / f                                 # back to metres
    cc = z["cc"] / f
    cov = z["cov"] / z["cov"].max()
    alpha = None
    if "coverage" in res:                                      # pyGIMLi coverage(), shown as pyGIMLi does
        from scipy.spatial import cKDTree
        cv = res["coverage"]
        cov_pg = doi_coverage(name, cv, f)[cKDTree(cv["cc"]).query(cc)[1]]
        alpha = coverage_alpha(cv["cov"])[cKDTree(cv["cc"]).query(cc)[1]]
        below = cov_pg <= COV_CUT                              # outside pyGIMLi's standardizedCoverage()
        alpha = np.where(below, np.minimum(alpha, 0.3), alpha)
        doi_field, doi_level = cov_pg, COV_CUT
        doi_txt = (f"opacity: pyGIMLi coverage()\ndashed: depth of investigation, pyGIMLi standardizedCoverage() "
                   f"(coverage > {COV_CUT:g}" + (", cell areas in benchmark units" if name in COV_SCALE_FREE else "")
                   + "), faded below")
    else:                                                      # fall back: sensitivity threshold
        below = cov < 0.01
        doi_field, doi_level = np.log10(np.maximum(cov, 1e-6)), -2
        doi_txt = "grey: below the depth of investigation (normalized sensitivity < 0.01)"
    from matplotlib.collections import PolyCollection
    from matplotlib.cm import ScalarMappable
    verts = nodes[z["tri_cells"]]                               # (n_cells, 3, 2) triangles in metres

    def cells(ax, values, cmap, nrm, fade=True):
        """Each triangle drawn once, edges in its own colour (no seams). With pyGIMLi coverage: colours
        blended towards white with the coverage opacity (as a transparent model over a white page);
        otherwise cells below the depth of investigation blended 70 % towards light grey."""
        cm = matplotlib.colormaps[cmap]
        col = cm(nrm(values))
        if fade and alpha is not None:
            col[:, :3] = alpha[:, None] * col[:, :3] + (1 - alpha[:, None]) * 1.0
        elif fade:
            col[below, :3] = 0.3 * col[below, :3] + 0.7 * 0.86
        ax.add_collection(PolyCollection(verts, facecolors=col, edgecolors=col, linewidths=0.25))
        return ScalarMappable(norm=nrm, cmap=cm)
    keys = [k for k in METHODS if k in res]
    allm = np.concatenate([res[k]["m"][~below] for k in keys]) / LN10
    norm = rho_norm(name, allm)
    zmax = 0.45 * L
    elec = np.linspace(0, L, info["n_elec"])

    def frame(ax, title):
        if doi_field is not None:
            ax.tricontour(cc[:, 0], cc[:, 1], doi_field, levels=[doi_level], colors="k", linewidths=0.6,
                          linestyles="--")
        ax.plot(elec, np.zeros_like(elec), "kv", ms=2.5)
        ax.set_xlim(-0.05 * L, 1.05 * L)
        ax.set_ylim(-zmax, 0)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_ylabel("depth (m)")
    head = (f"{name}: {info['n_elec']} electrodes, {info['spacing']:.2f} m spacing, {info['n_data']} data, "
            f"data error {err:.1%}" + "\n" + doi_txt)
    # 1. models
    fig, axs = plt.subplots(len(keys), 1, figsize=(7.2, 1.9 * len(keys) + 0.7), constrained_layout=True)
    for ax, k in zip(axs, keys):
        im = cells(ax, np.exp(res[k]["m"]), "turbo", norm)
        frame(ax, f"{LAB[k]}:  χ² = {res[k]['chi2']:.2f},  λ = {res[k]['lam']:g}")
    axs[-1].set_xlabel("x (m)")
    rho_colorbar(fig, im, name, ax=axs, shrink=0.5, pad=0.03, aspect=30, label="resistivity (Ω·m)")
    fig.suptitle(head, fontsize=7.5)
    for ext in ("png", "pdf"):
        _save(fig, os.path.join(out, f"{name}_models.{ext}"), dpi=160)
    plt.close(fig)
    # 2. differences to pyGIMLi
    fig, axs = plt.subplots(len(keys) - 1, 1, figsize=(7.2, 1.9 * (len(keys) - 1) + 0.7), constrained_layout=True)
    for ax, k in zip(np.atleast_1d(axs), keys[1:]):
        im = cells(ax, (res[k]["m"] - res["pygimli"]["m"]) / LN10, "RdBu_r", TwoSlopeNorm(0, -0.6, 0.6))
        frame(ax, f"{LAB[k]} minus pyGIMLi")
    np.atleast_1d(axs)[-1].set_xlabel("x (m)")
    fig.colorbar(im, ax=axs, shrink=0.5, pad=0.03, aspect=30, label="log10(ρ / ρ_pyGIMLi)")
    fig.suptitle(head, fontsize=7.5)
    for ext in ("png", "pdf"):
        _save(fig, os.path.join(out, f"{name}_differences.{ext}"), dpi=160)
    plt.close(fig)
    # 3. data fit
    dobs = z["dobs"]
    ncol = (len(keys) + 2) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7), constrained_layout=True)
    axs = axs.ravel()
    lo, hi = dobs.min() * 0.8, dobs.max() * 1.25
    for ax, k in zip(axs, keys):
        ax.loglog(dobs, res[k]["resp"], ".", ms=3, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("observed ρa (Ω·m)")
        ax.set_ylabel("calculated ρa (Ω·m)")
        ax.set_title(f"{LAB[k]}\nχ² = {res[k]['chi2']:.2f}", fontsize=8)
    ax = axs[len(keys)]
    for k in keys:
        ax.hist((np.log(res[k]["resp"]) - np.log(dobs)) / err, bins=np.linspace(-6, 6, 49), histtype="step",
                lw=1.4, label=LAB[k])
    ax.set_xlabel(f"normalized residual (log ρa / {err:.1%})")
    ax.set_title("residual distribution", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    for ax in axs[len(keys) + 1:]:
        ax.set_axis_off()
    for ext in ("png", "pdf"):
        _save(fig, os.path.join(out, f"{name}_datafit.{ext}"), dpi=160)
    plt.close(fig)
    # 4. error estimate
    if "err_path" in info:
        p = info["err_path"]
        fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
        ax.semilogx(p["lams"], 100 * p["rms"], "o-")
        ax.axhline(100 * err, color="C3", ls="--", label=f"estimated error {err:.1%}")
        ax.plot(p["lams"][p["k"]], 100 * p["rms"][p["k"]], "o", color="C3", ms=9, mfc="none")
        ax.invert_xaxis()
        ax.set_xlabel("λ (pyGIMLi, blocky, zWeight 0.3)")
        ax.set_ylabel("relative RMS misfit (%)")
        ax.set_title(f"{name}: noise floor of the data", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        for ext in ("png", "pdf"):
            _save(fig, os.path.join(out, f"{name}_error_estimate.{ext}"), dpi=160)
        plt.close(fig)
    # 5. topography check
    if "pygimli_topo" in res:
        t = res["pygimli_topo"]
        fig, axs = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
        im = cells(axs[0], np.exp(res["pygimli"]["m"]), "turbo", norm)
        frame(axs[0], f"pyGIMLi, flat surface: χ² = {res['pygimli']['chi2']:.2f}")
        tt = Triangulation(t["tri_nodes"][:, 0], t["tri_nodes"][:, 1], t["tri_cells"])
        axs[1].tripcolor(tt, np.exp(t["m"]), cmap="turbo", norm=norm, shading="flat", edgecolors="face",
                         linewidth=0.25)
        axs[1].plot(t["elev"][:, 0], t["elev"][:, 1], "kv", ms=2.5)
        axs[1].set_title(f"pyGIMLi, with topography: χ² = {t['chi2']:.2f}", loc="left", fontsize=8.5)
        ztop = t["elev"][:, 1].max()
        axs[1].set_ylim(ztop - zmax, ztop + 0.3)
        axs[1].set_xlim(-0.05 * L, 1.05 * L)
        axs[1].set_aspect("equal")
        axs[1].set_ylabel("elevation (m)")
        axs[1].set_xlabel("x (m)")
        rho_colorbar(fig, im, name, ax=axs, shrink=0.5, pad=0.03, aspect=30, label="resistivity (Ω·m)")
        for ext in ("png", "pdf"):
            _save(fig, os.path.join(out, f"{name}_topography_check.{ext}"), dpi=160)
        plt.close(fig)
    lines = [f"{name}: {info}"] + [f"  {k:16s} chi2 {res[k]['chi2']:.2f}  lambda {res[k]['lam']:g}  "
                                   f"time {res[k].get('time', 0) / 60:.1f} min" for k in res if k in METHODS or k == "pygimli_topo"]
    if "coverage" in res:
        lines.append(f"  pyGIMLi coverage: share of the section with coverage > {COV_CUT:g} "
                     f"(standardizedCoverage): {np.mean(doi_coverage(name, res['coverage'], f) > COV_CUT):.0%}")
    open(os.path.join(out, f"{name}_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--error", default="auto", help="'auto' (estimated per line) or a relative error, e.g. 0.03")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="fine-tune on the CPU, never use the GPU")
    ap.add_argument("--seed", type=int, default=None, help="repeat with this seed (training models and "
                    "fine-tuning) in <run>/repeats/seed<N>, starting from the pyGIMLi result of <run>")
    a = ap.parse_args()
    out = os.path.join(OUTR, "auto_error" if a.error == "auto" else f"err{float(a.error) * 100:g}pct")
    base = out
    if a.seed is not None:
        out = os.path.join(base, "repeats", f"seed{a.seed}")
        os.makedirs(out, exist_ok=True)
    for name, fn in FILES.items():
        if a.only and name not in a.only.split(","):
            continue
        if a.seed is not None and not os.path.exists(os.path.join(out, f"{name}_results.npz")):
            z = dict(np.load(os.path.join(base, f"{name}_results.npz"), allow_pickle=True))
            r = z["res"].item()                               # same pyGIMLi model, coverage and data error
            keep = ["pygimli", "coverage"] + ([k for k in a.methods.split(",") if k in r] if a.seed == 1 else [])
            z["res"] = np.array({k: r[k] for k in keep}, dtype=object)   # seed 1 = the run in <run>
            np.savez(os.path.join(out, f"{name}_results.npz"), **z)
        if not a.plot_only:
            invert_file(name, fn, out, error=a.error, methods=a.methods.split(","), threads=a.threads, cpu=a.cpu,
                        seed=1 if a.seed is None else a.seed)
        plot_file(name, out)


if __name__ == "__main__":
    main()
