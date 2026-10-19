"""
structure_metrics.py
====================
Structure / target metrics for every saved benchmark result (complementing the
RMS of log10(rho_rec / rho_true)). Computed on a regular 0.25 m grid
(x 0-47 m, depth 0-12 m) restricted to the well-covered zone (sp.roi_mask);
recovered models are interpolated linearly from the cells, true models are
evaluated exactly at the grid points (sharp boundaries, incl. heterogeneity).

  ssim        structural similarity index (Wang et al. 2004) of the log10
              images, Gaussian window sigma = 1 m, data range 3 log10 units;
              mean over the covered zone. 1 = identical structure.
  if_err      interface depth error (m): for every grid column and every true
              interface (from the layer index of the model WITHOUT bodies and
              heterogeneity, so faults and truncations are included), the depth
              of the largest |d log rho / dz| of the recovered model within
              +-2.5 m (peak refined by a parabola); mean |depth difference| (each
              point capped at 2.5 m by the search window). Interface points within 1.5 m of a body are skipped.
  body_iou    per body: recovered anomaly = cells (grid points) within 4 m of
              the body whose deviation from the TRUE host model (model without
              bodies) has the body's sign and at least half its contrast;
              IoU = |anomaly AND body| / |anomaly OR body|. Mean over bodies.
  body_contr  per body: median recovered deviation inside the body / true
              contrast (1 = full contrast recovered). Mean over bodies.
  body_shift  per body: distance (m) between the centroids of the recovered
              anomaly and the body; median over bodies (nan if nothing found).

    python structure_metrics.py                    # every suite found in outputs/
    python structure_metrics.py --suites vae_benchmark,vae_benchmark_het50
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs")
OUTS = os.path.join(OUT, "structure_metrics")
DX = 0.25
XG = np.arange(0.0, 47.0 + 1e-9, DX)
ZG = np.arange(0.0, 12.0 + 1e-9, DX)
SUITES = ["vae_benchmark", "vae_benchmark_het10", "vae_benchmark_het30", "vae_benchmark_het50",
          "vae_benchmark_het10_tex", "vae_benchmark_het30_tex", "vae_benchmark_het50_tex"]
LN10 = np.log(10.0)
_G = {}


def _ctx():
    if not _G:
        import sparse_torch_inversion_ert as sp
        import vae_benchmark_ert as vb
        from scipy.spatial import Delaunay
        from scipy.ndimage import distance_transform_edt
        from pygimli.physics import ert
        scheme = ert.createData(elecs=sp.ELEC_X, schemeName="dd")
        _, _, cc, _ = sp.build_inversion_mesh(scheme)
        X, Z = np.meshgrid(XG, ZG)
        pts = np.column_stack([X.ravel(), -Z.ravel()])
        _G.update(sp=sp, vb=vb, cc=cc, tri=Delaunay(cc), X=X, Z=Z, pts=pts,
                  roi=sp.roi_mask(pts).reshape(X.shape), edt=distance_transform_edt)
    return _G


def _image(vals):
    """Linear interpolation of cell values to the grid; nearest cell where outside the
    hull of the cell centres (e.g. the surface row, above the shallowest centres)."""
    g = _ctx()
    if "near" not in g:
        from scipy.spatial import cKDTree
        g["near"] = cKDTree(g["cc"]).query(g["pts"])[1]
    from scipy.interpolate import LinearNDInterpolator
    v = LinearNDInterpolator(g["tri"], vals)(g["pts"])
    v = np.where(np.isnan(v), vals[g["near"]], v)
    return v.reshape(g["X"].shape) / LN10


def truth(spec):
    """True log10 image, host (no bodies) image, layer index (no bodies, no texture), body masks."""
    g, vb = _ctx(), _ctx()["vb"]
    pts, X = g["pts"], g["X"]
    true = vb.eval_model(spec, pts).reshape(X.shape) / LN10
    host = vb.eval_model(dict(spec, bodies=[]), pts).reshape(X.shape) / LN10
    x, z = pts[:, 0], -pts[:, 1]
    zq = z
    f = spec.get("fault")
    if f:
        zq = np.where(x > f["x0"] + z / np.tan(np.deg2rad(f["dip"])), z - f["throw"], z)
    layer = np.zeros(len(x), dtype=int)
    for it in spec["layers"]:
        layer += zq > vb.interface_depth(it, x)
    bodies = [vb.body_mask(x, z, b).reshape(X.shape) for b in spec.get("bodies", [])]
    return true, host, layer.reshape(X.shape), bodies


def ssim(true, rec, roi):
    from skimage.metrics import structural_similarity
    _, S = structural_similarity(true, rec, data_range=3.0, gaussian_weights=True, sigma=1.0 / DX,
                                 use_sample_covariance=False, full=True)
    return float(S[roi].mean())


def interface_error(rec, layer, bodies, roi, window=2.5, body_gap=1.5):
    g = _ctx()
    near_body = np.zeros_like(roi)
    if bodies:
        near_body = g["edt"](~np.any(bodies, 0)) * DX < body_gap
    grad = np.abs(np.gradient(rec, DX, axis=0))
    w = int(round(window / DX))
    err = []
    for j in range(rec.shape[1]):
        for i in np.nonzero(np.diff(layer[:, j]) != 0)[0]:          # boundary between rows i and i+1
            if not (roi[i, j] and roi[i + 1, j]) or near_body[i, j] or near_body[i + 1, j]:
                continue
            lo, hi = max(i - w, 0), min(i + w + 1, rec.shape[0] - 1)
            k = lo + int(np.argmax(grad[lo:hi, j]))
            if 0 < k < rec.shape[0] - 1:                             # sub-grid peak (parabola)
                a, b, c = grad[k - 1, j], grad[k, j], grad[k + 1, j]
                k = k + 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else k
            err.append(abs(k - (i + 0.5)) * DX)
    return float(np.mean(err)) if err else np.nan


def body_metrics(rec, true, host, bodies, roi, reach=4.0):
    g = _ctx()
    iou, contr, shift = [], [], []
    allb = np.any(bodies, 0) if bodies else None
    for k, B in enumerate(bodies):
        B = B & roi
        if B.sum() < 4:                                             # body (almost) outside the covered zone
            continue
        c = float(np.median((true - host)[B]))
        if abs(c) < 0.05:
            continue
        others = allb & ~bodies[k]
        near = (g["edt"](~B) * DX <= reach) & roi & ~others
        dev = rec - host
        D = near & (dev * np.sign(c) >= 0.5 * abs(c))
        iou.append((D & B).sum() / max((D | B).sum(), 1))
        contr.append(float(np.median(dev[B])) / c)
        if D.any():
            cb = np.array([g["X"][B].mean(), g["Z"][B].mean()])
            cd = np.array([g["X"][D].mean(), g["Z"][D].mean()])
            shift.append(float(np.hypot(*(cb - cd))))
        else:
            shift.append(np.nan)
    if not iou:
        return dict(body_iou=np.nan, body_contr=np.nan, body_shift=np.nan)
    return dict(body_iou=float(np.mean(iou)), body_contr=float(np.mean(contr)),
                body_shift=float(np.nanmedian(shift)) if np.isfinite(shift).any() else np.nan)


def model_rows(fn):
    z = np.load(fn, allow_pickle=True)
    spec, res = z["spec"].item(), z["res"].item()
    true, host, layer, bodies = truth(spec)
    roi = _ctx()["roi"]
    rows = []
    for k, r in res.items():
        if "m" not in r:
            continue
        rec = _image(np.asarray(r["m"], float))
        row = dict(model=spec["name"], group=spec["name"].split("_")[0], method=k, rms=r.get("rms", np.nan),
                   ssim=ssim(true, rec, roi), if_err=interface_error(rec, layer, bodies, roi))
        row.update(body_metrics(rec, true, host, bodies, roi))
        rows.append(row)
    return rows


def _work(fn):
    try:
        return model_rows(fn)
    except Exception as e:
        import traceback
        with open(os.path.join(OUTS, "errors.log"), "a") as fh:
            fh.write(f"=== {fn}\n{traceback.format_exc()}\n")
        print("ERROR", fn, e, flush=True)
        return []


def run_suite(suite, workers=4):
    import csv
    mdir = os.path.join(OUT, suite, "models")
    files = [os.path.join(mdir, f) for f in sorted(os.listdir(mdir)) if f.endswith(".npz")]
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = [r for rr in pool.map(_work, files) for r in rr]
    keys = ["model", "group", "method", "rms", "ssim", "if_err", "body_iou", "body_contr", "body_shift"]
    with open(os.path.join(OUTS, f"{suite}_structure.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(suite, rows):
    import vae_benchmark_ert as vb
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["method"]] = r
    methods = [m for m in vb.METHODS_ALL if any(m in v for v in by.values())] if hasattr(vb, "METHODS_ALL") \
        else sorted({r["method"] for r in rows}, key=lambda k: (k != "pygimli", k))
    L = [f"STRUCTURE METRICS  {suite}  ({len(by)} models)",
         "ssim: higher = better (1 = identical);  if_err: interface depth error in m (lower = better);",
         "body_iou: overlap of recovered anomaly and body (higher = better); body_contr: recovered share of",
         "the true contrast (1 = full); body_shift: centroid offset in m.  'wins' = share of models better than pyGIMLi.",
         "",
         f"{'method':18s}{'RMS':>7s}{'SSIM':>7s}{'wins':>6s}{'if_err':>8s}{'wins':>6s}"
         f"{'IoU':>7s}{'wins':>6s}{'contr':>7s}{'shift':>7s}"]

    def med(k, q):
        x = [v[k][q] for v in by.values() if k in v and np.isfinite(v[k][q])]
        return np.median(x) if x else np.nan

    def wins(k, q, higher):
        x = [(v[k][q] > v["pygimli"][q]) if higher else (v[k][q] < v["pygimli"][q])
             for v in by.values() if k in v and "pygimli" in v and np.isfinite(v[k][q]) and np.isfinite(v["pygimli"][q])]
        return 100 * np.mean(x) if x and k != "pygimli" else np.nan
    for k in methods:
        L.append(f"{k:18s}{med(k, 'rms'):7.3f}{med(k, 'ssim'):7.3f}{wins(k, 'ssim', True):5.0f}%"
                 f"{med(k, 'if_err'):8.2f}{wins(k, 'if_err', False):5.0f}%"
                 f"{med(k, 'body_iou'):7.2f}{wins(k, 'body_iou', True):5.0f}%"
                 f"{med(k, 'body_contr'):7.2f}{med(k, 'body_shift'):7.2f}")
    txt = "\n".join(L).replace("nan%", "  - ").replace("   nan", "     -")
    open(os.path.join(OUTS, f"{suite}_structure.txt"), "w").write(txt)
    print(txt + "\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUTS, exist_ok=True)
    for s in (a.suites.split(",") if a.suites else SUITES):
        if os.path.isdir(os.path.join(OUT, s, "models")):
            summarize(s, run_suite(s, a.workers))


if __name__ == "__main__":
    main()
