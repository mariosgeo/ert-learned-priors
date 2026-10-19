"""
sparse_torch_inversion_ert.py
=============================
ERT inversion in PyTorch on a SPARSE set of control cells; every other cell
of the inversion mesh is filled by biharmonic inpainting. pyGIMLi is the
forward model. The result is compared with pyGIMLi's own ERTManager
inversion on the same mesh and the same data.

Method
------
* Parameters  p : log-resistivity at ~100 "control" cells instead of all
  ~2000 cells. Their layout follows the resolution of surface ERT: dense near
  the surface, sparse at depth. Rows of control points sit at depths whose
  spacing grows geometrically (dz_k = dz0 * growth^k); the horizontal spacing
  in a row is aspect * dz_k (staggered row to row), and each row only spans
  the part of the line the data actually see at that depth (it shrinks by
  `shrink * z` on both ends, like the pseudosection footprint). Each row also
  gets one anchor point at either lateral edge of the paraDomain so the
  interpolant never extrapolates.
* Inpainting  m = W p : scipy's RBFInterpolator with the thin-plate-spline
  kernel r^2 log r - the Green's function of the 2-D biharmonic operator, i.e.
  the minimum-bending-energy surface through the control values (the same
  principle as skimage.restoration.inpaint_biharmonic, selectable with
  --interp skimage). Both are LINEAR in p, so the inpainting step is a fixed
  (n_cells x n_ctrl) matrix W: exactly differentiable, one mat-vec per
  iteration. --aniso > 1 stretches the interpolation horizontally (more
  lateral continuity, as expected for layered ground).
* Forward : pyGIMLi (ERTModelling), wrapped for torch. J = dlog(rhoa)/dlog(rho)
  from pyGIMLi; the chain rule through W and the soft bounds is done in torch.
* Optimizer (--optimizer):
    gn    (default) Gauss-Newton on the REDUCED Jacobian  Jp = J diag(b') W
          (n_data x ~100), solved in torch with a backtracking line search.
          With ~100 unknowns the normal equations are tiny, so each iteration
          costs one forward + Jacobian and the inversion converges in a few
          iterations per lambda - an order of magnitude fewer forward solves
          than L-BFGS.
    lbfgs torch.optim.LBFGS (strong Wolfe) with pyGIMLi as a
          torch.autograd.Function (backward = J^T g).
* Regularization on differences of neighbouring control points (Delaunay
  edges): edge weights 1 - (1 - zweight) |sin(angle)|, so with zweight < 1
  vertical changes (layer boundaries) are cheaper than lateral ones;
  --norm l1 uses IRLS (blocky, sharper boundaries) instead of L2.
  lambda is cooled stage-wise; we keep the first stage with
  chi2 <= --chi2-target (discrepancy principle, default 1.05: even the TRUE
  model only reaches chi2 ~ 1.5 on this mesh because sharp boundaries are
  stair-cased) or the last stage before chi2 stops improving (< 5 % gain).
* Adaptive refinement (--refine N): after an inversion, add control cells
  where the recovered model changes fastest, inside the well-covered zone and
  with a depth-dependent minimum spacing, then continue the inversion. A new
  control cell starts at the current interpolated value, which leaves the
  biharmonic interpolant unchanged, so refinement is a warm start.

Synthetic model: three layers (lower interface dipping 9 -> 12 m) plus a
rectangular target in the middle layer. The data are simulated on a FINER,
different mesh than the inversion mesh (no inverse crime), dipole-dipole,
48 electrodes at 1 m, 3 % Gaussian noise.

Usage
-----
    conda activate pg2
    python sparse_torch_inversion_ert.py                    # resistive target (500 Ohm.m)
    python sparse_torch_inversion_ert.py --target-rho 10    # conductive target
    python sparse_torch_inversion_ert.py --interp skimage   # image-inpainting variant
    python sparse_torch_inversion_ert.py --quick            # smoke test
"""

import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── experiment definition ──────────────────────────────────────────────────
ELEC_X = np.linspace(0, 47, 48)
NOISE = 0.03                       # relative data error (= error of log rhoa)
# inversion domain: "standard" (2 m beyond the electrodes, 20 m deep) or, with the
# environment variable ERT_DOMAIN=large, 12 m beyond the electrodes and 30 m deep
DOMAIN = os.environ.get("ERT_DOMAIN", "standard")
PARA_DEPTH = 30.0 if DOMAIN == "large" else 20.0
PARA_BOUNDARY = 12.0 if DOMAIN == "large" else 2.0
SIM_DEPTH = 30.0                   # depth of the fine SIMULATION mesh (same for every domain)
INV_CELL = 1.0                     # max cell area of the inversion mesh (m^2)
SYN_CELL = 0.2                     # finer mesh for the synthetic data
RHO_LO, RHO_HI = 1.0, 1.0e4        # soft bounds on resistivity (Ohm.m)
SB_BETA = 4.0                      # sharpness of the soft bounds

RHO_LAYERS = (250.0, 80.0, 30.0)   # top, middle, bottom layer
Z1 = 2.5                           # depth of the first interface (m)
TARGET = dict(xc=23.5, w=8.0, top=4.0, h=3.0)   # inside the middle layer
XLIM = (-14.0, 61.0) if DOMAIN == "large" else (-5.0, 52.0)
YLIM = (-PARA_DEPTH, 0.5)
PROFILES_X = (23.5, 8.0)           # through the target / layers only
CMAP = "turbo"                     # rainbow colour scale for all maps


def z2(x):
    """Depth of the second (dipping) interface: 9 m at x=0 -> 12 m at x=47."""
    t = np.clip((np.asarray(x) - ELEC_X[0]) / (ELEC_X[-1] - ELEC_X[0]), 0, 1)
    return 9.0 + 3.0 * t


def target_mask(cc):
    z = -cc[:, 1]
    return ((np.abs(cc[:, 0] - TARGET["xc"]) <= TARGET["w"] / 2)
            & (z >= TARGET["top"]) & (z <= TARGET["top"] + TARGET["h"]))


def true_log_rho(cc, rho_t):
    x, z = cc[:, 0], -cc[:, 1]
    rho = np.where(z < Z1, RHO_LAYERS[0], np.where(z < z2(x), RHO_LAYERS[1], RHO_LAYERS[2]))
    rho = np.where(target_mask(cc), rho_t, rho)
    return np.log(rho)


def roi_mask(cc, shrink=0.5, zmax=12.0):
    """Well-covered region used for the model-error statistics."""
    z = -cc[:, 1]
    return ((z <= zmax) & (cc[:, 0] >= ELEC_X[0] + shrink * z)
            & (cc[:, 0] <= ELEC_X[-1] - shrink * z))


def _threads(n):
    os.environ["OMP_NUM_THREADS"] = str(n)
    import pygimli as pg
    import torch
    try:
        pg.setThreadCount(n)
    except Exception:
        pass
    torch.set_num_threads(n)


# ── synthetic data ─────────────────────────────────────────────────────────
def simulate_data(rho_t, seed=0):
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    scheme = ert.createData(elecs=ELEC_X, schemeName="dd")
    fine = mt.createParaMesh(scheme, paraDepth=SIM_DEPTH, quality=34,
                             paraMaxCellSize=SYN_CELL, boundaryMaxCellSize=500)
    cc = np.array([[c.center().x(), c.center().y()] for c in fine.cells()])
    data = ert.simulate(fine, scheme=scheme, res=np.exp(true_log_rho(cc, rho_t)),
                        noiseLevel=0, noiseAbs=0, verbose=False)
    clean = np.array(data["rhoa"])
    rng = np.random.default_rng(seed)
    data["rhoa"] = clean * (1 + rng.normal(0, NOISE, size=clean.shape))
    data["err"] = np.full(len(clean), NOISE)
    bad = np.where(np.array(data["rhoa"]) <= 0)[0]
    if len(bad):
        data.markInvalid(bad)
        data.removeInvalid()
        print(f"removed {len(bad)} non-positive data")
    return data, fine.cellCount()


def build_inversion_mesh(data):
    import pygimli.meshtools as mt
    from ert_common import ERTForward
    mesh = mt.createParaMesh(data, paraDepth=PARA_DEPTH, paraBoundary=PARA_BOUNDARY, quality=33,
                             paraMaxCellSize=INV_CELL, boundaryMaxCellSize=50)
    fwd = ERTForward(data, mesh)
    pd = fwd.fop.paraDomain
    cc = np.array([[c.center().x(), c.center().y()] for c in pd.cells()])
    area = np.array([c.size() for c in pd.cells()])
    return mesh, fwd, cc, area


# ── control-point layout: dense at the surface, sparse at depth ────────────
def place_controls(cc, dz0=1.0, growth=1.3, aspect=2.0, shrink=0.5, edges=True):
    """Indices of the inversion cells whose values are free parameters."""
    from scipy.spatial import cKDTree
    depth = -cc[:, 1]
    zmax = depth.max()
    x0, x1 = ELEC_X[0], ELEC_X[-1]
    pts, k, z, dz = [], 0, dz0 / 2, dz0
    while True:
        last = z + dz >= zmax
        z = min(z, zmax - 0.25)
        band = np.abs(depth - z) < max(0.5 * dz, 0.75)
        xs_min, xs_max = cc[band, 0].min(), cc[band, 0].max()
        dx = aspect * dz
        lo, hi = max(x0 + shrink * z, xs_min), min(x1 - shrink * z, xs_max)
        if hi - lo < dx:
            xs = np.array([0.5 * (lo + hi)])
        else:
            n = int(round((hi - lo) / dx))
            step = (hi - lo) / n
            xs = lo + (np.arange(n + 1) + 0.5 * (k % 2)) * step   # staggered rows
            xs = xs[xs <= hi + 1e-9]
        if edges:
            xs = np.concatenate([[xs_min], xs, [xs_max]])
        pts += [(x, -z) for x in xs]
        if last:
            break
        z += dz
        dz *= growth
        k += 1
    idx = cKDTree(cc).query(np.array(pts))[1]
    return np.unique(idx)


def refine_controls(cc, ctrl, x_cells, cov_n, n_add, zmax=12.0, cov_min=1e-2):
    """Add up to n_add control cells where the current model (x_cells, log-rho)
    changes fastest, inside the well-covered zone (normalized sensitivity per
    area >= cov_min, depth <= zmax), keeping a minimum spacing of 1 + 0.25 z m
    to every other control cell (finer near the surface, coarser at depth)."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(cc).query(cc, k=9)
    grad = np.max(np.abs(x_cells[:, None] - x_cells[j[:, 1:]]) / d[:, 1:], axis=1)
    z = -cc[:, 1]
    cand = (cov_n >= cov_min) & (z <= zmax)
    cand[ctrl] = False
    chosen = [cc[i] for i in ctrl]
    new = []
    for i in np.where(cand)[0][np.argsort(-grad[cand])]:
        if np.min(np.hypot(*(np.array(chosen) - cc[i]).T)) >= 1.0 + 0.25 * z[i]:
            chosen.append(cc[i])
            new.append(i)
            if len(new) >= n_add:
                break
    return np.unique(np.concatenate([ctrl, new]).astype(int)), np.array(new, dtype=int)


# ── biharmonic inpainting as a fixed linear operator  m = W p ──────────────
def weights_tps(cc, ctrl, aniso=1.0):
    """Thin-plate spline (2-D biharmonic Green's function) through the control
    cells, evaluated at every cell. Columns of W = response to a unit spike.
    aniso > 1 compresses horizontal distances (more lateral continuity)."""
    from scipy.interpolate import RBFInterpolator
    s = np.array([1.0 / aniso, 1.0])
    P = cc[ctrl] * s
    W = RBFInterpolator(P, np.eye(len(P)), kernel="thin_plate_spline", degree=1)(cc * s)
    W[ctrl] = np.eye(len(P))          # exact at the control cells (no round-off)
    return W


def weights_skimage(cc, ctrl, aniso=1.0, h=0.5):
    """skimage.restoration.inpaint_biharmonic on a regular image covering the
    paraDomain (control cells = known pixels), sampled back to the cells."""
    from skimage.restoration import inpaint_biharmonic
    from scipy.interpolate import RegularGridInterpolator
    hx = h * aniso                    # wider pixels = horizontally stretched
    xg = np.arange(cc[:, 0].min() - hx, cc[:, 0].max() + 2 * hx, hx)
    yg = np.arange(cc[:, 1].min() - h, cc[:, 1].max() + 2 * h, h)
    ix = np.rint((cc[ctrl, 0] - xg[0]) / hx).astype(int)
    iy = np.rint((cc[ctrl, 1] - yg[0]) / h).astype(int)
    if len(set(zip(iy, ix))) < len(ctrl):
        raise ValueError("two control cells share a pixel; reduce h")
    mask = np.ones((len(yg), len(xg)), dtype=bool)
    mask[iy, ix] = False
    G = np.empty((len(yg), len(xg), len(ctrl)))
    for j in range(len(ctrl)):
        img = np.zeros(mask.shape)
        img[iy[j], ix[j]] = 1.0
        G[:, :, j] = inpaint_biharmonic(img, mask)
    W = RegularGridInterpolator((yg, xg), G)(cc[:, ::-1])
    W[ctrl] = np.eye(len(ctrl))
    return W


def regularization_operator(P, zweight=1.0):
    """Weighted (n_edges x n_ctrl) difference operator over the Delaunay edges
    of the control points. Vertical edges are weighted by zweight."""
    from scipy.spatial import Delaunay
    edges = set()
    for s in Delaunay(P).simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edges.add(tuple(sorted((s[a], s[b]))))
    edges = np.array(sorted(edges))
    v = P[edges[:, 1]] - P[edges[:, 0]]
    sin = np.abs(v[:, 1]) / np.hypot(v[:, 0], v[:, 1])
    w = 1.0 - (1.0 - zweight) * sin
    D = np.zeros((len(edges), len(P)))
    D[np.arange(len(edges)), edges[:, 0]] = w
    D[np.arange(len(edges)), edges[:, 1]] = -w
    return D


# ── PyTorch side ───────────────────────────────────────────────────────────
def make_pygimli_function():
    import torch

    class PyGIMLiLogRhoa(torch.autograd.Function):
        """log(rhoa) = f(log rho) with pyGIMLi; backward = J^T g."""
        n_calls = 0

        @staticmethod
        def forward(ctx, log_rho, fwd):
            J, dcalc = fwd.jacobian_loglog(log_rho.detach().cpu().numpy())
            ctx.save_for_backward(torch.from_numpy(J))
            PyGIMLiLogRhoa.n_calls += 1
            return torch.from_numpy(np.log(dcalc))

        @staticmethod
        def backward(ctx, g):
            (J,) = ctx.saved_tensors
            return J.T @ g, None

    return PyGIMLiLogRhoa


def soft_bounds(x, lo, hi, beta=SB_BETA):
    """~identity inside [lo, hi], smoothly saturating outside."""
    import torch.nn.functional as F
    return lo + F.softplus(x - lo, beta=beta) - F.softplus(x - hi, beta=beta)


def soft_bounds_deriv(x, lo, hi, beta=SB_BETA):
    import torch
    return torch.sigmoid(beta * (x - lo)) - torch.sigmoid(beta * (x - hi))


class Counter:
    jac = 0      # forward + Jacobian
    fwd = 0      # forward only (line search / final check)

    @classmethod
    def total(cls):
        return cls.jac + cls.fwd


def _stage_done(stages, chi2_target, stall, verbose):
    """Discrepancy / stall rule. Returns the stage to keep, or None to go on."""
    if stages[-1]["chi2"] <= chi2_target:
        return stages[-1]
    if len(stages) > 1 and stages[-1]["chi2"] > stall * stages[-2]["chi2"]:
        if verbose:
            print(f"  chi2 stalled (< {100 * (1 - stall):.0f} % gain): keeping "
                  f"lambda={stages[-2]['lam']:g}", flush=True)
        return stages[-2]
    return None


def invert_gn(fwd, W, D, dobs, p0, lams, chi2_target=1.05, stall=0.95, norm="l2",
              max_iter=8, eps=0.05, verbose=True, hist=None):
    """Gauss-Newton (IRLS for l1) on the control values, in torch."""
    import torch
    Wt, Dt = torch.from_numpy(W), torch.from_numpy(D)
    dlog = torch.from_numpy(np.log(dobs))
    lo, hi = np.log(RHO_LO), np.log(RHO_HI)
    N, E, n = len(dobs), Dt.shape[0], Wt.shape[1]
    p = torch.tensor(p0, dtype=torch.float64)
    hist = [] if hist is None else hist
    stages = []

    def chi2_of(p_):
        Counter.fwd += 1
        m = soft_bounds(Wt @ p_, lo, hi).numpy()
        return float(np.mean(((np.log(fwd.response(m)) - dlog.numpy()) / NOISE) ** 2))

    for lam in lams:
        chi2 = None
        for it in range(max_iter):
            x = Wt @ p
            J, dcalc = fwd.jacobian_loglog(soft_bounds(x, lo, hi).numpy())
            Counter.jac += 1
            r = (torch.from_numpy(np.log(dcalc)) - dlog) / NOISE
            chi2 = float((r ** 2).mean())
            Jp = (torch.from_numpy(J) * soft_bounds_deriv(x, lo, hi)) @ Wt / NOISE
            g = Dt @ p
            if norm == "l1":                           # IRLS weights ~ 1/|g|
                wr = 1.0 / torch.sqrt(g ** 2 + eps ** 2)
                wr = wr / wr.mean()
            else:
                wr = torch.ones_like(g)
            reg = lambda q: float((wr * (Dt @ q) ** 2).sum()) / E
            phi0 = chi2 + lam * reg(p)
            A = Jp.T @ Jp / N + lam / E * Dt.T @ (wr[:, None] * Dt)
            b = Jp.T @ r / N + lam / E * Dt.T @ (wr * g)
            dp = -torch.linalg.solve(A + 1e-9 * torch.eye(n, dtype=A.dtype), b)
            for step in (1.0, 0.5, 0.25, 0.125):
                pn = p + step * dp
                chi2n = chi2_of(pn)
                phin = chi2n + lam * reg(pn)
                if phin < phi0:
                    break
            else:
                break                                   # no descent: converged
            p, chi2 = pn, chi2n
            hist.append((lam, chi2, Counter.total()))
            if (phi0 - phin) / phi0 < 0.01:
                break
        m = soft_bounds(Wt @ p, lo, hi).numpy()
        stages.append(dict(lam=lam, chi2=chi2, p=p.numpy().copy(), m=m))
        if verbose:
            print(f"  GN stage lambda={lam:<7g} chi2={chi2:7.3f}  "
                  f"pyGIMLi calls so far: {Counter.jac} fwd+J, {Counter.fwd} fwd", flush=True)
        keep = _stage_done(stages, chi2_target, stall, verbose)
        if keep is not None:
            return keep, stages, hist
    return stages[-1], stages, hist


def invert_lbfgs(fwd, W, D, dobs, p0, lams, chi2_target=1.05, stall=0.95, norm="l2",
                 max_iter=25, eps=0.05, verbose=True, hist=None):
    import torch
    fn = make_pygimli_function()
    Wt, Dt = torch.from_numpy(W), torch.from_numpy(D)
    dlog = torch.from_numpy(np.log(dobs))
    lo, hi = np.log(RHO_LO), np.log(RHO_HI)
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    hist = [] if hist is None else hist
    stages = []

    def model(p_):
        return soft_bounds(Wt @ p_, lo, hi)

    for lam in lams:
        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=max_iter, history_size=20,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-9)

        def closure():
            opt.zero_grad()
            r = (fn.apply(model(p), fwd) - dlog) / NOISE
            Counter.jac += 1
            chi2 = (r ** 2).mean()
            g = Dt @ p
            reg = (torch.sqrt(g ** 2 + eps ** 2).mean() if norm == "l1" else (g ** 2).mean())
            phi = chi2 + lam * reg
            phi.backward()
            hist.append((lam, chi2.item(), Counter.total()))
            return phi

        opt.step(closure)
        with torch.no_grad():
            m = model(p).numpy()
        Counter.fwd += 1
        chi2 = float(np.mean(((np.log(fwd.response(m)) - np.log(dobs)) / NOISE) ** 2))
        stages.append(dict(lam=lam, chi2=chi2, p=p.detach().numpy().copy(), m=m))
        if verbose:
            print(f"  L-BFGS stage lambda={lam:<7g} chi2={chi2:7.3f}  "
                  f"pyGIMLi calls so far: {Counter.total()}", flush=True)
        keep = _stage_done(stages, chi2_target, stall, verbose)
        if keep is not None:
            return keep, stages, hist
    return stages[-1], stages, hist


# ── pyGIMLi reference ──────────────────────────────────────────────────────
def invert_pygimli(data, mesh, lams, chi2_target=1.05, zweight=1.0, blocky=False,
                   run_all=False, verbose=True, robust=False, start=None, dip=None, corr=None, inv_kw=None):
    """start: log-resistivity used as starting AND reference model (pyGIMLi
    isReference). dip/corr: geostatistical regularization of the parameter
    region along an angle dip (degrees, pyGIMLi convention) with correlation
    lengths corr (along, across)."""
    """Largest lambda with chi2 <= chi2_target (discrepancy principle). With
    run_all, every lambda is inverted (trade-off curve); returns (best, runs).
    inv_kw: extra options of pyGIMLi's Inversion.run, e.g. stopAtChi1=False, maxIter=40."""
    import pygimli as pg
    from pygimli.physics import ert
    dobs = np.array(data["rhoa"])
    best, runs = None, []
    for lam in lams:
        mgr =ert.ERTManager(pg.DataContainerERT(data), verbose=False)
        if start is not None or dip is not None:
            mgr.setMesh(pg.Mesh(mesh))
            kw = {}
            if dip is not None:
                rm = mgr.fop.regionManager()
                para = [r for r in rm.regionIdxs() if not rm.region(r).isBackground()]
                mgr.fop.setRegionProperties(para[0], correlationLengths=corr, dip=dip)
            if start is not None:
                kw.update(startModel=pg.Vector(np.exp(np.asarray(start, float))), isReference=True)
            mgr.invert(lam=lam, zWeight=zweight, blockyModel=blocky, robustData=robust, verbose=False, **kw,
                       **(inv_kw or {}))
        else:
            mgr.invert(mesh=pg.Mesh(mesh), lam=lam, zWeight=zweight, blockyModel=blocky, robustData=robust,
                   verbose=False, **(inv_kw or {}))
        resp = np.array(mgr.inv.response)
        chi2 = float(np.mean(((np.log(resp) - np.log(dobs)) / NOISE) ** 2))
        pd = mgr.paraDomain
        run = dict(lam=lam, chi2=chi2, m=np.log(np.array(mgr.paraModel())),
                   cc=np.array([[c.center().x(), c.center().y()] for c in pd.cells()]),
                   tri=_triangles(pd), n_iter=mgr.inv.inv.iter())
        runs.append(run)
        if verbose:
            print(f"  pyGIMLi lambda={lam:<6g} chi2={chi2:7.3f}  iterations={run['n_iter']}",
                  flush=True)
        if best is None and chi2 <= chi2_target:
            best = run
            if not run_all:
                break
    return (best or runs[-1]), runs


def _triangles(pd):
    nodes = np.array([[n.pos().x(), n.pos().y()] for n in pd.nodes()])
    cells = np.array([[n.id() for n in c.nodes()] for c in pd.cells()])
    return nodes, cells


# ── evaluation ─────────────────────────────────────────────────────────────
def model_metrics(cc, log_rho, rho_t):
    true = true_log_rho(cc, rho_t)
    err10 = (log_rho - true) / np.log(10)
    roi, tgt = roi_mask(cc), target_mask(cc)
    out = dict(rms_log10_roi=float(np.sqrt(np.mean(err10[roi] ** 2))),
               target_median=float(np.exp(np.median(log_rho[tgt]))))
    z = -cc[:, 1]
    for name, sel in [("layer1", z < Z1), ("layer2", (z >= Z1) & (z < z2(cc[:, 0]))),
                      ("layer3", z >= z2(cc[:, 0]))]:
        s = sel & roi & ~tgt
        out[name + "_median"] = float(np.exp(np.median(log_rho[s]))) if s.any() else np.nan
    return out


def vertical_profile(cc, log_rho, x, z):
    from scipy.interpolate import griddata
    return griddata(cc, log_rho, np.column_stack([np.full_like(z, x), -z]), method="linear")


# ── plotting ───────────────────────────────────────────────────────────────
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"
C_TRUE, C_PG, C_TORCH = INK, "#2a78d6", "#eb6834"


def _overlay(ax):
    import matplotlib.pyplot as plt
    xs = np.linspace(*XLIM, 100)
    rect = ((TARGET["xc"] - TARGET["w"] / 2, -TARGET["top"] - TARGET["h"]), TARGET["w"], TARGET["h"])
    # black solid under white dashes: one line readable on every rainbow colour
    for col, lw, ls in (("black", 1.8, "-"), ("white", 0.9, (0, (4, 3)))):
        ax.plot(xs, np.full_like(xs, -Z1), color=col, lw=lw, ls=ls)
        ax.plot(xs, -z2(xs), color=col, lw=lw, ls=ls)
        ax.add_patch(plt.Rectangle(*rect, fill=False, ec=col, lw=lw, ls=ls))
    ax.plot(ELEC_X, np.zeros_like(ELEC_X), "v", color=INK, ms=2.5)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=INK2); ax.set_ylabel("depth (m)", color=INK2)
    ax.tick_params(colors=INK2, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(GRID)


def _cellplot(ax, tri, vals, **kw):
    import matplotlib.tri as mtri
    nodes, cells = tri
    return ax.tripcolor(mtri.Triangulation(nodes[:, 0], nodes[:, 1], cells), facecolors=vals, **kw)


def _controls(ax, P, new=None, ms=4.5):
    ax.plot(P[:, 0], P[:, 1], "o", ms=ms, mfc="white", mec="black", mew=0.7,
            label="initial control cells")
    if new is not None and len(new):
        ax.plot(new[:, 0], new[:, 1], "D", ms=ms * 0.85, mfc="magenta", mec="black", mew=0.6,
                label="added by refinement")


def _save(fig, fn, dpi=140):
    try:
        fig.savefig(fn, dpi=dpi)
    except OSError:
        alt = fn.replace(".png", "_1.png")
        fig.savefig(alt, dpi=dpi)
        print(f"  {fn} is locked (open in a viewer?) -> saved as {alt}")


def plot_tradeoff(curves, chosen, rho_t, fn):
    """Model error and target recovery vs data fit, along the lambda path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    style = {"pyGIMLi": C_PG, "PyTorch sparse": C_TORCH}
    for name, pts in curves.items():
        chi = np.array([p["chi2"] for p in pts])
        for ax, key in zip(axs, ["rms_log10_roi", "target_median"]):
            v = np.array([p["metrics"][key] for p in pts])
            ax.plot(chi, v, "o-", color=style[name], lw=1.8, ms=4, label=f"{name} (lambda path)")
            for c, y, p in zip(chi, v, pts):
                ax.annotate(f"{p['lam']:g}", (c, y), textcoords="offset points", xytext=(4, 4),
                            fontsize=6.5, color=style[name])
    for name, p in chosen.items():
        for ax, key in zip(axs, ["rms_log10_roi", "target_median"]):
            ax.plot(p["chi2"], p["metrics"][key], "*", ms=14, mfc=style[name], mec="black",
                    label=f"{name}: shown in the maps")
    for ax in axs:
        ax.axvspan(0, 1.05, color=GRID, alpha=0.5, lw=0, label="chi2 <= 1.05 (fits the noise)")
        ax.set_xscale("log"); ax.set_xlabel("data misfit chi2  (labels: lambda)")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
        ax.grid(color=GRID, lw=0.6, which="both")
    axs[0].set_ylabel("RMS log10(rho_rec / rho_true) in the covered zone")
    axs[0].set_title("Model error vs data fit (lower is better)")
    axs[1].axhline(rho_t, color=C_TRUE, ls="--", lw=1.5, label=f"true target ({rho_t:g} Ohm.m)")
    axs[1].set_yscale("log"); axs[1].set_ylabel("median resistivity in the target (Ohm.m)")
    axs[1].set_title("Target recovery vs data fit (closer to the dashed line is better)")
    h, l = axs[1].get_legend_handles_labels()
    keep = dict(zip(l, h))
    axs[1].legend(keep.values(), keep.keys(), fontsize=7, frameon=False)
    _save(fig, fn)
    plt.close(fig)


def plot_results(R, fn_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.titlecolor": INK})
    rho_t = R["rho_t"]
    vals = [RHO_LAYERS[0], RHO_LAYERS[1], RHO_LAYERS[2], rho_t]
    norm = LogNorm(min(vals) / 1.5, max(vals) * 1.5)
    rnorm = Normalize(-1, 1)
    tri, cc = R["tri"], R["cc"]
    pg_, tc = R["pg"], R["torch"]
    P0, Pn = cc[R["ctrl0"]], cc[R["added"]] if len(R["added"]) else None

    # ---------- Figure 1: models ----------
    fig, axs = plt.subplots(3, 2, figsize=(13, 9.2), constrained_layout=True)
    ax = axs[0, 0]
    pc = _cellplot(ax, tri, np.exp(true_log_rho(cc, rho_t)), norm=norm, cmap=CMAP)
    ax.set_title("(a) True model on the inversion mesh")
    fig.colorbar(pc, ax=ax, shrink=0.8, label="resistivity (Ohm.m)")
    ax = axs[0, 1]
    ps = _cellplot(ax, tri, R["cov"] / R["cov"].max(), norm=LogNorm(1e-3, 1), cmap=CMAP)
    _controls(ax, P0, Pn)
    ax.set_title(f"(b) {len(R['ctrl'])} control cells over data sensitivity "
                 f"({len(cc)} cells in total)")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    fig.colorbar(ps, ax=ax, shrink=0.8, label="normalized sensitivity per m$^2$")
    ax = axs[1, 0]
    _cellplot(ax, pg_["tri"], np.exp(pg_["m"]), norm=norm, cmap=CMAP)
    ax.set_title(f"(c) pyGIMLi ERTManager ({pg_.get('label', '')})   lambda={pg_['lam']:g}, "
                 f"chi2={pg_['chi2']:.2f}, "
                 f"{len(pg_['m'])} parameters")
    ax = axs[1, 1]
    _cellplot(ax, tri, np.exp(tc["m"]), norm=norm, cmap=CMAP)
    _controls(ax, P0, Pn, ms=2.5)
    ax.set_title(f"(d) PyTorch sparse + biharmonic inpainting   lambda={tc['lam']:g}, "
                 f"chi2={tc['chi2']:.2f}, {len(R['ctrl'])} parameters")
    fig.colorbar(pc, ax=axs[1].tolist(), shrink=0.8, label="resistivity (Ohm.m)")
    for ax, (lab, name, r) in zip(axs[2], [("e", "pyGIMLi", pg_), ("f", "PyTorch sparse", tc)]):
        d = (r["m"] - true_log_rho(r.get("cc", cc), rho_t)) / np.log(10)
        pr = _cellplot(ax, r.get("tri", tri), d, norm=rnorm, cmap=CMAP)
        ax.set_title(f"({lab}) {name}: log10(recovered / true)   "
                     f"RMS in covered zone = {r['metrics']['rms_log10_roi']:.3f}")
    fig.colorbar(pr, ax=axs[2].tolist(), shrink=0.8, label="log10(rho_rec / rho_true)")
    for ax in axs.ravel():
        _overlay(ax)
    _save(fig, fn_prefix + "_models.png")
    plt.close(fig)

    # ---------- Figure 2: profiles + convergence ----------
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    z = np.linspace(0.05, 16, 320)
    for ax, x in zip(axs[:2], PROFILES_X):
        tr = true_log_rho(np.column_stack([np.full_like(z, x), -z]), rho_t)
        ax.plot(np.exp(tr), z, color=C_TRUE, lw=2, ls="--", label="true")
        ax.plot(np.exp(vertical_profile(pg_["cc"], pg_["m"], x, z)), z, color=C_PG, lw=2,
                label="pyGIMLi")
        ax.plot(np.exp(vertical_profile(cc, tc["m"], x, z)), z, color=C_TORCH, lw=2,
                label="PyTorch sparse")
        zc = -cc[R["ctrl"], 1][np.abs(cc[R["ctrl"], 0] - x) < 2.0]
        ax.plot(np.full_like(zc, norm.vmin * 1.05), zc, ">", color=C_TORCH, ms=5,
                label="control cells within 2 m")
        ax.set_xscale("log"); ax.set_xlim(norm.vmin, norm.vmax); ax.set_ylim(16, 0)
        ax.set_xlabel("resistivity (Ohm.m)"); ax.set_ylabel("depth (m)")
        what = "through the target" if abs(x - TARGET["xc"]) < TARGET["w"] / 2 else "layers only"
        ax.set_title(f"Vertical profile at x = {x:g} m ({what})")
        ax.grid(color=GRID, lw=0.6, which="both"); ax.legend(fontsize=7.5, frameon=False)
    ax = axs[2]
    h = np.array(R["hist"])
    ax.semilogy(h[:, 2], h[:, 1], "o-", ms=3, color=C_TORCH, lw=1.5, label="PyTorch chi2")
    for i in np.where(np.diff(h[:, 0]) != 0)[0]:
        ax.axvline(0.5 * (h[i, 2] + h[i + 1, 2]), color=GRID, lw=0.8)
    if R["refine_at"] is not None:
        ax.axvline(R["refine_at"], color="magenta", lw=1, ls=":", label="control refinement")
    ax.axhline(1.0, color=INK2, lw=1, ls="--", label="chi2 = 1 (noise level)")
    ax.axhline(pg_["chi2"], color=C_PG, lw=1.5, label=f"pyGIMLi final chi2 ({pg_['chi2']:.2f})")
    ax.set_xlabel("pyGIMLi forward calls (grey lines: lambda stages)")
    ax.set_ylabel("chi2"); ax.set_title("Convergence of the sparse PyTorch inversion")
    ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=7.5, frameon=False)
    _save(fig, fn_prefix + "_profiles.png")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--target-rho", type=float, default=500.0)
    ap.add_argument("--interp", choices=["tps", "skimage"], default="tps")
    ap.add_argument("--aniso", type=float, default=2.0,
                    help="horizontal stretch of the biharmonic interpolation")
    ap.add_argument("--optimizer", choices=["gn", "lbfgs"], default="gn")
    ap.add_argument("--norm", choices=["l2", "l1"], default="l1")
    ap.add_argument("--zweight", type=float, default=0.3,
                    help="weight of vertical control-point differences (<1 favours layers)")
    ap.add_argument("--refine", type=int, default=1, help="adaptive refinement rounds")
    ap.add_argument("--refine-add", type=int, default=30, help="control cells added per round")
    ap.add_argument("--dz0", type=float, default=1.0, help="first row spacing (m)")
    ap.add_argument("--growth", type=float, default=1.3, help="row-spacing growth per row")
    ap.add_argument("--aspect", type=float, default=2.0, help="horizontal / vertical spacing")
    ap.add_argument("--shrink", type=float, default=0.5, help="row narrowing per m depth")
    ap.add_argument("--lams", type=str, default="100,30,10,3,1,0.3,0.1,0.03,0.01")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="iterations per lambda stage (default gn 8, lbfgs 25)")
    ap.add_argument("--pg-lams", type=str, default="300,100,50,20,10,5,2")
    ap.add_argument("--tradeoff", action="store_true",
                    help="also run the full lambda path of both methods (trade-off figure)")
    ap.add_argument("--pg-zweight", type=float, default=0.3)
    ap.add_argument("--pg-smooth", action="store_true",
                    help="pyGIMLi with smooth (L2) instead of blocky (L1) model norm")
    ap.add_argument("--chi2-target", type=float, default=1.05)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default=None, help="output file prefix")
    ap.add_argument("--no-pg", action="store_true", help="reuse the saved pyGIMLi result")
    ap.add_argument("--quick", action="store_true", help="smoke test: few stages")
    a = ap.parse_args()
    if a.max_iter is None:
        a.max_iter = 8 if a.optimizer == "gn" else 25
    if a.quick:
        a.lams, a.max_iter, a.pg_lams = "1,0.1", 4, "20"

    _threads(a.threads)
    out = os.path.join(HERE, "outputs", "sparse_torch")
    os.makedirs(out, exist_ok=True)
    tag = a.tag or (f"rho{a.target_rho:g}_{a.interp}" + ("_quick" if a.quick else ""))
    prefix = os.path.join(out, tag)
    config = (f"interp={a.interp} aniso={a.aniso:g} optimizer={a.optimizer} norm={a.norm} "
              f"zweight={a.zweight:g} refine={a.refine}x{a.refine_add} "
              f"layout dz0={a.dz0:g} growth={a.growth:g} aspect={a.aspect:g} shrink={a.shrink:g}")
    t0 = time.time()

    print(f"Simulating data (target {a.target_rho:g} Ohm.m) on a fine mesh ...", flush=True)
    data, n_fine = simulate_data(a.target_rho, seed=a.seed)
    dobs = np.array(data["rhoa"])
    mesh, fwd, cc, area = build_inversion_mesh(data)
    m_true = true_log_rho(cc, a.target_rho)
    floor = np.mean(((np.log(fwd.response(m_true)) - np.log(dobs)) / NOISE) ** 2)
    print(f"  {len(dobs)} data, fine mesh {n_fine} cells, inversion paraDomain {fwd.n_cells} cells; "
          f"true model on the inversion mesh gives chi2 = {floor:.2f}", flush=True)
    print(f"  {config}", flush=True)

    m0 = np.full(fwd.n_cells, np.log(np.median(dobs)))
    J0, _ = fwd.jacobian_loglog(m0)
    cov = np.sqrt(np.sum((J0 / NOISE) ** 2, axis=0)) / np.sqrt(area)

    def operators(ctrl):
        t = time.time()
        W = (weights_tps(cc, ctrl, a.aniso) if a.interp == "tps"
             else weights_skimage(cc, ctrl, a.aniso))
        D = regularization_operator(cc[ctrl], a.zweight)
        print(f"  {len(ctrl)} control cells ({100 * len(ctrl) / len(cc):.1f} % of the mesh), "
              f"{len(D)} smoothness edges, W ({a.interp}) built in {time.time() - t:.1f}s", flush=True)
        return W, D

    invert = invert_gn if a.optimizer == "gn" else invert_lbfgs
    lams = [float(v) for v in a.lams.split(",")]
    kw = dict(chi2_target=a.chi2_target, norm=a.norm, max_iter=a.max_iter)

    print(f"PyTorch inversion ({a.optimizer}) on control cells ...", flush=True)
    t = time.time()
    ctrl0 = ctrl = place_controls(cc, a.dz0, a.growth, a.aspect, a.shrink)
    W, D = operators(ctrl)
    hist = []
    best, stages, hist = invert(fwd, W, D, dobs, np.full(len(ctrl), np.log(np.median(dobs))),
                                lams, hist=hist, **kw)
    added, refine_at = np.array([], dtype=int), None
    for k in range(a.refine):
        x_cells = W @ best["p"]
        ctrl, new = refine_controls(cc, ctrl, x_cells, cov / cov.max(), a.refine_add)
        if not len(new):
            break
        added = np.concatenate([added, new])
        refine_at = Counter.total() + 0.5 if refine_at is None else refine_at
        print(f"Refinement round {k + 1}: +{len(new)} control cells", flush=True)
        W, D = operators(ctrl)
        lams_r = [l for l in lams if l <= best["lam"]]
        best, st, hist = invert(fwd, W, D, dobs, x_cells[ctrl], lams_r, hist=hist, **kw)
        stages += st
    t_torch = time.time() - t
    n_calls = (Counter.jac, Counter.fwd)

    pg_tag = f"zw{a.pg_zweight:g}" + ("" if a.pg_smooth else "_blocky")
    pg_cache = os.path.join(out, f"pygimli_rho{a.target_rho:g}_{pg_tag}.npz")
    t = time.time()
    if a.no_pg and os.path.exists(pg_cache):
        z = np.load(pg_cache, allow_pickle=True)
        pgr, pg_runs = z["pgr"].item(), list(z["runs"])
        print("pyGIMLi result reused from", pg_cache)
    else:
        print("pyGIMLi inversion ...", flush=True)
        pgr, pg_runs = invert_pygimli(data, mesh, [float(v) for v in a.pg_lams.split(",")],
                                      a.chi2_target, a.pg_zweight, not a.pg_smooth,
                                      run_all=a.tradeoff)
        pgr["time"] = time.time() - t
        if not a.quick:
            np.savez(pg_cache, pgr=np.array(pgr, dtype=object),
                     runs=np.array(pg_runs, dtype=object))

    pgr["label"] = pg_tag
    best["metrics"] = model_metrics(cc, best["m"], a.target_rho)
    pgr["metrics"] = model_metrics(pgr["cc"], pgr["m"], a.target_rho)
    R = dict(rho_t=a.target_rho, cc=cc, tri=_triangles(fwd.fop.paraDomain), ctrl=ctrl,
             ctrl0=ctrl0, added=added, cov=cov, pg=pgr, torch=best, hist=hist,
             refine_at=refine_at)
    plot_results(R, prefix)

    if a.tradeoff:
        if len(pg_runs) < 3:
            print("  (trade-off: pyGIMLi cache holds only the chosen lambda; rerun without --no-pg)")
        print("Trade-off path of the sparse inversion (no early stop, initial controls) ...",
              flush=True)
        W0, D0 = operators(ctrl0)
        _, path, _ = invert(fwd, W0, D0, dobs, np.full(len(ctrl0), np.log(np.median(dobs))),
                            lams, chi2_target=-1.0, stall=1e9, norm=a.norm,
                            max_iter=a.max_iter, verbose=False)
        for p in path:
            p["metrics"] = model_metrics(cc, p["m"], a.target_rho)
        for p in pg_runs:
            p["metrics"] = model_metrics(p["cc"], p["m"], a.target_rho)
        plot_tradeoff({"pyGIMLi": pg_runs, "PyTorch sparse": path},
                      {"pyGIMLi": pgr, "PyTorch sparse": best}, a.target_rho,
                      prefix + "_tradeoff.png")
        print("  lambda path  (chi2 / RMS log10 / target):")
        for name, pts in [("pyGIMLi", pg_runs), ("PyTorch", path)]:
            print(f"  {name:8s} " + "  ".join(
                f"{p['lam']:g}:{p['chi2']:.2f}/{p['metrics']['rms_log10_roi']:.3f}/"
                f"{p['metrics']['target_median']:.0f}" for p in pts))

    lines = [f"SPARSE PYTORCH vs PYGIMLI  ({tag}; layers {RHO_LAYERS} Ohm.m, target "
             f"{a.target_rho:g} Ohm.m, {len(dobs)} dd data, {NOISE:.0%} noise)",
             f"PyTorch config: {config}",
             f"chi2 of the TRUE model on the inversion mesh (discretization floor): {floor:.2f}",
             f"{'':22s}{'params':>8s}{'lambda':>8s}{'chi2':>7s}{'RMS log10':>11s}"
             f"{'target':>9s}{'layer1':>8s}{'layer2':>8s}{'layer3':>8s}{'time':>8s}"]
    lines.append(f"{'true':22s}{'':>8s}{'':>8s}{'':>7s}{'':>11s}{a.target_rho:9.0f}"
                 + "".join(f"{v:8.0f}" for v in RHO_LAYERS))
    for name, r, npar, tt in [(f"pyGIMLi ({pg_tag})", pgr, len(pgr["m"]), pgr["time"]),
                              ("PyTorch sparse", best, len(ctrl), t_torch)]:
        mm = r["metrics"]
        lines.append(f"{name:22s}{npar:8d}{r['lam']:8g}{r['chi2']:7.2f}{mm['rms_log10_roi']:11.3f}"
                     f"{mm['target_median']:9.0f}{mm['layer1_median']:8.0f}"
                     f"{mm['layer2_median']:8.0f}{mm['layer3_median']:8.0f}{tt:7.0f}s")
    lines.append(f"PyTorch: {n_calls[0]} forward+Jacobian and {n_calls[1]} forward-only pyGIMLi "
                 f"calls over {len(stages)} lambda stages; RMS log10 is over the covered zone "
                 "(z <= 12 m, inside the data footprint); layer values are medians there, "
                 "target excluded.")
    txt = "\n".join(lines)
    open(prefix + "_summary.txt", "w").write(txt)
    np.savez(prefix + "_results.npz", cc=cc, ctrl=ctrl, ctrl0=ctrl0, added=added, W=W,
             m_torch=best["m"], p_torch=best["p"], m_pg=pgr["m"], cc_pg=pgr["cc"],
             hist=np.array(hist), dobs=dobs, metrics_torch=best["metrics"],
             metrics_pg=pgr["metrics"], config=config,
             stages_lam=[s["lam"] for s in stages], stages_chi2=[s["chi2"] for s in stages])
    print("\n" + txt)
    print(f"\nsaved {prefix}_models.png, _profiles.png, _summary.txt, _results.npz  "
          f"(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
