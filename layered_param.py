"""
layered_param.py
================
Explicit dipping-layer inversion ("layered_param" method of the benchmark).

Model: K interfaces, each a straight line  depth_k(x) = d_k + s_k (x - 23.5),
and K+1 layer resistivities (top down). Unknowns: d_k, s_k, log rho_0..K
(3K + 1 numbers). The model lives on the LARGER inversion domain (12 m beyond
the electrodes, 30 m deep; the outer region is filled by pyGIMLi from the
domain edges). The data misfit is always computed with the SHARP layered model:
the data are extremely sensitive to how a high-contrast boundary near the
surface is represented (smoothing it over 0.05 m raised the true model's chi2
from 2.8 to 17 for D2_02). A smoothed version (sigmoid, w = 0.4 m)
    log rho = r_0 + sum_k sigmoid((z - depth_k(x)) / w) (r_k - r_{k-1})
is used only to obtain search directions.

Inversion: Gauss-Newton / Levenberg-Marquardt with pyGIMLi sensitivities
(d log rhoa / d log rho per cell) chained with the derivatives of the layer
function (torch autograd), plus a weak pull towards the starting values.
Start: the dip-aware analysis of the pyGIMLi model (interface depths, dips,
layer values). Number of interfaces: the analysed K and K-1, chosen by
BIC = N chi2 + p ln N.
"""
import numpy as np

XM, W_SMOOTH = 23.5, 0.4
LO, HI = 0.0, np.log(1e4)
_MESH = {}


def full_mesh(scheme):
    """Larger mesh with every cell in ONE parameter region (no extrapolated
    background); cell centres in pyGIMLi's parameter order."""
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    if "mesh" not in _MESH:
        mesh = mt.createParaMesh(scheme, paraDepth=30.0, paraBoundary=12.0, quality=33,
                                 paraMaxCellSize=1.0, boundaryMaxCellSize=50)
        fop = ert.ERTModelling()
        fop.setData(scheme)
        fop.setMesh(mesh)
        _MESH["mesh"] = mesh
        _MESH["cc"] = np.array([[c.center().x(), c.center().y()] for c in fop.paraDomain.cells()])
    return _MESH["mesh"], _MESH["cc"]


class FullForward:
    def __init__(self, data, mesh):
        import pygimli as pg
        from pygimli.physics import ert
        self.pg = pg
        self.fop = ert.ERTModelling()
        self.fop.setData(data)
        self.fop.setMesh(mesh)                         # parameter domain; outer region filled by pyGIMLi

    def response(self, logrho):
        return np.array(self.fop.response(self.pg.Vector(np.exp(logrho))))

    def jacobian_loglog(self, logrho):
        res = np.exp(logrho)
        d = self.response(logrho)                    # potentials for createJacobian
        self.fop.createJacobian(self.pg.Vector(res))
        J = np.array(self.fop.jacobian())
        return J * res[None, :] / d[:, None], d


def layer_logrho(theta, K, x, z, w=None):
    """torch: log rho at points (x, z depth) for parameters theta. w = boundary
    smoothing width (m); w = 0 gives the SHARP layered model."""
    import torch
    w = W_SMOOTH if w is None else w
    d, s, r = theta[:K], theta[K:2 * K], theta[2 * K:]
    out = r[0] + 0 * x
    for k in range(K):
        arg = z - (d[k] + s[k] * (x - XM))
        sig = (arg > 0).to(x.dtype) if w == 0 else torch.sigmoid(arg / w)
        out = out + sig * (r[k + 1] - r[k])
    return torch.clamp(out, LO, HI)


def _invert_k(fwd, dobs, theta0, K, x, z, noise, n_iter=40, verbose=False):
    import torch
    from torch.func import jacrev
    th0 = torch.tensor(theta0, dtype=torch.float64)
    th = th0.clone()
    sig = torch.tensor([5.0] * K + [0.3] * K + [2.0] * (K + 1), dtype=torch.float64)   # prior scales
    dlog = np.log(dobs)
    N = len(dobs)

    # misfit with the SHARP model (the data are very sensitive to how boundaries
    # are represented); the smoothed model is used only for the search direction
    def f(t):
        return layer_logrho(t, K, x, z)

    def f_sharp(t):
        return layer_logrho(t, K, x, z, w=0)

    def chi2_of(t):
        with torch.no_grad():
            m = f_sharp(t).numpy()
        return float(np.mean(((np.log(fwd.response(m)) - dlog) / noise) ** 2))

    mu, prior_w = 1.0, 1e-3
    chi = None
    for it in range(n_iter):
        with torch.no_grad():
            m = f_sharp(th).numpy()
        J, d = fwd.jacobian_loglog(m)
        r = (np.log(d) - dlog) / noise
        chi = float(np.mean(r ** 2))
        Jt = torch.from_numpy(J / noise) @ jacrev(f)(th)                 # (N, p)
        rt = torch.from_numpy(r)
        pr = (th - th0) / sig
        phi0 = chi + prior_w * float((pr ** 2).sum())
        A = Jt.T @ Jt / N + prior_w * torch.diag(1 / sig ** 2)
        g = Jt.T @ rt / N + prior_w * pr / sig
        improved = False
        for _ in range(6):                                             # Levenberg-Marquardt damping
            step = -torch.linalg.solve(A + mu * torch.diag(torch.diag(A)), g)
            tn = th + step
            tn[K:2 * K] = torch.clamp(tn[K:2 * K], -np.tan(np.radians(40)), np.tan(np.radians(40)))
            tn[:K] = torch.clamp(tn[:K], 0.2, 29.0)
            tn[2 * K:] = torch.clamp(tn[2 * K:], LO + 0.1, HI - 0.1)
            cn = chi2_of(tn)
            phin = cn + prior_w * float((((tn - th0) / sig) ** 2).sum())
            if phin < phi0:
                th, chi, mu, improved = tn, cn, max(mu / 3, 1e-4), True
                break
            mu *= 4
        if verbose:
            print(f"    K={K} it {it}: chi2 {chi:.3f} mu {mu:.1e}", flush=True)
        if not improved or (phi0 - phin) / phi0 < 1e-3:
            break
    return th.numpy(), chi


def invert_layered(data, info, noise=0.03, verbose=False):
    """info: dip-aware analysis of the pyGIMLi model (z_if, slopes, trend, zg).
    Returns dict(theta, K, chi2, bic, model function)."""
    import torch
    from pygimli.physics import ert
    import sparse_torch_inversion_ert as sp
    scheme = ert.createData(elecs=sp.ELEC_X, schemeName="dd")
    mesh, cc = full_mesh(scheme)
    fwd = FullForward(data, mesh)
    dobs = np.array(data["rhoa"])
    x = torch.tensor(cc[:, 0], dtype=torch.float64)
    z = torch.tensor(-cc[:, 1], dtype=torch.float64)
    zi = list(np.asarray(info["z_if"], float))
    si = list(np.asarray(info.get("slopes", np.zeros(len(zi))), float))
    if not zi:                                         # no interface found: start with one at 5 m
        zi, si = [5.0], [0.0]
    zg, trend = info["zg"], info["trend"]

    def start(zs, ss):
        order = np.argsort(zs)
        zs, ss = list(np.array(zs)[order]), list(np.array(ss)[order])
        edges = [0.0] + zs + [zg[-1] + 4]
        r = [np.interp(0.5 * (a + b), zg, trend) * np.log(10) for a, b in zip(edges[:-1], edges[1:])]
        return np.array(zs + ss + r), len(zs)

    cands = [start(zi, si)]
    if len(zi) >= 2:                                   # also try one interface fewer (drop each)
        for j in range(len(zi)):
            cands.append(start([z_ for i, z_ in enumerate(zi) if i != j], [s_ for i, s_ in enumerate(si) if i != j]))
    N, best = len(dobs), None
    for th0, K in cands:
        th, chi = _invert_k(fwd, dobs, th0, K, x, z, noise, verbose=verbose)
        bic = N * chi + (3 * K + 1) * np.log(N)
        if verbose:
            print(f"  K={K}: chi2 {chi:.2f}, BIC {bic:.0f}", flush=True)
        if best is None or bic < best["bic"]:
            best = dict(theta=th, K=K, chi2=chi, bic=bic)
    return best


def evaluate(best, pts):
    """log rho of the fitted layered model at points pts (x, y = -depth)."""
    import torch
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    z = torch.tensor(-pts[:, 1], dtype=torch.float64)
    with torch.no_grad():
        return layer_logrho(torch.tensor(best["theta"]), best["K"], x, z, w=0).numpy()
