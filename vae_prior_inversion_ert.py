"""
vae_prior_inversion_ert.py
==========================
ERT inversion with a LEARNED prior. A variational autoencoder (VAE) is trained
on random synthetic subsurface models; its decoder then replaces the fixed
biharmonic interpolation of sparse_torch_inversion_ert.py, and the inversion
searches its 32 latent variables. pyGIMLi stays the forward model, so every
result still has to fit the data.

Pipeline
--------
1. Prior samples (no forward simulations needed): random log10-resistivity
   images (32 x 96 pixels over the inversion domain) with 2-4 layers separated
   by planar dipping interfaces, plus 0-2 rectangular or elliptical bodies,
   resistive or conductive. Rho ranges 10-1000 Ohm.m (layers) and 3-3000
   Ohm.m (bodies).
2. VAE (convolutional encoder/decoder, latent size 32), trained on the GPU if
   available. Gaussian likelihood with fixed sigma (--sigma, log10 units).
3. Inversion: model on the inversion mesh  m = soft_bounds(S dec(z)), S =
   bilinear sampling of the image at the cell centres. Gauss-Newton on z:
       Jz = J_pyGIMLi diag(b') S d dec / dz      (d dec/dz by torch.func.jacfwd)
       Phi = chi2 + lambda * mean(z^2)           (z ~ N(0, I) is the prior)
   lambda cooled until chi2 <= 1.05 or chi2 stalls (same rules as the sparse
   script); a few random restarts guard against local minima in latent space.
4. Comparison on three test models: the two in-family cases of the sparse
   script (500 / 10 Ohm.m target, three layers) and an OUT-OF-FAMILY case
   (sinusoidal interface + 60 deg dipping dyke) that the prior has never seen,
   against pyGIMLi (zWeight 0.3, blocky) and the sparse biharmonic inversion.
5. --learning-curve: train on 1k / 3k / 10k / 30k models and record the
   reconstruction error on held-out models and the inversion error on the
   three test cases - i.e. how many synthetic models the prior needs.

Usage
-----
    conda activate pg2
    python vae_prior_inversion_ert.py                    # train (30k) + invert 3 cases
    python vae_prior_inversion_ert.py --learning-curve   # + training-set-size study
    python vae_prior_inversion_ert.py --quick            # smoke test
"""

import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sparse_torch_inversion_ert as sp            # noqa: E402  (mesh, data, pyGIMLi, sparse)

# ERT_TEXTURE=1: every training model gets within-layer/body geostatistical texture
TEXTURE = os.environ.get("ERT_TEXTURE") == "1"
OUT = os.path.join(HERE, "outputs", "vae_prior" + ("_large" if sp.DOMAIN == "large" else "")
                   + ("_tex" if TEXTURE else ""))
# image rows (depth) x columns (x); the large domain keeps roughly the same pixel size
NZ_GRID, NX_GRID = (48, 128) if sp.DOMAIN == "large" else (32, 96)
H0, W0 = NZ_GRID // 8, NX_GRID // 8                # size after the 3 stride-2 encoder layers
LN10 = np.log(10.0)
CMAP = "turbo"
C_TRUE, C_PG, C_SP, C_VAE = "#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a"
GRID_C, INK2 = "#d9d8d2", "#5f5e5a"


# ── test cases ─────────────────────────────────────────────────────────────
DYKE = dict(x0=30.0, z0=2.0, dip=60.0, width=2.0, length_z=12.0)


def dyke_mask(cc):
    x, z = cc[:, 0], -cc[:, 1]
    a = np.deg2rad(DYKE["dip"])
    ux, uz = -np.cos(a), np.sin(a)                 # down-dip direction (z positive down)
    dx, dz = x - DYKE["x0"], z - DYKE["z0"]
    s = dx * ux + dz * uz
    d = np.abs(dx * uz - dz * ux)
    return (d <= DYKE["width"] / 2) & (s >= 0) & (s <= DYKE["length_z"] / np.sin(a))


def ood_log_rho(cc):
    """Out-of-family: sinusoidal interface + dipping dyke (not in the prior)."""
    x, z = cc[:, 0], -cc[:, 1]
    rho = np.where(z < 6.0 + 2.0 * np.sin(2 * np.pi * x / 25.0), 150.0, 40.0)
    rho = np.where(dyke_mask(cc), 800.0, rho)
    return np.log(rho)


FAULT = dict(x0=25.0, dip=70.0, throw=3.0)         # fault through x=25 m at the surface


def _unfaulted(cc):
    z = -cc[:, 1]
    return np.log(np.where(z < 3.0, 200.0, np.where(z < 9.0, 60.0, 25.0)))


def _hanging(cc):
    x, z = cc[:, 0], -cc[:, 1]
    return x > FAULT["x0"] + z / np.tan(np.deg2rad(FAULT["dip"]))


def fault_log_rho(cc):
    """3 layers (200/60/25 Ohm.m, interfaces 3 and 9 m) cut by a 70 deg fault,
    right block down-thrown by 3 m."""
    shifted = cc.copy()
    shifted[:, 1] = cc[:, 1] + FAULT["throw"]            # sample the layering 3 m shallower
    return np.where(_hanging(cc), _unfaulted(shifted), _unfaulted(cc))


def fault_mask(cc):
    return np.abs(fault_log_rho(cc) - _unfaulted(cc)) > 1e-9


def trunc_log_rho(cc):
    """Middle layer ends abruptly at x = 25 m (top layer continues down to the
    lower interface beyond it)."""
    x, z = cc[:, 0], -cc[:, 1]
    rho = np.where(z < sp.Z1, 250.0, np.where(z < sp.z2(x), np.where(x < 25.0, 80.0, 250.0), 30.0))
    return np.log(rho)


def trunc_mask(cc):
    x, z = cc[:, 0], -cc[:, 1]
    return (x >= 25.0) & (z >= sp.Z1) & (z < sp.z2(x))


def grad_mask(cc):
    x, z = cc[:, 0], -cc[:, 1]
    return ((x - 15.0) / 3.0) ** 2 + ((z - 5.0) / 1.5) ** 2 <= 1


def grad_log_rho(cc):
    """Weathering-type gradient (400 -> ~10 Ohm.m over 20 m, gentle lateral
    undulation) with a 15 Ohm.m conductive lens."""
    x, z = cc[:, 0], -cc[:, 1]
    l10 = 2.6 - 0.08 * np.minimum(z, 20.0) + 0.15 * np.sin(2 * np.pi * x / 40.0)   # 10 Ohm.m below 20 m
    l10 = np.where(grad_mask(cc), np.log10(15.0), l10)
    return l10 * LN10


CASES = {
    "res500": dict(fn=lambda cc: sp.true_log_rho(cc, 500.0), mask=sp.target_mask,
                   title="in-family: 3 layers + resistive target (500 Ohm.m)", xs=(23.5, 8.0)),
    "con10": dict(fn=lambda cc: sp.true_log_rho(cc, 10.0), mask=sp.target_mask,
                  title="in-family: 3 layers + conductive target (10 Ohm.m)", xs=(23.5, 8.0)),
    "ood": dict(fn=ood_log_rho, mask=dyke_mask,
                title="sinusoidal interface + 60 deg dyke (800 Ohm.m)", xs=(26.0, 10.0)),
    "fault": dict(fn=fault_log_rho, mask=fault_mask,
                  title="3 layers cut by a 70 deg normal fault (3 m throw)", xs=(20.0, 32.0)),
    "trunc": dict(fn=trunc_log_rho, mask=trunc_mask,
                  title="middle layer ending abruptly at x = 25 m", xs=(15.0, 35.0)),
    "grad": dict(fn=grad_log_rho, mask=grad_mask,
                 title="weathering gradient (400 -> 10 Ohm.m) + conductive lens", xs=(15.0, 35.0)),
}


def simulate(logrho_fn, seed=0):
    """Same recipe as sp.simulate_data (fine, different mesh; 3 % noise)."""
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    scheme = ert.createData(elecs=sp.ELEC_X, schemeName="dd")
    fine = mt.createParaMesh(scheme, paraDepth=sp.SIM_DEPTH, quality=34,
                             paraMaxCellSize=sp.SYN_CELL, boundaryMaxCellSize=500)
    cc = np.array([[c.center().x(), c.center().y()] for c in fine.cells()])
    data = ert.simulate(fine, scheme=scheme, res=np.exp(logrho_fn(cc)),
                        noiseLevel=0, noiseAbs=0, verbose=False)
    clean = np.array(data["rhoa"])
    rng = np.random.default_rng(seed)
    data["rhoa"] = clean * (1 + rng.normal(0, sp.NOISE, size=clean.shape))
    data["err"] = np.full(len(clean), sp.NOISE)
    bad = np.where(np.array(data["rhoa"]) <= 0)[0]
    if len(bad):
        data.markInvalid(bad)
        data.removeInvalid()
    return data


def metrics(cc, m, case):
    true = case["fn"](cc)
    e = (m - true) / LN10
    roi, an = sp.roi_mask(cc), case["mask"](cc)
    return dict(rms=float(np.sqrt(np.mean(e[roi] ** 2))),
                anomaly=float(np.exp(np.median(m[an]))),
                anomaly_true=float(np.exp(np.median(true[an]))))


# ── image grid <-> inversion cells ─────────────────────────────────────────
def make_grid(cc):
    xg = np.linspace(cc[:, 0].min(), cc[:, 0].max(), NX_GRID)   # pixel centres
    zg = np.linspace(0.0, -cc[:, 1].min(), NZ_GRID)             # depth, row 0 = surface
    return xg, zg


def bilinear_matrix(cc, xg, zg):
    """(n_cells x n_pix) bilinear sampling of a (NZ, NX) image at the cells."""
    n = len(cc)
    fx = np.clip((cc[:, 0] - xg[0]) / (xg[1] - xg[0]), 0, NX_GRID - 1 - 1e-9)
    fz = np.clip((-cc[:, 1] - zg[0]) / (zg[1] - zg[0]), 0, NZ_GRID - 1 - 1e-9)
    ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
    tx, tz = fx - ix, fz - iz
    S = np.zeros((n, NZ_GRID * NX_GRID))
    r = np.arange(n)
    for dz_, dx_, w in [(0, 0, (1 - tz) * (1 - tx)), (0, 1, (1 - tz) * tx),
                        (1, 0, tz * (1 - tx)), (1, 1, tz * tx)]:
        np.add.at(S, (r, (iz + dz_) * NX_GRID + ix + dx_), w)
    return S


# ── prior: random layered models with bodies ───────────────────────────────
def random_models(n, xg, zg, seed):
    """n random log10-resistivity images (NZ, NX)."""
    rng = np.random.default_rng(seed)
    X, Z = np.meshgrid(xg, zg)
    xm = 0.5 * (xg[0] + xg[-1])
    out = np.empty((n, NZ_GRID, NX_GRID), dtype=np.float32)
    for k in range(n):
        n_int = rng.integers(1, 4)                           # 2-4 layers
        layer = np.zeros(X.shape, dtype=int)
        for _ in range(n_int):
            d, s = rng.uniform(1.0, 16.0), rng.uniform(-0.15, 0.15)   # depth, dip (+-8.5 deg)
            layer += Z > d + s * (X - xm)
        img = rng.uniform(1.0, 3.0, n_int + 1)[layer]       # 10-1000 Ohm.m
        for _ in range(rng.choice(3, p=[0.2, 0.6, 0.2])):    # 0-2 bodies
            xc, top = rng.uniform(3, 44), rng.uniform(0.5, 12)
            w, h = rng.uniform(2, 14), rng.uniform(1, 6)
            if rng.random() < 0.6:
                m = (np.abs(X - xc) <= w / 2) & (Z >= top) & (Z <= top + h)
            else:
                m = ((X - xc) / (w / 2)) ** 2 + ((Z - top - h / 2) / (h / 2)) ** 2 <= 1
            img[m] = rng.uniform(0.5, 3.5)                   # 3-3000 Ohm.m
        out[k] = img
    return out


def _grf(rng, X, Z, L, aniso=1.0, n_modes=32):
    """Smooth Gaussian random field (unit variance, squared-exponential
    correlation length L, horizontally stretched by aniso) via random Fourier
    features."""
    kx = rng.normal(0, 1.0 / (L * aniso), n_modes)
    kz = rng.normal(0, 1.0 / L, n_modes)
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    f = np.cos(np.multiply.outer(X, kx) + np.multiply.outer(Z, kz) + ph).sum(-1)
    return f * np.sqrt(2.0 / n_modes)


def _curve(rng, x, amp):
    """Smooth 1-D undulation (3 sinusoids, wavelengths 8-60 m), max |.| ~ amp."""
    f = sum(rng.uniform(0, 1) * np.sin(2 * np.pi * x / rng.uniform(8, 60) + rng.uniform(0, 2 * np.pi))
            for _ in range(3))
    return amp * f / max(np.abs(f).max(), 1e-9)


def _add_body(rng, img, X, Z):
    """One body: rotated rectangle (incl. thin dykes), rotated ellipse, curved
    channel or irregular blob; absolute or host-relative resistivity."""
    xc, zc = rng.uniform(0, 47), rng.uniform(0.5, 16)
    w, h = rng.uniform(1, 25), rng.uniform(0.5, 8)
    th = np.deg2rad(rng.uniform(-90, 90)) if rng.random() < 0.6 else 0.0
    u = (X - xc) * np.cos(th) + (Z - zc) * np.sin(th)
    v = -(X - xc) * np.sin(th) + (Z - zc) * np.cos(th)
    shape = rng.choice(4, p=[0.35, 0.3, 0.15, 0.2])
    if shape == 0:                                           # rectangle / dyke
        m = (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)
    elif shape == 1:                                         # ellipse
        m = (u / (w / 2)) ** 2 + (v / (h / 2)) ** 2 <= 1
    elif shape == 2:                                         # channel (meandering band)
        v0 = rng.uniform(0.5, 3) * np.sin(2 * np.pi * u / rng.uniform(8, 30) + rng.uniform(0, 6.3))
        m = (np.abs(u) <= w / 2) & (np.abs(v - v0) <= max(h / 4, 0.4))
    else:                                                    # irregular blob
        g = _grf(rng, X, Z, rng.uniform(1.5, 5))
        m = ((u / (w / 2)) ** 2 + (v / (h / 2)) ** 2 <= 1.5) & (g > rng.uniform(-0.3, 0.5))
    if not m.any():
        return img
    img = img.copy()
    if rng.random() < 0.5:
        img[m] = rng.uniform(0.0, 4.0)                       # 1-10 000 Ohm.m
    else:
        img[m] = np.median(img[m]) + rng.uniform(-1.5, 1.5)  # relative to the host
    if rng.random() < 0.2:                                   # internal texture
        img[m] += rng.uniform(0.05, 0.2) * _grf(rng, X, Z, rng.uniform(1, 4))[m]
    return img


def random_models_general(n, xg, zg, seed):
    """Broad family: 1-6 layers (dip +-25 deg, curved / truncated interfaces,
    faults), within-layer trends and texture, smooth non-layered fields, and
    0-5 bodies of four shapes. log10 rho in [0, 4] (1-10 000 Ohm.m)."""
    rng = np.random.default_rng(seed)
    X, Z = np.meshgrid(xg, zg)
    x, xm = xg, 0.5 * (xg[0] + xg[-1])
    out = np.empty((n, NZ_GRID, NX_GRID), dtype=np.float32)
    for k in range(n):
        if rng.random() < 0.15:                              # non-layered smooth field
            img = rng.uniform(1.0, 3.0) + rng.uniform(0.2, 0.8) * _grf(
                rng, X, Z, rng.uniform(2, 15), aniso=rng.uniform(1, 4))
            nb_p = [0.4, 0.3, 0.2, 0.1]
        else:
            n_int = rng.integers(0, 6)                       # 1-6 layers
            deps = []
            for _ in range(n_int):
                dep = rng.uniform(0.5, 18.0) + np.tan(np.deg2rad(rng.uniform(-25, 25))) * (x - xm)
                if rng.random() < 0.4:
                    dep = dep + _curve(rng, x, rng.uniform(0.3, 3.0))
                if rng.random() < 0.15:                      # interface exists on one side only
                    xt = rng.uniform(5, 42)
                    keep = (x < xt) if rng.random() < 0.5 else (x > xt)
                    dep = np.where(keep, dep, np.inf)
                deps.append(dep)
            Zq = Z
            if n_int and rng.random() < 0.25:                # fault: shift one side
                plane = rng.uniform(5, 42) + rng.choice([-1, 1]) * Z / np.tan(np.deg2rad(rng.uniform(45, 90)))
                Zq = np.where(X > plane, Z - rng.uniform(-5, 5), Z)
            layer = np.zeros(X.shape, dtype=int)
            for dep in deps:
                layer += Zq > dep[None, :]
            img = rng.uniform(0.3, 3.7, n_int + 1)[layer]  # 2-5000 Ohm.m
            if rng.random() < 0.5:                           # linear trends inside layers
                a_, b_ = rng.uniform(-0.3, 0.3, (2, n_int + 1))
                img = img + a_[layer] * (X - xm) / 25.0 + b_[layer] * (Z - 10.0) / 10.0
            if rng.random() < 0.4:                           # correlated texture
                img = img + rng.uniform(0.03, 0.2) * _grf(rng, X, Z, rng.uniform(2, 10),
                                                          aniso=rng.uniform(1, 3))
            nb_p = [0.15, 0.3, 0.25, 0.15, 0.1, 0.05]
        for _ in range(rng.choice(len(nb_p), p=nb_p)):
            img = _add_body(rng, img, X, Z)
        if TEXTURE:                                          # texture inside every layer and body
            img = img + rng.uniform(0.02, 0.2) * _grf(rng, X, Z, rng.uniform(1.5, 4), aniso=rng.uniform(2, 8))
        out[k] = np.clip(img, 0.0, 4.0)
    return out


def grid_from_cells(ccs, vals, xg, zg):
    """Linear interpolation of cell values onto the image grid (nearest outside)."""
    from scipy.interpolate import griddata
    Xg, Zg = np.meshgrid(xg, zg)
    pts = np.column_stack([Xg.ravel(), -Zg.ravel()])
    v = griddata(ccs, vals, pts, method="linear")
    v = np.where(np.isnan(v), griddata(ccs, vals, pts, method="nearest"), v)
    return v.reshape(Xg.shape)


TRACK_MODE = "dominant"          # interface tracking for the dip-aware prior: "dominant" | "predictive"


def _dip_layering(img, covn, xg, zg, central, max_if=3):
    """Dip-aware layering of a model image: dominant slope s (dz/dx) from the
    image gradients in the covered zone (layers tilted by s satisfy
    df/dx = -s df/dz), the depth profile read ALONG that slope, interfaces from
    it, and an individual slope per interface (depth of the steepest vertical
    change tracked column by column, weighted line fit)."""
    xm = 0.5 * (xg[0] + xg[-1])
    gz, gx = np.gradient(img, zg, xg)
    w = (covn >= 0.01) & central[None, :]
    s = -np.sum(w * gx * gz) / max(np.sum(w * gz ** 2), 1e-12)
    s = float(np.clip(s, -np.tan(np.radians(35)), np.tan(np.radians(35))))
    cols = np.where(central)[0]
    # profile along the slope: value at depth z_k + s (x_j - xm), median over columns
    prof = np.array([[np.interp(zk + s * (xg[j] - xm), zg, img[:, j]) for j in cols] for zk in zg])
    trend = np.median(prof, axis=1)
    g = np.abs(np.gradient(trend, zg))
    cand = sorted([i for i in range(1, len(zg) - 1)
                   if g[i] >= g[i - 1] and g[i] >= g[i + 1] and g[i] > 0.03], key=lambda i: -g[i])
    ifs = []
    for i in cand:
        if all(abs(zg[i] - zg[j]) >= 1.5 for j in ifs):
            ifs.append(i)
        if len(ifs) == max_if:
            break
    found = []
    jc = cols[np.argmin(np.abs(xg[cols] - xm))]
    for i in sorted(ifs):
        dd, ww, xx = [], [], []
        # track the interface from the centre outwards: each column searches +-1.5 m
        # around the depth found in the previous column, so the interface may
        # deviate gradually from the dominant dip (opposite dips, wedges)
        for side in (cols[cols >= jc], cols[cols < jc][::-1]):
            px, pz = [xg[jc]], [zg[i]]                   # points found on this side (start: centre)
            for j in side:
                if TRACK_MODE == "predictive":
                    # predicted depth: last point + local slope (dominant dip until enough points)
                    s_loc = s
                    if len(px) >= 4:
                        s_loc = float(np.clip(np.polyfit(px[-6:], pz[-6:], 1)[0], -np.tan(np.radians(35)),
                                              np.tan(np.radians(35))))
                    zpred, win = pz[-1] + s_loc * (xg[j] - px[-1]), 1.2
                else:                                    # window around the dominant-dip line
                    zpred, win = zg[i] + s * (xg[j] - xm), 2.0
                sel = np.abs(zg - zpred) <= win
                if sel.sum() < 2 or covn[sel, j].max() < 0.01:
                    continue
                k = np.argmax(np.abs(gz[sel, j]))
                px.append(xg[j]); pz.append(zg[sel][k])
                dd.append(zg[sel][k]); ww.append(np.abs(gz[sel, j][k])); xx.append(xg[j] - xm)
        if len(dd) >= 5:
            sl, ic = np.polyfit(np.array(xx), np.array(dd), 1, w=np.sqrt(np.array(ww)))
            sl = float(np.clip(sl, -np.tan(np.radians(35)), np.tan(np.radians(35))))
            strength = float(np.mean(ww))
        else:
            sl, ic, strength = s, zg[i], 0.0
        found.append((float(ic), sl, strength))
    # merge near-duplicates (the same interface found twice): keep the stronger one
    found.sort()
    z_if, slopes, last = [], [], None
    for ic, sl, st in found:
        if last is not None and abs(ic - last[0]) < 1.0 and abs(np.arctan(sl) - np.arctan(last[1])) < np.radians(6):
            if st > last[2]:
                z_if[-1], slopes[-1], last = ic, sl, (ic, sl, st)
            continue
        z_if.append(ic); slopes.append(sl); last = (ic, sl, st)
    zs = np.meshgrid(xg, zg)[1] - s * (np.meshgrid(xg, zg)[0] - xm)   # sheared depth of every pixel
    resid = img - np.interp(zs, zg, trend)
    return dict(trend=trend, z_if=np.array(z_if), slopes=np.array(slopes), shear=s, resid=resid)


def analyse_pg(img, covn, xg, zg, max_bodies=4, edge_margin=None, dip_aware=False, min_amp=None,
               drop_below_doi=False):
    """Structure of a pyGIMLi model image (log10 rho): 1-D depth trend,
    interfaces (strongest vertical changes of the trend), depth of
    investigation per column (normalized sensitivity < 0.01) and anomalies
    (connected regions of the lateral residual img - trend) with sign,
    strength, centre, extent and orientation (weighted principal axes).
    Stricter version (field data; off by default): min_amp drops anomalies whose peak
    residual is below min_amp (log10), drop_below_doi those centred below the DOI."""
    from scipy import ndimage
    central = (xg >= sp.ELEC_X[0]) & (xg <= sp.ELEC_X[-1])
    trend = np.median(img[:, central], axis=1)
    g = np.abs(np.gradient(trend, zg))
    cand = sorted([i for i in range(1, len(zg) - 1)
                   if g[i] >= g[i - 1] and g[i] >= g[i + 1] and g[i] > 0.03], key=lambda i: -g[i])
    ifs = []
    for i in cand:
        if all(abs(zg[i] - zg[j]) >= 1.5 for j in ifs):
            ifs.append(i)
        if len(ifs) == 3:
            break
    z_if = np.sort(zg[ifs]) if ifs else np.array([])
    low = covn < 0.01
    doi = np.array([zg[np.argmax(low[:, j])] if low[:, j].any() else zg[-1] for j in range(len(xg))])
    r = img - trend[:, None]
    dip = None
    if dip_aware:                                       # layering read along the dip
        dip = _dip_layering(img, covn, xg, zg, central)
        trend, z_if, r = dip["trend"], dip["z_if"], dip["resid"]
    thr = max(0.08, 0.35 * np.abs(r[:, central]).max())
    bodies = []
    for sgn in (1, -1):
        lab, nl = ndimage.label(sgn * r > thr)
        for k in range(1, nl + 1):
            mk = lab == k
            if mk.sum() < 6:
                continue
            iz, ix = np.nonzero(mk)
            w = np.abs(r[mk])
            xc, zc = np.average(xg[ix], weights=w), np.average(zg[iz], weights=w)
            C = np.cov(np.vstack([xg[ix] - xc, zg[iz] - zc]), aweights=w)
            ev, evec = np.linalg.eigh(C)
            major = evec[:, 1] if evec[1, 1] >= 0 else -evec[:, 1]      # points down-dip
            bodies.append(dict(sign=sgn, amp=float(w.max()), xc=float(xc), zc=float(zc),
                               L=float(3.5 * np.sqrt(max(ev[1], 1e-6))),
                               W=float(3.5 * np.sqrt(max(ev[0], 1e-6))),
                               theta=float(np.arctan2(major[1], major[0])),
                               top=float(zg[iz].min()), bottom=float(zg[iz].max()),
                               area=int(mk.sum()), doi=float(doi[int(round(ix.mean()))]),
                               inside=float(np.mean((xg[ix] >= sp.ELEC_X[0]) & (xg[ix] <= sp.ELEC_X[-1])))))
    # edge filter (anomaly-removal version): pyGIMLi values at the line ends are
    # unreliable; keep an anomaly only if its centre is >= edge_margin inside the
    # electrode spread and at least half of it lies under the electrodes
    dropped = []
    if edge_margin is not None:
        e0, e1 = sp.ELEC_X[0], sp.ELEC_X[-1]
        keep = []
        for b in bodies:
            ok = (e0 + edge_margin <= b["xc"] <= e1 - edge_margin) and b["inside"] >= 0.5
            (keep if ok else dropped).append(b)
        bodies = keep
    if min_amp is not None or drop_below_doi:
        keep = []
        for b in bodies:
            ok = (min_amp is None or b["amp"] >= min_amp) and not (drop_below_doi and b["zc"] > b["doi"])
            (keep if ok else dropped).append(b)
        bodies = keep
    bodies = sorted(bodies, key=lambda b: -b["amp"] * b["area"])[:max_bodies]
    # resolution index per depth (0 = data blind, 1 = best), from the sensitivity
    sens = np.median(covn[:, central], axis=1)
    res_z = np.clip((np.log10(np.maximum(sens, 1e-6)) + 3.0) / 3.0, 0.0, 1.0)
    out = dict(trend=trend, z_if=z_if, doi=doi, bodies=bodies, resid=r, thr=thr,
               dropped=dropped, res_z=res_z)
    if dip is not None:
        out.update(slopes=dip["slopes"], shear=dip["shear"])
    return out


def _pg_body(rng, b, extend):
    """Randomized version of one pyGIMLi anomaly: narrower, stronger and
    (usually) continued deeper. Returns centre, along/across lengths, angle
    and the vertical extension applied."""
    th = b["theta"] + np.deg2rad(rng.normal(0, 10))
    if np.sin(th) < 0:
        th += np.pi                                          # keep the axis pointing down
    length, width = b["L"] * rng.uniform(0.6, 1.1), max(b["W"] * rng.uniform(0.4, 0.9), 0.7)
    xc, zc = b["xc"] + rng.normal(0, 1.0), b["zc"] + rng.normal(0, 0.5)
    ext = 0.0
    if rng.random() < 0.8:                                   # deeper continuation
        ext = rng.uniform(0, extend * max(b["bottom"] - b["top"], b["doi"] - b["bottom"], 2.0))
        steep = abs(np.sin(th)) > 0.5 and b["L"] > 1.3 * b["W"]
        if steep:                                            # along the dip (dyke-like)
            dl = ext / abs(np.sin(th))
            length += dl
            xc, zc = xc + np.cos(th) * dl / 2, zc + np.sin(th) * dl / 2
        else:                                                # straight down (thicker body)
            th = np.arctan2(np.sin(th), np.cos(th))
            if abs(np.sin(th)) <= 0.5:                       # near-horizontal major axis
                width += ext
            else:
                length += ext
            zc += ext / 2
    return xc, zc, length, width, th, ext


def random_models_pg(n, xg, zg, info, seed, mix=0.15, extend=1.0, adaptive=False, contrast=(1.0, 1.8),
                     extras=(0.6, 0.3, 0.1)):
    """Training models around a pyGIMLi model (see analyse_pg): jittered and
    sharpened layering, the detected anomalies as narrower / stronger bodies
    continued deeper into the low-sensitivity zone, plus random extras; a
    fraction `mix` of fully general models keeps the prior broad.
    adaptive=True (anomaly-removal version): the contrast increase depends on the
    resolution at that depth - layers x U(1, 1 + 0.8 (1 - res)), bodies
    x U(1, 1.5 + 1.5 (1 - res)) - instead of fixed U(1, 1.8) and U(1, 3).
    extras: probabilities of 0, 1 and 2 unexpected random bodies per model."""
    res_z = info.get("res_z")
    rng = np.random.default_rng(seed)
    X, Z = np.meshgrid(xg, zg)
    x, xm = xg, 0.5 * (xg[0] + xg[-1])
    trend = info["trend"]
    out = np.empty((n, NZ_GRID, NX_GRID), dtype=np.float32)
    for k in range(n):
        if rng.random() < mix:
            out[k] = random_models_general(1, xg, zg, seed=int(rng.integers(1 << 31)))[0]
            continue
        deps = []
        slopes = info.get("slopes")
        for i_, d in enumerate(info["z_if"]):
            if rng.random() < 0.1:                           # the interface may be an artefact
                continue
            s0 = 0.0 if slopes is None else float(slopes[i_])      # dip-aware: around the estimated dip
            dep = d + rng.normal(0, 0.15 * abs(d) + 0.3) + (s0 + np.tan(np.deg2rad(rng.uniform(-6, 6)))) * (x - xm)
            if rng.random() < 0.3:
                dep = dep + _curve(rng, x, rng.uniform(0.2, 1.5))
            deps.append(dep)
        if rng.random() < 0.2:                               # an interface pyGIMLi smoothed away
            s_ex = info.get("shear", 0.0)
            deps.append(rng.uniform(1, 18) + (s_ex + np.tan(np.deg2rad(rng.uniform(-10, 10)))) * (x - xm))
        deps.sort(key=lambda d: d.mean())
        layer = np.zeros(X.shape, dtype=int)
        for dep in deps:
            layer += Z > dep[None, :]
        edges = [0.0] + [float(d.mean()) for d in deps] + [float(zg[-1]) + 4.0]
        vals = np.array([np.interp(0.5 * (a_ + b_), zg, trend) for a_, b_ in zip(edges[:-1], edges[1:])])
        if adaptive:
            mids = np.array([0.5 * (a_ + b_) for a_, b_ in zip(edges[:-1], edges[1:])])
            fac = rng.uniform(1.0, 1.0 + 0.8 * (1.0 - np.interp(mids, zg, res_z)))
        else:
            fac = rng.uniform(*contrast)                    # layer-contrast factor range
        vals = vals.mean() + fac * (vals - vals.mean()) + rng.normal(0, 0.1, len(vals))
        img = vals[layer]
        if rng.random() < 0.3:
            img = img + rng.uniform(0.03, 0.15) * _grf(rng, X, Z, rng.uniform(2, 10), aniso=2.0)
        for b in info["bodies"]:
            # the anomaly may be an artefact; more likely below the depth of investigation
            if rng.random() < (0.15 if b["zc"] <= b["doi"] else 0.5):
                continue
            xc, zc, length, width, th, _ = _pg_body(rng, b, extend)
            u = (X - xc) * np.cos(th) + (Z - zc) * np.sin(th)
            v = -(X - xc) * np.sin(th) + (Z - zc) * np.cos(th)
            if rng.random() < 0.6:
                m = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)
            else:
                m = (u / (length / 2)) ** 2 + (v / (width / 2)) ** 2 <= 1
            if m.any():
                host = np.median(img[m])
                img = img.copy()
                hi = 1.5 + 1.5 * (1.0 - float(np.interp(b["zc"], zg, res_z))) if adaptive else 3.0
                img[m] = host + b["sign"] * b["amp"] * rng.uniform(1.0, hi)
        for _ in range(rng.choice(3, p=list(extras))):       # unexpected extras
            img = _add_body(rng, img, X, Z)
        if TEXTURE:                                          # texture inside every layer and body
            img = img + rng.uniform(0.02, 0.2) * _grf(rng, X, Z, rng.uniform(1.5, 4), aniso=rng.uniform(2, 8))
        out[k] = np.clip(img, 0.0, 4.0)
    return out


def plot_pg_prior(info, img, Xs, xg, zg, fn, extend=1.0):
    """What was read from the pyGIMLi model and examples of the training set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Polygon
    ext = (xg[0], xg[-1], -zg[-1], -zg[0])
    norm = LogNorm(10 ** max(img.min() - 0.5, 0), 10 ** min(img.max() + 0.5, 4))
    fig, axs = plt.subplots(3, 4, figsize=(17, 6.6), constrained_layout=True)
    ax = axs[0, 0]
    im = ax.imshow(10 ** img, extent=ext, aspect="equal", cmap=CMAP, norm=norm)
    xm_ = 0.5 * (xg[0] + xg[-1])
    for i_, d in enumerate(info["z_if"]):
        sl = 0.0 if info.get("slopes") is None else info["slopes"][i_]
        ax.plot(xg, -(d + sl * (xg - xm_)), color="white", lw=1.2, ls="--")
    ax.plot(xg, -info["doi"], color="black", lw=1.2, label="depth of investigation")
    for b in info["bodies"]:
        th = b["theta"]
        c, s_ = np.cos(th), np.sin(th)
        corners = [(-b["L"] / 2, -b["W"] / 2), (b["L"] / 2, -b["W"] / 2), (b["L"] / 2, b["W"] / 2),
                   (-b["L"] / 2, b["W"] / 2)]
        pts = [(b["xc"] + u * c - v * s_, -(b["zc"] + u * s_ + v * c)) for u, v in corners]
        ax.add_patch(Polygon(pts, fill=False, ec="white", lw=1.5))
        dmax = extend * max(b["bottom"] - b["top"], b["doi"] - b["bottom"], 2.0)
        ax.annotate("", xy=(b["xc"], -(b["bottom"] + dmax)), xytext=(b["xc"], -b["bottom"]),
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5))
    for b in info.get("dropped", []):
        th = b["theta"]
        c, s_ = np.cos(th), np.sin(th)
        corners = [(-b["L"] / 2, -b["W"] / 2), (b["L"] / 2, -b["W"] / 2), (b["L"] / 2, b["W"] / 2),
                   (-b["L"] / 2, b["W"] / 2)]
        pts = [(b["xc"] + u * c - v * s_, -(b["zc"] + u * s_ + v * c)) for u, v in corners]
        ax.add_patch(Polygon(pts, fill=False, ec="0.6", lw=1.3, ls="--"))
    extra = ", ignored edge anomalies (grey)" if info.get("dropped") else ""
    ax.set_title("pyGIMLi model: interfaces (--), anomalies (boxes),\n"
                 f"max. deeper continuation (arrows), DOI (black){extra}", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    ax = axs[0, 1]
    ax.imshow(info["resid"], extent=ext, aspect="equal", cmap=CMAP, norm=Normalize(-0.8, 0.8))
    ax.contour(xg, -zg, np.abs(info["resid"]), levels=[info["thr"]], colors="white", linewidths=1)
    ax.set_title(f"lateral residual (log10), anomaly threshold {info['thr']:.2f}", fontsize=8)
    for ax, sm in zip(axs.ravel()[2:], Xs):
        ax.imshow(10 ** sm, extent=ext, aspect="equal", cmap=CMAP, norm=norm)
        ax.set_title("training model", fontsize=8)
    for ax in axs.ravel():
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7, label="resistivity (Ohm.m)")
    sp._save(fig, fn)
    plt.close(fig)


FAMILIES = {"layered": random_models, "general": random_models_general, "pg": None}
FAMILY_DEFAULTS = {
    "layered": dict(n_train=30000, nz=32, steps=12000, batch=128, ns="1000,3000,10000,30000"),
    "general": dict(n_train=200000, nz=64, steps=40000, batch=256, ns="10000,30000,100000,200000"),
    "pg": dict(n_train=40000, nz=32, steps=12000, batch=128, ns=""),
}


# ── VAE ────────────────────────────────────────────────────────────────────
def make_vae(nz):
    import torch.nn as nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.nz = nz
            self.enc = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(),        # 16 x 48
                nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),       # 8 x 24
                nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),      # 4 x 12
                nn.Flatten(), nn.Linear(128 * H0 * W0, 2 * nz))
            self.fc = nn.Linear(nz, 128 * H0 * W0)

            def up(ci, co):
                return [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        nn.Conv2d(ci, co, 3, 1, 1), nn.GELU()]
            self.dec = nn.Sequential(*up(128, 64), *up(64, 32), *up(32, 16),
                                     nn.Conv2d(16, 1, 3, 1, 1))

        def encode(self, x):
            mu, logvar = self.enc(x).chunk(2, dim=1)
            return mu, logvar

        def decode(self, z):
            return self.dec(self.fc(z).view(-1, 128, H0, W0))

    return VAE()


def train_vae(X, nz, steps, batch, sigma, device, seed=0, verbose=True):
    """X: (n, NZ, NX) log10 images. Returns (model on CPU, mean, std)."""
    import torch
    torch.manual_seed(seed)
    mean, std = float(X.mean()), float(X.std())
    Xt = torch.from_numpy((X[:, None] - mean) / std).float().to(device)
    model = make_vae(nz).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    s2 = 2 * (sigma / std) ** 2
    t0 = time.time()
    for step in range(steps):
        x = Xt[torch.randint(0, len(Xt), (batch,), device=device)]
        mu, logvar = model.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec = ((model.decode(z) - x) ** 2).sum((1, 2, 3)) / s2
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        warm = min(1.0, step / (0.2 * steps))                # KL warm-up
        loss = (rec + warm * kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step + 1) % max(1, steps // 5) == 0:
            print(f"    step {step + 1:6d}/{steps}  rec={rec.mean().item():9.1f}  "
                  f"kl={kl.mean().item():6.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return model.cpu().eval(), mean, std


def recon_rms(model, X, mean, std):
    """Held-out reconstruction RMS (log10) using the posterior mean."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy((X[:, None] - mean) / std).float()
        mu, _ = model.encode(x)
        xr = model.decode(mu).numpy()[:, 0] * std + mean
    return float(np.sqrt(np.mean((xr - X) ** 2)))


def load_or_train(path, X, a, device, verbose=True):
    import torch
    if os.path.exists(path) and not a.retrain:
        ck = torch.load(path, weights_only=False)
        model = make_vae(ck["nz"])
        model.load_state_dict(ck["state"])
        return model.eval(), ck["mean"], ck["std"]
    if verbose:
        print(f"  training VAE on {len(X)} models ({a.steps} steps, {device}) ...", flush=True)
    model, mean, std = train_vae(X, a.nz, a.steps, a.batch, a.sigma, device, verbose=verbose)
    torch.save(dict(state=model.state_dict(), nz=a.nz, mean=mean, std=std,
                    n_train=len(X), steps=a.steps, sigma=a.sigma), path)
    return model, mean, std


# ── inversion in latent space ──────────────────────────────────────────────
def invert_latent(fwd, model, mean, std, S, dobs, lams, restarts=3, chi2_target=1.05,
                  stall=0.95, max_iter=10, seed=0, verbose=True, z_starts=None):
    import torch
    from torch.func import jacfwd
    dec = model.double()
    for prm in dec.parameters():                             # decoder is fixed: no autograd graph
        prm.requires_grad_(False)
    St = torch.from_numpy(S)
    dlog = torch.from_numpy(np.log(dobs))
    lo, hi = np.log(sp.RHO_LO), np.log(sp.RHO_HI)
    N, nz = len(dobs), model.nz
    calls = [0, 0]                                           # fwd+J, fwd

    def cells(z):                                            # pre-bounds log-rho per cell
        return St @ (dec.decode(z[None])[0, 0].reshape(-1) * std + mean) * LN10

    def chi2_of(z):
        calls[1] += 1
        with torch.no_grad():
            m = sp.soft_bounds(cells(z), lo, hi).numpy()
        return float(np.mean(((np.log(fwd.response(m)) - dlog.numpy()) / sp.NOISE) ** 2))

    def run(z, lam_list, hist):
        stages = []
        for lam in lam_list:
            chi2 = None
            for _ in range(max_iter):
                with torch.no_grad():
                    x = cells(z)
                Jd = jacfwd(cells)(z)                        # (n_cells, nz)
                J, dcalc = fwd.jacobian_loglog(sp.soft_bounds(x, lo, hi).numpy())
                calls[0] += 1
                r = (torch.from_numpy(np.log(dcalc)) - dlog) / sp.NOISE
                chi2 = float((r ** 2).mean())
                Jz = (torch.from_numpy(J) * sp.soft_bounds_deriv(x, lo, hi)) @ Jd / sp.NOISE
                phi0 = chi2 + lam * float((z ** 2).mean())
                A = Jz.T @ Jz / N + lam / nz * torch.eye(nz, dtype=Jz.dtype)
                b = Jz.T @ r / N + lam / nz * z
                dz = -torch.linalg.solve(A, b)
                for step in (1.0, 0.5, 0.25, 0.125):
                    zn = z + step * dz
                    chi2n = chi2_of(zn)
                    phin = chi2n + lam * float((zn ** 2).mean())
                    if phin < phi0:
                        break
                else:
                    break
                z, chi2 = zn, chi2n
                hist.append((lam, chi2, calls[0] + calls[1]))
                if (phi0 - phin) / phi0 < 0.01:
                    break
            with torch.no_grad():
                m = sp.soft_bounds(cells(z), lo, hi).numpy()
            stages.append(dict(lam=lam, chi2=chi2, z=z.clone(), m=m))
            if verbose:
                print(f"    latent GN lambda={lam:<6g} chi2={chi2:7.3f}  "
                      f"calls {calls[0]} fwd+J, {calls[1]} fwd", flush=True)
            keep = sp._stage_done(stages, chi2_target, stall, False)
            if keep is not None:
                return keep, stages
        return stages[-1], stages

    # restarts: the first lambda stage from z = 0 and from random latent points
    # (z_starts: extra given starting points, e.g. the encoded pyGIMLi model)
    rng = np.random.default_rng(seed)
    starts = [torch.zeros(nz, dtype=torch.float64)] + [
        torch.as_tensor(np.asarray(z, dtype=float)) for z in (z_starts or [])]
    while len(starts) < restarts:
        starts.append(torch.from_numpy(rng.normal(0, 1, nz)))
    hist, first = [], []
    for z0 in starts:
        st, _ = run(z0, lams[:1], hist)
        first.append(st)
    z_best = min(first, key=lambda s: s["chi2"])["z"]
    if verbose and restarts > 1:
        print("    restart chi2 after first stage: "
              + ", ".join(f"{s['chi2']:.2f}" for s in first), flush=True)
    best, path = run(z_best, lams, hist)
    best["calls"] = tuple(calls)
    best["hist"] = hist
    best["path"] = [dict(lam=s["lam"], chi2=s["chi2"], m=s["m"]) for s in path]   # every lambda stage
    return best


# ── reference methods ──────────────────────────────────────────────────────
def run_sparse(fwd, cc, dobs, cov):
    """sparse_torch_inversion_ert defaults: TPS aniso 2, L1, zweight 0.3, 1 refinement."""
    lams = [100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]
    ctrl = sp.place_controls(cc)
    W, D = sp.weights_tps(cc, ctrl, 2.0), sp.regularization_operator(cc[ctrl], 0.3)
    p0 = np.full(len(ctrl), np.log(np.median(dobs)))
    best, _, _ = sp.invert_gn(fwd, W, D, dobs, p0, lams, norm="l1", verbose=False)
    x = W @ best["p"]
    ctrl, _ = sp.refine_controls(cc, ctrl, x, cov / cov.max(), 30)
    W, D = sp.weights_tps(cc, ctrl, 2.0), sp.regularization_operator(cc[ctrl], 0.3)
    best, _, _ = sp.invert_gn(fwd, W, D, dobs, x[ctrl], [l for l in lams if l <= best["lam"]],
                              norm="l1", verbose=False)
    best["n_par"] = len(ctrl)
    return best


def run_pygimli(name, data, mesh):
    fn = os.path.join(OUT, f"pygimli_{name}.npz")
    if os.path.exists(fn):
        return np.load(fn, allow_pickle=True)["pgr"].item()
    t = time.time()
    pgr, _ = sp.invert_pygimli(data, mesh, [300, 100, 50, 20, 10, 5, 2], 1.05, 0.3, True,
                               verbose=False)
    pgr["time"] = time.time() - t
    np.savez(fn, pgr=np.array(pgr, dtype=object))
    return pgr


# ── plotting ───────────────────────────────────────────────────────────────
def _frame(ax, case):
    """Electrodes, true boundaries (black + white dashes), limits."""
    xs = np.linspace(*sp.XLIM, 400)
    zs = np.linspace(0, sp.PARA_DEPTH, 200)
    Xf, Zf = np.meshgrid(xs, zs)
    T = case["fn"](np.column_stack([Xf.ravel(), -Zf.ravel()])).reshape(Xf.shape)
    u = np.unique(np.round(T, 6))
    if len(u) > 1:
        lev = 0.5 * (u[:-1] + u[1:])
        ax.contour(Xf, -Zf, T, levels=lev, colors="black", linewidths=1.8)
        ax.contour(Xf, -Zf, T, levels=lev, colors="white", linewidths=0.9, linestyles="dashed")
    ax.plot(sp.ELEC_X, np.zeros_like(sp.ELEC_X), "v", color=C_TRUE, ms=2.5)
    ax.set_xlim(*sp.XLIM); ax.set_ylim(*sp.YLIM); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=INK2); ax.set_ylabel("depth (m)", color=INK2)
    ax.tick_params(colors=INK2, labelsize=7.5)
    for s_ in ax.spines.values():
        s_.set_color(GRID_C)


def plot_case(name, case, R, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.2})
    cc, tri = R["cc"], R["tri"]
    true = case["fn"](cc)
    norm = LogNorm(np.exp(true.min()) / 1.5, np.exp(true.max()) * 1.5)
    rnorm = Normalize(-1, 1)
    rows = [("pyGIMLi (zW 0.3, blocky)", R["pg"], R["pg"]["cc"], R["pg"]["tri"]),
            ("Sparse biharmonic (PyTorch GN)", R["sp"], cc, tri),
            (f"VAE prior '{R['family']}' (PyTorch GN, {R['vae']['n_par']} latent)", R["vae"], cc, tri)]
    fig, axs = plt.subplots(4, 2, figsize=(13, 11.5), constrained_layout=True)
    fig.suptitle(case["title"], fontsize=11)
    pc = sp._cellplot(axs[0, 0], tri, np.exp(true), norm=norm, cmap=CMAP)
    axs[0, 0].set_title("True model")
    sp._cellplot(axs[0, 1], tri, np.exp(R["proj"]), norm=norm, cmap=CMAP)
    axs[0, 1].set_title(f"True model as the VAE can represent it (encode -> decode)   "
                        f"RMS = {R['proj_rms']:.3f}")
    for i, (lab, r, rcc, rtri) in enumerate(rows, start=1):
        sp._cellplot(axs[i, 0], rtri, np.exp(r["m"]), norm=norm, cmap=CMAP)
        axs[i, 0].set_title(f"{lab}   lambda={r['lam']:g}, chi2={r['chi2']:.2f}, "
                            f"{r['n_par']} parameters")
        pr = sp._cellplot(axs[i, 1], rtri, (r["m"] - case["fn"](rcc)) / LN10, norm=rnorm, cmap=CMAP)
        axs[i, 1].set_title(f"log10(recovered / true)   RMS in covered zone = {r['metrics']['rms']:.3f}")
    fig.colorbar(pc, ax=axs[:, 0].tolist(), shrink=0.5, label="resistivity (Ohm.m)")
    fig.colorbar(pr, ax=axs[1:, 1].tolist(), shrink=0.6, label="log10(rho_rec / rho_true)")
    for ax in axs.ravel():
        _frame(ax, case)
    sp._save(fig, fn + "_models.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    z = np.linspace(0.05, 16, 320)
    for ax, x in zip(axs, case["xs"]):
        pts = np.column_stack([np.full_like(z, x), -z])
        ax.plot(np.exp(case["fn"](pts)), z, color=C_TRUE, lw=2, ls="--", label="true")
        for lab, r, rcc, col in [("pyGIMLi", R["pg"], R["pg"]["cc"], C_PG),
                                 ("sparse biharmonic", R["sp"], cc, C_SP),
                                 ("VAE prior", R["vae"], cc, C_VAE)]:
            ax.plot(np.exp(sp.vertical_profile(rcc, r["m"], x, z)), z, color=col, lw=2, label=lab)
        ax.set_xscale("log"); ax.set_xlim(norm.vmin, norm.vmax); ax.set_ylim(16, 0)
        ax.set_xlabel("resistivity (Ohm.m)"); ax.set_ylabel("depth (m)")
        ax.set_title(f"Vertical profile at x = {x:g} m")
        ax.grid(color=GRID_C, lw=0.6, which="both"); ax.legend(fontsize=7.5, frameon=False)
    sp._save(fig, fn + "_profiles.png")
    plt.close(fig)


def plot_prior(model, mean, std, X, xg, zg, fn, n=4, seed=1):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    with torch.no_grad():
        x = torch.from_numpy((X[:n, None] - mean) / std).float()
        rec = model.float().decode(model.encode(x)[0]).numpy()[:, 0] * std + mean
        g = torch.Generator().manual_seed(seed)
        smp = model.decode(torch.randn(n, model.nz, generator=g)).numpy()[:, 0] * std + mean
    fig, axs = plt.subplots(3, n, figsize=(3.4 * n, 4.6), constrained_layout=True)
    ext = (xg[0], xg[-1], -zg[-1], -zg[0])
    for j in range(n):
        for i, (img, lab) in enumerate([(X[j], "training model"), (rec[j], "VAE reconstruction"),
                                        (smp[j], "random draw z ~ N(0, I)")]):
            im = axs[i, j].imshow(10 ** img, extent=ext, aspect="equal", cmap=CMAP,
                                  norm=LogNorm(3, 3000), interpolation="nearest")
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(lab, fontsize=8)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7, label="resistivity (Ohm.m)")
    sp._save(fig, fn)
    plt.close(fig)


def plot_learning_curve(LC, pg_rms, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = np.array(LC["n"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axs[0].plot(ns, LC["val"], "o-", color=C_VAE, lw=2)
    axs[0].set_xscale("log"); axs[0].set_xlabel("number of training models")
    axs[0].set_ylabel("RMS log10 error on 2000 held-out models")
    axs[0].set_title("(a) How well the prior represents unseen models of its family")
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7c5cd6", "#5f5e5a"]
    cols = {name: pal[i % len(pal)] for i, name in enumerate(LC["inv"])}
    for name, v in LC["inv"].items():
        axs[1].plot(ns, v, "o-", color=cols[name], lw=2, label=f"VAE inversion: {name}")
        axs[1].axhline(pg_rms[name], color=cols[name], lw=1, ls="--", label=f"pyGIMLi: {name}")
    axs[1].set_xscale("log"); axs[1].set_xlabel("number of training models")
    axs[1].set_ylabel("RMS log10(rec / true) in the covered zone")
    axs[1].set_title("(b) Inversion error on the test models (lower is better)")
    axs[1].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axs:
        ax.grid(color=GRID_C, lw=0.6, which="both")
    sp._save(fig, fn)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--family", choices=list(FAMILIES), default="layered",
                    help="training models: layered (narrow), general (broad) or pg (built "
                         "around a pyGIMLi inversion of each data set)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--nz", type=int, default=None, help="latent dimension")
    ap.add_argument("--steps", type=int, default=None, help="VAE training steps")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=0.1, help="likelihood std (log10 units)")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lams", type=str, default="10,3,1,0.3,0.1,0.03,0.01")
    ap.add_argument("--cases", type=str, default="res500,con10,ood,fault,trunc,grad")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--ns", type=str, default=None)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--pg-mix", type=float, default=0.15,
                    help="pg family: fraction of fully general models mixed in")
    ap.add_argument("--pg-extend", type=float, default=1.0,
                    help="pg family: scale of the deeper continuation of anomalies")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for k, v in FAMILY_DEFAULTS[a.family].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.quick:
        a.n_train, a.n_val, a.steps, a.restarts = 2000, 200, 600, 1
        a.cases, a.ns, a.lams = "res500", "500,2000", "1,0.1"
    import torch
    sp._threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    OUTF = OUT if a.family == "layered" else os.path.join(OUT, a.family)
    os.makedirs(OUTF, exist_ok=True)
    tag = "quick_" if a.quick else ""
    lams = [float(v) for v in a.lams.split(",")]
    names = a.cases.split(",")
    t0 = time.time()

    # data + mesh for every case
    print("Simulating test data ...", flush=True)
    data = {n: simulate(CASES[n]["fn"]) for n in names}
    mesh, fwd0, cc, area = sp.build_inversion_mesh(data[names[0]])
    from ert_common import ERTForward
    fwds = {n: ERTForward(data[n], mesh) for n in names}
    tri = sp._triangles(fwd0.fop.paraDomain)
    xg, zg = make_grid(cc)
    S = bilinear_matrix(cc, xg, zg)
    J0, _ = fwd0.jacobian_loglog(np.full(len(cc), np.log(np.median(np.array(data[names[0]]["rhoa"])))))
    cov = np.sqrt(np.sum((J0 / sp.NOISE) ** 2, axis=0)) / np.sqrt(area)

    # prior samples (cached); the pg family builds one prior per data set below
    fn_x = os.path.join(OUTF, f"{tag}prior_models_{a.n_train}.npz")
    if a.family == "pg":
        pass
    elif os.path.exists(fn_x):
        z_ = np.load(fn_x); X, Xv = z_["X"], z_["Xv"]
    else:
        t = time.time()
        X = FAMILIES[a.family](a.n_train, xg, zg, seed=1)
        Xv = FAMILIES[a.family](a.n_val, xg, zg, seed=2)
        np.savez(fn_x, X=X, Xv=Xv)
        print(f"  generated {a.n_train} + {a.n_val} prior models in {time.time() - t:.0f}s", flush=True)

    # references (cached per case)
    refs = {}
    for n in names:
        dobs = np.array(data[n]["rhoa"])
        print(f"[{n}] pyGIMLi + sparse references ...", flush=True)
        pgr = run_pygimli(tag + n, data[n], mesh)
        pgr["n_par"] = len(pgr["m"])
        pgr["metrics"] = metrics(pgr["cc"], pgr["m"], CASES[n])
        fn_sp = os.path.join(OUT, f"{tag}sparse_{n}.npz")
        if os.path.exists(fn_sp):
            spr = np.load(fn_sp, allow_pickle=True)["spr"].item()
        else:
            t = time.time()
            spr = run_sparse(fwds[n], cc, dobs, cov)
            spr["time"] = time.time() - t
            np.savez(fn_sp, spr=np.array({k: spr[k] for k in ("m", "lam", "chi2", "n_par", "time")},
                                         dtype=object))
        spr["metrics"] = metrics(cc, spr["m"], CASES[n])
        refs[n] = dict(pg=pgr, sp=spr)

    def invert_case(n, model, mean, std, restarts, verbose):
        t = time.time()
        r = invert_latent(fwds[n], model, mean, std, S, np.array(data[n]["rhoa"]), lams,
                          restarts=restarts, verbose=verbose)
        r["time"] = time.time() - t
        r["n_par"] = model.nz
        r["metrics"] = metrics(cc, r["m"], CASES[n])
        return r

    # learning curve
    if a.learning_curve and a.family == "pg":
        print("  (learning curve not available for the pg family: one prior per data set)")
    elif a.learning_curve:
        LC = dict(n=[], val=[], inv={n: [] for n in names})
        for N in [int(v) for v in a.ns.split(",")]:
            print(f"Learning curve: N = {N}", flush=True)
            model, mean, std = load_or_train(os.path.join(OUTF, f"{tag}vae_N{N}.pt"), X[:N], a,
                                             device, verbose=False)
            LC["n"].append(N)
            LC["val"].append(recon_rms(model, Xv, mean, std))
            for n in names:
                r = invert_case(n, model, mean, std, 1, False)
                LC["inv"][n].append(r["metrics"]["rms"])
            print(f"  val RMS {LC['val'][-1]:.3f}   inversion RMS "
                  + "  ".join(f"{n}={LC['inv'][n][-1]:.3f}" for n in names), flush=True)
        plot_learning_curve(LC, {n: refs[n]["pg"]["metrics"]["rms"] for n in names},
                            os.path.join(OUTF, f"{tag}learning_curve.png"))
        np.savez(os.path.join(OUTF, f"{tag}learning_curve.npz"), **{k: np.array(v, dtype=object)
                                                                  for k, v in LC.items()})

    # main model(s)
    if a.family != "pg":
        print(f"VAE prior with N = {a.n_train} training models", flush=True)
        model, mean, std = load_or_train(os.path.join(OUTF, f"{tag}vae_N{a.n_train}.pt"), X, a, device)
        vrms = recon_rms(model, Xv, mean, std)
        print(f"  held-out reconstruction RMS (log10): {vrms:.3f}", flush=True)
        plot_prior(model, mean, std, Xv, xg, zg, os.path.join(OUTF, f"{tag}prior_samples.png"))
    covg = grid_from_cells(cc, cov / cov.max(), xg, zg)

    def get_prior(n):
        """Shared VAE, or (pg family) a VAE trained around this data set's
        pyGIMLi model."""
        if a.family != "pg":
            return model, mean, std, vrms
        pgr = refs[n]["pg"]
        img = grid_from_cells(pgr["cc"], pgr["m"] / LN10, xg, zg)
        info = analyse_pg(img, covg, xg, zg)
        print(f"[{n}] pyGIMLi structure: interfaces at "
              + ", ".join(f"{d:.1f}" for d in info["z_if"]) + " m; "
              + f"{len(info['bodies'])} anomalies "
              + ", ".join(f"({'+' if b['sign'] > 0 else '-'} x={b['xc']:.0f} z={b['zc']:.1f} m)"
                          for b in info["bodies"]), flush=True)
        fn_xc = os.path.join(OUTF, f"{tag}prior_{n}_{a.n_train}.npz")
        if os.path.exists(fn_xc) and not a.retrain:
            z_ = np.load(fn_xc); Xc, Xvc = z_["X"], z_["Xv"]
        else:
            Xc = random_models_pg(a.n_train, xg, zg, info, 1, a.pg_mix, a.pg_extend)
            Xvc = random_models_pg(a.n_val, xg, zg, info, 2, a.pg_mix, a.pg_extend)
            np.savez(fn_xc, X=Xc, Xv=Xvc)
        plot_pg_prior(info, img, Xvc[:10], xg, zg, os.path.join(OUTF, f"{tag}{n}_prior.png"), a.pg_extend)
        mdl, mn, sd = load_or_train(os.path.join(OUTF, f"{tag}vae_{n}_N{a.n_train}.pt"), Xc, a, device)
        vr_ = recon_rms(mdl, Xvc, mn, sd)
        print(f"  [{n}] held-out reconstruction RMS (log10): {vr_:.3f}", flush=True)
        return mdl, mn, sd, vr_

    lines = [f"VAE PRIOR ({a.family}) vs SPARSE vs PYGIMLI  (VAE: {a.n_train} training models"
             + (" per data set" if a.family == "pg" else "") + f", latent {a.nz})",
             f"{'case':8s}{'method':34s}{'params':>7s}{'lambda':>8s}{'chi2':>7s}{'RMS log10':>11s}"
             f"{'anomaly':>9s}{'true':>7s}{'calls':>9s}{'time':>7s}"]
    for n in names:
        case = CASES[n]
        model, mean, std, vrms = get_prior(n)
        print(f"[{n}] VAE latent inversion ...", flush=True)
        vr = invert_case(n, model, mean, std, a.restarts, True)
        # what the prior can represent: encode the true model image, decode, sample to cells
        Xg, Zg = np.meshgrid(xg, zg)
        timg = case["fn"](np.column_stack([Xg.ravel(), -Zg.ravel()])).reshape(Xg.shape) / LN10
        with torch.no_grad():
            mf = model.float()
            mu, _ = mf.encode(torch.from_numpy((timg[None, None] - mean) / std).float())
            pimg = mf.decode(mu).numpy()[0, 0] * std + mean
        proj = S @ pimg.ravel() * LN10
        proj_rms = metrics(cc, proj, case)["rms"]
        R = dict(family=a.family, cc=cc, tri=tri, pg=refs[n]["pg"], sp=refs[n]["sp"], vae=vr, proj=proj,
                 proj_rms=proj_rms)
        plot_case(n, case, R, os.path.join(OUTF, f"{tag}{n}"))
        for lab, r, calls in [("pyGIMLi (zW 0.3, blocky)", R["pg"], "-"),
                              ("sparse biharmonic", R["sp"], "-"),
                              ("VAE prior", vr, f"{vr['calls'][0]}+{vr['calls'][1]}")]:
            mm = r["metrics"]
            lines.append(f"{n:8s}{lab:34s}{r['n_par']:7d}{r['lam']:8g}{r['chi2']:7.2f}"
                         f"{mm['rms']:11.3f}{mm['anomaly']:9.0f}{mm['anomaly_true']:7.0f}"
                         f"{calls:>9s}{r.get('time', np.nan):6.0f}s")
        lines.append(f"{n:8s}{'(VAE representation of the truth)':34s}{'':7s}{'':8s}{'':7s}"
                     f"{proj_rms:11.3f}   (held-out reconstruction RMS {vrms:.3f})")
        np.savez(os.path.join(OUTF, f"{tag}{n}_results.npz"), m_vae=vr["m"], z=vr["z"].numpy(),
                 m_sp=R["sp"]["m"], m_pg=R["pg"]["m"], proj=proj, cc=cc)
    txt = "\n".join(lines + ["anomaly = median resistivity inside the true target / dyke (Ohm.m); "
                             "calls = pyGIMLi forward+Jacobian + forward-only"])
    open(os.path.join(OUTF, f"{tag}summary.txt"), "w").write(txt)
    print("\n" + txt + f"\n\ntotal {time.time() - t0:.0f}s; outputs in {OUTF}")


if __name__ == "__main__":
    main()
