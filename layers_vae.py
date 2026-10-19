"""
layers_vae.py
=============
"Sharp layers + body VAE" inversion (benchmark method layers_vae).

Model (log resistivity):
    m(theta, z) = L(theta) + ln(10) * S B(z)
  L(theta)  explicit dipping layers (layered_param.py): interface depths, dips and
            layer resistivities; SHARP boundaries for the data misfit, a smoothed
            version (sigmoid, 0.4 m) only for the search directions
  B(z)      decoder of a VAE trained only on BODIES on a zero background (lenses,
            dykes, channels, blobs; log10 deviations from the host), i.e. what the
            layering does not explain; S samples the image on the mesh cells
Both are inverted together (Gauss-Newton / Levenberg-Marquardt) on the larger
inversion domain, with a weak pull of theta towards the explicit-layer result
and the prior lambda * mean(z^2) on the body latent vector (lambda cooled until
chi2 <= 1.05 or chi2 stalls). Starts: z = 0 and the encoded difference
pyGIMLi model - layered model.

    python layers_vae.py --train     # train the body VAE (GPU), outputs/layers_vae/body_vae.pt
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTL = os.path.join(HERE, "outputs", "layers_vae")
BODY_CKPT = os.path.join(OUTL, "body_vae.pt")
LN10 = np.log(10.0)


# ── body-only training images ──────────────────────────────────────────────
def body_models(n, xg, zg, seed):
    """log10 deviation images: 0-5 bodies (rotated boxes / dykes, ellipses,
    channels, blobs) of -2..+2 log10 on a zero background; 15 % empty."""
    import vae_prior_inversion_ert as v
    rng = np.random.default_rng(seed)
    X, Z = np.meshgrid(xg, zg)
    out = np.zeros((n, len(zg), len(xg)), dtype=np.float32)
    for k in range(n):
        img = np.zeros(X.shape)
        nb = 0 if rng.random() < 0.15 else rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.3, 0.2, 0.1, 0.05])
        for _ in range(nb):
            xc, zc = rng.uniform(0, 47), rng.uniform(0.5, 16)
            w, h = rng.uniform(1, 25), rng.uniform(0.5, 8)
            th = np.deg2rad(rng.uniform(-90, 90)) if rng.random() < 0.6 else 0.0
            u = (X - xc) * np.cos(th) + (Z - zc) * np.sin(th)
            vv = -(X - xc) * np.sin(th) + (Z - zc) * np.cos(th)
            shape = rng.choice(4, p=[0.35, 0.3, 0.15, 0.2])
            if shape == 0:
                m = (np.abs(u) <= w / 2) & (np.abs(vv) <= h / 2)
            elif shape == 1:
                m = (u / (w / 2)) ** 2 + (vv / (h / 2)) ** 2 <= 1
            elif shape == 2:
                v0 = rng.uniform(0.5, 3) * np.sin(2 * np.pi * u / rng.uniform(8, 30) + rng.uniform(0, 6.3))
                m = (np.abs(u) <= w / 2) & (np.abs(vv - v0) <= max(h / 4, 0.4))
            else:
                g = v._grf(rng, X, Z, rng.uniform(1.5, 5))
                m = ((u / (w / 2)) ** 2 + (vv / (h / 2)) ** 2 <= 1.5) & (g > rng.uniform(-0.3, 0.5))
            val = rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)
            img[m] = val
        out[k] = img
    return out


def train(n=100000, steps=20000, nz=32, batch=256):
    import torch
    import vae_benchmark_ert as vb
    os.makedirs(OUTL, exist_ok=True)
    vb.setup(threads=4)
    v, G = vb.G["v"], vb.G
    t = time.time()
    X = body_models(n, G["xg"], G["zg"], seed=11)
    print(f"generated {n} body images in {time.time() - t:.0f}s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with vb._GPULock():
        model, mean, std = v.train_vae(X, nz, steps, batch, 0.05, dev)
    torch.save(dict(state=model.state_dict(), nz=nz, mean=mean, std=std, n_train=n, steps=steps), BODY_CKPT)
    Xv = body_models(500, G["xg"], G["zg"], seed=12)
    print(f"body VAE trained in {(time.time() - t) / 60:.0f} min; held-out reconstruction RMS "
          f"{v.recon_rms(model, Xv, mean, std):.3f} log10 -> {BODY_CKPT}", flush=True)


# ── joint inversion ────────────────────────────────────────────────────────
def _load_vae(path):
    """Load a VAE checkpoint without the benchmark's shared state (works inside
    pool workers, where vae_benchmark_ert runs as __main__ and its G is empty)."""
    import torch
    import vae_prior_inversion_ert as v
    ck = torch.load(path, weights_only=False)
    model = v.make_vae(ck["nz"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck["mean"], ck["std"]


def _sampling(cc, xg, zg):
    """Bilinear sampling of the image at cells cc; zero for cells outside it."""
    import vae_prior_inversion_ert as v
    S = v.bilinear_matrix(cc, xg, zg)
    out = (cc[:, 0] < xg[0]) | (cc[:, 0] > xg[-1]) | (-cc[:, 1] > zg[-1])
    S[out] = 0.0
    return S


def invert_layers_vae(data, layered, body_starts, xg, zg, lams=(10, 3, 1, 0.3, 0.1),
                      noise=0.03, max_iter=10, verbose=False):
    """layered: explicit-layer result (theta, K). body_starts: list of latent
    starting vectors. Returns dict(theta, K, z, chi2, lam)."""
    import torch
    from torch.func import jacfwd
    from pygimli.physics import ert
    import layered_param as lp
    import sparse_torch_inversion_ert as sp
    scheme = ert.createData(elecs=sp.ELEC_X, schemeName="dd")
    mesh, cc = lp.full_mesh(scheme)
    fwd = lp.FullForward(data, mesh)
    dobs = np.array(data["rhoa"])
    dlog = np.log(dobs)
    N = len(dobs)
    St = torch.from_numpy(_sampling(cc, xg, zg))
    model, mean, std = _load_vae(BODY_CKPT)
    dec = model.double()
    for p_ in dec.parameters():
        p_.requires_grad_(False)
    nz, K = model.nz, layered["K"]
    nth = 3 * K + 1
    x = torch.tensor(cc[:, 0], dtype=torch.float64)
    zc = torch.tensor(-cc[:, 1], dtype=torch.float64)
    th0 = torch.tensor(layered["theta"], dtype=torch.float64)
    sig = torch.tensor([3.0] * K + [0.2] * K + [1.0] * (K + 1), dtype=torch.float64)

    def body(z):
        return St @ (dec.decode(z[None])[0, 0].reshape(-1) * std + mean) * LN10

    def m_of(q, w):
        th, z = q[:nth], q[nth:]
        return torch.clamp(lp.layer_logrho(th, K, x, zc, w=w) + body(z), lp.LO, lp.HI)

    def chi2_of(q):
        with torch.no_grad():
            m = m_of(q, 0).numpy()
        return float(np.mean(((np.log(fwd.response(m)) - dlog) / noise) ** 2))

    def phi(q, chi, lam):
        th, z = q[:nth], q[nth:]
        return chi + 1e-2 * float((((th - th0) / sig) ** 2).sum()) + lam * float((z ** 2).mean())

    def run(q, lam_list):
        stages = []
        for lam in lam_list:
            mu, chi = 1.0, None
            for _ in range(max_iter):
                with torch.no_grad():
                    m = m_of(q, 0).numpy()
                J, d = fwd.jacobian_loglog(m)
                r = torch.from_numpy((np.log(d) - dlog) / noise)
                chi = float((r ** 2).mean())
                Jq = torch.from_numpy(J / noise) @ jacfwd(lambda qq: m_of(qq, lp.W_SMOOTH))(q)
                th, z = q[:nth], q[nth:]
                pw = torch.cat([1e-2 / sig ** 2, torch.full((nz,), lam / nz, dtype=torch.float64)])
                pg_ = torch.cat([1e-2 * (th - th0) / sig ** 2, lam / nz * z])
                A = Jq.T @ Jq / N + torch.diag(pw)
                g = Jq.T @ r / N + pg_
                f0 = phi(q, chi, lam)
                ok = False
                for _ in range(6):
                    qn = q - torch.linalg.solve(A + mu * torch.diag(torch.diag(A)), g)
                    qn[K:2 * K] = torch.clamp(qn[K:2 * K], -np.tan(np.radians(40)), np.tan(np.radians(40)))
                    qn[:K] = torch.clamp(qn[:K], 0.2, 29.0)
                    cn = chi2_of(qn)
                    fn = phi(qn, cn, lam)
                    if fn < f0:
                        q, chi, mu, ok = qn, cn, max(mu / 3, 1e-4), True
                        break
                    mu *= 4
                if not ok or (f0 - fn) / f0 < 1e-3:
                    break
            stages.append(dict(lam=lam, chi2=chi, q=q.clone()))
            if verbose:
                print(f"    layers+VAE lambda={lam:<5g} chi2={chi:.3f}", flush=True)
            keep = sp._stage_done(stages, 1.05, 0.95, False)
            if keep is not None:
                return keep, stages
        return stages[-1], stages

    first = []
    for z0 in body_starts:                               # pick the best start after the first stage
        q0 = torch.cat([th0, torch.as_tensor(np.asarray(z0, float))])
        st, _ = run(q0, lams[:1])
        first.append(st)
    best, _ = run(min(first, key=lambda s: s["chi2"])["q"], list(lams))
    q = best["q"].numpy()
    return dict(theta=q[:nth], K=K, z=q[nth:], chi2=best["chi2"], lam=best["lam"])


def evaluate(res, P, S_std, model_ckpt=BODY_CKPT):
    """log rho on the standard cells: sharp layers by area fraction (sample
    points P) + body image sampled with S_std."""
    import torch
    import layered_param as lp
    lay = lp.evaluate(dict(theta=res["theta"], K=res["K"]), P.reshape(-1, 2)).reshape(P.shape[:2]).mean(1)
    model, mean, std = _load_vae(model_ckpt)
    with torch.no_grad():
        img = model.float().decode(torch.from_numpy(np.asarray(res["z"], np.float32))[None]).numpy()[0, 0] * std + mean
    return np.clip(lay + S_std @ img.ravel() * LN10, lp.LO, lp.HI)


if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
