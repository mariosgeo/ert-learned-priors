"""
ttl_inversion.py
================
Test-time-learning (TTL) baselines for the benchmark (methods ttl_cnn, ttl_field),
after Xu & Heagy (2024, CNN reparameterisation; 2025, neural fields):
the model is the output of a neural network that is NOT trained on any
synthetic models; its weights are fitted to the one data set being inverted.
There is no explicit regularisation term - the implicit bias of the network
plus early stopping at the noise level (chi2 <= 1.05) regularise the result.

  ttl_cnn    the VAE decoder ARCHITECTURE (vae_prior_inversion_ert.make_vae),
             randomly initialised, fed a fixed random input vector; image
             (log10 rho on the 32 x 96 grid) sampled on the cells. Same network
             as the learned-prior VAE, but without the learned weights - so the
             comparison isolates what the training set contributes.
  ttl_field  coordinate MLP m(x, z) with random Fourier features (anisotropic,
             longer horizontally), evaluated at the cell centres.

Optimisation: outer Gauss-Newton-like linearisation (one pyGIMLi forward +
Jacobian per outer step), inner Adam steps on the network weights against the
linearised misfit with a trust-region penalty; a step is accepted only if the
true chi2 decreases. Stops at chi2 <= 1.05, on stall (< 1 % in 3 outer steps)
or after max_outer linearisations.
"""
import copy
import numpy as np

LN10 = np.log(10.0)


def _cnn(seed, m_ref10):
    import torch
    import vae_prior_inversion_ert as v
    torch.manual_seed(seed)
    net = v.make_vae(64)
    del net.enc                                              # decoder only
    z_in = torch.randn(1, 64)

    def img():
        return m_ref10 + 0.5 * net.decode(z_in)[0, 0]        # log10 rho image
    return list(net.fc.parameters()) + list(net.dec.parameters()), img


def _field(seed, cc, m_ref10, n_freq=64, width=128, kx=1 / 32.0, kz=1 / 8.0):
    """Fourier-feature frequency scales kx, kz (cycles/m) chosen from (1/8, 1/2), (1/16, 1/4),
    (1/32, 1/8) on two benchmark models (D2_02, E_07): lowest frequencies were best."""
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    B = torch.randn(2, n_freq, generator=g) * torch.tensor([[kx], [kz]])   # cycles / m (x, z)
    X = torch.tensor(cc, dtype=torch.float32)
    feat = torch.cat([torch.sin(2 * np.pi * X @ B), torch.cos(2 * np.pi * X @ B)], 1)
    net = nn.Sequential(nn.Linear(2 * n_freq, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                        nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def cells10():
        return m_ref10 + 0.5 * net(feat)[:, 0]               # log10 rho per cell
    return list(net.parameters()), cells10


def invert_ttl(fwd, dobs, cc, S, kind="cnn", seed=0, noise=0.03, chi2_target=1.05,
               max_outer=40, inner=200, lr=1e-3, verbose=False):
    import torch
    import sparse_torch_inversion_ert as sp
    dlog = np.log(dobs)
    m_ref10 = float(np.log10(np.median(dobs)))
    lo, hi = np.log(sp.RHO_LO), np.log(sp.RHO_HI)
    if kind == "cnn":
        params, img = _cnn(seed, m_ref10)
        St = torch.from_numpy(S).float()

        def cells():                                          # log rho per cell
            return sp.soft_bounds((St @ img().reshape(-1)) * LN10, lo, hi)
    else:
        params, f = _field(seed, cc, m_ref10)

        def cells():
            return sp.soft_bounds(f() * LN10, lo, hi)
    opt = torch.optim.Adam(params, lr=lr)
    calls, mu, hist = 0, 0.1, []

    def chi2_of(m):
        nonlocal calls
        calls += 1
        return float(np.mean(((np.log(fwd.response(m)) - dlog) / noise) ** 2))

    with torch.no_grad():
        m0 = cells().double().numpy()
    best = dict(m=m0, chi2=chi2_of(m0))
    for outer in range(max_outer):
        J, d0 = fwd.jacobian_loglog(best["m"])
        calls += 1
        hist.append(best["chi2"])
        if verbose:
            print(f"    TTL-{kind} step {outer:2d}  chi2 {best['chi2']:8.3f}  mu {mu:.3g}", flush=True)
        if best["chi2"] <= chi2_target:
            break
        if len(hist) > 3 and best["chi2"] > 0.99 * hist[-4]:
            break
        Jt = torch.from_numpy(J).float()
        r0 = torch.from_numpy((np.log(d0) - dlog) / noise).float()
        m_lin = torch.from_numpy(best["m"]).float()
        saved = (copy.deepcopy([p.detach().clone() for p in params]), copy.deepcopy(opt.state_dict()))
        for k in range(inner):
            m = cells()
            rl = r0 + Jt @ (m - m_lin) / noise
            chi_lin = (rl ** 2).mean()
            loss = chi_lin + mu * ((m - m_lin) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if chi_lin.item() <= 0.95 * chi2_target:           # do not fit below the noise
                break
        with torch.no_grad():
            m_new = cells().double().numpy()
        c_new = chi2_of(m_new)
        if c_new < best["chi2"]:
            best = dict(m=m_new, chi2=c_new)
            mu = max(mu / 2, 1e-3)
        else:                                                  # reject: restore weights, damp more
            with torch.no_grad():
                for p, q in zip(params, saved[0]):
                    p.copy_(q)
            opt.load_state_dict(saved[1])
            mu *= 5
    best.update(calls=calls, hist=hist)
    return best
