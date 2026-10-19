"""
gan_prior_ert.py
================
GAN prior for ERT inversion: a Wasserstein GAN with gradient penalty (WGAN-GP)
trained on the SAME 200 000 random models (general family) as the random-prior
VAE. Its generator has the VAE decoder's architecture and latent size (64), so
the two priors differ only in how they were trained:

  VAE  reconstruction + KL: smooth, well-organized latent space, softer images
  GAN  adversarial (critic): sharper, more realistic images, rougher latent space

The generator is used exactly like the VAE decoder: the inversion searches its
latent vector z with Gauss-Newton + pyGIMLi (vae_prior_inversion_ert.invert_latent,
prior term lambda * mean(z^2), z ~ N(0, I)). A GAN has no encoder, so an image
(e.g. the pyGIMLi model, used as a starting point) is mapped to z by
optimization ("projection").

Usage
-----
    conda activate pg2
    python gan_prior_ert.py --train          # train (GPU), save outputs/gan_prior/gan_general.pt
    python gan_prior_ert.py --check          # samples, projection error on held-out models
    python vae_benchmark_ert.py              # adds the "gan" method to the 100-model benchmark
"""

import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTG = os.path.join(HERE, "outputs", "gan_prior")
GAN_CKPT = os.path.join(OUTG, "gan_general.pt")
PRIOR = os.path.join(HERE, "outputs", "vae_prior", "general", "prior_models_200000.npz")


def make_gan(nz=64):
    import torch.nn as nn

    class Generator(nn.Module):
        """Same layout as the VAE decoder (+ batch norm, standard for GAN generators)."""
        def __init__(self):
            super().__init__()
            self.nz = nz
            self.fc = nn.Linear(nz, 128 * 4 * 12)

            def up(ci, co):
                return [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        nn.Conv2d(ci, co, 3, 1, 1), nn.BatchNorm2d(co), nn.GELU()]
            self.net = nn.Sequential(nn.BatchNorm2d(128), nn.GELU(), *up(128, 64), *up(64, 32),
                                     *up(32, 16), nn.Conv2d(16, 1, 3, 1, 1))

        def decode(self, z):                                 # same interface as the VAE
            return self.net(self.fc(z).view(-1, 128, 4, 12))

        def forward(self, z):
            return self.decode(z)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),       # 16 x 48
                nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),     # 8 x 24
                nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),    # 4 x 12
                nn.Flatten(), nn.Linear(256 * 4 * 12, 1))

        def forward(self, x):
            return self.net(x)

    return Generator(), Critic()


def train(steps=20000, n_critic=5, batch=128, gp=10.0, nz=64, seed=0, verbose=True):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z_ = np.load(PRIOR)
    X = z_["X"]
    mean, std = float(X.mean()), float(X.std())             # same normalization as the VAE
    Xt = torch.from_numpy((X[:, None] - mean) / std).float().to(dev)
    del X
    G, D = make_gan(nz)
    G, D = G.to(dev), D.to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    hist, t0 = [], time.time()
    for step in range(steps):
        for _ in range(n_critic):
            real = Xt[torch.randint(0, len(Xt), (batch,), device=dev)]
            fake = G(torch.randn(batch, nz, device=dev)).detach()
            eps = torch.rand(batch, 1, 1, 1, device=dev)
            mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
            grad = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
            pen = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
            lossD = D(fake).mean() - D(real).mean() + gp * pen
            oD.zero_grad()
            lossD.backward()
            oD.step()
        lossG = -D(G(torch.randn(batch, nz, device=dev))).mean()
        oG.zero_grad()
        lossG.backward()
        oG.step()
        if (step + 1) % 100 == 0:
            hist.append((step + 1, -lossD.item() + gp * pen.item(), pen.item()))
        if verbose and (step + 1) % max(1, steps // 20) == 0:
            el = time.time() - t0
            print(f"  step {step + 1:6d}/{steps}  Wasserstein estimate {hist[-1][1]:8.3f}  "
                  f"({el / 60:.1f} min, ~{el / (step + 1) * (steps - step - 1) / 60:.0f} min left)", flush=True)
            torch.save(dict(state=G.state_dict(), nz=nz, mean=mean, std=std, step=step + 1,
                            hist=hist), GAN_CKPT)            # checkpoint (resumable output)
    torch.save(dict(state=G.state_dict(), nz=nz, mean=mean, std=std, step=steps, hist=hist), GAN_CKPT)
    return G


def load_gan(path=GAN_CKPT):
    import torch
    ck = torch.load(path, weights_only=False)
    G, _ = make_gan(ck["nz"])
    G.load_state_dict(ck["state"])
    return G.eval(), ck["mean"], ck["std"]


def project(G, imgs, mean, std, steps=400, starts=4, reg=1e-3, seed=0):
    """Latent vectors whose generated images best match imgs (log10, (n, NZ, NX)
    or a single image), by Adam from several random starts."""
    import torch
    single = imgs.ndim == 2
    imgs = imgs[None] if single else imgs
    G = G.float().eval()
    for p in G.parameters():
        p.requires_grad_(False)
    x = torch.from_numpy((imgs[:, None] - mean) / std).float()
    g = torch.Generator().manual_seed(seed)
    best_z, best_l = None, None
    for s in range(starts):
        z = (torch.randn(len(x), G.nz, generator=g) * (0 if s == 0 else 1)).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(steps):
            loss = ((G.decode(z) - x) ** 2).mean((1, 2, 3)) + reg * (z ** 2).mean(1)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            l_ = ((G.decode(z) - x) ** 2).mean((1, 2, 3))
        if best_z is None:
            best_z, best_l = z.detach().clone(), l_
        else:
            better = l_ < best_l
            best_z[better], best_l[better] = z.detach()[better], l_[better]
    zs = best_z.numpy().astype(float)
    return zs[0] if single else zs


def check(n=300):
    """Random samples next to VAE samples, and projection error on held-out
    models (comparable to the VAE's held-out reconstruction error)."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import vae_prior_inversion_ert as v
    G, mean, std = load_gan()
    Xv = np.load(PRIOR)["Xv"][:n]
    zs = project(G, Xv, mean, std)
    with torch.no_grad():
        rec = G.decode(torch.from_numpy(zs).float()).numpy()[:, 0] * std + mean
        smp = G.decode(torch.randn(8, G.nz, generator=torch.Generator().manual_seed(1))).numpy()[:, 0] * std + mean
    rms = float(np.sqrt(np.mean((rec - Xv) ** 2)))
    vae, vm, vs = None, None, None
    vck = os.path.join(HERE, "outputs", "vae_prior", "general", "vae_N200000.pt")
    if os.path.exists(vck):
        ck = torch.load(vck, weights_only=False)
        vae = v.make_vae(ck["nz"]); vae.load_state_dict(ck["state"]); vae.eval()
        vm, vs = ck["mean"], ck["std"]
        with torch.no_grad():
            vsmp = vae.decode(torch.randn(8, vae.nz, generator=torch.Generator().manual_seed(1))).numpy()[:, 0] * vs + vm
    fig, axs = plt.subplots(4, 8, figsize=(22, 5.6), constrained_layout=True)
    for j in range(8):
        for i, (img, lab) in enumerate([(Xv[j], "held-out model"), (rec[j], "GAN projection"),
                                        (smp[j], "GAN random draw"),
                                        (vsmp[j] if vae is not None else smp[j], "VAE random draw")]):
            im = axs[i, j].imshow(10 ** img, cmap="turbo", norm=LogNorm(1, 1e4), aspect="auto")
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(lab, fontsize=8)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="resistivity (Ohm.m)")
    fig.suptitle(f"GAN prior: projection RMS on {n} held-out models = {rms:.3f} log10 "
                 f"(VAE reconstruction RMS on held-out models: 0.211)")
    fig.savefig(os.path.join(OUTG, "gan_samples.png"), dpi=100)
    plt.close(fig)
    print(f"GAN projection RMS on {n} held-out models: {rms:.3f} log10; figure {OUTG}\\gan_samples.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--steps", type=int, default=20000, help="generator steps (5 critic steps each)")
    a = ap.parse_args()
    os.makedirs(OUTG, exist_ok=True)
    if a.train:
        t = time.time()
        train(steps=a.steps)
        print(f"trained in {(time.time() - t) / 60:.0f} min -> {GAN_CKPT}")
    if a.check or a.train:
        check()


if __name__ == "__main__":
    main()
