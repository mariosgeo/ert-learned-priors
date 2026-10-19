"""Figure 13: how many training models does the prior need? Learning curves of the general VAE (64 latent,
10 000-200 000 random models) and of a narrower layered-family VAE (32 latent, 1 000-30 000 models), from
outputs/vae_prior/general/learning_curve.npz and outputs/vae_prior/learning_curve.npz (six development models,
not part of the benchmark; one inversion per network and model)."""
import os as _os_rel
_ROOT = _os_rel.path.abspath(_os_rel.path.join(_os_rel.path.dirname(_os_rel.path.abspath(__file__)), "..", ".."))
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(_ROOT, "outputs")
P = os.path.join(ROOT, "paper")
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
LAB = {"res500": "3 layers + resistive target", "con10": "3 layers + conductive target",
       "ood": "sinusoidal interface + dyke", "fault": "normal fault", "trunc": "truncated layer",
       "grad": "weathering gradient + lens"}
PG = {"res500": 0.158, "con10": 0.154, "ood": 0.185, "fault": 0.112, "trunc": 0.127, "grad": 0.145}   # pyGIMLi


def load(fn):
    z = np.load(fn, allow_pickle=True)
    return np.array(z["n"], float), np.array(z["val"], float), {k: np.array(v, float) for k, v in z["inv"].item().items()}


ng, vg, ig = load(os.path.join(ROOT, "vae_prior", "general", "learning_curve.npz"))
nl, vl, il = load(os.path.join(ROOT, "vae_prior", "learning_curve.npz"))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
ax = axs[0]
ax.semilogx(ng, vg, "o-", color="C0", label="general prior (64 latent)")
ax.semilogx(nl, vl, "s-", color="C1", label="layered family (32 latent)")
ax.set_xlabel("number of training models")
ax.set_ylabel("held-out reconstruction error (log10)")
ax.set_title("(a) how well the network represents unseen models", loc="left", fontsize=8)
ax.legend(frameon=False)
for ax, n, inv, title in [(axs[1], ng, ig, "(b) general prior: inversion error"),
                          (axs[2], nl, il, "(c) layered family: inversion error")]:
    for i, (k, v) in enumerate(inv.items()):
        ax.semilogx(n, v, "o-", color=f"C{i}", label=LAB[k], ms=4)
        ax.axhline(PG[k], color=f"C{i}", ls=":", lw=0.8)
    med = np.median(np.array(list(inv.values())), axis=0)
    ax.semilogx(n, med, "k-", lw=2.2, label="median")
    ax.axhline(np.median([PG[k] for k in inv]), color="k", ls=":", lw=1.2)
    ax.set_xlabel("number of training models")
    ax.set_ylabel("model error (RMS log10)")
    ax.set_title(title + " (dotted: pyGIMLi)", loc="left", fontsize=8)
axs[1].legend(frameon=False, fontsize=6.5, loc="upper right")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(P, f"fig13_learning_curve.{ext}"), dpi=170)
plt.close(fig)
print("general: median inversion", np.round(np.median(np.array(list(ig.values())), axis=0), 3),
      "| layered: median", np.round(np.median(np.array(list(il.values())), axis=0), 3))
