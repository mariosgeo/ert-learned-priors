# Data-adaptive learned priors for ERT inversion

Code, data, trained networks and results of

> Karaoulis, M. & Tsourlos, P. (submitted). Data-adaptive learned priors for electrical resistivity tomography:
> a 100-model benchmark against conventional regularized inversion. *Geophysical Journal International*.

The study compares conventional regularized inversion (pyGIMLi) with latent-space inversion in a variational
autoencoder (VAE): a random prior, a data-adaptive (guided) prior built from the pyGIMLi image of each data set, and a
hybrid inversion that runs pyGIMLi with the guided-VAE model as starting and reference model. It uses a benchmark of
100 synthetic two-dimensional models (with 0, +-10, +-30 and +-50 % variation inside layers and bodies) and two field
lines.

## Contents

| Path | Content |
|---|---|
| `*.py` | inversion code (see the docstring at the top of each file) |
| `outputs/paper/*.py` | scripts that reproduce the figures and statistics of the paper |
| `real_data/` | the two field lines (`g4_02.pyg`, `n2dd.pyg`, pyGIMLi format) and the topography of n2dd |
| `outputs/vae_benchmark*/data/` | the synthetic data sets of the 100 benchmark models, per suite |
| `outputs/vae_benchmark*/models/`, `outputs/hetero_tuning*/models/` | per-model results of every method (models, errors, data fit) |
| `outputs/vae_prior*/`, `outputs/hetero_tuning/*.pt`, ... | trained networks (general VAE 64 / 128 latent, plain and textured; GAN; body VAE; learning-curve networks) |
| `outputs/real_data/` | field-data results (3 % and estimated errors, five-seed repeats) and the misfit table |
| `outputs/timing*/` | computing time and memory |

Suites: `vae_benchmark` (constant layers and bodies), `vae_benchmark_het10/30/50` (+-10/30/50 % variation),
`*_tex` (prior retrained on textured models), `hetero_tuning*` (128-dimensional priors and their hybrids;
`hetero_tuning` = +-50 %).

## Installation

    conda create -n ertvae python=3.11
    conda activate ertvae
    pip install -r requirements.txt

A CUDA GPU is used for training and fine-tuning if available (an 8 GB card is sufficient); everything also runs on the CPU.

## Reproducing the results

    python vae_benchmark_ert.py --workers 6              # benchmark, constant layers (results are resumable)
    ERT_HETERO=30 python vae_benchmark_ert.py            # heterogeneous suites (10, 30, 50)
    python hetero_tuning.py --level 50 --all --variants plain128 --hybrid --stage run   # 128-dimensional priors
    python real_data_inversion.py --error auto           # field data (estimated errors)
    python real_data_inversion.py --error auto --seed 2 --methods pygimli,vae_pg_dip_strict,hybrid_vae_dip_strict
    python timing_benchmark.py; python timing_realdata.py
    python outputs/paper/paper_stats.py; python outputs/paper/figs.py   # statistics and figures

Existing results are reused, so the figure scripts run immediately on the included results. The 200 000 training
images of the general VAE are not included; they are regenerated exactly from seed 1 by
`vae_prior_inversion_ert.py`. Fine-tuned networks of the guided prior are recreated with the seeds stored per model.

## Licence

Code: MIT (`LICENSE`). Data, trained networks and results: CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the paper and this archive (DOI on Zenodo).
