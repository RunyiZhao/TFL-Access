"""
run_ablation.py -- Component ablation for the D-LA-JSSA method (Section VI, Q4).

We isolate the contribution of the three method components by toggling them:

  Full          : CVaR distributional critic (M1) + continuous belief state
                  (M2) + uncertainty-aware reward (M3 surrogate).
  -M1 (no CVaR) : mean-constrained Lagrangian BeliefDQN (agents.BeliefDQN).
                  Removes the distributional/CVaR critic; keeps belief state.
  -M2 (coarsened state) : CVaR critic but the belief map is hard-thresholded
                  into a binary occupancy vector before the encoder, mimicking
                  the loss of the rich conv+LRU belief encoding.
  -M3 (no uncertainty penalty) : Full but with lam_u = 0 in the reward.

Metric: net objective = throughput - KAPPA * collision_rate, plus the raw
throughput and collision rate, averaged over seeds. Produces a grouped bar
chart figures/fig_ablation.pdf.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import ContinuousTFEnv
from agents import BeliefDQN
from agents_cvar import CVaRDistDQN, CVaRDistDQN_TwoStep

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

TRAIN_STEPS = 18000
EVAL_STEPS = 4000
KAPPA = 3.0
SEEDS = [0, 1]


def make_env(seed, lam_u=0.3):
    return ContinuousTFEnv(Nf=64, n_pu=3, snr_quality=1.2, eta=0.5,
                           lam_c=2.0, lam_u=lam_u, seed=seed)


def run_variant(name, seed):
    """Return (throughput, collision_rate) for a trained variant."""
    if name == "Full":
        env = make_env(seed, lam_u=0.3)
        agent = CVaRDistDQN(env, seed=seed, cvar_alpha=0.6, cvar_beta=0.2,
                            pcol_max=0.05)
    elif name == "-M1":
        env = make_env(seed, lam_u=0.3)
        agent = BeliefDQN(env, seed=seed, pcol_max=0.05)
    elif name == "-M2":
        env = make_env(seed, lam_u=0.3)
        agent = CVaRDistDQN_TwoStep(env, seed=seed, cvar_alpha=0.6,
                                    cvar_beta=0.2, pcol_max=0.05)
    elif name == "-M3":
        env = make_env(seed, lam_u=0.0)   # no uncertainty penalty
        agent = CVaRDistDQN(env, seed=seed, cvar_alpha=0.6, cvar_beta=0.2,
                            pcol_max=0.05)
    else:
        raise ValueError(name)

    b = env.reset()
    for _ in range(TRAIN_STEPS):
        a = agent.act(b)
        b2, r, _, info = env.step(a)
        agent.learn(b, a, r, b2, collision=info["collision"])
        b = b2

    # evaluate
    env_eval = make_env(seed + 100)
    agent.env = env_eval
    b = env_eval.reset()
    thru = coll = 0.0
    for _ in range(EVAL_STEPS):
        a = agent.act(b, greedy=True)
        b, r, _, info = env_eval.step(a)
        thru += info["thru"]; coll += info["collision"]
    return thru / EVAL_STEPS, coll / EVAL_STEPS


def main():
    variants = ["Full", "-M1", "-M2", "-M3"]
    results = {}
    for v in variants:
        ths, cls = [], []
        for sd in SEEDS:
            t, c = run_variant(v, sd)
            ths.append(t); cls.append(c)
        results[v] = {
            "thru": (np.mean(ths), np.std(ths)),
            "coll": (np.mean(cls), np.std(cls)),
            "obj": (np.mean([t - KAPPA * (1.0/64) * c for t, c in zip(ths, cls)]),
                    0.0),
        }
        print(f"{v:6s} thru={results[v]['thru'][0]:.4f}  "
              f"coll={results[v]['coll'][0]:.4f}  obj={results[v]['obj'][0]:.4f}")

    # ---- grouped bar chart: throughput (up good) and collision (down good) ----
    labels = variants
    thr = [results[v]["thru"][0] for v in labels]
    thr_e = [results[v]["thru"][1] for v in labels]
    col = [results[v]["coll"][0] for v in labels]
    col_e = [results[v]["coll"][1] for v in labels]

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(6.4, 4))
    w = 0.38
    b1 = ax1.bar(x - w/2, thr, w, yerr=thr_e, capsize=3,
                 color="tab:blue", label="throughput (↑)")
    ax1.set_ylabel("normalized throughput", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("ablation variant")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, col, w, yerr=col_e, capsize=3,
                 color="tab:red", label="PU collision (↓)")
    ax2.set_ylabel("PU collision rate", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Component ablation: removing M1/M2/M3 degrades performance")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_ablation.pdf"))
    plt.close()

    np.savez(os.path.join(OUT, "ablation.npz"),
             labels=labels, thr=thr, thr_e=thr_e, col=col, col_e=col_e)
    print("saved fig_ablation.pdf + ablation.npz")


if __name__ == "__main__":
    main()
