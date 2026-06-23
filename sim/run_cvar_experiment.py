"""
run_cvar_experiment.py -- Compare the CVaR-constrained distributional agent
(M1) against the mean-constrained (expectation) agent.

Produces:
  figures/fig_cvar_tail.pdf : histogram of per-window collision counts for
                              mean- vs CVaR-constrained -> shows the CVaR
                              agent clips the heavy tail (the key M1 plot).
  figures/cvar_results.npz
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import ContinuousTFEnv
from agents import BeliefDQN          # mean-constrained baseline (Lagrangian)
from agents_cvar import CVaRDistDQN   # CVaR-constrained distributional (M1)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

TRAIN_STEPS = 40000
EVAL_WINDOWS = 400
WINDOW = 25      # collisions are counted per window of this many steps
SEEDS = list(range(2))


def make_env(seed):
    # burst_prob>0 injects correlated heavy-occupancy events, creating a
    # GENUINE collision tail so that tail-clipping is non-trivial to verify.
    return ContinuousTFEnv(Nf=64, n_pu=3, snr_quality=1.2, eta=0.5,
                           lam_c=2.0, lam_u=0.3,
                           burst_prob=0.02, burst_len=8, seed=seed)


def train(agent_cls, seed=0, **kw):
    env = make_env(seed)
    agent = agent_cls(env, seed=seed, **kw)
    b = env.reset()
    for _ in range(TRAIN_STEPS):
        a = agent.act(b)
        b2, r, _, info = env.step(a)
        agent.learn(b, a, r, b2, collision=info["collision"])
        b = b2
    return agent


def eval_windows(agent, seed=999):
    env = make_env(seed)
    agent.env = env
    b = env.reset()
    counts = []
    thru_total = 0.0
    for _ in range(EVAL_WINDOWS):
        c = 0
        for _ in range(WINDOW):
            a = agent.act(b, greedy=True)
            b, r, _, info = env.step(a)
            c += int(info["collision"]); thru_total += info["thru"]
        counts.append(c)
    return np.array(counts), thru_total / (EVAL_WINDOWS * WINDOW)


def main():
    mc_all, cc_all = [], []
    mthru_all, cthru_all = [], []
    for sd in SEEDS:
        mean_agent = train(BeliefDQN, seed=sd, pcol_max=0.05)
        cvar_agent = train(CVaRDistDQN, seed=sd, pcol_max=0.02,
                           cvar_alpha=0.4, cvar_beta=0.15,
                           mu_lr=0.05, mu_max=50.0)
        mc, mthru = eval_windows(mean_agent, seed=900 + sd)
        cc, cthru = eval_windows(cvar_agent, seed=900 + sd)
        mc_all.append(mc); cc_all.append(cc)
        mthru_all.append(mthru); cthru_all.append(cthru)
    mc = np.concatenate(mc_all); cc = np.concatenate(cc_all)
    mthru = float(np.mean(mthru_all)); cthru = float(np.mean(cthru_all))

    def tail(x, q=0.95):
        return np.quantile(x, q)

    print(f"\n mean-constr : mean_coll/window={mc.mean():.2f} "
          f"95%tail={tail(mc):.1f} max={mc.max()} thru={mthru:.3f}")
    print(f" CVaR-constr : mean_coll/window={cc.mean():.2f} "
          f"95%tail={tail(cc):.1f} max={cc.max()} thru={cthru:.3f}")

    np.savez(os.path.join(OUT, "cvar_results.npz"),
             mean_counts=mc, cvar_counts=cc, mthru=mthru, cthru=cthru)
    print("saved cvar_results.npz (figure via make_figures.py)")


if __name__ == "__main__":
    main()
