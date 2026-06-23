"""
run_experiments.py -- Train LA-JSSA vs. baselines, produce figures.

Outputs (into figures/):
  fig_learning.pdf    : reward learning curves
  fig_metrics.pdf     : throughput / collision / utilization bars
  results.npz         : raw arrays for reproducibility
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import ContinuousTFEnv
from agents import (BeliefDQN, TwoStepDQN, FixedChannelDQN,
                    MyopicPolicy, OraclePolicy)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

TRAIN_STEPS = 40000
EVAL_STEPS = 5000
ETA = 0.5
SNR_Q = 1.3


def make_env(seed):
    return ContinuousTFEnv(Nf=64, n_pu=3, snr_quality=SNR_Q, eta=ETA,
                           lam_c=2.0, lam_u=0.3, seed=seed)


def train_agent(AgentCls, seed=0, learning=True):
    env = make_env(seed)
    agent = AgentCls(env, seed=seed) if learning else AgentCls(env, seed=seed)
    belief = env.reset()
    rewards = []
    window = []
    for t in range(TRAIN_STEPS):
        a = agent.act(belief)
        belief2, r, _, info = env.step(a)
        agent.learn(belief, a, r, belief2, collision=info["collision"])
        belief = belief2
        window.append(r)
        if len(window) == 200:
            rewards.append(np.mean(window)); window = []
    return agent, env, np.array(rewards)


def eval_agent(agent, env, n=EVAL_STEPS):
    belief = env.reset()
    thru = coll = util = 0.0
    rsum = 0.0
    for _ in range(n):
        a = agent.act(belief, greedy=True)
        belief, r, _, info = env.step(a)
        thru += info["thru"]; coll += info["collision"]
        util += info["idle"] / env.Nf
        rsum += r
    return dict(throughput=thru / n, collision=coll / n,
               utilization=util / n, reward=rsum / n)


def main():
    learners = [BeliefDQN, TwoStepDQN, FixedChannelDQN]
    nonlearners = [MyopicPolicy, OraclePolicy]
    SEEDS = [0, 1, 2]

    curves = {}
    metrics = {}

    for Cls in learners:
        print(f"training {Cls.name} over {len(SEEDS)} seeds ...")
        seed_curves = []
        seed_metrics = []
        for sd in SEEDS:
            agent, env, curve = train_agent(Cls, seed=sd)
            seed_curves.append(curve)
            env_eval = make_env(123 + sd)
            agent.env = env_eval
            seed_metrics.append(eval_agent(agent, env_eval))
        L = min(len(c) for c in seed_curves)
        curves[Cls.name] = np.mean([c[:L] for c in seed_curves], axis=0)
        metrics[Cls.name] = {
            k: (np.mean([m[k] for m in seed_metrics]),
                np.std([m[k] for m in seed_metrics]))
            for k in seed_metrics[0]
        }

    for Cls in nonlearners:
        seed_metrics = []
        for sd in SEEDS:
            env = make_env(7 + sd)
            agent = Cls(env, seed=7 + sd)
            env_eval = make_env(123 + sd)
            agent.env = env_eval
            seed_metrics.append(eval_agent(agent, env_eval))
        metrics[Cls.name] = {
            k: (np.mean([m[k] for m in seed_metrics]),
                np.std([m[k] for m in seed_metrics]))
            for k in seed_metrics[0]
        }

    for name, m in metrics.items():
        print(f"{name:28s} thru={m['throughput'][0]:.4f}+-{m['throughput'][1]:.4f} "
              f"coll={m['collision'][0]:.4f}+-{m['collision'][1]:.4f} "
              f"util={m['utilization'][0]:.4f}")

    # ---- Fig 1: learning curves ----
    plt.figure(figsize=(6, 4))
    for name, c in curves.items():
        x = np.arange(len(c)) * 200
        plt.plot(x, c, label=name, linewidth=1.8)
    plt.xlabel("training step"); plt.ylabel("avg reward (window=200)")
    plt.title("Learning curves"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_learning.pdf"))
    plt.close()

    # ---- Fig 2: metric bars with error bars ----
    names = list(metrics.keys())
    thr = [metrics[n]["throughput"][0] for n in names]
    thr_e = [metrics[n]["throughput"][1] for n in names]
    col = [metrics[n]["collision"][0] for n in names]
    col_e = [metrics[n]["collision"][1] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(range(len(names)), thr, yerr=thr_e, capsize=4, color="tab:blue")
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right")
    ax[0].set_ylabel("normalized throughput"); ax[0].set_title("Throughput")
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(range(len(names)), col, yerr=col_e, capsize=4, color="tab:red")
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=30, ha="right")
    ax[1].set_ylabel("PU collision rate"); ax[1].set_title("PU Collision")
    ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_metrics.pdf"))
    plt.close()

    np.savez(os.path.join(OUT, "results.npz"),
             names=names, thr=thr, thr_e=thr_e, col=col, col_e=col_e,
             **{f"curve_{i}": c for i, c in enumerate(curves.values())})
    print("saved figures + results.npz")


if __name__ == "__main__":
    main()
