"""
run_baselines.py -- head-to-head comparison of LA-JSSA against baselines on a
matched environment, state, and discrete action set. Pure NumPy (CPU).

Methods compared (all act()/learn() compatible, same action set):
  Random            : uniform access (floor)
  Myopic            : greedy min-belief (conservative)
  Fixed-channel DQN : binary grid state (prior-work premise)
  Belief-DQN        : continuous belief state, mean-constrained
  PPO-lite          : clipped policy gradient + Lagrangian collision penalty
  LA-JSSA (CVaR)    : full method (distributional critic + CVaR constraint)
  Oracle            : perfect sensing (upper bound)

Run ONE method/seed at a time (cached), then aggregate -> figure + table.

Usage:
  python run_baselines.py <method> <seed>     # e.g. python run_baselines.py PPO 0
  python run_baselines.py --combine           # build fig_baselines.pdf + table
"""
import os, sys, json
import numpy as np

from env import ContinuousTFEnv
from agents import BeliefDQN, FixedChannelDQN, MyopicPolicy, OraclePolicy
from agents_cvar import CVaRDistDQN
from agents_pg import PPOLiteAgent, RandomAgent

CACHE = os.path.join(os.path.dirname(__file__), "baselines_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
TRAIN = 16000
EVAL = 4000
KAPPA = 3.0 / 64
METHODS = ["Random", "Myopic", "FixedDQN", "BeliefDQN", "PPO-lite",
           "LA-JSSA", "Oracle"]


def make_env(seed):
    return ContinuousTFEnv(Nf=64, n_pu=3, snr_quality=1.2, eta=0.5,
                           lam_c=2.0, lam_u=0.3, seed=seed)


def build(method, env, seed):
    if method == "Random":    return RandomAgent(env, seed=seed)
    if method == "Myopic":    return MyopicPolicy(env)
    if method == "FixedDQN":  return FixedChannelDQN(env, seed=seed, pcol_max=0.05)
    if method == "BeliefDQN": return BeliefDQN(env, seed=seed, pcol_max=0.05)
    if method == "PPO-lite":  return PPOLiteAgent(env, seed=seed, pcol_max=0.05)
    if method == "LA-JSSA":   return CVaRDistDQN(env, seed=seed, cvar_alpha=0.6,
                                                 cvar_beta=0.2, pcol_max=0.05)
    if method == "Oracle":    return OraclePolicy(env)
    raise ValueError(method)


def run(method, seed):
    env = make_env(seed)
    ag = build(method, env, seed)
    b = env.reset()
    learns = method not in ("Random", "Myopic", "Oracle")
    for _ in range(TRAIN if learns else 1):
        a = ag.act(b)
        b2, r, _, info = env.step(a)
        ag.learn(b, a, r, b2, collision=info["collision"])
        b = b2
    # evaluate
    ev = make_env(seed + 100)
    if hasattr(ag, "env"): ag.env = ev
    b = ev.reset()
    thru = coll = 0.0
    for _ in range(EVAL):
        a = ag.act(b, greedy=True)
        b, r, _, info = ev.step(a)
        thru += info["thru"]; coll += info["collision"]
    return thru / EVAL, coll / EVAL


def combine():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from twc_style import apply_twc_style, COL_W, DBL_W, PALETTE
    apply_twc_style()

    agg = {}
    for m in METHODS:
        ts = [v[0] for k, v in cache.items() if k.split("|")[0] == m]
        cs = [v[1] for k, v in cache.items() if k.split("|")[0] == m]
        if ts:
            agg[m] = (np.mean(ts), np.std(ts), np.mean(cs), np.std(cs), len(ts))

    print(f"{'method':12s} {'thru':>16s} {'collision':>16s}  n")
    for m in METHODS:
        if m in agg:
            t, te, c, ce, n = agg[m]
            print(f"{m:12s} {t:.4f}+/-{te:.4f}   {c:.4f}+/-{ce:.4f}  {n}")

    present = [m for m in METHODS if m in agg]
    thr = [agg[m][0] for m in present]; thr_e = [agg[m][1] for m in present]
    col = [agg[m][2] for m in present]; col_e = [agg[m][3] for m in present]
    x = np.arange(len(present)); w = 0.38
    fig, ax1 = plt.subplots(figsize=(DBL_W, 2.7))
    ax1.bar(x - w/2, thr, w, yerr=thr_e, capsize=2.5, color=PALETTE["blue"],
            label="throughput", error_kw={"lw": 0.7})
    ax1.set_ylabel("throughput", color=PALETTE["blue"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax1.set_xticks(x); ax1.set_xticklabels(present, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, col, w, yerr=col_e, capsize=2.5, color=PALETTE["red"],
            label="PU collision", error_kw={"lw": 0.7})
    ax2.set_ylabel("PU collision rate", color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"]); ax2.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_baselines.pdf")); plt.close()

    # emit LaTeX table rows
    name_map = {"FixedDQN": "Fixed-channel DQN~\\cite{wang2018}",
                "BeliefDQN": "Belief-DQN (ours, mean)",
                "PPO-lite": "PPO~\\cite{schulman2017ppo} (Lagrangian)",
                "LA-JSSA": "\\textbf{LA-JSSA (ours)}",
                "Myopic": "Myopic (min-belief)",
                "Random": "Random access",
                "Oracle": "Perfect-sensing oracle"}
    rows = []
    for m in present:
        t, te, c, ce, n = agg[m]
        rows.append(f"{name_map.get(m,m)} & ${t:.3f}\\pm{te:.3f}$ & "
                    f"${c:.3f}\\pm{ce:.3f}$ \\\\")
    open(os.path.join(OUT, "baselines_table_rows.tex"), "w").write("\n".join(rows))
    print("\nwrote fig_baselines.pdf and baselines_table_rows.tex")


if __name__ == "__main__":
    if sys.argv[1] == "--combine":
        combine()
    else:
        method, seed = sys.argv[1], int(sys.argv[2])
        t, c = run(method, seed)
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        cache[f"{method}|{seed}"] = [t, c]
        json.dump(cache, open(CACHE, "w"))
        print(f"{method} seed{seed}: thru={t:.4f} coll={c:.4f} [{len(cache)} cached]")
