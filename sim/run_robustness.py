"""
run_robustness.py -- Cross-environment robustness stress suite (TWC must-have).

Answers the reviewer's most serious concern: "Are the results specific to one
synthetic setup?" We stress the environment along the three axes the reviewer
named and check that the belief-map representation keeps its advantage over the
prior-work fixed-channel grid in EVERY regime:

  (A) PU traffic diversity      : nominal Markov vs. bursty/heavy-tail traffic
  (B) spectrum geometry         : sparse (1 PU) vs. nominal (3) vs. dense (6 PU)
  (C) SNR / channel condition   : low-SNR vs. nominal vs. high-SNR sensing

To isolate the REPRESENTATION from the (high-variance) learning dynamics -- the
same methodology as the representation-richness test in run_representation.py --
both representations are consumed by the SAME fixed greedy "most-likely-idle"
access rule. We compare the proposed full-resolution SOFT belief map against the
prior-work fixed-channel grid (binarized K=8). The robustness claim is that the
soft belief map holds a higher throughput AND a lower-or-equal collision rate
across all regimes, i.e. the structural advantage of Proposition 1 is not an
artifact of the nominal instance.

Pure NumPy (CPU). Fast: no training.
"""
import os
import numpy as np

from env import ContinuousTFEnv
from run_representation import _rep_vector, _action_spans

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

EVAL = 8000
SEEDS = list(range(12))

REGIMES = {
    "nominal":   dict(n_pu=3, snr_quality=1.2, burst_prob=0.0),
    "low-SNR":   dict(n_pu=3, snr_quality=0.7, burst_prob=0.0),
    "high-SNR":  dict(n_pu=3, snr_quality=2.0, burst_prob=0.0),
    "sparse":    dict(n_pu=1, snr_quality=1.2, burst_prob=0.0),
    "dense":     dict(n_pu=6, snr_quality=1.2, burst_prob=0.0),
    "bursty":    dict(n_pu=3, snr_quality=1.2, burst_prob=0.05, burst_len=10),
}
REGIME_ORDER = ["nominal", "low-SNR", "high-SNR", "sparse", "dense", "bursty"]
# representation under test : label
METHODS = {"soft-64": "Belief map (ours)", "grid-8": "Fixed-channel grid"}


def make_env(regime, seed):
    kw = dict(Nf=64, eta=0.5, lam_c=2.0, lam_u=0.3, seed=seed)
    kw.update(REGIMES[regime])
    return ContinuousTFEnv(**kw)


def run(regime, rep, seed):
    env = make_env(regime, seed)
    lo, hi = _action_spans(env)
    rbar = 1.0 / env.Nf
    b = env.reset()
    thru = coll = rew = 0.0
    for _ in range(EVAL):
        v = _rep_vector(rep, b, env.eta, env.Nf)
        idle = np.concatenate([[0.0], np.cumsum(1.0 - v)])
        logsafe = np.concatenate([[0.0], np.cumsum(np.log(np.clip(1.0 - v, 1e-6, 1.0)))])
        score = rbar * (idle[hi + 1] - idle[lo]) \
            - env.lam_c * (1.0 - np.exp(logsafe[hi + 1] - logsafe[lo]))
        a = int(np.argmax(score))
        b, r, _, info = env.step(a)
        thru += info["thru"]; coll += info["collision"]; rew += r
    return thru / EVAL, coll / EVAL, rew / EVAL


def main():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from twc_style import apply_twc_style, DBL_W, PALETTE
    apply_twc_style()

    res = {(r, m): ([], [], []) for r in REGIME_ORDER for m in METHODS}
    for r in REGIME_ORDER:
        for m in METHODS:
            for sd in SEEDS:
                t, c, rw = run(r, m, sd)
                res[(r, m)][0].append(t); res[(r, m)][1].append(c)
                res[(r, m)][2].append(rw)

    def agg(r, m, i):
        v = res[(r, m)][i]; return np.mean(v), np.std(v)

    print(f"{'regime':10s} {'method':18s} {'reward':>14s} {'collision':>14s}")
    for r in REGIME_ORDER:
        for m in METHODS:
            rw, rwe = agg(r, m, 2); c, ce = agg(r, m, 1)
            print(f"{r:10s} {METHODS[m]:18s} {rw:.4f}+/-{rwe:.4f} {c:.4f}+/-{ce:.4f}")

    x = np.arange(len(REGIME_ORDER)); w = 0.38
    sr = [agg(r, "soft-64", 2)[0] for r in REGIME_ORDER]
    sre = [agg(r, "soft-64", 2)[1] for r in REGIME_ORDER]
    gr = [agg(r, "grid-8", 2)[0] for r in REGIME_ORDER]
    gre = [agg(r, "grid-8", 2)[1] for r in REGIME_ORDER]
    sc = [agg(r, "soft-64", 1)[0] for r in REGIME_ORDER]
    sce = [agg(r, "soft-64", 1)[1] for r in REGIME_ORDER]
    gc = [agg(r, "grid-8", 1)[0] for r in REGIME_ORDER]
    gce = [agg(r, "grid-8", 1)[1] for r in REGIME_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.9))
    ax1.bar(x - w/2, sr, w, yerr=sre, capsize=2.5, color=PALETTE["blue"],
            label="Belief map (ours)", error_kw={"lw": 0.7})
    ax1.bar(x + w/2, gr, w, yerr=gre, capsize=2.5, color=PALETTE["gray"],
            label="Fixed-channel grid", error_kw={"lw": 0.7})
    ax1.set_ylabel("access reward (eq. 9)"); ax1.set_xticks(x)
    ax1.set_xticklabels(REGIME_ORDER, rotation=20, ha="right")
    ax1.legend(loc="best", fontsize=6); ax1.set_title("(a) access reward")

    ax2.bar(x - w/2, sc, w, yerr=sce, capsize=2.5, color=PALETTE["blue"],
            label="Belief map (ours)", error_kw={"lw": 0.7})
    ax2.bar(x + w/2, gc, w, yerr=gce, capsize=2.5, color=PALETTE["red"],
            label="Fixed-channel grid", error_kw={"lw": 0.7})
    ax2.set_ylabel("PU collision rate"); ax2.set_xticks(x)
    ax2.set_xticklabels(REGIME_ORDER, rotation=20, ha="right")
    ax2.legend(loc="best", fontsize=6); ax2.set_title("(b) PU collision")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_robustness.pdf")); plt.close()

    np.savez(os.path.join(OUT, "robustness.npz"),
             regimes=np.array(REGIME_ORDER),
             soft_rew=np.array(sr), grid_rew=np.array(gr),
             soft_col=np.array(sc), grid_col=np.array(gc))

    rwins = sum(1 for i in range(len(REGIME_ORDER)) if sr[i] >= gr[i] - 1e-9)
    cwins = sum(1 for i in range(len(REGIME_ORDER)) if sc[i] <= gc[i] + 1e-9)
    print(f"\nbelief reward    >= grid in {rwins}/{len(REGIME_ORDER)} regimes")
    print(f"belief collision <= grid in {cwins}/{len(REGIME_ORDER)} regimes")
    print("wrote fig_robustness.pdf and robustness.npz")


if __name__ == "__main__":
    main()
