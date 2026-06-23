"""
run_representation.py -- Representation-richness test (TWC must-have).

Answers the reviewer's question: "Is the belief map truly necessary, or just
better feature engineering?" We hold the DECISION RULE, environment, and action
set fixed and vary ONLY the state representation, along a single axis of
increasing information richness:

  1. grid-4    : belief averaged into 4 fixed channels, binarized  (coarsest)
  2. grid-8    : belief averaged into 8 fixed channels, binarized
  3. grid-16   : belief averaged into 16 fixed channels, binarized
  4. binary-64 : full-resolution HARD occupancy (belief thresholded at eta)
  5. soft-64   : full-resolution SOFT belief map  (proposed)

Crucially, every representation is consumed by the SAME fixed greedy
"most-likely-idle" access rule -- so the experiment isolates the information the
representation preserves from any learning dynamics (which are high-variance at
this compact CPU scale). This is the direct test of Proposition 1 (information
refinement): a coarsening g(.) of the belief can only lose value, so the
fixed-rule access value should be monotone in representation richness, with the
soft map on top. (The learned-agent view is the matched-action baseline study,
Q2/Q4.)

Pure NumPy (CPU), vectorized over the action set (fast: no training).
"""
import os
import numpy as np

from env import ContinuousTFEnv

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

EVAL = 8000
SEEDS = list(range(12))
REPS = ["grid-4", "grid-8", "grid-16", "binary-64", "soft-64"]
RICHNESS = {"grid-4": 4, "grid-8": 8, "grid-16": 16,
            "binary-64": 64, "soft-64": 256}  # soft ~ multi-bit per cell


def _action_spans(env):
    """Precompute (lo, hi) cell span for every action (fixed for the env)."""
    spans = np.array([env.action_cells(a) for a in env.actions], dtype=int)
    return spans[:, 0], spans[:, 1]


def _rep_vector(rep, belief, eta, Nf):
    """Per-cell occupancy score vector of length Nf for the representation
    (lower = more-likely idle). Grids are upsampled back to Nf cells so a single
    cumulative-sum scorer handles all representations uniformly."""
    if rep == "soft-64":
        return belief
    if rep == "binary-64":
        return (belief >= eta).astype(float)
    if rep.startswith("grid-"):
        K = int(rep.split("-")[1])
        chunks = np.array_split(belief, K)
        g = np.array([1.0 if c.mean() >= eta else 0.0 for c in chunks])
        # upsample channel decision back to Nf cells
        sizes = [len(c) for c in chunks]
        return np.repeat(g, sizes)
    raise ValueError(rep)


def run(rep, seed):
    env = make_env(seed)
    lo, hi = _action_spans(env)
    rbar = 1.0 / env.Nf
    b = env.reset()
    thru = coll = rew = 0.0
    for _ in range(EVAL):
        v = _rep_vector(rep, b, env.eta, env.Nf)
        # one-step reward-greedy treating the representation as an occupancy
        # probability: maximize  rbar * E[idle cells] - lam_c * P(collision).
        idle = np.concatenate([[0.0], np.cumsum(1.0 - v)])
        logsafe = np.concatenate([[0.0], np.cumsum(np.log(np.clip(1.0 - v, 1e-6, 1.0)))])
        idle_est = idle[hi + 1] - idle[lo]
        coll_risk = 1.0 - np.exp(logsafe[hi + 1] - logsafe[lo])
        score = rbar * idle_est - env.lam_c * coll_risk
        a = int(np.argmax(score))
        b, r, _, info = env.step(a)
        thru += info["thru"]; coll += info["collision"]; rew += r
    return thru / EVAL, coll / EVAL, rew / EVAL


def make_env(seed):
    return ContinuousTFEnv(Nf=64, n_pu=3, snr_quality=1.2, eta=0.5,
                           lam_c=2.0, lam_u=0.3, seed=seed)


# ---- shared helpers reused by run_robustness.py (vectorized greedy) ----
def represent(rep, belief, eta, Nf):
    """Return a per-action scorer: spans (lo,hi) -> score. Kept for API
    compatibility; the vectorized path in run()/eval_regime is preferred."""
    v = _rep_vector(rep, belief, eta, Nf)

    def score(lo, hi):
        return v[lo:hi + 1].mean()
    return score


def main():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from twc_style import apply_twc_style, COL_W, PALETTE
    apply_twc_style()

    res = {rep: ([], [], []) for rep in REPS}
    for rep in REPS:
        for sd in SEEDS:
            t, c, rw = run(rep, sd)
            res[rep][0].append(t); res[rep][1].append(c); res[rep][2].append(rw)

    thr = [np.mean(res[r][0]) for r in REPS]
    thr_e = [np.std(res[r][0]) for r in REPS]
    col = [np.mean(res[r][1]) for r in REPS]
    col_e = [np.std(res[r][1]) for r in REPS]
    rew = [np.mean(res[r][2]) for r in REPS]
    rew_e = [np.std(res[r][2]) for r in REPS]

    print(f"{'representation':12s} {'thru':>14s} {'collision':>14s} {'reward':>14s}")
    for i, r in enumerate(REPS):
        print(f"{r:12s} {thr[i]:.4f}+/-{thr_e[i]:.4f} {col[i]:.4f}+/-{col_e[i]:.4f} "
              f"{rew[i]:.4f}+/-{rew_e[i]:.4f}")

    rich = np.array([RICHNESS[r] for r in REPS], dtype=float)

    def spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    sp_rew = spearman(rich, rew)
    sp_col = spearman(rich, [-c for c in col])
    print(f"\nSpearman(richness, reward)     = {sp_rew:+.3f}")
    print(f"Spearman(richness, -collision) = {sp_col:+.3f}")

    x = np.arange(len(REPS))
    fig, ax1 = plt.subplots(figsize=(COL_W, 2.7))
    ax1.errorbar(x, rew, yerr=rew_e, fmt="o-", color=PALETTE["blue"],
                 capsize=2.5, lw=1.2, label="access reward (eq. 9)")
    ax1.set_ylabel("access reward", color=PALETTE["blue"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax1.set_xticks(x); ax1.set_xticklabels(REPS, rotation=15, ha="right")
    ax1.set_xlabel("state representation (increasing information richness $\\to$)")
    ax2 = ax1.twinx()
    ax2.errorbar(x, col, yerr=col_e, fmt="s--", color=PALETTE["red"],
                 capsize=2.5, lw=1.2, label="PU collision")
    ax2.set_ylabel("PU collision rate", color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"]); ax2.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_representation.pdf")); plt.close()

    np.savez(os.path.join(OUT, "representation.npz"),
             reps=np.array(REPS), thr=np.array(thr), thr_e=np.array(thr_e),
             col=np.array(col), col_e=np.array(col_e),
             rew=np.array(rew), rew_e=np.array(rew_e), richness=rich,
             sp_rew=sp_rew, sp_col=sp_col)
    print("wrote fig_representation.pdf and representation.npz")


if __name__ == "__main__":
    main()
