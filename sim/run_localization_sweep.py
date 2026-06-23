"""
run_localization_sweep.py -- Validate the TFL-grounded theory (Section V).

Sweeps the localizer quality, and for each quality level measures:
  * mean IoU and eps_loc = 1 - IoU         (localization accuracy)
  * induced cell-level Pmd, Pfa at eta*     (eq. loc_to_cell)
  * access throughput and PU-collision of a fixed greedy belief policy

Produces fig_localization.pdf with two panels:
  (a) eps_loc -> (Pmd, Pfa): confirms cell-level error grows ~linearly in
      localization error, validating Assumption (Localization-error model).
  (b) eps_loc -> access throughput gap: confirms Corollary (Access cost of
      localization error): throughput gap grows ~linearly in eps_loc.

Pure numpy (no torch). The access policy here is a fixed, non-learning greedy
rule (pick the lowest-belief block that meets the bandwidth demand), so the
measured relationship isolates the SENSING->ACCESS map, not the learner.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_loc import LocalizationTFEnv

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

ETA = 0.5
EVAL_STEPS = 4000
QUALITIES = [0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97]
SEEDS = list(range(10))


def greedy_lowest_belief(env, B):
    """Fixed policy: among actions meeting the largest bandwidth, choose the
    block with the lowest summed belief (most-likely-idle)."""
    best, best_score = 0, np.inf
    for idx, (fc, bw) in enumerate(env.actions):
        lo, hi = env.action_cells((fc, bw))
        score = B[lo:hi + 1].sum() - 0.001 * (hi - lo + 1)  # prefer wider if equal
        if score < best_score:
            best_score, best = score, idx
    return best


def eval_quality(q, seed):
    env = LocalizationTFEnv(Nf=64, n_pu=3, loc_quality=q, eta=ETA, seed=seed)
    meas = env.measure_loc_to_cell(ETA, n=8000, seed=seed + 500)
    B = env.reset()
    thru = coll = 0.0
    for _ in range(EVAL_STEPS):
        a = greedy_lowest_belief(env, B)
        B, r, _, info = env.step(a)
        thru += info["thru"]; coll += info["collision"]
    meas["thru"] = thru / EVAL_STEPS
    meas["coll"] = coll / EVAL_STEPS
    return meas


def main():
    rows = {q: [] for q in QUALITIES}
    for q in QUALITIES:
        for sd in SEEDS:
            rows[q].append(eval_quality(q, sd))

    def agg(q, key):
        v = [r[key] for r in rows[q]]
        return np.mean(v), np.std(v)

    eps = np.array([agg(q, "eps_loc")[0] for q in QUALITIES])
    pmd = np.array([agg(q, "Pmd")[0] for q in QUALITIES])
    pfa = np.array([agg(q, "Pfa")[0] for q in QUALITIES])
    thr = np.array([agg(q, "thru")[0] for q in QUALITIES])
    thr_e = np.array([agg(q, "thru")[1] for q in QUALITIES])

    # oracle throughput (loc_quality -> 1 proxy): take best measured
    thr_oracle = thr.max()
    gap = thr_oracle - thr

    print(" q     eps_loc   Pmd     Pfa     thru    gap")
    for i, q in enumerate(QUALITIES):
        print(f"{q:.2f}  {eps[i]:.3f}   {pmd[i]:.3f}  {pfa[i]:.3f}  "
              f"{thr[i]:.4f}  {gap[i]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))

    # (a) eps_loc -> Pmd, Pfa
    order = np.argsort(eps)
    ax1.plot(eps[order], pmd[order], "o-", color="tab:red", label=r"$P_{md}$")
    ax1.plot(eps[order], pfa[order], "s-", color="tab:blue", label=r"$P_{fa}$")
    ax1.set_xlabel(r"localization error $\varepsilon_{loc}=1-\mathrm{IoU}$")
    ax1.set_ylabel("induced cell-level error")
    ax1.set_title("(a) Localization error $\\to$ detection error")
    ax1.legend(); ax1.grid(alpha=0.3)

    # (b) eps_loc -> throughput gap (linear fit)
    ax2.errorbar(eps[order], gap[order], yerr=thr_e[order], fmt="o",
                 color="tab:purple", capsize=3, label="measured")
    A = np.vstack([eps[order], np.ones_like(eps[order])]).T
    slope, intercept = np.linalg.lstsq(A, gap[order], rcond=None)[0]
    xs = np.linspace(eps.min(), eps.max(), 50)
    ax2.plot(xs, slope * xs + intercept, "--", color="black",
             label=f"linear fit (slope={slope:.2f})")
    ax2.set_xlabel(r"localization error $\varepsilon_{loc}=1-\mathrm{IoU}$")
    ax2.set_ylabel("access throughput gap")
    ax2.set_title("(b) Localization error $\\to$ access cost")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_localization.pdf"))
    plt.close()
    np.savez(os.path.join(OUT, "localization_sweep.npz"),
             eps=eps, pmd=pmd, pfa=pfa, thr=thr, gap=gap, q=np.array(QUALITIES))
    print(f"\nsaved fig_localization.pdf  (gap-vs-eps slope = {slope:.3f})")


if __name__ == "__main__":
    main()
