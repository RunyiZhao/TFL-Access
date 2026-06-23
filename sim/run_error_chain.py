"""
run_error_chain.py -- Full sensing-error propagation chain (TWC must-have).

Validates the COMPLETE theoretical chain of Section V end-to-end:

        eps_loc  -->  (Pmd, Pfa)  -->  Pcol  -->  throughput

The reviewer asked for a single, multi-panel "error-propagation consistency"
plot that shows every link of the chain as a function of the localization
error eps_loc = 1 - E[IoU], and checks that each link is (piecewise) linear,
as Assumptions 1-2 and Lemmas 1-2 predict. Earlier we only showed the first
and last links (fig_localization); this closes the chain by adding the
collision link Pcol and reporting the goodness-of-fit (Pearson r, slope) of
every link.

Pure NumPy (no torch). The access policy is the same fixed greedy belief rule
used in run_localization_sweep.py, so the measured relationship isolates the
SENSING -> ACCESS map, not the learner.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_loc import LocalizationTFEnv
from twc_style import apply_twc_style, DBL_W, PALETTE, panel_label

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

ETA = 0.5
EVAL_STEPS = 4000
QUALITIES = [0.50, 0.60, 0.68, 0.76, 0.83, 0.89, 0.94, 0.97]
SEEDS = list(range(10))


def greedy_lowest_belief(env, B):
    best, best_score = 0, np.inf
    for idx, (fc, bw) in enumerate(env.actions):
        lo, hi = env.action_cells((fc, bw))
        score = B[lo:hi + 1].sum() - 0.001 * (hi - lo + 1)
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
    meas["pcol"] = coll / EVAL_STEPS
    return meas


def _fit(x, y):
    """Return (slope, intercept, pearson_r) of a linear fit."""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, intercept, r


def main():
    rows = {q: [eval_quality(q, sd) for sd in SEEDS] for q in QUALITIES}

    def agg(q, key):
        v = [r[key] for r in rows[q]]
        return np.mean(v), np.std(v)

    eps = np.array([agg(q, "eps_loc")[0] for q in QUALITIES])
    pmd = np.array([agg(q, "Pmd")[0] for q in QUALITIES])
    pfa = np.array([agg(q, "Pfa")[0] for q in QUALITIES])
    pcol = np.array([agg(q, "pcol")[0] for q in QUALITIES])
    pcol_e = np.array([agg(q, "pcol")[1] for q in QUALITIES])
    thr = np.array([agg(q, "thru")[0] for q in QUALITIES])
    thr_e = np.array([agg(q, "thru")[1] for q in QUALITIES])
    gap = thr.max() - thr

    o = np.argsort(eps)
    eps, pmd, pfa, pcol, pcol_e, thr, thr_e, gap = (
        eps[o], pmd[o], pfa[o], pcol[o], pcol_e[o], thr[o], thr_e[o], gap[o])

    s_md, _, r_md = _fit(eps, pmd)
    s_fa, _, r_fa = _fit(eps, pfa)
    s_col, _, r_col = _fit(eps, pcol)
    s_gap, _, r_gap = _fit(eps, gap)

    print(" eps_loc   Pmd     Pfa     Pcol    thru    gap")
    for i in range(len(eps)):
        print(f"{eps[i]:.3f}   {pmd[i]:.3f}  {pfa[i]:.3f}  {pcol[i]:.3f}  "
              f"{thr[i]:.4f}  {gap[i]:.4f}")
    print(f"\nlinearity (Pearson r): Pmd={r_md:.3f} Pfa={r_fa:.3f} "
          f"Pcol={r_col:.3f} gap={r_gap:.3f}")

    apply_twc_style()
    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 4.4))
    xs = np.linspace(eps.min(), eps.max(), 50)

    def link(ax, y, yerr, color, ylabel, label, slope, r):
        A = np.vstack([eps, np.ones_like(eps)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        if yerr is not None:
            ax.errorbar(eps, y, yerr=yerr, fmt="o", color=color, capsize=2.5,
                        ms=4, lw=0.8, label=label)
        else:
            ax.plot(eps, y, "o", color=color, ms=4, label=label)
        ax.plot(xs, m * xs + c, "--", color="black", lw=1.0,
                label=f"fit ($r$={r:.3f})")
        ax.set_xlabel(r"localization error $\varepsilon_{\mathrm{loc}}=1-\mathrm{IoU}$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    link(axes[0, 0], pmd, None, PALETTE["red"], "miss rate $P_{\\mathrm{md}}$",
         "$P_{\\mathrm{md}}$", s_md, r_md)
    # overlay Pfa on the first panel for the cell-level link
    A = np.vstack([eps, np.ones_like(eps)]).T
    mfa, cfa = np.linalg.lstsq(A, pfa, rcond=None)[0]
    axes[0, 0].plot(eps, pfa, "s", color=PALETTE["blue"], ms=4,
                    label="$P_{\\mathrm{fa}}$")
    axes[0, 0].plot(xs, mfa * xs + cfa, ":", color=PALETTE["blue"], lw=1.0,
                    label=f"$P_{{\\mathrm{{fa}}}}$ fit ($r$={r_fa:.3f})")
    axes[0, 0].set_ylabel("cell-level error")
    axes[0, 0].legend(loc="best", fontsize=6)

    link(axes[0, 1], pcol, pcol_e, PALETTE["orange"],
         "collision $P_{\\mathrm{col}}$", "$P_{\\mathrm{col}}$", s_col, r_col)
    link(axes[1, 0], thr, thr_e, PALETTE["green"], "throughput",
         "throughput", 0.0, float(np.corrcoef(eps, thr)[0, 1]))
    link(axes[1, 1], gap, thr_e, PALETTE["purple"], "throughput gap",
         "gap", s_gap, r_gap)

    for ax, lab in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
        panel_label(ax, lab)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_error_chain.pdf")); plt.close()
    np.savez(os.path.join(OUT, "error_chain.npz"),
             eps=eps, pmd=pmd, pfa=pfa, pcol=pcol, thr=thr, gap=gap,
             r_md=r_md, r_fa=r_fa, r_col=r_col, r_gap=r_gap,
             s_md=s_md, s_fa=s_fa, s_col=s_col, s_gap=s_gap)
    print("\nsaved fig_error_chain.pdf and error_chain.npz")


if __name__ == "__main__":
    main()
