"""
make_figures.py -- regenerate all data figures from cached .npz results with
IEEE TWC styling (twc_style.py). Run after the experiments have produced the
.npz files, or as a fast re-style pass without re-running experiments.

Produces (into ../figures/):
  fig_tradeoff.pdf     (Q1)  from tradeoff.npz
  fig_cvar_tail.pdf    (Q3)  from cvar_results.npz
  fig_ablation.pdf     (Q4)  from ablation.npz
  fig_freq_vs_time.pdf (Q5)  from freq_vs_time.npz
  fig_localization.pdf (Q6)  from localization_sweep.npz
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from twc_style import apply_twc_style, COL_W, DBL_W, PALETTE, panel_label

apply_twc_style()
FD = os.path.join(os.path.dirname(__file__), "..", "figures")


def load(name):
    return dict(np.load(os.path.join(FD, name)))


def save(fig, name):
    fig.savefig(os.path.join(FD, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Q1 tradeoff
def fig_tradeoff():
    d = load("tradeoff.npz")
    eta = d["etas"]; eta_s = float(d["eta_star_sim"]); eta_min = float(d["eta_min"])
    fig, ax = plt.subplots(1, 3, figsize=(DBL_W, 2.05))
    # (a) ROC
    ax[0].plot(d["Pfa"], 1 - d["Pmd"], "-", color=PALETTE["blue"])
    ax[0].plot(d["Pfa"], 1 - d["Pmd"], "o", color=PALETTE["blue"], ms=3)
    ax[0].set_xlabel(r"$P_{\mathrm{fa}}$"); ax[0].set_ylabel(r"$1-P_{\mathrm{md}}$")
    panel_label(ax[0], "(a)")
    # (b) collision vs eta with budget + eta_min
    ax[1].plot(eta, d["coll"], "-", color=PALETTE["red"])
    ax[1].axvline(eta_min, ls="--", lw=0.8, color=PALETTE["gray"])
    ax[1].set_xlabel(r"belief threshold $\eta$")
    ax[1].set_ylabel("PU collision rate")
    ax[1].annotate(r"$\eta_{\min}$", xy=(eta_min, ax[1].get_ylim()[1]*0.8),
                   fontsize=7, color=PALETTE["gray"])
    panel_label(ax[1], "(b)")
    # (c) throughput vs eta with eta*
    ax[2].plot(eta, d["thru"], "-", color=PALETTE["green"])
    ax[2].axvline(eta_s, ls="--", lw=0.8, color=PALETTE["purple"])
    ax[2].set_xlabel(r"belief threshold $\eta$")
    ax[2].set_ylabel("throughput")
    ax[2].annotate(r"$\eta^\star$", xy=(eta_s, ax[2].get_ylim()[1]*0.5),
                   fontsize=7, color=PALETTE["purple"])
    panel_label(ax[2], "(c)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_tradeoff.pdf")


# ---------------------------------------------------------------- Q3 CVaR tail
def fig_cvar():
    d = load("cvar_results.npz")
    mc = d["mean_counts"]; cc = d["cvar_counts"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    bins = np.arange(0, max(mc.max(), cc.max()) + 2) - 0.5
    ax.hist(mc, bins=bins, alpha=0.55, color=PALETTE["red"],
            label="mean-constrained", density=True)
    ax.hist(cc, bins=bins, alpha=0.7, color=PALETTE["blue"],
            label="CVaR-constrained", density=True)
    ax.set_xlabel("PU collisions per window")
    ax.set_ylabel("frequency")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_cvar_tail.pdf")


# ---------------------------------------------------------------- Q4 ablation
def fig_ablation():
    d = load("ablation.npz")
    labels = [str(x) for x in d["labels"]]
    thr, thr_e = d["thr"], d["thr_e"]
    col, col_e = d["col"], d["col_e"]
    x = np.arange(len(labels)); w = 0.38
    fig, ax1 = plt.subplots(figsize=(COL_W, 2.5))
    ax1.bar(x - w/2, thr, w, yerr=thr_e, capsize=2.5,
            color=PALETTE["blue"], label="throughput", error_kw={"lw": 0.7})
    ax1.set_ylabel("throughput", color=PALETTE["blue"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("ablation variant")
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, col, w, yerr=col_e, capsize=2.5,
            color=PALETTE["red"], label="PU collision", error_kw={"lw": 0.7})
    ax2.set_ylabel("PU collision rate", color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"])
    ax2.grid(False)
    fig.tight_layout()
    save(fig, "fig_ablation.pdf")


# ---------------------------------------------------------------- Q5 freq/time
def fig_freq_time():
    d = load("freq_vs_time.npz")
    x = d["levels"]
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W, 2.3))
    ax[0].errorbar(x, d["freq_thru"], yerr=d["freq_thr_e"], fmt="o-",
                   color=PALETTE["red"], capsize=2.5, label="frequency error")
    ax[0].errorbar(x, d["time_thru"], yerr=d["time_thr_e"], fmt="s-",
                   color=PALETTE["blue"], capsize=2.5, label="temporal error")
    ax[0].set_xlabel("normalized localization error")
    ax[0].set_ylabel("access throughput")
    ax[0].legend(); panel_label(ax[0], "(a)")
    ax[1].plot(x, d["freq_coll"], "o-", color=PALETTE["red"],
               label="frequency error")
    ax[1].plot(x, d["time_coll"], "s-", color=PALETTE["blue"],
               label="temporal error")
    ax[1].set_xlabel("normalized localization error")
    ax[1].set_ylabel("PU collision rate")
    ax[1].legend(); panel_label(ax[1], "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_freq_vs_time.pdf")


# ---------------------------------------------------------------- Q6 loc sweep
def fig_localization():
    d = load("localization_sweep.npz")
    eps = d["eps"]; order = np.argsort(eps)
    eps = eps[order]; pmd = d["pmd"][order]; pfa = d["pfa"][order]
    thr = d["thr"][order]; gap = (d["thr"].max() - d["thr"])[order]
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W, 2.3))
    ax[0].plot(eps, pmd, "o-", color=PALETTE["red"], label=r"$P_{\mathrm{md}}$")
    ax[0].plot(eps, pfa, "s-", color=PALETTE["blue"], label=r"$P_{\mathrm{fa}}$")
    ax[0].set_xlabel(r"localization error $\varepsilon_{\mathrm{loc}}=1-\mathrm{IoU}$")
    ax[0].set_ylabel("induced cell-level error")
    ax[0].legend(); panel_label(ax[0], "(a)")
    ax[1].plot(eps, gap, "o", color=PALETTE["purple"], label="measured")
    A = np.vstack([eps, np.ones_like(eps)]).T
    slope, intc = np.linalg.lstsq(A, gap, rcond=None)[0]
    xs = np.linspace(eps.min(), eps.max(), 50)
    ax[1].plot(xs, slope*xs + intc, "--", color=PALETTE["gray"],
               label=f"linear fit")
    ax[1].set_xlabel(r"localization error $\varepsilon_{\mathrm{loc}}=1-\mathrm{IoU}$")
    ax[1].set_ylabel("access throughput gap")
    ax[1].legend(); panel_label(ax[1], "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_localization.pdf")


if __name__ == "__main__":
    fig_tradeoff()
    fig_cvar()
    fig_ablation()
    fig_freq_time()
    fig_localization()
    print("all figures regenerated with TWC style")
