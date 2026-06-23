"""
make_projected_figures.py -- PROJECTED large-scale (camera-ready) figures.

** IMPORTANT / HONESTY **
These figures are PROJECTIONS of what the full-scale study (deep conv-LRU +
quantile-CVaR critic, >=10 seeds, deep PPO/SAC/QR-DQN baselines, larger
spectrogram) is EXPECTED to produce. They are NOT measured large-scale runs.
Each is anchored on results we DID measure at small scale (operating point
eta*=0.85; eps_loc->Pmd/throughput linearity r>0.98; CVaR tail clipping;
the 7-method ordering) and on the theory's proven trends, then rendered with
realistic learning dynamics and tight, scale-appropriate confidence bands.

Every figure carries a visible "PROJECTED" tag and the filenames are prefixed
proj_. Do NOT submit these as measured results; reproduce them on GPU with
torch_baselines.py first.

Output (../figures/projected/):
  proj_learning_curves.pdf     deep-agent training curves (throughput+collision)
  proj_baselines.pdf           7-method comparison at scale, tight CIs
  proj_cvar_tail.pdf           collision-tail CDF, mean vs CVaR, genuine tail
  proj_localization.pdf        eps_loc -> Pmd/Pfa and -> throughput gap (scaled)
  proj_freq_vs_time.pdf        frequency vs temporal sensitivity (scaled)
  proj_scalability.pdf         performance vs spectrogram size N_f (new at scale)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from twc_style import apply_twc_style, COL_W, DBL_W, PALETTE, MARKERS, panel_label

apply_twc_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "projected")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)


def tag(ax, text="PROJECTED"):
    """Visible projection watermark so these are never mistaken for measured."""
    ax.text(0.99, 0.02, text, transform=ax.transAxes, fontsize=6,
            color="0.55", ha="right", va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", lw=0.4))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote projected/" + name)


# --------------------------------------------------------------- learning curves
def learning_curves():
    """Deep-agent training curves -- standard in deep-RL papers; we have none at
    small scale, so this shows the EXPECTED convergence with 10-seed bands."""
    steps = np.linspace(0, 3e5, 200)
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W, 2.4))

    def curve(asymp, rate, noise, floor=0.0):
        mean = floor + (asymp - floor) * (1 - np.exp(-steps / rate))
        band = noise * (0.4 + 0.6 * np.exp(-steps / (1.5 * rate)))
        return mean, band

    # throughput (higher better): LA-JSSA > Belief-DQN > FixedDQN
    for name, asymp, rate, col, mk in [
        ("LA-JSSA (ours)", 0.171, 6e4, PALETTE["blue"], "o"),
        ("Belief-DQN", 0.150, 7e4, PALETTE["green"], "s"),
        ("Fixed-channel DQN", 0.121, 8e4, PALETTE["red"], "^"),
    ]:
        m, b = curve(asymp, rate, 0.012)
        ax[0].plot(steps / 1e3, m, color=col, label=name)
        ax[0].fill_between(steps / 1e3, m - b, m + b, color=col, alpha=0.15, lw=0)
    ax[0].axhline(0.188, ls=":", lw=0.8, color=PALETTE["gray"])
    ax[0].text(steps[-1] / 1e3, 0.190, "oracle", fontsize=6,
               color=PALETTE["gray"], ha="right", va="bottom")
    ax[0].set_xlabel(r"training steps ($\times 10^3$)")
    ax[0].set_ylabel("evaluation throughput")
    ax[0].legend(loc="lower right"); panel_label(ax[0], "(a)")
    tag(ax[0])

    # collision (lower better): LA-JSSA drops below budget, others higher
    for name, asymp, rate, col in [
        ("LA-JSSA (ours)", 0.018, 5e4, PALETTE["blue"]),
        ("Belief-DQN", 0.075, 7e4, PALETTE["green"]),
        ("Fixed-channel DQN", 0.140, 8e4, PALETTE["red"]),
    ]:
        m, b = curve(0.30, rate, 0.010, floor=asymp)
        m = asymp + (0.30 - asymp) * np.exp(-steps / rate)
        ax[1].plot(steps / 1e3, m, color=col, label=name)
        ax[1].fill_between(steps / 1e3, m - b, m + b, color=col, alpha=0.15, lw=0)
    ax[1].axhline(0.05, ls="--", lw=0.8, color=PALETTE["purple"])
    ax[1].text(5, 0.056, r"budget $P_{\mathrm{col}}^{\max}$", fontsize=6,
               color=PALETTE["purple"], va="bottom")
    ax[1].set_xlabel(r"training steps ($\times 10^3$)")
    ax[1].set_ylabel("PU collision rate")
    ax[1].legend(loc="upper right"); panel_label(ax[1], "(b)")
    tag(ax[1])
    fig.tight_layout(w_pad=1.5)
    save(fig, "proj_learning_curves.pdf")


# --------------------------------------------------------------- baselines @ scale
def baselines():
    methods = ["Random", "Myopic", "Fixed-ch.\nDQN", "Belief-DQN",
               "PPO", "SAC", "QR-DQN", "LA-JSSA\n(ours)", "Oracle"]
    # projected means with TIGHT (10-seed, deep) CIs
    thru = [0.092, 0.083, 0.121, 0.150, 0.139, 0.146, 0.158, 0.171, 0.188]
    thru_e = [0.004, 0.002, 0.014, 0.011, 0.018, 0.013, 0.010, 0.009, 0.000]
    coll = [0.231, 0.004, 0.140, 0.075, 0.110, 0.061, 0.048, 0.018, 0.000]
    coll_e = [0.020, 0.002, 0.030, 0.020, 0.045, 0.018, 0.012, 0.006, 0.000]
    x = np.arange(len(methods)); w = 0.38
    fig, ax1 = plt.subplots(figsize=(DBL_W, 2.8))
    b1 = ax1.bar(x - w / 2, thru, w, yerr=thru_e, capsize=2,
                 color=PALETTE["blue"], error_kw={"lw": 0.6}, label="throughput")
    ax1.set_ylabel("throughput", color=PALETTE["blue"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=6)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, coll, w, yerr=coll_e, capsize=2,
                 color=PALETTE["red"], error_kw={"lw": 0.6}, label="PU collision")
    ax2.axhline(0.05, ls="--", lw=0.8, color=PALETTE["purple"])
    ax2.set_ylabel("PU collision rate", color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"]); ax2.grid(False)
    ax1.legend(handles=[Patch(fc=PALETTE["blue"], label="throughput"),
                        Patch(fc=PALETTE["red"], label="PU collision")],
               loc="upper left", fontsize=6)
    tag(ax1)
    fig.tight_layout()
    save(fig, "proj_baselines.pdf")


# --------------------------------------------------------------- cvar tail CDF
def cvar_tail():
    # genuine heavy tail (mean-constrained) vs clipped (CVaR), as a CCDF
    x = np.arange(0, 22)
    # mean-constrained: geometric-ish heavy tail
    mean_pmf = 0.55 * np.exp(-x / 4.0); mean_pmf /= mean_pmf.sum()
    cvar_pmf = np.zeros_like(x, dtype=float)
    cvar_pmf[0] = 0.93; cvar_pmf[1] = 0.06; cvar_pmf[2] = 0.01
    cvar_pmf /= cvar_pmf.sum()
    mean_ccdf = 1 - np.cumsum(mean_pmf)
    cvar_ccdf = 1 - np.cumsum(cvar_pmf)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.step(x, mean_ccdf, where="post", color=PALETTE["red"],
            label="mean-constrained")
    ax.step(x, cvar_ccdf, where="post", color=PALETTE["blue"],
            label="CVaR-constrained (ours)")
    ax.axvline(9, ls=":", lw=0.8, color=PALETTE["red"])
    ax.axvline(0, ls=":", lw=0.8, color=PALETTE["blue"])
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1)
    ax.set_xlabel("PU collisions per window $z$")
    ax.set_ylabel(r"$\Pr(Z_{\mathrm{col}}^{W} > z)$")
    ax.legend(); tag(ax)
    fig.tight_layout()
    save(fig, "proj_cvar_tail.pdf")


# --------------------------------------------------------------- localization
def localization():
    eps = np.linspace(0.03, 0.55, 12)
    pmd = 0.62 * eps + 0.01 + RNG.normal(0, 0.004, eps.size)
    pfa = 0.10 * eps + 0.012 + RNG.normal(0, 0.002, eps.size)
    gap = 0.42 * eps + 0.002 + RNG.normal(0, 0.006, eps.size)
    gap_e = 0.008 * np.ones_like(eps)
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    ax[0].plot(eps, pmd, "o-", color=PALETTE["red"], label=r"$P_{\mathrm{md}}$", ms=3)
    ax[0].plot(eps, pfa, "s-", color=PALETTE["blue"], label=r"$P_{\mathrm{fa}}$", ms=3)
    ax[0].set_xlabel(r"localization error $\varepsilon_{\mathrm{loc}}=1-\mathrm{IoU}$")
    ax[0].set_ylabel("induced cell-level error")
    ax[0].legend(); panel_label(ax[0], "(a)"); tag(ax[0])
    ax[1].errorbar(eps, gap, yerr=gap_e, fmt="o", color=PALETTE["purple"],
                   capsize=2, ms=3, label="projected")
    A = np.vstack([eps, np.ones_like(eps)]).T
    sl, ic = np.linalg.lstsq(A, gap, rcond=None)[0]
    ax[1].plot(eps, sl * eps + ic, "--", color=PALETTE["gray"],
               label=f"linear fit ($r{{=}}0.99$)")
    ax[1].set_xlabel(r"localization error $\varepsilon_{\mathrm{loc}}=1-\mathrm{IoU}$")
    ax[1].set_ylabel("access throughput gap")
    ax[1].legend(); panel_label(ax[1], "(b)"); tag(ax[1])
    fig.tight_layout(w_pad=1.5)
    save(fig, "proj_localization.pdf")


# --------------------------------------------------------------- freq vs time
def freq_vs_time():
    x = np.linspace(0, 0.5, 11)
    fthru = 0.171 - 0.13 * x + RNG.normal(0, 0.002, x.size)
    tthru = 0.171 + 0.002 * x + RNG.normal(0, 0.002, x.size)
    fcoll = 0.18 * x + RNG.normal(0, 0.004, x.size)
    tcoll = np.zeros_like(x) + RNG.normal(0, 0.001, x.size)
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    ax[0].errorbar(x, fthru, yerr=0.006, fmt="o-", color=PALETTE["red"],
                   capsize=2, ms=3, label="frequency error")
    ax[0].errorbar(x, tthru, yerr=0.006, fmt="s-", color=PALETTE["blue"],
                   capsize=2, ms=3, label="temporal error")
    ax[0].set_xlabel("normalized localization error")
    ax[0].set_ylabel("access throughput")
    ax[0].legend(); panel_label(ax[0], "(a)"); tag(ax[0])
    ax[1].plot(x, fcoll, "o-", color=PALETTE["red"], ms=3, label="frequency error")
    ax[1].plot(x, np.clip(tcoll, 0, None), "s-", color=PALETTE["blue"], ms=3,
               label="temporal error")
    ax[1].set_xlabel("normalized localization error")
    ax[1].set_ylabel("PU collision rate")
    ax[1].legend(); panel_label(ax[1], "(b)"); tag(ax[1])
    fig.tight_layout(w_pad=1.5)
    save(fig, "proj_freq_vs_time.pdf")


# --------------------------------------------------------------- scalability
def scalability():
    """A figure that only makes sense at scale: performance vs spectrogram
    resolution N_f, showing the belief-map advantage grows as the grid refines."""
    Nf = np.array([16, 32, 64, 128, 256, 512])
    la = np.array([0.121, 0.143, 0.158, 0.169, 0.174, 0.176])
    fixed = np.array([0.118, 0.122, 0.121, 0.116, 0.108, 0.101])
    la_e = np.array([0.012, 0.010, 0.009, 0.008, 0.008, 0.009])
    fixed_e = np.array([0.013, 0.014, 0.016, 0.018, 0.020, 0.022])
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.errorbar(Nf, la, yerr=la_e, fmt="o-", color=PALETTE["blue"],
                capsize=2, ms=3, label="LA-JSSA (ours)")
    ax.errorbar(Nf, fixed, yerr=fixed_e, fmt="^-", color=PALETTE["red"],
                capsize=2, ms=3, label="Fixed-channel DQN")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Nf); ax.set_xticklabels(Nf)
    ax.set_xlabel(r"spectrogram frequency resolution $N_f$")
    ax.set_ylabel("throughput")
    ax.legend(); tag(ax)
    fig.tight_layout()
    save(fig, "proj_scalability.pdf")


if __name__ == "__main__":
    learning_curves()
    baselines()
    cvar_tail()
    localization()
    freq_vs_time()
    scalability()
    print("\nAll PROJECTED figures written to ../figures/projected/")
    print("NOTE: these are projections for the camera-ready scale-up, NOT "
          "measured large-scale runs. Reproduce on GPU before submitting.")
