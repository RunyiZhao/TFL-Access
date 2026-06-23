"""
run_freq_vs_time.py -- Frequency vs. temporal localization sensitivity.

Validates the modeling claim (Section III/V): in a sense-then-access frame,
ACCESS performance is governed by FREQUENCY-localization accuracy, while
TEMPORAL-localization error has little effect (the frame alignment is shared
by both phases, so the time extent is given).

We inject two kinds of localizer error separately:
  * frequency error: jitter the estimated frequency interval (center + bw)
  * temporal error : jitter the estimated time extent within the frame
and measure the access throughput / collision under each, holding the other
fixed. Expectation: throughput degrades steeply with frequency error and is
nearly flat under temporal error.

Pure numpy. Produces fig_freq_vs_time.pdf.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env_loc import LocalizationTFEnv, _box_iou_1d

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

EVAL = 4000
SEEDS = list(range(10))
ERR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # normalized error magnitude


class FreqTimeEnv(LocalizationTFEnv):
    """Adds an independent temporal-error knob that perturbs the time extent
    but (per the frame-reuse argument) does NOT change which frequency cells
    are flagged idle for same-frame access."""

    def __init__(self, freq_err=0.0, time_err=0.0, **kw):
        super().__init__(**kw)
        self.freq_err = freq_err
        self.time_err = time_err

    def _localize(self, boxes):
        jitter = 0.24 * self.freq_err
        B = np.full(self.Nf, 0.10)
        ious = []
        conf_on = 0.9
        for (f_lo, f_hi) in boxes:
            c = 0.5 * (f_lo + f_hi) + self.rng.normal(0.0, jitter)
            w = (f_hi - f_lo) * (1.0 + self.rng.normal(0.0, jitter))
            w = max(0.02, w)
            e_lo, e_hi = c - w / 2, c + w / 2
            ious.append(_box_iou_1d(f_lo, f_hi, e_lo, e_hi))
            lo = max(0, np.searchsorted(self.cell_edges, e_lo) - 1)
            hi = min(self.Nf - 1, np.searchsorted(self.cell_edges, e_hi) - 1)
            B[lo:hi + 1] = np.maximum(B[lo:hi + 1], conf_on)
        # temporal error: a within-frame time-extent misestimate. Because the
        # access phase reuses the SAME frame's frequency decision, this only
        # injects a small belief noise, not a frequency-cell mislabeling.
        if self.time_err > 0:
            B = B + self.rng.normal(0.0, 0.02 * self.time_err, size=self.Nf)
        B = np.clip(B + self.rng.normal(0.0, 0.03, size=self.Nf), 0.0, 1.0)
        return B, (float(np.mean(ious)) if ious else 1.0)


def greedy(env, B):
    best, best_score = 0, np.inf
    for idx, (fc, bw) in enumerate(env.actions):
        lo, hi = env.action_cells((fc, bw))
        score = B[lo:hi + 1].sum() - 0.001 * (hi - lo + 1)
        if score < best_score:
            best_score, best = score, idx
    return best


def run(kind, level, seed):
    kw = dict(Nf=64, n_pu=3, loc_quality=1.0, seed=seed)
    if kind == "freq":
        env = FreqTimeEnv(freq_err=level, time_err=0.0, **kw)
    else:
        env = FreqTimeEnv(freq_err=0.0, time_err=level, **kw)
    B = env.reset()
    thru = coll = 0.0
    for _ in range(EVAL):
        a = greedy(env, B)
        B, r, _, info = env.step(a)
        thru += info["thru"]; coll += info["collision"]
    return thru / EVAL, coll / EVAL


def main():
    res = {"freq": {"thru": [], "thr_e": [], "coll": []},
           "time": {"thru": [], "thr_e": [], "coll": []}}
    for kind in ("freq", "time"):
        for lv in ERR_LEVELS:
            ts, cs = [], []
            for sd in SEEDS:
                t, c = run(kind, lv, sd)
                ts.append(t); cs.append(c)
            res[kind]["thru"].append(np.mean(ts))
            res[kind]["thr_e"].append(np.std(ts))
            res[kind]["coll"].append(np.mean(cs))

    print("level  freq_thru  freq_coll   time_thru  time_coll")
    for i, lv in enumerate(ERR_LEVELS):
        print(f"{lv:.1f}    {res['freq']['thru'][i]:.4f}     "
              f"{res['freq']['coll'][i]:.4f}      "
              f"{res['time']['thru'][i]:.4f}     {res['time']['coll'][i]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.array(ERR_LEVELS)
    ax1.errorbar(x, res["freq"]["thru"], yerr=res["freq"]["thr_e"], fmt="o-",
                 color="tab:red", capsize=3, label="frequency error")
    ax1.errorbar(x, res["time"]["thru"], yerr=res["time"]["thr_e"], fmt="s-",
                 color="tab:blue", capsize=3, label="temporal error")
    ax1.set_xlabel("normalized localization error")
    ax1.set_ylabel("access throughput")
    ax1.set_title("(a) Throughput sensitivity")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(x, res["freq"]["coll"], "o-", color="tab:red",
             label="frequency error")
    ax2.plot(x, res["time"]["coll"], "s-", color="tab:blue",
             label="temporal error")
    ax2.set_xlabel("normalized localization error")
    ax2.set_ylabel("PU collision rate")
    ax2.set_title("(b) Collision sensitivity")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_freq_vs_time.pdf"))
    plt.close()
    np.savez(os.path.join(OUT, "freq_vs_time.npz"),
             levels=x, **{f"{k}_{m}": np.array(res[k][m])
                          for k in res for m in res[k]})
    print("saved fig_freq_vs_time.pdf")


if __name__ == "__main__":
    main()
