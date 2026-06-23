import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = os.path.join(os.path.dirname(__file__), "ablation_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
cache = json.load(open(CACHE))
variants = ["Full", "-M1", "-M2", "-M3"]
agg = {}
for v in variants:
    ts, cs = [], []
    for k, (t, c) in cache.items():
        if k.split("|")[0] == v:
            ts.append(t); cs.append(c)
    agg[v] = (np.mean(ts), np.std(ts), np.mean(cs), np.std(cs), len(ts))
    print(f"{v:5s} n={agg[v][4]} thru={agg[v][0]:.4f}±{agg[v][1]:.4f} "
          f"coll={agg[v][2]:.4f}±{agg[v][3]:.4f}")

labels = variants
thr  = [agg[v][0] for v in labels]; thr_e = [agg[v][1] for v in labels]
col  = [agg[v][2] for v in labels]; col_e = [agg[v][3] for v in labels]

x = np.arange(len(labels)); w = 0.38
fig, ax1 = plt.subplots(figsize=(6.4, 4))
ax1.bar(x - w/2, thr, w, yerr=thr_e, capsize=3, color="tab:blue",
        label="throughput")
ax1.set_ylabel("normalized throughput", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_xlabel("variant")
ax2 = ax1.twinx()
ax2.bar(x + w/2, col, w, yerr=col_e, capsize=3, color="tab:red",
        label="PU collision")
ax2.set_ylabel("PU collision rate", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax1.set_title("Component ablation: removing M1/M2/M3 degrades performance")
fig.tight_layout()
plt.savefig(os.path.join(OUT, "fig_ablation.pdf")); plt.close()
np.savez(os.path.join(OUT, "ablation.npz"),
         labels=labels, thr=thr, thr_e=thr_e, col=col, col_e=col_e)
print("saved fig_ablation.pdf")
