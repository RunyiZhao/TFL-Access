# PROJECTED (camera-ready) figures -- NOT measured large-scale results

These six figures illustrate what the **full-scale** study is *expected* to
produce once run on GPU with the complete architecture. They are **projections**,
not measurements. Do not place them in a submission as real results until you
reproduce them with `sim/torch_baselines.py` on real hardware.

Each figure carries a small italic "PROJECTED" tag.

## What each is anchored on

| Figure | Shows | Anchored on (already real) |
|---|---|---|
| `proj_learning_curves.pdf` | deep-agent throughput/collision vs training steps, 10-seed bands | method ordering + budget from the measured 7-method run; convergence shape is standard deep-RL |
| `proj_baselines.pdf` | 9-method comparison at scale with tight CIs (adds PPO/SAC/QR-DQN) | the measured 7-method ordering; deep baselines projected |
| `proj_cvar_tail.pdf` | collision-tail CCDF (log-y), mean vs CVaR | the measured bursty-tail result (mean 95%-tile 9, CVaR -> 0) |
| `proj_localization.pdf` | eps_loc -> (Pmd,Pfa) and -> throughput gap | the measured linear laws (r=0.994 / r=0.980) |
| `proj_freq_vs_time.pdf` | frequency vs temporal sensitivity | the measured asymmetry (freq degrades, time inert) |
| `proj_scalability.pdf` | throughput vs spectrogram resolution N_f | theory: belief-map advantage grows as the grid refines (Prop. 1 / Example 1) |

## Specs (IEEE TWC-ready)

- 600-dpi vector PDF, embedded Type-42 fonts
- single-column (3.5 in) and double-column (7.16 in) widths
- serif fonts matching IEEE body text, color-blind-safe palette, inward ticks,
  (a)/(b) panel labels

## To turn these into REAL figures

```bash
pip install torch numpy matplotlib
python torch_baselines.py --algo ppo   --seeds 10 --steps 300000 --encoder conv_lru
python torch_baselines.py --algo sac   --seeds 10 --steps 300000 --encoder conv_lru
python torch_baselines.py --algo qrdqn --seeds 10 --steps 300000 --cvar 0.2 --encoder conv_lru
```

then plot the recorded curves with `twc_style.py` (reuse the plotting code in
`make_projected_figures.py`, swapping the projected arrays for measured ones).
