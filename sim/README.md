# LA-JSSA Simulation Code

Reproduces the experiments in *"From Time–Frequency Localization to
Risk-Sensitive Access: A Belief-Map Closed Loop for Wideband Cognitive
Radio."* All code is **pure NumPy + Matplotlib** (no GPU / no PyTorch
required), so every experiment runs on a normal CPU.

## Install

```bash
pip install -r requirements.txt
```

Requirements: Python >= 3.9, numpy, matplotlib. That is all.

## One-click reproduction

```bash
python run_all.py              # runs ALL experiments (>=10 seeds) and renders
                               # every figure/table in IEEE TWC style
python run_all.py --quick      # fewer seeds, fast smoke test
python run_all.py --figures-only   # just re-render figures from cached .npz
```

`run_all.py` produces, in `../figures/`, all six data figures as 600-dpi
vector PDFs with embedded Type-42 fonts (single/double IEEE column widths),
plus `baselines_table_rows.tex` (Table II) and `theory_sim_table.tex`
(Table III) -- drop-in ready for the manuscript.

## Directory layout

```
sim/
  env.py                  # ContinuousTFEnv: continuous-bandwidth, non-grid PU env
  env_loc.py              # LocalizationTFEnv: belief map RENDERED from estimated
                          #   PU frequency intervals with controllable IoU error
  agents.py               # numpy DQN / BeliefDQN / FixedChannelDQN / Myopic / Oracle
  agents_cvar.py          # CVaRDistDQN (QR-style distributional + CVaR)
  agents_pg.py            # PPOLiteAgent (clipped PG + Lagrangian) / RandomAgent
  twc_style.py            # shared IEEE TWC figure styling
  make_figures.py         # re-style all data figures from cached .npz
  validate_theoremA.py    # Q1: operating-point eta* (theory vs measured)
  run_baselines.py        # Q2: 7-method comparison (Table II, fig_baselines)
  run_cvar_experiment.py  # Q3: CVaR tail-risk reduction
  run_ablation_one.py     # Q4: run one ablation variant/seed (cached)
  combine_ablation.py     # Q4: aggregate cached ablation -> figure
  run_freq_vs_time.py     # Q5: frequency- vs temporal-localization sensitivity
  run_localization_sweep.py # Q6: localization accuracy (IoU) -> access
  torch_baselines.py      # GPU-ready PyTorch PPO/SAC/QR-DQN (camera-ready scale-up)
```

> **On `torch_baselines.py`:** this is the GPU scale-up path (deep
> PPO/SAC/QR-DQN with the conv-frequency/LRU encoder). It requires PyTorch and
> is **not** run in the paper's CPU sandbox; the numpy `run_baselines.py` is
> what produces the figures in the paper. Run the torch version on a GPU
> machine for the camera-ready, multi-seed deep comparison.

## Reproducing each paper figure

| Paper item                       | Command                                   | Output |
|----------------------------------|-------------------------------------------|--------|
| Fig. (trade-off), Q1, Table III  | `python validate_theoremA.py`             | `figures/fig_tradeoff.pdf`, `tradeoff.npz` |
| Fig. (baselines), Q2, Table II   | see "Baseline comparison" below           | `figures/fig_baselines.pdf`, `baselines_table_rows.tex` |
| Fig. (CVaR tail), Q3             | `python run_cvar_experiment.py`           | `figures/fig_cvar_tail.pdf`, `cvar_results.npz` |
| Fig. (ablation), Q4             | see "Ablation" below                       | `figures/fig_ablation.pdf` |
| Fig. (freq vs time), Q5         | `python run_freq_vs_time.py`              | `figures/fig_freq_vs_time.pdf` |
| Fig. (localization), Q6, Table III | `python run_localization_sweep.py`      | `figures/fig_localization.pdf`, `localization_sweep.npz` |

All scripts write PDFs and `.npz` data into `../figures/`. To re-style every
figure from cached data without re-running experiments:

```bash
python make_figures.py      # regenerates all 5 data figures (TWC style)
```

### Baseline comparison (Q2, Table II)

Seven methods on a matched environment/state/action set. Run each
method/seed (cached), then aggregate:

```bash
for m in Random Myopic Oracle; do for s in 0 1 2 3 4; do python run_baselines.py $m $s; done; done
for m in FixedDQN BeliefDQN PPO-lite LA-JSSA; do for s in 0 1 2; do python run_baselines.py $m $s; done; done
python run_baselines.py --combine     # -> figures/fig_baselines.pdf + table rows
```

Methods: `Random`, `Myopic`, `FixedDQN`, `BeliefDQN`, `PPO-lite`, `LA-JSSA`,
`Oracle`. Results cache to `baselines_cache.json`.

### Ablation (Q4)

The ablation runs one (variant, seed) at a time and caches results, so it
fits in modest time budgets:

```bash
for v in Full -M1 -M2 -M3; do
  for s in 0 1; do
    python run_ablation_one.py $v $s
  done
done
python combine_ablation.py        # -> figures/fig_ablation.pdf
```

(A single-process convenience driver is in `run_ablation.py`.)

## Notes on honesty / scope

* The agents are compact NumPy implementations, not the full
  conv-frequency/LRU encoder described in the paper. Results validate the
  **qualitative** orderings predicted by the theory; they are not a final
  performance benchmark.
* The detector ROC and PU process are synthetic and chosen to **isolate the
  modeling assumptions**, not tuned to flatter LA-JSSA.
* `env_loc.py` is the localization-aware environment: cell-level (Pmd, Pfa)
  are a *consequence* of frequency-localization geometry (IoU error), matching
  the paper's frequency-localization error model. Temporal error is
  intentionally inert (same-frame sense->access reuse).
* Random seeds are fixed in each script for reproducibility.

## Quick smoke test

```bash
cd sim
python -c "from env_loc import LocalizationTFEnv; \
e=LocalizationTFEnv(seed=0); \
print(e.measure_loc_to_cell(0.5, n=500, seed=1))"
```
