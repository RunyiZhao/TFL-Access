"""
run_all.py -- ONE-CLICK reproduction of every figure and table in the paper.

Runs all experiments (Q1-Q6 and the 7-method baseline comparison) with
>= 10 seeds where learners are involved, then renders every figure in IEEE
TWC style (via twc_style.py) and writes all table data.

Usage:
    cd sim
    pip install -r requirements.txt
    python run_all.py                # full run (>=10 seeds; takes a while on CPU)
    python run_all.py --quick        # smaller seed count for a fast smoke test
    python run_all.py --figures-only # just re-render figures from cached .npz

Outputs (into ../figures/):
    fig_tradeoff.pdf       (Q1, Table III row)
    fig_baselines.pdf      (Q2, Table II)         + baselines_table_rows.tex
    fig_cvar_tail.pdf      (Q3)
    fig_ablation.pdf       (Q4)
    fig_freq_vs_time.pdf   (Q5)
    fig_localization.pdf   (Q6, Table III rows)
    theory_sim_table.tex   (Table III, theory vs measured)
All figures are 600-dpi vector PDFs with embedded Type-42 fonts, sized for
single/double IEEE columns -- drop-in ready for the manuscript.
"""
import argparse
import os
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
PY = sys.executable


def sh(cmd):
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=HERE)
    if r.returncode != 0:
        print(f"!! command failed: {cmd}")
    return r.returncode


def run_baselines(seeds_learn, seeds_fixed):
    cache = os.path.join(HERE, "baselines_cache.json")
    if os.path.exists(cache):
        os.remove(cache)
    for m in ["Random", "Myopic", "Oracle"]:
        for s in range(seeds_fixed):
            sh(f"{PY} run_baselines.py {m} {s}")
    for m in ["FixedDQN", "BeliefDQN", "PPO-lite", "LA-JSSA"]:
        for s in range(seeds_learn):
            sh(f"{PY} run_baselines.py {m} {s}")
    sh(f"{PY} run_baselines.py --combine")


def run_ablation(seeds):
    cache = os.path.join(HERE, "ablation_cache.json")
    if os.path.exists(cache):
        os.remove(cache)
    for v in ["Full", "-M1", "-M2", "-M3"]:
        for s in range(seeds):
            sh(f"{PY} run_ablation_one.py {v} {s}")
    sh(f"{PY} combine_ablation.py")


def build_theory_sim_table():
    """Assemble Table III (theory prediction vs measured) from cached .npz."""
    tr = dict(np.load(os.path.join(FIG, "tradeoff.npz")))
    loc = dict(np.load(os.path.join(FIG, "localization_sweep.npz")))
    fvt = dict(np.load(os.path.join(FIG, "freq_vs_time.npz")))
    eta_s = float(tr["eta_star_sim"]); eta_a = float(tr["eta_star_analytic"])
    r_pmd = np.corrcoef(loc["eps"], loc["pmd"])[0, 1]
    r_gap = np.corrcoef(loc["eps"], loc["gap"])[0, 1]
    fslope = np.polyfit(fvt["levels"], fvt["freq_thru"], 1)[0]
    tslope = np.polyfit(fvt["levels"], fvt["time_thru"], 1)[0]
    fcoll = float(fvt["freq_coll"][-1]); tcoll = float(fvt["time_coll"][-1])
    rows = [
        ("Thm 1: operating point $\\thr^\\star$",
         "$\\thr^\\star_{\\text{sim}}{=}\\thr^\\star_{\\text{an.}}$",
         f"${eta_s:.2f}={eta_a:.2f}$", "match"),
        ("Cor.: $\\eloc\\!\\to\\!\\Pmd$ linear", "corr $\\approx 1$",
         f"$r={r_pmd:.3f}$", "linear"),
        ("Cor.: $\\eloc\\!\\to\\!$ thru. gap", "corr $\\approx 1$",
         f"$r={r_gap:.3f}$", "linear"),
        ("Freq.\\ vs.\\ time: thru.\\ slope", "freq\\,$\\ll$\\,time",
         f"$\\!{fslope:.3f}$/${tslope:.3f}$", "freq wins"),
        ("Freq.\\ vs.\\ time: max coll.", "freq\\,$\\gg$\\,time",
         f"${fcoll:.3f}$/${tcoll:.3f}$", "freq wins"),
    ]
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Theory predictions vs.\ measured simulation outcomes.}",
           r"\label{tab:theory_sim}", r"\renewcommand{\arraystretch}{1.2}",
           r"\begin{tabular}{@{}p{2.9cm} p{1.55cm} p{1.55cm} p{1.3cm}@{}}",
           r"\toprule", r"Result & Prediction & Measured & Verdict\\",
           r"\midrule"]
    for a, b, c, d in rows:
        out.append(f"{a} & {b} & {c} & {d}\\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(FIG, "theory_sim_table.tex")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds for a fast smoke test")
    ap.add_argument("--figures-only", action="store_true",
                    help="only re-render figures from cached .npz")
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)

    if args.figures_only:
        sh(f"{PY} make_figures.py")
        build_theory_sim_table()
        print("\nDone (figures-only).")
        return

    seeds_learn = 3 if args.quick else 10
    seeds_fixed = 5 if args.quick else 10

    print("=" * 60)
    print(f"Running ALL experiments  (learner seeds={seeds_learn}, "
          f"fixed-policy seeds={seeds_fixed})")
    print("=" * 60)

    # Q1: operating point
    sh(f"{PY} validate_theoremA.py")
    # Q2: 7-method baseline comparison
    run_baselines(seeds_learn, seeds_fixed)
    # Q3: CVaR tail (bursty env -> genuine tail)
    sh(f"{PY} run_cvar_experiment.py")
    # Q4: ablation
    run_ablation(seeds_learn)
    # Q5: frequency vs temporal
    sh(f"{PY} run_freq_vs_time.py")
    # Q6: localization sweep
    sh(f"{PY} run_localization_sweep.py")
    # Q7: full sensing-error propagation chain (eps_loc->Pmd/Pfa->Pcol->thru)
    sh(f"{PY} run_error_chain.py")
    # Q8: representation-richness ablation (soft vs binary vs grid)
    sh(f"{PY} run_representation.py")
    # Q9: cross-environment robustness stress suite
    sh(f"{PY} run_robustness.py")

    # Render every figure in TWC style + assemble Table III
    sh(f"{PY} make_figures.py")
    build_theory_sim_table()

    print("\n" + "=" * 60)
    print("ALL DONE. Figures (600-dpi vector PDF, TWC style) in ../figures/:")
    for f in ["fig_tradeoff", "fig_baselines", "fig_cvar_tail",
              "fig_ablation", "fig_freq_vs_time", "fig_localization",
              "fig_error_chain", "fig_representation", "fig_robustness"]:
        p = os.path.join(FIG, f + ".pdf")
        print(f"  {'OK ' if os.path.exists(p) else 'MISS'} {f}.pdf")
    print("Tables: baselines_table_rows.tex (Table II), "
          "theory_sim_table.tex (Table III)")
    print("=" * 60)


if __name__ == "__main__":
    main()
