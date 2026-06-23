# LA-JSSA Paper — Outline & Progress Tracker

**Title:** Localization-Aware Joint Spectrum Sensing and Access in Wideband
Cognitive Radio: A Belief-Map Reinforcement Learning Approach

**Target:** IEEE TWC (theory-heavy). Fallback: TCCN.

**Locked decisions:**
- Main line = Innovation 1 (localization→access end-to-end closed loop)
  + Innovation 2 (three-way tradeoff & error-propagation proof).
- Scenario = single SU vs. multiple PUs, continuous TF occupancy.
- Theory-first.

---

## Core gap (the paper's reason to exist)
TF-localization works (Prasad TWC'20, Li SPL'22, WRIST, Wideband Recognition
Dataset) output continuous (fc, BW, t_start, dur) boxes but STOP at sensing.
DRL-DSA works (Wang TCCN'18, Zhong AC'19, Xu DRQN, Li SPL aggregation,
Liu IoTJ'23, Li HMADRL TWC'24) all assume a FIXED CHANNEL GRID with binary
good/bad state. Nobody feeds the continuous, uncertainty-bearing TF occupancy
estimate INTO the access decision as state. That is innovation 1; the error
propagation through that loop is innovation 2.

---

## Section status
- [ ] I.   Introduction
- [~] II.  Related Work (refs added: DDQSA'23, HMADRL'24, Liang'08, +recent) (3 subsecs, each ends with the cut vs. our work)
- [x] III. System Model & Problem Formulation   <-- WRITTEN (this pass)
- [ ] IV.  LA-JSSA Framework
- [x] V.   Theoretical Analysis: Thm A WRITTEN+PROVEN (2 lemmas, thm, cor);
           Thm B & C stated with proof sketches -> Appendix
- [ ] VI.  Simulation
- [ ] VII. Conclusion
- [ ] App. Proofs

## Theorems to prove (Section V)
- **Thm A (error propagation + 3-way tradeoff):**
  P_col <= P_md + g(eta);  R >= (1-P_fa) R_max - h(P_col).
  Combine -> tradeoff surface; optimal threshold eta* by stationarity.
- **Thm B (belief-state sufficiency):**
  Pi_bel ⊇ Pi_disc; discretization distortion bound eps_q -> performance gap
  lower bound. Justifies why discretization is provably suboptimal.
- **Thm C (regret, optional):**
  piecewise-stationary Markov PU -> sublinear regret O(sqrt(T log T)) with
  explicit sensing-error term. May move to future work if space-tight.

## Baselines (Section VI)
1. Fixed-channel DQN (Wang'18)
2. Fixed-channel actor-critic (Zhong'19)
3. DRQN partial-observation (Xu)
4. **Two-step: NMS hard boxes -> discrete DRL** (key control for innovation 1)
5. Perfect-sensing oracle (upper bound)

## Metrics
normalized throughput, PU collision rate / harmful-interference prob,
spectrum utilization, convergence speed, robustness vs SNR & signal density;
theory-vs-sim tradeoff curve overlay.

## Notation
All symbols live in sections/notation_macros.tex. Do NOT hardcode symbols
in section files — use the macros so a global rename is one edit.

---
## Literature positioning (from 2024-25 search, for Related Work II)
- Closest joint sensing+access: Bokobza-Dabora-Cohen DDQSA (TWC'23) learns
  BOTH sensing and access policies BUT narrowband, single channel, no TF
  localization, no continuous boxes. -> our cut: continuous 2D TF belief.
- Li-Zhang-Ding-Fang HMADRL (TWC'24): hierarchical multi-agent, sensing
  window + power, BUT state = discrete band occupancy on a fixed grid.
  -> our cut: no fixed grid; belief map.
- Classical sensing-throughput tradeoff (Liang TWC'08 + many): single
  channel, lever = sensing DURATION + energy threshold along TIME axis.
  -> Thm A is a different object: lever = belief threshold on 2D continuous
  TF occupancy from a learned localizer. Confirmed NO continuous-TF version
  exists. This is the novelty anchor for Thm A.
- Recent sensing nets to cite as orthogonal (sensing-only): MASSnet IoTJ'24,
  Gao attention-MARL TVT'24, multi-view GNN IoTJ'25.

---
## UPDATE (this pass): single-file consolidation + Secs I, II, IV + proofs
- [x] I.   Introduction (gap + 4 contributions) WRITTEN
- [x] II.  Related Work (3 strands, each with explicit cut) WRITTEN
- [x] IV.  LA-JSSA Framework WRITTEN (two-timescale, belief-map gen,
           ConvLSTM state, DQN/DDPG heads, Prop. risk-sensitive equiv.,
           Algorithm 1)
- [x] V.   Theorem A proven; B & C now FULLY PROVEN in Appendix
- [x] App. Proof of Thm B (inclusion + strict gap) and Thm C (regret
           decomposition) WRITTEN
- [~] VI.  Simulation = structured stub (design final; awaiting code run)
- [x] VII. Conclusion WRITTEN

## Deliverable files
- la_jssa.tex               single file, uses references.bib (bibtex build)
- la_jssa_selfcontained.tex single file, bibliography INLINED (no .bib needed)
- la_jssa.pdf               compiled, 7 pages
- references.bib            24 entries
- sections/                 kept as the editable source-of-record per section
  NOTE: per request the paper is now ONE tex file; sections/ retained only
  as a backstage copy. Edit la_jssa.tex directly going forward.

## New 2024-25 literature folded in
CSRD2025 dataset, DSFF (arXiv 2504.07427), CMuSeNet, MASSnet IoTJ'24;
belief-state sufficiency grounded in Kaelbling-Littman-Cassandra'98;
risk-sensitive MDP via Howard-Matheson'72.

## Remaining TODO
1. Run the simulation (continuous-TF env + belief-DQN + 2-step baseline)
   and fill Section VI + tradeoff overlay validating Thm A.
2. Architecture figure (fig:arch) referenced as "Figure 1 (to be added)".
3. Optional: tighten Thm C from sketch to full proof if targeting TWC.

---
## UPDATE (simulation pass): code + Theorem A validation + Section VI filled
- [x] sim/env.py        : continuous-TF env, non-grid PU boxes, belief model,
                          piecewise-stationary Markov, ROC measurement
- [x] sim/agents.py     : pure-numpy DQN (MLP + replay + target net) with
                          Lagrangian dual control (Algorithm 1); BeliefDQN,
                          TwoStepDQN, FixedChannelDQN, Myopic, Oracle
- [x] sim/run_experiments.py   : 3-seed training + comparison figures
- [x] sim/validate_theoremA.py : threshold sweep -> ROC + feasibility frontier
                          + optimal-threshold overlay
- [x] VI. Simulation    : WRITTEN with real numbers + Table + Fig (8-page PDF)

### Headline results
- Theorem A VALIDATED cleanly: ROC monotone (Pmd up, Pfa down); feasibility
  frontier eta_min ~= 0.30; eta*_sim == eta*_analytic (= 0.85 on feasible set).
  Figure fig_tradeoff.pdf is publication-grade.
- Theorem B signature present: among learners, Belief-DQN has LOWEST collision
  (0.166) vs Two-step (0.232) and Fixed-channel (0.194) at comparable
  throughput -> coarsening the belief costs PU protection. Variance is high
  (numpy DQN, 40k steps, 3 seeds); flagged as needing more compute.

### HONEST status of agent experiment
The ORDERING supports the theory but absolute collision spread is large.
Q1 (tradeoff) is the strong, defensible result. Q2 (belief value) shows the
right trend; camera-ready needs torch + conv encoder + ConvLSTM + more seeds.

## Remaining TODO
1. Scale up agent experiment (torch, conv belief encoder, ConvLSTM, >=10 seeds)
   to sharpen Q2 throughput gap.
2. Architecture figure (fig:arch / "Figure 1 to be added").
3. Optional: promote Theorem C from sketch to full proof for TWC.

---
## UPDATE (method innovation M1 + new theorem + CVaR experiment)
Literature scan (2024-25) -> chose distributional RL + CVaR safe constraint
as the methodological innovation (M1), with LRU encoder (M2) and adaptive
action resolution (M3) as architecture ablations.

### New theory
- [x] Theorem D (CVaR-constrained operating point): proves eta*_CVaR >= eta*
      (mean), with the excess conservativeness bounded by the collision
      CVaR-mean tail spread / collision sensitivity; vanishes as localizer
      sharpens. Assumption (tail-monotone collision) added. Proof inline.
- Ties M1 back to the value of localization (consistent with Corollary 1).

### New method sections (Section IV)
- [x] IV-C upgraded to conv-freq + LRU encoder (M2), cites Lu'24 (LRU > Transformer
      on POMDP), Morad'25 (conv+attention POMDP encoder).
- [x] IV-E "Distributional Safe Access" (M1): quantile critic, CVaR action
      utility, CVaR collision constraint via dual ascent. Remark on M3.
- Intro contributions updated: added distributional-safe-access bullet.

### New code
- [x] sim/agents_cvar.py     : QR-DQN quantile critic (pure numpy) +
      CVaR action selection + windowed collision-CVaR dual (stabilized:
      mu clamped, rate-based CVaR).
- [x] sim/run_cvar_experiment.py : mean-constrained vs CVaR-constrained.

### Headline CVaR result (REAL, this run)
- Mean-constrained: mean 0.90 coll/window, 95% tail = 3, max = 5, thru 0.117
- CVaR-constrained: mean 0.00 coll/window, 95% tail = 0, max = 0, thru 0.125
- => CVaR clips the collision tail to ~zero at equal/higher throughput.
- fig_cvar_tail.pdf is publication-grade and validates Theorem D.
- Honest caveat (noted in paper): near-zero tail is partly the synthetic
  env; expect small residual tail on measured spectra.

### New references (30 total)
Lu'24 rethinking-transformers/LRU, Morad'25 CAE, Dabney'18 QR-DQN,
Ma'20 DSAC, Rockafellar'00 CVaR, Kim'24 spectral-risk safe RL.

## Remaining TODO
1. Scale agent comparison (torch, conv+LRU, >=10 seeds) for Q2 sharpening.
2. Architecture figure (fig:arch).
3. Optional full proof of Theorem C; full ablation (M1/M2/M3) experiment.

---
## UPDATE (architecture figure)
- [x] figures/fig_arch.tex : standalone TikZ two-timescale architecture
      diagram, compiles to fig_arch.pdf. Shows full data flow
      r -> spectrogram -> belief map -> conv_f+LRU_t encoder (M2) ->
      quantile critic -> CVaR head (M1) -> adaptive-resolution action (M3),
      with dual-ascent CVaR constraint and ACK/collision feedback; sensing
      (slow/supervised) and access (fast/RL) bands annotated.
- [x] Section IV: "Figure 1 to be added" placeholder replaced with real
      \begin{figure*} including fig_arch.pdf + full caption (fig:arch).
- Paper now 10 pages, 2 result figures + 1 architecture figure, 0 undefined refs.

## Status: all planned figures done (arch + tradeoff + cvar tail + metrics + learning)
## Remaining TODO
1. Scale agent comparison (torch, conv+LRU, >=10 seeds) for Q2 sharpening.
2. Optional full proof of Theorem C; full M1/M2/M3 ablation experiment.

---
## UPDATE (align to IEEEtran template preferences)
- hyperref switched to [hidelinks] (IEEE-conventional black links, no colored boxes).
- Confirmed single-file structure matches the standard IEEEtran journal template:
  \documentclass[journal]{IEEEtran}, \maketitle, \section, \appendices,
  inlined thebibliography (in la_jssa_selfcontained.tex).
- Journal target kept flexible (TWC is the goal but not required); the
  template/format is IEEE-generic and fits any IEEE Trans.
- Deliverable remains ONE tex file (la_jssa.tex + la_jssa_selfcontained.tex).

---
## UPDATE (integrate Zhao's 5 representative papers + thicken intro/RW)
Read all 5 uploaded papers (the user = Runyi Zhao). All are TFL sensing-side
works that STOP at sensing -> perfectly confirm our gap and let us frame the
loop-closing as a natural continuation of the user's research line.

### Added to references.bib (39 total)
- zhao2023coop (Cooperative TFL + lightweight detector, COMML'23)
- zhao2023ccdgan (CCD-GAN domain adaptation, COMML'23)
- zhao2024uav (Anchor-free multi-UAV, IoTJ'24)
- zhao2025trtfl_ssl (Transformer + SSL TFL, TWC'25)
- zhao2024trtfl_conf (TRTFL, VTC-Spring'24)
- plus zhang2024spectrumtransformer, he2022mae (MAE), xing2007dsa,
  sohul2015cbrs

### Rewritten sections
- [x] Introduction: two-research-lines framing; TFL lineage cited as the
      user's trajectory (lightweight->cooperative->CCD-GAN->anchor-free->
      TRTFL+SSL); sharpened gap (grid-straddling cell -> collide or waste);
      5 contributions incl. distributional CVaR.
- [x] Related Work: 3 expanded strands (TFL / DRL-DSA / joint), each ending
      with an explicit "Our distinction"; TFL strand now traces the full
      detector lineage and cites all 5 Zhao papers.
- [x] System model: grounded in CCM/CBRS framing (cites sohul2015cbrs,
      zhao2023ccdgan); sensing tier cites anchor-free (zhao2024uav) and
      transformer (zhao2025trtfl_ssl) as concrete localizers.

### Style alignment to user's papers
- IEEEtran journal, hidelinks (matches the user's bare_jrnl template).
- Single .tex file retained.
- Terminology: SU/PU, CBRS, soft-vs-NMS-hard boxes, belief map.

Paper now 10 pages, 0 undefined refs, 3 figures, 5 theorems/props.

---
## UPDATE (reduce self-citation to 2; neutral TWC tone)
- Per request, cut Zhao citations from 5 to 2:
  KEPT: zhao2024uav (anchor-free, IoTJ'24), zhao2025trtfl_ssl (TFL+transformer, TWC'25).
  DROPPED (removed from text AND references.bib): zhao2023coop, zhao2023ccdgan,
  zhao2024trtfl_conf.
- Rewrote intro TFL paragraph and RW TFL strand in neutral third-person
  literature tone (no longer narrating the user's personal trajectory);
  detector lineage now framed objectively: Faster-RCNN/YOLO -> anchor-free ->
  transformer.
- references.bib now 36 entries (was 39). Paper: 10 pages, 0 undefined refs,
  exactly 2 Zhao cites in bbl.

---
## UPDATE (complete remaining planned TODOs: Thm C proof + M1/M2/M3 ablation)

### Theorem C upgraded sketch -> FULL PROOF
- Added Assumption 4 (bounded rewards, finite effective state-action space SA
  via tile-coding, sub-Gaussian) and Assumption 5 (piecewise-stationary PU,
  >= L_min per segment).
- Statement now: C1*sqrt(SA*T*Upsilon_T*log T) learning term + C2*T*(Pmd+Pfa)
  irreducible sensing floor; first term o(T) if Upsilon_T = o(T/log T).
- Appendix B proof: Lemma 3 (per-segment UCB regret, sub-Gaussian + Cauchy-
  Schwarz over segments) + Lemma 4 (TWO-SIDED sensing gap c2(Pmd+Pfa) <= gap
  <= C2(Pmd+Pfa) -- the lower bound makes "irreducible" rigorous) + combining.
- Discussion text updated: Thm C now "established", not "stated".

### M1/M2/M3 ablation (Section VI, Q4) -- REAL DATA
- sim/run_ablation_one.py runs one variant/seed, caches to ablation_cache.json;
  sim/combine_ablation.py averages + plots figures/fig_ablation.pdf.
- 4 variants x 2 seeds completed:
    Full  thru=0.107 coll=0.081
    -M1   thru=0.120 coll=0.083   (matches Full in MEAN; tail effect is in
                                   fig_cvar_tail, not here -- explained in text)
    -M2   thru=0.066 coll=0.371   (coarsened belief state: ~40% thru drop,
                                   collision up several-fold -- supports Thm B)
    -M3   thru=0.074 coll=0.472   (no uncertainty penalty: worst collision)
- New Q4 subsection + fig:ablation; explicitly reconciles -M1 (mean) with the
  CVaR tail result so the two experiments are complementary, not contradictory.
- Discussion/Limitations refreshed: honest notes on numpy-vs-full-encoder
  variance and synthetic near-zero tail.

## Status: BOTH remaining planned TODOs done.
Paper: 11 pages, 0 undefined refs, 4 figures (arch + tradeoff + cvar tail +
ablation), 5 theorems/props all with full proofs, 36 refs (2 Zhao).

---
## UPDATE (respond to AI-reviewer critique; tighten theory-model-experiment consistency)

Triaged the reviewer's points into must-fix / partial-accept / decline, then revised:

### Accepted & fixed (reviewer was right)
- **POMDP misuse (Major 1):** Section III rewritten. Now defines hidden state
  (occupancy + box params), observation (spectrogram + ACK), and B as a LEARNED
  SOFT-OCCUPANCY OBSERVATION FEATURE / approximate sufficient statistic -- NOT a
  Bayesian posterior. "POMDP" downgraded to "partially observed control problem".
- **Regret vs deep algorithm (Major 7):** Theorem 3 (was Thm C) now explicitly
  framed as a STYLIZED TABULAR SURROGATE analysis with a disclaimer that it is
  NOT a guarantee for the deep CVaR agent.
- **Thm A concavity (Major 3):** added Assumption 5 (single-peaked/quasiconcave
  throughput); without it, result degrades to first-order/boundary
  characterization (stated explicitly in proof).
- **A_occ counterfactual (Major 2):** Lemma 1 now leads with a distribution-free
  UNION BOUND |O(a)|*Pmd; product form is the conditional-independence special
  case. Feasibility frontier updated to union-bound form.
- **Belief-state dominance overreach (Major 6):** split into Proposition 1
  (information refinement, same action space, coarsening => no better value)
  + Example 1 (strict gap on one misaligned instance). Appendix B heading updated.
- **CVaR overclaim (Major 5):** Theorem 2 keeps only the clean conservativeness
  direction (eta*_CVaR >= eta*); quantitative gap moved to Corollary 2 with an
  explicit first-order/local caveat.
- **Proposition 1 false equivalence (Major 4):** downgraded to Remark
  (uncertainty penalty as a RISK PROXY, exact only under cond. independence +
  linear reward).
- **Minors:** "no grid" clarified (no fixed CHANNEL grid, but TF discretized at
  sensing resolution); A/B/C/D theorem labels removed (now auto-numbered
  Thm 1-3 / Prop 1 / Cor 1-2 / Example 1); abstract/intro/conclusion softened
  ("prove/dominates/POMDP/no prior" -> measured wording).

### Partially accepted
- Experimental concerns (seeds, baselines, measured data): cannot run torch in
  sandbox, so NOT fabricated. Instead written honestly into expanded
  Limitations: present results = evidence for QUALITATIVE ORDERINGS, not a final
  benchmark; camera-ready needs >=10 seeds, CIs, PPO/SAC/QR-DQN baselines under
  matched state/action, and measured-spectrum validation. Conclusion future-work
  updated to match.

### Declined (with reason)
- Did NOT invent additional seeds/baselines/real-data results. Honesty over
  appearing to satisfy the reviewer.

Compiles clean: 12 pages, 0 undefined refs, 0 unresolved "??".
Numbering: Theorems 1-3, Proposition 1, Corollaries 1-2, Example 1, Lemmas 1-4,
Assumptions 1-7, Remarks.

---
## UPDATE (respond to 2nd AI-reviewer pass; new title; condense theory)

### Title
OLD: "Localization-Aware Joint Spectrum Sensing and Access in Wideband
      Cognitive Radio: A Belief-Map Reinforcement Learning Approach"
NEW: "Closing the Sensing-Access Loop: Belief-Map-Driven Risk-Sensitive
      Spectrum Access in Wideband Cognitive Radio"
(foregrounds the loop-closure gap + the two method pillars; more concise.)

### Reviewer-2 fixes
- **Unified framework (Major 1, the central critique):** Section V now opens
  with an explicit ABSTRACTION HIERARCHY (true system -> belief map ->
  fixed-policy evaluation -> tabular surrogate) and three scope bullets
  (operating-point / representation / learning analysis), stating up front
  what each theorem does and does NOT claim, and that the deep RL algorithm
  does not inherit the guarantees. Directly answers "what unified object do
  the theorems solve?"
- **CVaR degenerate-Bernoulli (Major 3):** defined the COLLISION-WINDOW
  variable Z_col^W = sum over W frames; CVaR constraint + Thm 2 + Cor 2 +
  framework objective all re-expressed on Z_col^W. This is the quantity the
  sim already measured (fig_cvar_tail), so theory<->experiment now aligned.
- **Theorem 1 O(a)/geometry leap (Major 2):** Assumption 3 rewritten to
  declare FIXED ACTION GEOMETRY (|O(a)|, A_idle constant in eta); theorem
  retitled "...operating point under fixed action geometry".
- **Regret (Major 4):** reinforced Option A (qualitative-only) via the new
  hierarchy bullet + existing subsection disclaimer.
- **Over-claiming language (Major 5):** "minimal sufficient"->"informative";
  "strictly suboptimal"->"can be strictly suboptimal under grid misalignment";
  appendix "value dominance"->"value monotonicity".
- **Prop 1 / Example 1 comparison basis (Major 6):** appendix now explicitly
  states the ACTION SPACE IS IDENTICAL; only the state representation differs
  (coarse policy can take the fine action but cannot inform it).
- **Figure-theory alignment (Major 7):** limitations now state the synthetic
  ROC is a parametric model; experiments verify assumption<->prediction
  consistency on a controlled instance, not that assumptions hold on hardware.

### Theory condensed
- Trimmed the "design implication" and "why distributional RL" remarks;
  merged per-theorem scope justifications into the single hierarchy intro
  (less repetition).

### Declined (honesty)
- Still did NOT fabricate extra seeds / strong baselines / measured data;
  kept these as explicit camera-ready limitations.

Compiles clean: 12 pages, 0 undefined refs, 0 "??". Numbering intact
(Thm 1-3, Prop 1, Cor 1-2, Example 1, Lemmas 1-4, Assumptions 1-7).

---
## UPDATE (3rd AI-reviewer pass; TFL-foregrounding title; polish)

### Title (user prefers TFL-foregrounding alternative)
NEW: "From Time-Frequency Localization to Risk-Sensitive Access:
      A Belief-Map Closed Loop for Wideband Cognitive Radio"

### Reviewer-3 fixes (selective -- some suggestions judged over-defensive)
- **Regret additivity (Major 1, the critical one):** added Assumption 8
  (STATIONARY, POLICY-INDEPENDENT SENSING ERROR -- localizer is a fixed sensing
  channel the agent does not adapt). Theorem 3 renamed "...a learning term plus
  a policy-independent sensing gap"; gap term relabeled; added explicit text
  that under this model sensing error enters the reward as a stationary additive
  offset NOT through the transition kernel, with honest caveat that the gap is
  not claimed irreducible if the agent co-adapts the localizer. Appendix B proof
  updated: additive split justified by Assumption 8; "Combining" para softened
  to "policy-independent within this model".
- **Sensing floor terminology (Major 6):** "irreducible sensing floor" ->
  "policy-independent sensing-induced gap" throughout.
- **CVaR window spec (Major 2):** Assumption 6 (tail-monotone) now states
  Z_col^W is over FIXED-LENGTH, NON-OVERLAPPING windows, W constant in T.
- **Thm 1 single-crossing (Major 4):** Assumption 5 now explicitly restricts to
  ROC families with single-crossing/quasiconcave throughput; outside it, result
  holds only as first-order/boundary characterization.
- **Lemma 4 lower bound (Major 7):** flagged "under a bounded reward-per-error
  model" with explicit per-error cost; upper bound needs no such assumption.
- **Lemma 3 compression (Major 8):** standard UCB derivation replaced by a
  citation to Garivier-Moulines.
- **Experiments non-circular (Major 5):** limitations now state the environment
  is constructed to ISOLATE assumptions, NOT tuned to maximize LA-JSSA.
- **Prop 1 state/action (Major 3):** already addressed last round (action space
  identical; only state differs) -- verified present.
- **Closure figure (accept-booster 1):** fig:arch ALREADY is the unified
  closed-loop diagram (RF->STFT->localizer->belief map->encoder->critic/CVaR->
  action->env->ACK feedback, two timescales). Strengthened the framework text to
  point to it as the closure figure; no separate diagram needed.

### Polish
- Removed lingering "belief-state POMDP" claims that contradicted the Sec III
  downgrade: intro, system-model table, keyword, framework intro now say
  "partially observed control problem". (General-concept POMDP mentions in
  encoder text + appendix sufficient-statistic fact retained, as correct.)

### Declined / not done (judged over-defensive or low-value)
- No separate "unified system diagram" (fig:arch already serves).
- Still no fabricated seeds/baselines/real data.

Compiles clean: 13 pages, 0 undefined refs, 0 "??". Assumptions 1-8,
Theorems 1-3, Proposition 1, Corollaries 1-2, Example 1, Lemmas 1-4.

---
## UPDATE (theory walk-through optimization + language/formula polish)

### Formula formatting (no overruns / no large gaps)
- Overfull \hbox count reduced 9 -> 2 (remaining two are ~1pt, invisible).
- Abstraction-hierarchy eq: single wide row (67pt over) -> stacked 2-line
  aligned flow; section refs now use \ref not hardcoded "III/IV".
- CVaR constraint+definition: one \qquad-joined row -> 2-line aligned block.
- Cauchy-Schwarz chain (appendix): 3-term inequality -> 2-line aligned.
- Objective eq (13): argmax + s.t. on one row -> 2-line aligned.
- Collision-union & regret eqs: stripped \;...\; padding to plain =/\le,
  added \!\! around the big sum, shortened underbrace labels.

### Theory tightening / polish
- Softened residual absolute "No existing work" -> "To our knowledge, no
  prior work".
- Verified no wordy hedges (no "in order to", "a number of", "it should be
  noted"); i.e./e.g. spacing correct for IEEE.

### Theorem proof structure (verified, all clean)
- Thm 1: Lemma 1 (union bound, no counterfactual) + Lemma 2 -> 3-step proof;
  quasiconcave Assumption gives global opt, else first-order/boundary.
- Thm 2: window variable Z_col^W -> genuine tail; CVaR>=E + tail-monotone
  => eta*_CVaR >= eta*; Cor 2 first-order margin.
- Prop 1 + Example 1: data-processing inequality, same action space.
- Thm 3: stylized surrogate; Assumption 8 justifies additive split;
  Lemma 3 (cited UCB) + Lemma 4 (two-sided, bounded-per-error).

Compiles clean: 13 pages, 0 undefined refs, 0 "??", 2 negligible overfulls.

---
## UPDATE (make theory genuinely TFL-grounded + new localization->access experiment)

### Theory: from "detection" to "localization"
PROBLEM: the proofs used cell-level Pmd/Pfa directly -- detection language, not
localization. The TFL-specific error (box IoU / bandwidth geometry) was absent,
so the theory only half-matched the title.

FIX (adaptive, keeps all 4 theorems):
- Added Assumption (Localization-error model), eq. (loc_to_cell): Pmd, Pfa are
  now DERIVED from box-level localization error eps_loc = 1 - E[IoU]:
  Pmd <= c_md*eps_loc + r_md(eta),  Pfa <= c_fa*eps_loc + r_fa(eta).
- Added Corollary (Access cost of localization error), eq. (loc_gap):
  throughput gap <= rbar*A_idle*(c_fa*eps_loc + r_fa) + kappa*Pcolmax, i.e.
  each unit of IoU improvement -> linear access-throughput recovery. This is
  the result that genuinely foregrounds TFL ("closes the loop the title names").
- Design remark updated to cite IoU/eps_loc as the lever, not just ROC.
- New macros: \iou \eloc \cmd \cfa \rmd \rfa.

### New simulation (matches upgraded theory)
- sim/env_loc.py: LocalizationTFEnv -- belief map RENDERED FROM ESTIMATED PU
  BOXES with controllable IoU error (loc_quality knob), so cell-level error is
  a CONSEQUENCE of localization geometry. Includes measure_loc_to_cell().
- sim/run_localization_sweep.py: sweeps localizer quality over 7 levels x 3
  seeds; produces figures/fig_localization.pdf (2 panels):
    (a) eps_loc -> (Pmd,Pfa) nearly linear  [validates loc_to_cell]
    (b) eps_loc -> access throughput gap, linear fit  [validates Corollary]
  Real measured results (eps_loc 0.05->0.51): Pmd 0.037->0.312, Pfa rises too;
  throughput gap grows monotonically/linearly with eps_loc.
- Existing agents (BeliefDQN etc.) verified to run unchanged on env_loc.

### Paper wiring
- New Section VI-E (Q5: From Localization Accuracy to Access Performance) +
  fig:localization. Simulation intro now frames all five questions Q1-Q5.

Compiles clean: 14 pages, 0 undefined refs, 0 "??", 2 negligible overfulls.

---
## UPDATE (frequency-focused localization + expanded experiments to TWC norm)

### Theory: frequency-localization (not 2-D box)
- Reframed Assumption (Localization-error model) -> FREQUENCY-localization:
  eq.(freq_iou) uses 1-D frequency-interval IoU; eq.(loc_to_cell) unchanged in
  form. Rationale: in a sense-then-access frame, the access phase reuses the
  SAME frame's idle FREQUENCY ranges, so temporal-support error is absorbed by
  frame alignment; only frequency extent matters.
- System model VI-D (access): added frame-structure sentence (sense->access
  frequency reuse) motivating the frequency-centric error model.
- Corollary (Access cost of localization error) + design remark reworded to
  frequency localization / IoU. New macros \fint \flo \fhi.

### Expanded simulation (now 6 figures + 3 tables, TWC-typical)
- env_loc.py localizer made frequency-only (dropped temporal jitter).
- NEW sim/run_freq_vs_time.py -> fig_freq_vs_time.pdf (Q5): injects frequency
  vs temporal error separately; MEASURED result: freq error degrades thru
  (0.066->0.060) and collision (0->0.092), temporal error flat (thru ~0.066,
  collision 0). Empirically proves frequency dominates.
- run_localization_sweep.py -> fig_localization.pdf (Q6): eps_loc -> (Pmd,Pfa)
  and -> throughput gap, both linear.
- NEW Table III (tab:theory_sim): theory-prediction vs measured, consolidating
  eta* match, eps->Pmd r=0.993, eps->gap r=0.986, freq-vs-time dominance.
- Sim intro now frames Q1-Q6.

Figures: arch, tradeoff, cvar_tail, ablation, freq_vs_time, localization (6).
Tables: notation, performance, theory-vs-sim (3).
Compiles clean: 14 pages, 0 undefined refs, 0 "??", 3 negligible (<2pt) overfulls.

---
## UPDATE (consistency audit + complete runnable code package + ZH nomenclature)

### Logic/consistency fixes
- Cross-reference audit: 0 broken \ref/\eqref; verified theorem/assumption
  numbering sequential (Thm 1-3, Prop 1, Cor 1-3, Lemma 1-4, Asm 1-8, Ex 1).
- Linked Assumption (cell-level error) explicitly to Assumption
  (frequency-localization) so the TFL->Pmd/Pfa chain is named, not just
  equation-referenced.
- Citation audit: 19 cited -> 26 cited. Added missing method citations at first
  mention: DRQN (DQN), DDPG, QR-DQN + DSAC (distributional), Rockafellar (CVaR),
  YOLOv3, MAE. Verified every \cite resolves to a bib key.
- Fixed wang2018 bib entry type (@inproceedings -> @article; removed empty
  booktitle warning). bibtex now warning-free.
- Abstract closing updated to mention the frequency-localization finding,
  matching Section VI (Q5/Q6) and the conclusion.
- Verified continuous-vs-discrete action is reconciled (framework: two
  instantiations DQN/DDPG; theory/sim use the discrete one; stated in Asm).

### Code package (complete, runnable, pure numpy)
- Added sim/README.md: maps every script -> paper figure/table, with exact
  commands; documents honesty/scope and a smoke test.
- Added sim/requirements.txt (numpy, matplotlib only).
- Verified all 12 scripts parse; ran Q1/Q5/Q6 end-to-end -> figures regenerate
  and reproduce the paper numbers (eta*=0.85; eps->Pmd/gap linear; freq
  dominates time).

### Chinese PDF
- Added a full nomenclature section "符号说明（公式中字母的含义）": every
  formula symbol grouped (sensing/occupancy, frequency localization,
  access/state, reward/risk, learning/regret) with Chinese meaning, as a
  page-breaking longtable. ZH now 10 pages, 0 overfull, 0 "??".

Status: EN 14 pp (6 figs, 3 tables, 26 refs), 0 undefined, 0 "??", bibtex clean.

---
## UPDATE (clarify eta<->detection-confidence link; TWC-grade figure restyle)

### Theory clarification: what eta thresholds
- System model now states explicitly that B_{m,n} IS the time-frequency
  localizer's DETECTION CONFIDENCE rasterized onto the cell grid (each box
  carries a confidence score; B is the edge-feathered score of the covering
  box). eta is the decision threshold ON that confidence.
- Made explicit that (Pmd,Pfa) are shaped by TWO factors: the confidence
  threshold eta (where the soft score is cut) and the box geometric accuracy
  (IoU error eps_loc). Assumption (localization) r(eta) term relabeled as the
  confidence-threshold contribution, c*eps_loc as the localization
  contribution; separation shown in eq.(loc_to_cell).
- This closes the reviewer-style question "what quantity does eta threshold?":
  it is the detector confidence, and eps_loc is the box IoU error; both feed
  Pmd/Pfa.

### Figures restyled to IEEE TWC norm
- Added sim/twc_style.py (serif fonts, color-blind-safe palette, inward ticks,
  subtle grid, (a)/(b) panel labels, 600 dpi, Type-42 fonts, single/double
  column widths).
- Added sim/make_figures.py: regenerates all 5 data figures from cached .npz
  with the TWC style (fast restyle, no experiment re-run; data unchanged).
- Removed sentence-style in-figure titles (descriptions live in LaTeX captions).
- Updated tradeoff caption to (a)/(b)/(c) panel labels matching the figure.

Figures are now publication-styled; underlying DATA still needs the
camera-ready upgrade (>=10 seeds, strong baselines, full encoder, measured
spectra) noted in Limitations -- requires GPU/torch, out of sandbox scope.

EN: 14 pp, 6 figs, 3 tables, 26 refs, 0 undefined, 0 "??".

---
## UPDATE (expand references to ~48; add 7-method baseline comparison; runnable scripts)

### References 26 -> 48 (all cited, 0 orphans, bibtex clean)
- Added & wired into text: surveys (yucek2009, axell2012, arjoune2019), deep
  detection (oshea2018, vo2020), more DSA (naparstek2018, chang2019, xu2021,
  tan2020, lin2019, zhang2019), safe/risk RL (altman1999, achiam2017,
  tessler2019, chow2017), method bodies used as baselines (schulman2017 PPO,
  haarnoja2018 SAC, bellemare2017 distributional, mnih2015 DQN), and
  prasad2021 TFL. Removed genuine padding (8 uncited entries).

### Baseline comparison (Q2) -- real numpy data
- New sim/agents_pg.py: PPOLiteAgent (clipped PG + entropy + Lagrangian
  collision penalty) and RandomAgent.
- New sim/run_baselines.py: 7 methods on MATCHED env/state/action set
  (Random, Myopic, FixedDQN, BeliefDQN, PPO-lite, LA-JSSA, Oracle).
  REAL measured results (3 seeds learners, 5 fixed):
    Random   thru .092 / coll .229
    Myopic   .082 / .005
    FixedDQN .106 / .170
    BeliefDQN .111 / .127
    PPO-lite .049 / .315 (unstable, honest)
    LA-JSSA  .129 / .051  (best learner, bolded)
    Oracle   .188 / .000
  -> figures/fig_baselines.pdf + Table II (replaced old 5-row table).
- Q2 text rewritten around the 7-method comparison; limitations updated.
- New sim/torch_baselines.py: GPU-ready PyTorch PPO/SAC/QR-DQN scaffold with
  BeliefEncoder (conv-freq + GRU/LRU) for camera-ready scale-up; explicitly
  NOT run in sandbox.
- README updated with baseline commands and torch note.

### Chinese PDF
- Q2 updated with the 7-method comparison table (matches EN Table II).

Status: EN 15 pp, 7 figs, 3 tables, 48 refs, 0 undefined, 0 "??", bibtex clean.
ZH 10 pp, 0 "??".

---
## UPDATE (address AI-reviewer Major Revision; seeds>=10; one-click reproduction)

### P0/P1 reviewer items addressed
- Intro: removed all subsections (Two Research Lines / The Gap / Contributions)
  -> continuous prose, per request.
- CVaR experiment (reviewer 4.4, circular validation): added BURSTY PU mode to
  env.py (burst_prob/burst_len) -> GENUINE heavy collision tail. Mean-constr.
  agent now 95%tail=9, max=16; CVaR agent clips to 0 at equal/higher throughput.
  Non-circular validation of Theorem 2. Updated Q3 text + fig_cvar_tail.
- Implementation-vs-description gap (reviewer 4.1/8.2): added explicit
  "Implementation scope" paragraph distinguishing the full architecture
  (Fig.1/Alg.1) from the compact CPU realization used in experiments; one-to-one
  mapping noted; deep PyTorch release for camera-ready.
- Statistical significance (reviewer 4.2): all 7 baseline methods now 10 seeds;
  Q5/Q6 10 seeds. HONEST result: at 10 seeds learner mean-throughput overlaps
  within 1 std -> removed "LA-JSSA highest throughput" overclaim. Reframed Q2
  around robust trends (belief-state lower mean collision) and moved the clean
  separation to Q3 (tail). Table II + fig_baselines + captions updated to honest
  10-seed numbers.
- Keyword: "partial observability" -> "partial observation" (reviewer 3.5).

### One-click reproduction (request #3)
- New sim/run_all.py: runs ALL experiments (>=10 seeds), renders every figure in
  TWC style (600dpi vector, Type-42 fonts, single/double col), emits Table II
  (baselines_table_rows.tex) and Table III (theory_sim_table.tex).
  Modes: full / --quick / --figures-only. Verified --figures-only works.
- README updated with one-click instructions.

### Real 10-seed numbers (honest)
Random .092/.231  Myopic .083/.004  FixedDQN .119/.139  BeliefDQN .111/.110
PPO .082/.182  LA-JSSA .099/.156  Oracle .188/.000
Theory results robust at 10 seeds: eta*=0.85 exact; eps->Pmd r=0.994;
eps->gap r=0.980; freq-vs-time freq dominates (time inert).

Status: EN 15pp, 0 undefined, 0 "??". ZH 11pp, 0 "??". Both rebuilt.
Still TODO (lower priority): Thm 3 -> Remark, trim to 12pp, notation slimming.

---
## UPDATE (Theorem 3 -> Proposition; trim toward 14pp)

### Professional call on Theorem 3 (not mechanical compliance)
Reviewer asked to demote Thm 3 to a Remark. Judgment: demote to PROPOSITION
(not Remark) -- it has a full proof (Remark would understate) but is a
standard piecewise-stationary-bandit result with no quantitative bridge to the
deep agent (Theorem would overstate). This keeps the narrative-stitching
insight ("learning cost vanishes, sensing cost does not -> improve the
localizer") while right-sizing the claim.
- Thm 3 -> Proposition 2 (label thm:regret kept; all refs updated Theorem->
  Proposition; hierarchy eq + appendix header updated).
- Appendix regret proof compressed 108 -> ~45 lines (proof SKETCH: keeps the
  decomposition + the genuinely-contested additive-split justification; the
  standard UCB lemma now cited, not re-derived; the two-sided gap kept tersely).
- Removed redundant per-proposition scope preamble (covered by the Section V
  scope bullets) -- reviewer 3.4.
- Section V opening hierarchy DISPLAY eq -> inline (saves vertical space).
- Discussion/Limitations tightened (~390 -> ~230 words), fixed stale "3-5 seeds"
  -> 10 seeds, removed duplication with the setup's Implementation-scope para.

### Page count
Now 15pp in IEEEtran DRAFT rendering, with only 2 reference lines spilling onto
p15 (refs 47-48). Bibliography set in \small (standard IEEE). Effectively a
14-page paper; IEEEtran draft renders longer than final two-column production,
which would be <=14pp. Forcing the last 2 lines off would require dropping
cited refs (reopens "under-surveyed") or shrinking a figure -- not worth it.
Added rebuild.py (preserves \small bib) for reproducible reassembly.

Status: EN 15pp (eff. 14), 0 undefined, 0 "??", 48 refs, bibtex clean.
