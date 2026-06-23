"""
env.py -- Continuous time-frequency spectrum environment for LA-JSSA.

Design goals (must mirror the paper, Sections III & V):
  * PU emissions occupy ARBITRARY center frequency / bandwidth, NOT aligned
    to any channel grid (this is the whole point vs. fixed-channel DRL-DSA).
  * Occupancy lives on a fine F x T cell grid; the true occupancy map O is
    binary; the sensing model emits a soft belief map B in [0,1] with
    controllable cell-level miss/false-alarm behaviour via threshold eta.
  * PU on/off dynamics are piecewise-stationary Markov (matches Thm C).
  * The SU action is a continuous resource block (fc, bw) quantized only at
    the action-head level, never in the state.

Symbols follow the paper's notation:
  Nf, Nt  : freq/time cells per frame (here we use Nf freq cells, 1 time
            step per decision for clarity; temporal correlation is in the
            Markov PU process).
  O       : true occupancy vector over Nf cells (0/1)
  B       : belief vector over Nf cells in [0,1]
  Pmd,Pfa : induced cell-level miss / false-alarm at a given threshold eta
"""

import numpy as np


class PUEmission:
    """A single primary-user emission as a continuous TF box on the freq axis.

    f_lo, f_hi are continuous in [0, 1] (normalized band). The emission turns
    on/off according to a 2-state Markov chain; in different stationary
    *segments* the on-probability changes (piecewise-stationary).
    """

    def __init__(self, rng, seg_on_probs, p01=0.15, p10=0.25):
        self.rng = rng
        # continuous, non-grid-aligned support
        self.f_lo = rng.uniform(0.02, 0.80)
        self.bw = rng.uniform(0.04, 0.18)
        self.f_hi = min(1.0, self.f_lo + self.bw)
        self.on = False
        self.p01 = p01  # off->on baseline
        self.p10 = p10  # on->off
        self.seg_on_probs = seg_on_probs  # list of target occupancy per segment

    def step(self, seg_idx):
        # modulate the off->on rate so that the stationary occupancy matches
        # the segment's target; keeps a Markov (correlated) on/off process.
        target = self.seg_on_probs[seg_idx % len(self.seg_on_probs)]
        # stationary occupancy of a 2-state chain = p01/(p01+p10); solve p01.
        p10 = self.p10
        p01 = max(1e-3, min(0.95, target * p10 / max(1e-3, (1.0 - target))))
        if self.on:
            if self.rng.random() < p10:
                self.on = False
        else:
            if self.rng.random() < p01:
                self.on = True
        return self.on


class ContinuousTFEnv:
    """Continuous-bandwidth spectrum sensing+access environment.

    One decision = one frame. The agent observes a belief map (length Nf) and
    chooses a resource block; reward follows the paper's reward (throughput
    minus collision and uncertainty penalties).
    """

    def __init__(self,
                 Nf=64,
                 n_pu=3,
                 snr_quality=1.0,      # higher => better localizer ROC
                 eta=0.5,              # belief decision threshold
                 lam_c=3.0,            # collision penalty weight
                 lam_u=0.5,            # uncertainty penalty weight
                 seg_len=400,          # steps per stationary segment
                 n_segments=6,
                 action_bws=(0.06, 0.10, 0.16),  # candidate SU block widths
                 n_fc=16,              # candidate center-freq positions
                 burst_prob=0.0,       # per-frame prob. of a PU activity burst
                 burst_len=8,          # frames a burst lasts
                 seed=0):
        self.Nf = Nf
        self.n_pu = n_pu
        self.snr_quality = snr_quality
        self.eta = eta
        self.lam_c = lam_c
        self.lam_u = lam_u
        self.seg_len = seg_len
        self.n_segments = n_segments
        self.burst_prob = burst_prob
        self.burst_len = burst_len
        self._burst_left = 0
        self.rng = np.random.default_rng(seed)

        # piecewise-stationary target occupancies per PU per segment
        self.seg_on_probs = [
            list(self.rng.uniform(0.1, 0.6, size=n_segments)) for _ in range(n_pu)
        ]
        self.pus = [PUEmission(self.rng, self.seg_on_probs[i]) for i in range(n_pu)]

        # discrete ACTION set (state stays continuous!). Action = (fc_idx, bw_idx)
        self.action_bws = list(action_bws)
        self.fc_grid = np.linspace(0.05, 0.95, n_fc)
        self.actions = [(fc, bw) for bw in self.action_bws for fc in self.fc_grid]
        self.n_actions = len(self.actions)

        self.cell_edges = np.linspace(0.0, 1.0, Nf + 1)
        self.cell_centers = 0.5 * (self.cell_edges[:-1] + self.cell_edges[1:])

        self.t = 0
        self.O = np.zeros(Nf)

    # ---- occupancy / sensing ----
    def _seg(self):
        return (self.t // self.seg_len) % self.n_segments

    def _true_occupancy(self):
        O = np.zeros(self.Nf)
        seg = self._seg()
        # bursty correlated activity: occasionally all PUs turn on together for
        # several frames, creating a genuine heavy tail in the collision return.
        if self._burst_left > 0:
            self._burst_left -= 1
            force_on = True
        elif self.burst_prob > 0 and self.rng.random() < self.burst_prob:
            self._burst_left = self.burst_len - 1
            force_on = True
        else:
            force_on = False
        for pu in self.pus:
            active = pu.step(seg)
            if force_on or active:
                lo = np.searchsorted(self.cell_edges, pu.f_lo) - 1
                hi = np.searchsorted(self.cell_edges, pu.f_hi) - 1
                lo = max(0, lo); hi = min(self.Nf - 1, hi)
                O[lo:hi + 1] = 1.0
        return O

    def _belief(self, O):
        """Soft belief map. Occupied cells get a high score, idle cells low,
        both corrupted by noise whose scale shrinks as snr_quality grows.
        This yields a smooth, monotone ROC so that raising eta trades Pfa for
        Pmd exactly as Assumption (ROC) in the paper requires.
        """
        sigma = 0.55 / self.snr_quality
        # latent score ~ N(O, sigma^2), squashed to [0,1] via logistic
        latent = O + self.rng.normal(0.0, sigma, size=self.Nf)
        B = 1.0 / (1.0 + np.exp(-(latent - 0.5) * 4.0))
        return np.clip(B, 0.0, 1.0)

    # ---- gym-like API ----
    def reset(self):
        self.t = 0
        for pu in self.pus:
            pu.on = False
        self.O = self._true_occupancy()
        self.B = self._belief(self.O)
        return self.B.copy()

    def action_cells(self, action):
        fc, bw = action
        lo = np.searchsorted(self.cell_edges, fc - bw / 2) - 1
        hi = np.searchsorted(self.cell_edges, fc + bw / 2) - 1
        lo = max(0, lo); hi = min(self.Nf - 1, hi)
        return lo, hi

    def step(self, action_idx):
        action = self.actions[action_idx]
        lo, hi = self.action_cells(action)
        cells = np.arange(lo, hi + 1)
        A = len(cells)

        collision = float(np.any(self.O[cells] > 0.5))
        idle_cells = int(np.sum(self.O[cells] < 0.5))
        # throughput: rate per truly-idle accessed cell, only if not colliding
        rbar = 1.0 / self.Nf
        thru = rbar * idle_cells * (1.0 - collision)
        # uncertainty penalty = summed Bernoulli variance of belief over block
        unc = float(np.sum(self.B[cells] * (1.0 - self.B[cells])))

        reward = thru - self.lam_c * collision - self.lam_u * unc

        # advance environment
        self.t += 1
        self.O = self._true_occupancy()
        self.B = self._belief(self.O)

        info = {"collision": collision, "thru": thru, "idle": idle_cells,
                "A": A, "seg": self._seg()}
        done = False
        return self.B.copy(), reward, done, info

    # ---- measurement helpers for Theorem A validation ----
    def measure_roc(self, eta, n=20000):
        """Empirically measure cell-level (Pmd, Pfa) at threshold eta."""
        rng = np.random.default_rng(12345)
        env = ContinuousTFEnv(Nf=self.Nf, n_pu=self.n_pu,
                              snr_quality=self.snr_quality, seed=999)
        env.reset()
        occ_total = occ_miss = idle_total = idle_fa = 0
        for _ in range(n):
            O = env._true_occupancy()
            B = env._belief(O)
            decided_occ = (B >= eta)
            occ_mask = O > 0.5
            idle_mask = ~occ_mask
            occ_total += occ_mask.sum()
            idle_total += idle_mask.sum()
            occ_miss += np.sum(occ_mask & (~decided_occ))   # truly occ, declared idle
            idle_fa += np.sum(idle_mask & decided_occ)       # truly idle, declared occ
            env.t += 1
        Pmd = occ_miss / max(1, occ_total)
        Pfa = idle_fa / max(1, idle_total)
        return Pmd, Pfa
