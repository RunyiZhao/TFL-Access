"""
env_loc.py -- Localization-aware TF spectrum environment for LA-JSSA.

This environment matches the UPGRADED, TFL-grounded theory (Section V): the
belief map is produced by a *time-frequency localizer* that emits estimated
boxes with controllable geometric error, rather than by per-cell detection
noise. The cell-level miss/false-alarm rates Pmd/Pfa are thus DERIVED from the
localization error eps_loc = 1 - E[IoU], exactly as Assumption
(Localization-error model), eq. (loc_to_cell).

Pipeline per frame:
  true continuous PU boxes  -->  localizer  -->  estimated boxes (IoU error)
       -->  rasterize est. box to soft belief map B in [0,1]  -->  agent state

Key control knob:
  loc_quality in (0, 1]  : higher => tighter boxes => higher IoU => lower
                           Pmd/Pfa floors. (loc_quality -> 1 approaches the
                           perfect-localization oracle.)

The action space and reward mirror env.py so the same agents run unchanged.
"""

import numpy as np


class PUBox:
    """A primary-user emission as a continuous TF box on the frequency axis,
    with a piecewise-stationary 2-state Markov on/off process."""

    def __init__(self, rng, seg_on_probs, p10=0.25):
        self.rng = rng
        self.f_lo = rng.uniform(0.02, 0.80)
        self.bw = rng.uniform(0.04, 0.18)
        self.f_hi = min(1.0, self.f_lo + self.bw)
        self.on = False
        self.p10 = p10
        self.seg_on_probs = seg_on_probs

    def step(self, seg_idx):
        target = self.seg_on_probs[seg_idx % len(self.seg_on_probs)]
        p10 = self.p10
        p01 = max(1e-3, min(0.95, target * p10 / max(1e-3, (1.0 - target))))
        if self.on:
            if self.rng.random() < p10:
                self.on = False
        else:
            if self.rng.random() < p01:
                self.on = True
        return self.on


def _box_iou_1d(lo1, hi1, lo2, hi2):
    """IoU of two 1-D intervals (frequency support)."""
    inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    union = (hi1 - lo1) + (hi2 - lo2) - inter
    return inter / union if union > 1e-9 else 0.0


class LocalizationTFEnv:
    """Localization-aware continuous-bandwidth sensing+access environment."""

    def __init__(self,
                 Nf=64,
                 n_pu=3,
                 loc_quality=0.85,     # localizer accuracy in (0,1]
                 eta=0.5,              # belief decision threshold
                 lam_c=3.0,
                 lam_u=0.5,
                 seg_len=400,
                 n_segments=6,
                 action_bws=(0.06, 0.10, 0.16),
                 n_fc=16,
                 p_detect=0.97,        # prob. localizer detects an ON box
                 p_ghost=0.02,         # prob. of a spurious (ghost) box / frame
                 seed=0):
        self.Nf = Nf
        self.n_pu = n_pu
        self.loc_quality = float(loc_quality)
        self.eta = eta
        self.lam_c = lam_c
        self.lam_u = lam_u
        self.seg_len = seg_len
        self.n_segments = n_segments
        self.p_detect = p_detect
        self.p_ghost = p_ghost
        self.rng = np.random.default_rng(seed)

        self.seg_on_probs = [
            list(self.rng.uniform(0.1, 0.6, size=n_segments)) for _ in range(n_pu)
        ]
        self.pus = [PUBox(self.rng, self.seg_on_probs[i]) for i in range(n_pu)]

        self.action_bws = list(action_bws)
        self.fc_grid = np.linspace(0.05, 0.95, n_fc)
        self.actions = [(fc, bw) for bw in self.action_bws for fc in self.fc_grid]
        self.n_actions = len(self.actions)

        self.cell_edges = np.linspace(0.0, 1.0, Nf + 1)
        self.cell_centers = 0.5 * (self.cell_edges[:-1] + self.cell_edges[1:])

        self.t = 0
        self.O = np.zeros(Nf)
        self.last_iou = []

    # ---- occupancy ----
    def _seg(self):
        return (self.t // self.seg_len) % self.n_segments

    def _true_boxes(self):
        """Return list of (f_lo, f_hi) for currently-ON PUs."""
        seg = self._seg()
        boxes = []
        for pu in self.pus:
            if pu.step(seg):
                boxes.append((pu.f_lo, pu.f_hi))
        return boxes

    def _true_occupancy(self, boxes):
        O = np.zeros(self.Nf)
        for (f_lo, f_hi) in boxes:
            lo = max(0, np.searchsorted(self.cell_edges, f_lo) - 1)
            hi = min(self.Nf - 1, np.searchsorted(self.cell_edges, f_hi) - 1)
            O[lo:hi + 1] = 1.0
        return O

    # ---- localizer: true boxes -> estimated freq intervals -> belief map ----
    def _localize(self, boxes):
        """Emit estimated FREQUENCY intervals with IoU error governed by
        loc_quality, then rasterize to a soft belief map over frequency cells.
        Per the frame structure (sense-then-access in the same frame), only the
        frequency extent is estimated; temporal alignment is given. Returns
        (B, mean_freq_iou)."""
        # frequency geometric jitter shrinks as loc_quality -> 1
        jitter = 0.12 * (1.0 - self.loc_quality)
        B = np.full(self.Nf, 0.10)            # background idle belief
        ious = []
        conf_on = 0.5 + 0.5 * self.loc_quality
        for (f_lo, f_hi) in boxes:
            if self.rng.random() > self.p_detect:
                continue  # missed detection of the whole emission
            # perturb frequency center and bandwidth (freq-localization error)
            c = 0.5 * (f_lo + f_hi) + self.rng.normal(0.0, jitter)
            w = (f_hi - f_lo) * (1.0 + self.rng.normal(0.0, jitter))
            w = max(0.02, w)
            e_lo, e_hi = c - w / 2, c + w / 2
            ious.append(_box_iou_1d(f_lo, f_hi, e_lo, e_hi))
            lo = max(0, np.searchsorted(self.cell_edges, e_lo) - 1)
            hi = min(self.Nf - 1, np.searchsorted(self.cell_edges, e_hi) - 1)
            # soft confidence with feathered edges (uncertainty at interval ends)
            B[lo:hi + 1] = np.maximum(B[lo:hi + 1], conf_on)
            if lo - 1 >= 0:
                B[lo - 1] = max(B[lo - 1], 0.5 * conf_on)
            if hi + 1 < self.Nf:
                B[hi + 1] = max(B[hi + 1], 0.5 * conf_on)
        # ghost (spurious) boxes -> raise belief on a random empty span
        if self.rng.random() < self.p_ghost:
            gc = self.rng.uniform(0.1, 0.9)
            gw = self.rng.uniform(0.03, 0.10)
            lo = max(0, np.searchsorted(self.cell_edges, gc - gw / 2) - 1)
            hi = min(self.Nf - 1, np.searchsorted(self.cell_edges, gc + gw / 2) - 1)
            B[lo:hi + 1] = np.maximum(B[lo:hi + 1], 0.6)
        # small belief noise
        B = np.clip(B + self.rng.normal(0.0, 0.03, size=self.Nf), 0.0, 1.0)
        mean_iou = float(np.mean(ious)) if ious else 1.0
        return B, mean_iou

    # ---- gym-like API ----
    def reset(self):
        self.t = 0
        for pu in self.pus:
            pu.on = False
        boxes = self._true_boxes()
        self.O = self._true_occupancy(boxes)
        self.B, iou = self._localize(boxes)
        self.last_iou = [iou]
        return self.B.copy()

    def action_cells(self, action):
        fc, bw = action
        lo = max(0, np.searchsorted(self.cell_edges, fc - bw / 2) - 1)
        hi = min(self.Nf - 1, np.searchsorted(self.cell_edges, fc + bw / 2) - 1)
        return lo, hi

    def step(self, action_idx):
        action = self.actions[action_idx]
        lo, hi = self.action_cells(action)
        cells = np.arange(lo, hi + 1)
        A = len(cells)

        collision = float(np.any(self.O[cells] > 0.5))
        idle_cells = int(np.sum(self.O[cells] < 0.5))
        rbar = 1.0 / self.Nf
        thru = rbar * idle_cells * (1.0 - collision)
        unc = float(np.sum(self.B[cells] * (1.0 - self.B[cells])))
        reward = thru - self.lam_c * collision - self.lam_u * unc

        self.t += 1
        boxes = self._true_boxes()
        self.O = self._true_occupancy(boxes)
        self.B, iou = self._localize(boxes)
        self.last_iou = [iou]

        info = {"collision": collision, "thru": thru, "idle": idle_cells,
                "A": A, "seg": self._seg(), "iou": iou}
        return self.B.copy(), reward, False, info

    # ---- measurement: localization error -> cell-level (Pmd,Pfa) ----
    def measure_loc_to_cell(self, eta, n=20000, seed=999):
        """Empirically measure mean IoU, eps_loc=1-IoU, and the induced
        cell-level (Pmd,Pfa) at threshold eta. Validates eq. (loc_to_cell)."""
        env = LocalizationTFEnv(Nf=self.Nf, n_pu=self.n_pu,
                                loc_quality=self.loc_quality,
                                p_detect=self.p_detect, p_ghost=self.p_ghost,
                                seed=seed)
        env.reset()
        occ_total = occ_miss = idle_total = idle_fa = 0
        iou_sum = iou_cnt = 0
        for _ in range(n):
            boxes = env._true_boxes()
            O = env._true_occupancy(boxes)
            B, iou = env._localize(boxes)
            if boxes:
                iou_sum += iou; iou_cnt += 1
            decided_occ = (B >= eta)
            occ_mask = O > 0.5
            idle_mask = ~occ_mask
            occ_total += occ_mask.sum()
            idle_total += idle_mask.sum()
            occ_miss += np.sum(occ_mask & (~decided_occ))
            idle_fa += np.sum(idle_mask & decided_occ)
            env.t += 1
        Pmd = occ_miss / max(1, occ_total)
        Pfa = idle_fa / max(1, idle_total)
        mean_iou = iou_sum / max(1, iou_cnt)
        return {"iou": mean_iou, "eps_loc": 1.0 - mean_iou,
                "Pmd": Pmd, "Pfa": Pfa}
