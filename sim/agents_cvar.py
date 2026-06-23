"""
agents_cvar.py -- Distributional (quantile) DQN with a CVaR collision
constraint, implementing method M1 of the paper, in pure NumPy.

Key ideas (paper Section IV-E, Theorem D):
  * The critic outputs N_QUANT quantiles of the action-value distribution
    (QR-DQN style) instead of a scalar Q. This is the "learn the full return
    distribution" step.
  * Action selection maximizes a CVaR utility of the throughput return:
        CVaR_alpha[Z] = mean of the lowest alpha-fraction of quantiles.
  * The PU-collision constraint is imposed on the TAIL of the collision
    return via a separate running CVaR estimate, enforced by dual ascent on
    a multiplier mu (Algorithm 1).

This reuses the MLP/replay infrastructure shape from agents.py but the head
is vectorized over quantiles.
"""

import numpy as np
from agents import Replay


def huber_quantile_grad(pred, target, taus, kappa=1.0):
    """Gradient of the quantile-Huber loss w.r.t. pred.
    pred:   (B, NQ)   predicted quantiles for taken action
    target: (B, NQ')  target quantiles (from next-state distribution)
    taus:   (NQ,)     quantile midpoints
    Returns grad wrt pred, shape (B, NQ).
    """
    B, NQ = pred.shape
    NQ2 = target.shape[1]
    # pairwise TD errors u_{ij} = target_j - pred_i
    u = target[:, None, :] - pred[:, :, None]          # (B, NQ, NQ2)
    # huber derivative
    hub = np.where(np.abs(u) <= kappa, -u, -kappa * np.sign(u))  # d/dpred
    rho = np.abs(taus[None, :, None] - (u < 0).astype(float)) * hub
    grad = rho.mean(axis=2)                             # average over targets
    return grad / B


class QuantileNet:
    def __init__(self, n_in, n_actions, n_quant, hidden=64, lr=5e-4, seed=0):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / n_in); s2 = np.sqrt(2.0 / hidden)
        self.na, self.nq = n_actions, n_quant
        self.W1 = rng.normal(0, s1, (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, (hidden, hidden)); self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, s2, (hidden, n_actions * n_quant))
        self.b3 = np.zeros(n_actions * n_quant)
        self.lr = lr

    def forward(self, x):
        z1 = x @ self.W1 + self.b1; a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(0, z2)
        out = a2 @ self.W3 + self.b3                    # (B, na*nq)
        q = out.reshape(-1, self.na, self.nq)           # (B, na, nq)
        return q, (x, z1, a1, z2, a2)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def train_step(self, x, a_idx, grad_q):
        """grad_q: (B, nq) gradient wrt quantiles of taken action."""
        q, (x_, z1, a1, z2, a2) = self.forward(x)
        B = x.shape[0]
        dout = np.zeros((B, self.na, self.nq))
        dout[np.arange(B), a_idx, :] = grad_q
        dout = dout.reshape(B, self.na * self.nq)
        dW3 = a2.T @ dout; db3 = dout.sum(0)
        da2 = dout @ self.W3.T; dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        dW1 = x_.T @ dz1; db1 = dz1.sum(0)
        for p, g in zip(self.params(), [dW1, db1, dW2, db2, dW3, db3]):
            np.clip(g, -1.0, 1.0, out=g)
            p -= self.lr * g


class CVaRDistDQN:
    """Distributional DQN with CVaR action selection + CVaR collision dual.

    State transform defaults to identity (continuous belief map). Subclass or
    set .transform to change the state representation.
    """
    name = "D-LA-JSSA (CVaR-Dist)"

    def __init__(self, env, n_quant=16, hidden=64, lr=5e-4, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=20000, batch=64,
                 target_sync=500, cvar_alpha=0.5, pcol_max=0.05,
                 cvar_beta=0.2, mu_lr=0.02, mu_max=20.0, seed=0):
        self.env = env
        self.gamma = gamma; self.batch = batch; self.target_sync = target_sync
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.nq = n_quant
        self.taus = (np.arange(n_quant) + 0.5) / n_quant
        dim = env.Nf
        self.q = QuantileNet(dim, env.n_actions, n_quant, hidden, lr, seed)
        self.qt = QuantileNet(dim, env.n_actions, n_quant, hidden, lr, seed + 1)
        self.qt.copy_from(self.q)
        self.buf = Replay(50000, dim, seed)
        self.steps = 0
        # risk levels
        self.cvar_alpha = cvar_alpha   # tail level for throughput utility
        self.cvar_beta = cvar_beta     # tail level for collision constraint
        # collision-tail dual control
        self.pcol_max = pcol_max
        self.mu = 1.0; self.mu_lr = mu_lr; self.mu_max = mu_max
        self.coll_hist = []            # recent collisions for CVaR estimate
        self.hist_len = 500

    def transform(self, belief):
        return belief

    def eps(self):
        f = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + (self.eps_end - self.eps_start) * f

    def _cvar_lower(self, quant):
        """CVaR_alpha of a distribution given its quantiles (lower tail)."""
        k = max(1, int(self.cvar_alpha * self.nq))
        return quant[..., :k].mean(axis=-1)

    def act(self, belief, greedy=False):
        s = self.transform(belief)
        if (not greedy) and self.rng.random() < self.eps():
            return self.rng.integers(self.env.n_actions)
        q, _ = self.q.forward(s[None, :])          # (1, na, nq)
        # risk-sensitive: pick action maximizing CVaR_alpha of return
        cvar = self._cvar_lower(q[0])              # (na,)
        return int(np.argmax(cvar))

    def collision_cvar(self):
        """Running CVaR_beta of the collision RATE, estimated over short
        windows so the quantity is on the same [0,1] scale as pcol_max. The
        tail (worst beta-fraction of windows) captures bursty interference."""
        W = 20
        if len(self.coll_hist) < W:
            return float(np.mean(self.coll_hist)) if self.coll_hist else 0.0
        arr = np.array(self.coll_hist)
        # non-overlapping window rates
        nwin = len(arr) // W
        rates = arr[:nwin * W].reshape(nwin, W).mean(axis=1)
        rates = np.sort(rates)                      # ascending
        k = max(1, int(self.cvar_beta * len(rates)))
        return float(rates[-k:].mean())             # mean of worst beta-fraction

    def learn(self, belief, a, r, belief2, collision=0.0):
        # dual-augmented reward using the collision-tail estimate
        self.coll_hist.append(collision)
        if len(self.coll_hist) > self.hist_len:
            self.coll_hist.pop(0)
        ccvar = self.collision_cvar()
        r_aug = r - self.mu * max(0.0, ccvar - self.pcol_max)

        s = self.transform(belief); s2 = self.transform(belief2)
        self.buf.add(s, a, r_aug, s2)
        self.steps += 1
        self.mu = min(self.mu_max,
                      max(0.0, self.mu + self.mu_lr * (ccvar - self.pcol_max)))

        if self.buf.size() < self.batch:
            return 0.0
        bs, ba, br, bs2 = self.buf.sample(self.batch)
        # target: greedy next action by CVaR, then distributional Bellman
        qn, _ = self.qt.forward(bs2)               # (B, na, nq)
        cvar_n = self._cvar_lower(qn)              # (B, na)
        a_star = np.argmax(cvar_n, axis=1)         # (B,)
        next_quant = qn[np.arange(len(ba)), a_star, :]   # (B, nq)
        target = br[:, None] + self.gamma * next_quant   # (B, nq)
        # current quantiles for taken action
        qc, _ = self.q.forward(bs)
        pred = qc[np.arange(len(ba)), ba, :]       # (B, nq)
        grad = huber_quantile_grad(pred, target, self.taus)
        self.q.train_step(bs, ba, grad)
        if self.steps % self.target_sync == 0:
            self.qt.copy_from(self.q)
        return 0.0


class CVaRDistDQN_TwoStep(CVaRDistDQN):
    name = "Two-step + CVaR"

    def transform(self, belief):
        return (belief >= self.env.eta).astype(float)
