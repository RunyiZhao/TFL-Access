"""
agents.py -- Pure-NumPy agents for LA-JSSA (no torch dependency).

Agents:
  BeliefDQN      : proposed. State = continuous belief map. Small MLP Q-net
                   trained with experience replay + target net (all in numpy).
  TwoStepDQN     : ablation/baseline. State = HARD binary occupancy obtained
                   by thresholding the belief (NMS-style hard boxes). Same
                   learner otherwise -> isolates the value of the soft belief.
  FixedChannelDQN: baseline. Belief is averaged into K fixed channels and
                   binarized -> the classic fixed-grid state. Action head is
                   the same continuous-block set (fair comparison).
  MyopicPolicy   : non-learning. Pick the block with lowest summed belief.
  OraclePolicy   : upper bound. Sees true occupancy, picks a feasible block
                   maximizing idle cells.
"""

import numpy as np


# ----------------------------- MLP Q-network -----------------------------
class MLPQNet:
    def __init__(self, n_in, n_out, hidden=64, lr=5e-4, seed=0):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / n_in); s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, s1, (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, (hidden, hidden)); self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, s2, (hidden, n_out)); self.b3 = np.zeros(n_out)
        self.lr = lr

    def forward(self, x):
        z1 = x @ self.W1 + self.b1; a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(0, z2)
        q = a2 @ self.W3 + self.b3
        cache = (x, z1, a1, z2, a2)
        return q, cache

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def train_step(self, x, a_idx, target):
        q, (x_, z1, a1, z2, a2) = self.forward(x)
        B = x.shape[0]
        dq = np.zeros_like(q)
        pred = q[np.arange(B), a_idx]
        dq[np.arange(B), a_idx] = (pred - target) / B  # MSE grad
        # backprop
        dW3 = a2.T @ dq; db3 = dq.sum(0)
        da2 = dq @ self.W3.T; dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        dW1 = x_.T @ dz1; db1 = dz1.sum(0)
        for p, g in zip(self.params(), [dW1, db1, dW2, db2, dW3, db3]):
            np.clip(g, -1.0, 1.0, out=g)
            p -= self.lr * g
        return float(np.mean((pred - target) ** 2))


# ----------------------------- replay buffer -----------------------------
class Replay:
    def __init__(self, cap, dim, seed=0):
        self.cap = cap; self.dim = dim; self.i = 0; self.full = False
        self.s = np.zeros((cap, dim)); self.s2 = np.zeros((cap, dim))
        self.a = np.zeros(cap, dtype=int); self.r = np.zeros(cap)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2):
        self.s[self.i] = s; self.a[self.i] = a; self.r[self.i] = r; self.s2[self.i] = s2
        self.i = (self.i + 1) % self.cap
        if self.i == 0: self.full = True

    def sample(self, n):
        hi = self.cap if self.full else self.i
        idx = self.rng.integers(0, hi, size=min(n, hi))
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx]

    def size(self):
        return self.cap if self.full else self.i


# ----------------------------- DQN core -----------------------------
class DQNCore:
    """Generic DQN learner; subclasses define state_transform(belief)."""
    name = "DQN"

    def __init__(self, env, hidden=64, lr=5e-4, gamma=0.95, eps_start=1.0,
                 eps_end=0.05, eps_decay=20000, batch=64, target_sync=500,
                 pcol_max=0.05, mu_lr=0.5, seed=0):
        self.env = env
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.batch = batch; self.target_sync = target_sync
        self.rng = np.random.default_rng(seed)
        dim = self.state_dim()
        self.q = MLPQNet(dim, env.n_actions, hidden, lr, seed)
        self.qt = MLPQNet(dim, env.n_actions, hidden, lr, seed + 1)
        self.qt.copy_from(self.q)
        self.buf = Replay(50000, dim, seed)
        self.steps = 0
        # --- Lagrangian dual control (Algorithm 1) ---
        self.pcol_max = pcol_max
        self.mu = 1.0           # Lagrange multiplier on collision constraint
        self.mu_lr = mu_lr
        self.coll_ema = 0.0     # running collision estimate
        self.ema_beta = 0.01

    def state_dim(self):
        return self.env.Nf

    def state_transform(self, belief):
        return belief

    def eps(self):
        f = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + (self.eps_end - self.eps_start) * f

    def act(self, belief, greedy=False):
        s = self.state_transform(belief)
        if (not greedy) and self.rng.random() < self.eps():
            return self.rng.integers(self.env.n_actions)
        q, _ = self.q.forward(s[None, :])
        return int(np.argmax(q[0]))

    def learn(self, belief, a, r, belief2, collision=0.0):
        # augment reward with the dual penalty on the collision constraint
        r_aug = r - self.mu * max(0.0, collision - self.pcol_max)
        s = self.state_transform(belief); s2 = self.state_transform(belief2)
        self.buf.add(s, a, r_aug, s2)
        self.steps += 1
        # dual ascent on the multiplier using the running collision estimate
        self.coll_ema += self.ema_beta * (collision - self.coll_ema)
        self.mu = max(0.0, self.mu + self.mu_lr * (self.coll_ema - self.pcol_max))
        if self.buf.size() < self.batch:
            return 0.0
        bs, ba, br, bs2 = self.buf.sample(self.batch)
        qn, _ = self.qt.forward(bs2)
        target = br + self.gamma * np.max(qn, axis=1)
        loss = self.q.train_step(bs, ba, target)
        if self.steps % self.target_sync == 0:
            self.qt.copy_from(self.q)
        return loss


class BeliefDQN(DQNCore):
    name = "LA-JSSA (Belief-DQN)"
    # uses the continuous belief map directly -- default transform


class TwoStepDQN(DQNCore):
    name = "Two-step (hard box)"

    def state_transform(self, belief):
        # NMS-style hard decision at the env threshold -> binary occupancy
        return (belief >= self.env.eta).astype(float)


class FixedChannelDQN(DQNCore):
    name = "Fixed-channel DQN"
    K = 8  # number of fixed channels

    def state_dim(self):
        return self.K

    def state_transform(self, belief):
        # average belief into K equal channels, then binarize (classic state)
        chunks = np.array_split(belief, self.K)
        avg = np.array([c.mean() for c in chunks])
        return (avg >= self.env.eta).astype(float)


# ----------------------------- non-learning policies -----------------------------
class MyopicPolicy:
    name = "Myopic (min-belief)"

    def __init__(self, env, seed=0):
        self.env = env

    def act(self, belief, greedy=True):
        best, best_score = 0, np.inf
        for i, act in enumerate(self.env.actions):
            lo, hi = self.env.action_cells(act)
            score = belief[lo:hi + 1].sum() / max(1, (hi - lo + 1))
            if score < best_score:
                best_score, best = score, i
        return best

    def learn(self, *a, **k):
        return 0.0


class OraclePolicy:
    name = "Perfect-sensing oracle"

    def __init__(self, env, seed=0):
        self.env = env

    def act(self, belief, greedy=True):
        O = self.env.O  # cheats: true occupancy
        best, best_idle = 0, -1
        for i, act in enumerate(self.env.actions):
            lo, hi = self.env.action_cells(act)
            if np.any(O[lo:hi + 1] > 0.5):
                continue  # would collide
            idle = int(np.sum(O[lo:hi + 1] < 0.5))
            if idle > best_idle:
                best_idle, best = idle, i
        return best

    def learn(self, *a, **k):
        return 0.0
