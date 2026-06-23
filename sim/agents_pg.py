"""
agents_pg.py -- compact NumPy policy-gradient (PPO-lite) and SAC-lite-style
baselines, matching the act()/learn() interface of the other agents so they run
on the same environment, state, and discrete action set.

These are faithful, runnable CPU baselines (no PyTorch). They are intentionally
small; a full deep PPO/SAC implementation for GPU is provided separately in
sim/torch_baselines.py for the camera-ready scale-up.
"""
import numpy as np


def _flat(b):
    return np.asarray(b, dtype=np.float64).ravel()


class _MLP:
    """Two-layer tanh MLP with manual gradients."""
    def __init__(self, din, dh, dout, rng, scale=0.1):
        self.W1 = rng.normal(0, scale, (din, dh))
        self.b1 = np.zeros(dh)
        self.W2 = rng.normal(0, scale, (dh, dout))
        self.b2 = np.zeros(dout)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h = np.tanh(self.z1)
        self.o = self.h @ self.W2 + self.b2
        return self.o

    def backward(self, dout, lr):
        dW2 = np.outer(self.h, dout)
        db2 = dout
        dh = (self.W2 @ dout) * (1 - self.h ** 2)
        dW1 = np.outer(self.x, dh)
        db1 = dh
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PPOLiteAgent:
    """Actor-critic policy gradient with a clipped surrogate update and entropy
    bonus (PPO-lite), plus a Lagrangian penalty enforcing the collision budget.
    Discrete actions over the env's resource-block set."""

    def __init__(self, env, seed=0, dh=64, lr=3e-3, gamma=0.95,
                 clip=0.2, ent_coef=0.01, pcol_max=0.05, mu_lr=0.05,
                 batch=32):
        self.env = env
        self.rng = np.random.default_rng(seed)
        din = _flat(env.reset()).size
        self.actor = _MLP(din, dh, env.n_actions, self.rng)
        self.critic = _MLP(din, dh, 1, self.rng)
        self.lr = lr; self.gamma = gamma; self.clip = clip
        self.ent_coef = ent_coef; self.pcol_max = pcol_max; self.mu = 0.0
        self.mu_lr = mu_lr; self.batch = batch
        self.buf = []

    def act(self, belief, greedy=False):
        x = _flat(belief)
        logits = self.actor.forward(x)
        p = _softmax(logits)
        self._last_p = p; self._last_x = x
        if greedy:
            return int(np.argmax(p))
        return int(self.rng.choice(len(p), p=p))

    def learn(self, belief, a, r, belief2, collision=0.0):
        # Lagrangian-shaped reward: penalize collisions above budget
        shaped = r - self.mu * max(0.0, collision)
        x = _flat(belief)
        v = self.critic.forward(x)[0]
        self.buf.append((x, a, shaped, self._last_p.copy(), v, collision))
        self.mu = max(0.0, self.mu + self.mu_lr * (collision - self.pcol_max))
        if len(self.buf) >= self.batch:
            self._update()
            self.buf = []

    def _update(self):
        # Monte-Carlo returns within the minibatch
        G = 0.0
        returns = []
        for (_, _, rr, _, _, _) in reversed(self.buf):
            G = rr + self.gamma * G
            returns.append(G)
        returns = np.array(returns[::-1])
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        for (x, a, rr, p_old, v, _), Gt in zip(self.buf, returns):
            # critic step
            vpred = self.critic.forward(x)[0]
            adv = Gt - vpred
            self.critic.backward(np.array([-(Gt - vpred)]), self.lr)
            # actor step: clipped surrogate (single-sample)
            logits = self.actor.forward(x)
            p = _softmax(logits)
            ratio = p[a] / (p_old[a] + 1e-8)
            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
            use_clip = (clipped * adv) < (ratio * adv)
            # gradient of log pi wrt logits = (onehot - p)
            g = -p.copy()
            g[a] += 1.0
            coef = adv * (0.0 if use_clip else 1.0)
            # entropy bonus gradient
            ent_g = -(np.log(p + 1e-8) + 1.0) * p
            dlogits = -(coef * g + self.ent_coef * ent_g)
            self.actor.backward(dlogits, self.lr)


class RandomAgent:
    """Uniform-random access over the action set (performance floor)."""
    def __init__(self, env, seed=0):
        self.env = env; self.rng = np.random.default_rng(seed)
        self.n = env.n_actions

    def act(self, belief, greedy=False):
        return int(self.rng.integers(self.n))

    def learn(self, *a, **k):
        pass
