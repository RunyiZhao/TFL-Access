"""Run ONE ablation variant (1 seed) and cache result to disk. Combine later."""
import sys, os, json
import numpy as np
from env import ContinuousTFEnv
from agents import BeliefDQN
from agents_cvar import CVaRDistDQN, CVaRDistDQN_TwoStep

TRAIN_STEPS = 16000
EVAL_STEPS = 3500
CACHE = os.path.join(os.path.dirname(__file__), "ablation_cache.json")

def make_env(seed, lam_u=0.3):
    return ContinuousTFEnv(Nf=64, n_pu=3, snr_quality=1.2, eta=0.5,
                           lam_c=2.0, lam_u=lam_u, seed=seed)

def run_variant(name, seed):
    if name == "Full":
        env = make_env(seed); agent = CVaRDistDQN(env, seed=seed, cvar_alpha=0.6, cvar_beta=0.2, pcol_max=0.05)
    elif name == "-M1":
        env = make_env(seed); agent = BeliefDQN(env, seed=seed, pcol_max=0.05)
    elif name == "-M2":
        env = make_env(seed); agent = CVaRDistDQN_TwoStep(env, seed=seed, cvar_alpha=0.6, cvar_beta=0.2, pcol_max=0.05)
    elif name == "-M3":
        env = make_env(seed, lam_u=0.0); agent = CVaRDistDQN(env, seed=seed, cvar_alpha=0.6, cvar_beta=0.2, pcol_max=0.05)
    b = env.reset()
    for _ in range(TRAIN_STEPS):
        a = agent.act(b); b2, r, _, info = env.step(a); agent.learn(b, a, r, b2, collision=info["collision"]); b = b2
    env_eval = make_env(seed + 100); agent.env = env_eval; b = env_eval.reset()
    thru = coll = 0.0
    for _ in range(EVAL_STEPS):
        a = agent.act(b, greedy=True); b, r, _, info = env_eval.step(a); thru += info["thru"]; coll += info["collision"]
    return thru / EVAL_STEPS, coll / EVAL_STEPS

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    t, c = run_variant(name, seed)
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    cache[f"{name}|{seed}"] = [t, c]
    json.dump(cache, open(CACHE, "w"))
    print(f"{name} seed{seed}: thru={t:.4f} coll={c:.4f}  [{len(cache)} cached]")
