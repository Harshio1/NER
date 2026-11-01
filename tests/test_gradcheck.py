"""Numerical gradient check of the hand-written BiGRU backward pass."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from gru_numpy import BiGRUTagger

rng = np.random.default_rng(1)
m = BiGRUTagger(V=12, S=7, n_tags=7, wd=5, sd=3, H=6, seed=3)
batch = []
for n in (5, 3, 4):
    batch.append((rng.integers(2, 12, n).tolist(), rng.integers(2, 7, n).tolist(),
                  rng.integers(0, 2, n).astype(float).tolist(), [1.0] + [0.0] * (n - 1), rng.integers(0, 7, n).tolist()))
args = BiGRUTagger.pad(batch)
loss, G = m.loss_and_grads(*args)
worst = 0
for k in m.P:
    flat = m.P[k].reshape(-1)
    for i in rng.choice(flat.size, min(6, flat.size), replace=False):
        o = flat[i]; eps = 1e-5
        flat[i] = o + eps; lp, _ = m.loss_and_grads(*args)
        flat[i] = o - eps; lm, _ = m.loss_and_grads(*args)
        flat[i] = o
        num = (lp - lm) / (2 * eps); ana = G[k].reshape(-1)[i]
        worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
print("max relative gradient error:", worst)
assert worst < 1e-4, "gradient check failed"
print("PASS")
