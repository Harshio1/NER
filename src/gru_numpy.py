"""Bidirectional GRU sequence tagger written in pure NumPy (manual BPTT + Adam).

Used for (a) the results reported in this project and (b) the in-browser demo, whose
JavaScript forward pass reads the weights exported by `export()`.
Input per token = [word emb | suffix emb | capitalised flag | first-token flag].
"""
import json
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def flip(a, idx):
    return np.take_along_axis(a, idx[:, :, None], 1)


def gru_forward(x, W, Uzr, Un, b, H):
    B, T, _ = x.shape
    xw = x @ W + b
    h = np.zeros((B, H)); out = np.zeros((B, T, H)); cache = []
    for t in range(T):
        zr = sigmoid(xw[:, t, :2 * H] + h @ Uzr)
        z, r = zr[:, :H], zr[:, H:]
        n = np.tanh(xw[:, t, 2 * H:] + (r * h) @ Un)
        hn = (1 - z) * n + z * h
        cache.append((z, r, n, h)); out[:, t] = hn; h = hn
    return out, cache


def gru_backward(dH, x, cache, W, Uzr, Un, H):
    B, T, D = x.shape
    dxw = np.zeros((B, T, 3 * H)); dUzr = np.zeros_like(Uzr); dUn = np.zeros_like(Un)
    dh_next = np.zeros((B, H))
    for t in reversed(range(T)):
        z, r, n, hp = cache[t]
        dh = dH[:, t] + dh_next
        dn_pre = dh * (1 - z) * (1 - n * n)
        dz_pre = dh * (hp - n) * z * (1 - z)
        dhp = dh * z
        drh = dn_pre @ Un.T
        dhp = dhp + drh * r
        dr_pre = drh * hp * r * (1 - r)
        dzr = np.concatenate([dz_pre, dr_pre], 1)
        dhp = dhp + dzr @ Uzr.T
        dxw[:, t, :2 * H] = dzr; dxw[:, t, 2 * H:] = dn_pre
        dUzr += hp.T @ dzr; dUn += (r * hp).T @ dn_pre
        dh_next = dhp
    dW = x.reshape(-1, D).T @ dxw.reshape(-1, 3 * H)
    return dxw @ W.T, dW, dUzr, dUn, dxw.sum((0, 1))


class BiGRUTagger:
    def __init__(self, V, S, n_tags, wd=32, sd=8, H=48, seed=0):
        rng = np.random.default_rng(seed)
        D = wd + sd + 2
        self.wd, self.sd, self.H, self.D, self.n_tags = wd, sd, H, D, n_tags
        P = {"Ew": rng.normal(0, .1, (V, wd)), "Es": rng.normal(0, .1, (S, sd))}
        P["Ew"][0] = 0; P["Es"][0] = 0
        for d in "fb":
            P["W" + d] = rng.normal(0, np.sqrt(2 / (D + 3 * H)), (D, 3 * H))
            P["Uzr" + d] = rng.normal(0, np.sqrt(1 / H), (H, 2 * H))
            P["Un" + d] = rng.normal(0, np.sqrt(1 / H), (H, H))
            P["b" + d] = np.zeros(3 * H)
        P["Wo"] = rng.normal(0, np.sqrt(2 / (2 * H + n_tags)), (2 * H, n_tags)); P["bo"] = np.zeros(n_tags)
        self.P = P
        self.m = {k: np.zeros_like(v) for k, v in P.items()}; self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.step = 0

    # ---- batch helpers ----
    @staticmethod
    def pad(batch):
        """batch: list of (wid, sid, cap, first, tag_ids) -> padded arrays + flip index."""
        L = np.array([len(b[0]) for b in batch]); T, B = L.max(), len(batch)
        wid = np.zeros((B, T), int); sid = np.zeros((B, T), int)
        cap = np.zeros((B, T)); first = np.zeros((B, T)); y = np.zeros((B, T), int); mask = np.zeros((B, T))
        for i, (w, s, c, f, t) in enumerate(batch):
            n = len(w); wid[i, :n] = w; sid[i, :n] = s; cap[i, :n] = c; first[i, :n] = f; y[i, :n] = t; mask[i, :n] = 1
        ar = np.arange(T)[None, :]
        idx = np.where(ar < L[:, None], L[:, None] - 1 - ar, ar)
        return wid, sid, cap, first, y, mask, idx

    def _inputs(self, wid, sid, cap, first):
        P = self.P
        return np.concatenate([P["Ew"][wid], P["Es"][sid], cap[..., None], first[..., None]], -1)

    def forward(self, wid, sid, cap, first, idx):
        P, H = self.P, self.H
        x = self._inputs(wid, sid, cap, first)
        hf, cf = gru_forward(x, P["Wf"], P["Uzrf"], P["Unf"], P["bf"], H)
        xr = flip(x, idx)
        hbr, cb = gru_forward(xr, P["Wb"], P["Uzrb"], P["Unb"], P["bb"], H)
        h = np.concatenate([hf, flip(hbr, idx)], -1)
        logits = h @ P["Wo"] + P["bo"]
        return logits, (x, xr, cf, cb, h)

    def loss_and_grads(self, wid, sid, cap, first, y, mask, idx):
        P, H = self.P, self.H
        logits, (x, xr, cf, cb, h) = self.forward(wid, sid, cap, first, idx)
        e = np.exp(logits - logits.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
        N = mask.sum()
        loss = -(np.log(np.take_along_axis(p, y[..., None], -1)[..., 0] + 1e-12) * mask).sum() / N
        dl = p.copy()
        np.put_along_axis(dl, y[..., None], np.take_along_axis(dl, y[..., None], -1) - 1, -1)
        dl *= mask[..., None] / N
        G = {"Wo": h.reshape(-1, 2 * H).T @ dl.reshape(-1, self.n_tags), "bo": dl.sum((0, 1))}
        dh = dl @ P["Wo"].T
        dxf, G["Wf"], G["Uzrf"], G["Unf"], G["bf"] = gru_backward(dh[..., :H], x, cf, P["Wf"], P["Uzrf"], P["Unf"], H)
        dxbr, G["Wb"], G["Uzrb"], G["Unb"], G["bb"] = gru_backward(flip(dh[..., H:], idx), xr, cb, P["Wb"], P["Uzrb"], P["Unb"], H)
        dx = dxf + flip(dxbr, idx)
        G["Ew"] = np.zeros_like(P["Ew"]); G["Es"] = np.zeros_like(P["Es"])
        np.add.at(G["Ew"], wid, dx[..., :self.wd]); np.add.at(G["Es"], sid, dx[..., self.wd:self.wd + self.sd])
        G["Ew"][0] = 0; G["Es"][0] = 0
        return loss, G

    def adam(self, G, lr=0.01, clip=5.0, b1=0.9, b2=0.999):
        norm = np.sqrt(sum((g ** 2).sum() for g in G.values()))
        s = min(1.0, clip / (norm + 1e-9)); self.step += 1
        for k, g in G.items():
            g = g * s
            self.m[k] = b1 * self.m[k] + (1 - b1) * g; self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.step); vh = self.v[k] / (1 - b2 ** self.step)
            self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def predict_batch(self, batch):
        wid, sid, cap, first, y, mask, idx = self.pad(batch)
        logits, _ = self.forward(wid, sid, cap, first, idx)
        return logits.argmax(-1), mask

    def export(self, path, vocab, tags):
        r = lambda a: np.round(a, 4).tolist()
        d = {"tags": tags, "words": vocab.words, "suffixes": vocab.sufs, "hidden": self.H,
             "wd": self.wd, "sd": self.sd, "params": {k: r(v) for k, v in self.P.items()}}
        with open(path, "w") as f:
            json.dump(d, f, separators=(",", ":"))
