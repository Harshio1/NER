"""Entity-level (strict span match, CoNLL-style) and token-level evaluation."""
from collections import defaultdict
from common import spans


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(gold_seqs, pred_seqs):
    c = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn per type
    tok_ok = tok_n = 0
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = spans(g), spans(p)
        for s in gs & ps: c[s[0]][0] += 1
        for s in ps - gs: c[s[0]][1] += 1
        for s in gs - ps: c[s[0]][2] += 1
        tok_ok += sum(a == b for a, b in zip(g, p)); tok_n += len(g)
    out = {"per_type": {}, "token_accuracy": tok_ok / max(tok_n, 1)}
    T = [0, 0, 0]
    for typ in ("PER", "ORG", "LOC"):
        tp, fp, fn = c[typ]
        p, r, f = prf(tp, fp, fn)
        out["per_type"][typ] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
        T[0] += tp; T[1] += fp; T[2] += fn
    p, r, f = prf(*T)
    out.update({"precision": p, "recall": r, "f1": f, "support": T[0] + T[2]})
    return out
