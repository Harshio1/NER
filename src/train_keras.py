"""Keras implementation for the assignment: Bidirectional SimpleRNN / LSTM / GRU sequence labeller.

    python src/train_keras.py --cell gru            # one model
    python src/train_keras.py --cell all            # compare SimpleRNN vs LSTM vs GRU
Uses exactly the same features, augmentation and evaluation as the NumPy model.
"""
import argparse, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from common import *
from augment import generate
from metrics import evaluate


def arrays(sents, vocab, rnd=None, lower=False):
    T = max(len(s["tokens"]) for s in sents); N = len(sents)
    W = np.zeros((N, T), "int32"); S = np.zeros((N, T), "int32"); F = np.zeros((N, T, 2), "float32")
    Y = np.zeros((N, T), "int32"); M = np.zeros((N, T), "float32")
    for i, s in enumerate(sents):
        w, sf, c, f = vocab.encode(s["tokens"]); n = len(w)
        y = [TAG2ID[t] for t in s["tags"]]
        if rnd is not None:
            if rnd.random() < 0.5: c = [0.0] * n
            w = [1 if rnd.random() < (0.35 if y[k] else 0.12) else w[k] for k in range(n)]
        if lower: c = [0.0] * n
        W[i, :n], S[i, :n], Y[i, :n], M[i, :n] = w, sf, y, 1
        F[i, :n, 0], F[i, :n, 1] = c, f
    return [W, S, F], Y, M


def build(cell, V, S, H=48):
    w = layers.Input((None,), dtype="int32"); s = layers.Input((None,), dtype="int32"); f = layers.Input((None, 2))
    x = layers.Concatenate()([layers.Embedding(V, 32, mask_zero=True)(w), layers.Embedding(S, 8)(s), f])
    RNN = {"simplernn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[cell]
    x = layers.Bidirectional(RNN(H, return_sequences=True))(x)
    out = layers.TimeDistributed(layers.Dense(len(TAGS), activation="softmax"))(x)
    m = Model([w, s, f], out)
    m.compile(tf.keras.optimizers.Adam(0.008), "sparse_categorical_crossentropy", weighted_metrics=["accuracy"])
    return m


def decode(model, sents, vocab, lower=False):
    X, _, M = arrays(sents, vocab, lower=lower)
    p = model.predict(X, verbose=0).argmax(-1)
    return [[TAGS[j] for j in row[: int(m.sum())]] for row, m in zip(p, M)]


def run(cell, a):
    real, test = load_sentences(a.train), load_sentences(a.test)
    aug, dev = generate(a.n_aug, "train", 1), generate(150, "dev", 2)
    vocab = Vocab.build(real + aug); data = real * 6 + aug
    rnd = np.random.default_rng(0); tf.keras.utils.set_random_seed(0)
    model = build(cell, len(vocab.words), len(vocab.sufs))
    for ep in range(a.epochs):
        X, Y, M = arrays(data, vocab, rnd)
        h = model.fit(X, Y, sample_weight=M, batch_size=32, epochs=1, verbose=0)
        if (ep + 1) % 5 == 0:
            print(f"[{cell}] epoch {ep+1} loss={h.history['loss'][0]:.4f}  dev F1={evaluate([s['tags'] for s in dev], decode(model, dev, vocab))['f1']:.3f}")
    gold = lambda S: [s["tags"] for s in S]
    return {"dev": evaluate(gold(dev), decode(model, dev, vocab)),
            "test": evaluate(gold(test), decode(model, test, vocab)),
            "test_lowercase": evaluate(gold(test), decode(model, test, vocab, True)),
            "test_predictions": [{"id": s["id"], "tokens": s["tokens"], "gold": s["tags"], "pred": p}
                                 for s, p in zip(test, decode(model, test, vocab))]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="gru", choices=["simplernn", "lstm", "gru", "all"])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--n_aug", type=int, default=700)
    ap.add_argument("--train", default=str(ROOT / "data/sentences.tsv")); ap.add_argument("--test", default=str(ROOT / "data/test_sentences.tsv"))
    a = ap.parse_args()
    cells = ["simplernn", "lstm", "gru"] if a.cell == "all" else [a.cell]
    res = {c: run(c, a) for c in cells}
    print(f"\n{'cell':10s}{'dev F1':>8s}{'test P':>8s}{'test R':>8s}{'test F1':>9s}{'lower F1':>10s}")
    for c, r in res.items():
        print(f"{c:10s}{r['dev']['f1']:8.3f}{r['test']['precision']:8.3f}{r['test']['recall']:8.3f}{r['test']['f1']:9.3f}{r['test_lowercase']['f1']:10.3f}")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results/keras_results.json", "w"), indent=1)
