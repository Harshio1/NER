"""Train the BiGRU tagger, run ablations, evaluate on the real test recordings, export everything."""
import json, sys, time, argparse
import numpy as np
from common import *
from augment import generate
from metrics import evaluate
from gru_numpy import BiGRUTagger

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--n_aug", type=int, default=700)
ap.add_argument("--train", default=str(ROOT / "data/sentences.tsv"))
ap.add_argument("--test", default=str(ROOT / "data/test_sentences.tsv"))
ap.add_argument("--out", default=str(ROOT / "results/evaluation_results.json"))
ap.add_argument("--no_web", action="store_true", help="do not overwrite website/data and website/model")
args = ap.parse_args()

real = load_sentences(args.train)
test = load_sentences(args.test)
aug = generate(args.n_aug, "train", seed=1)
dev = generate(150, "dev", seed=2)
vocab = Vocab.build(real + aug)


def enc(s, lower=False):
    w, sf, c, f = vocab.encode(s["tokens"])
    if lower: c = [0.0] * len(c)
    return (w, sf, c, f, [TAG2ID[t] for t in s["tags"]])


def decode(model, sents, lower=False):
    preds = []
    for i in range(0, len(sents), 64):
        chunk = sents[i:i + 64]
        ids, mask = model.predict_batch([enc(s, lower) for s in chunk])
        for row, m in zip(ids, mask):
            preds.append([TAGS[j] for j in row[: int(m.sum())]])
    return preds


def score(model, sents, lower=False):
    return evaluate([s["tags"] for s in sents], decode(model, sents, lower))


def train(data, seed, epochs, log=None):
    rnd = np.random.default_rng(seed)
    model = BiGRUTagger(len(vocab.words), len(vocab.sufs), len(TAGS), seed=seed)
    # real recordings are repeated so they are not drowned out by synthetic data
    pool = data
    for ep in range(epochs):
        order = rnd.permutation(len(pool)); losses = []
        for i in range(0, len(order), 32):
            batch = []
            for j in order[i:i + 32]:
                w, sf, c, f, y = enc(pool[j], lower=rnd.random() < 0.5)
                w = [1 if (rnd.random() < (0.35 if y[k] else 0.12)) else w[k] for k in range(len(w))]
                batch.append((w, sf, c, f, y))
            loss, G = model.loss_and_grads(*BiGRUTagger.pad(batch))
            model.adam(G, lr=0.008); losses.append(loss)
        if log is not None:
            log.append({"epoch": ep + 1, "loss": float(np.mean(losses)),
                        "dev_f1": score(model, dev)["f1"], "test_f1": score(model, test)["f1"]})
    return model


def agg(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


t0 = time.time()
configs = {"real_only_10": real * 6, "real_plus_augmented": real * 6 + aug}
summary, best = {}, None
for name, data in configs.items():
    runs = []
    for sd in range(args.seeds):
        log = [] if name == "real_plus_augmented" else None
        m = train(data, sd, args.epochs, log)
        r = {"dev": score(m, dev)["f1"], "test_cased": score(m, test)["f1"],
             "test_lower": score(m, test, True)["f1"]}
        runs.append(r)
        if name == "real_plus_augmented" and (best is None or r["dev"] > best[0]):
            best = (r["dev"], m, log, sd)   # model selection uses DEV only, never the test set
        print(f"{name} seed {sd}: {r}  [{time.time()-t0:.0f}s]", flush=True)
    summary[name] = {k: agg([r[k] for r in runs]) for k in runs[0]}

_, model, log, seed = best
preds_c, preds_l = decode(model, test), decode(model, test, True)
res = {
    "config": {"model": "Bidirectional GRU (NumPy)", "hidden": model.H, "word_dim": model.wd, "suffix_dim": model.sd,
               "epochs": args.epochs, "optimizer": "Adam lr=0.008", "seeds": args.seeds, "selected_seed": seed,
               "n_real_train": len(real), "n_aug_train": len(aug), "n_dev": len(dev), "n_test": len(test),
               "vocab": len(vocab.words), "params": int(sum(v.size for v in model.P.values()))},
    "test_cased": score(model, test), "test_lowercase_asr_style": score(model, test, True),
    "dev_unseen_entities": score(model, dev), "ablation_mean_std_f1": summary, "training_curve": log,
}
rows = []
for s, pc, pl in zip(test, preds_c, preds_l):
    rows.append({"id": s["id"], "audio": s["audio"], "tokens": s["tokens"], "gold": s["tags"], "pred": pc, "pred_lowercase": pl})
res["test_predictions"] = rows

from pathlib import Path
Path(args.out).parent.mkdir(exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
if args.no_web:
    print("saved", args.out); sys.exit(0)
web = ROOT / "website"
(web / "data").mkdir(parents=True, exist_ok=True); (web / "model").mkdir(parents=True, exist_ok=True)
(web / "data" / "results.json").write_text(json.dumps(res))
model.export(web / "model" / "bigru_ner.json", vocab, TAGS)
ds = [{"split": "train", **{k: s[k] for k in ("id", "audio", "tokens", "tags")}} for s in real] + \
     [{"split": "test", **{k: s[k] for k in ("id", "audio", "tokens", "tags")}} for s in test]
(web / "data" / "dataset.json").write_text(json.dumps(ds))
print(json.dumps({k: res[k] for k in ("test_cased", "test_lowercase_asr_style", "ablation_mean_std_f1")}, indent=1))
