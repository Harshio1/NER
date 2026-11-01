"""Step 5 - run the trained model on NEW voice recordings (audio -> text -> BIO tags).

    python src/predict_audio.py audio/test/test_01.wav --backend whisper
    python src/predict_audio.py --text "Deepika studied at Loyola College in Chennai"   # skip ASR
"""
import argparse, json
import numpy as np
from common import *
from gru_numpy import BiGRUTagger

MODEL = ROOT / "website/model/bigru_ner.json"


def load(path=MODEL):
    d = json.load(open(path))
    v = Vocab([], []); v.words, v.sufs = d["words"], d["suffixes"]
    v.w2i = {w: i for i, w in enumerate(v.words)}; v.s2i = {s: i for i, s in enumerate(v.sufs)}
    m = BiGRUTagger(len(v.words), len(v.sufs), len(d["tags"]), d["wd"], d["sd"], d["hidden"])
    m.P = {k: np.array(x) for k, x in d["params"].items()}
    return m, v


def tag(text, model, vocab):
    toks = tokenize(text)
    w, s, c, f = vocab.encode(toks)
    ids, _ = model.predict_batch([(w, s, c, f, [0] * len(toks))])
    return toks, [TAGS[i] for i in ids[0]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("wav", nargs="?"); ap.add_argument("--text")
    ap.add_argument("--backend", default="whisper"); ap.add_argument("--model_name", default="base.en")
    a = ap.parse_args()
    text = a.text
    if not text:
        from transcribe import transcribe_whisper, transcribe_google
        text = (transcribe_whisper if a.backend == "whisper" else lambda f, _: transcribe_google(f))([a.wav], a.model_name)[0]
    print("transcript:", text)
    m, v = load(); toks, tags = tag(text, m, v)
    for t, g in zip(toks, tags): print(f"{t:20s}{g}")
    print("entities:", [(typ, " ".join(toks[s:e])) for typ, s, e in sorted(spans(tags), key=lambda x: x[1])])
