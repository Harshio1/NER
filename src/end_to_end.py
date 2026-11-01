"""End-to-end check: audio -> Whisper text -> trained NER. For every entity in the typed reference
sentence, was the exact text AND type found in the tags predicted on the Whisper transcript?"""
import json
from common import *
from predict_audio import load, tag

model, vocab = load()
ref = load_sentences(ROOT / "data/test_sentences.tsv")
asr = {r["file"].rsplit(".", 1)[0]: r for r in read_tsv(ROOT / "data/transcripts_asr_test.tsv")}
rows, hit, total = [], 0, 0
for s in ref:
    key = Path(s["audio"]).stem
    toks, tags = tag(asr[key]["asr_text"], model, vocab)
    pred = {(t, " ".join(toks[a:b]).lower()) for t, a, b in spans(tags)}
    for t, a, b in sorted(spans(s["tags"]), key=lambda x: x[1]):
        txt = " ".join(s["tokens"][a:b]); ok = (t, txt.lower()) in pred
        hit += ok; total += 1
        rows.append({"id": s["id"], "entity": txt, "type": t, "found": ok, "asr_text": asr[key]["asr_text"],
                     "predicted": [f"{ty}:{' '.join(toks[x:y])}" for ty, x, y in sorted(spans(tags), key=lambda z: z[1])]})
out = {"found": hit, "total": total, "rate": hit / total, "rows": rows}
(ROOT / "results/end_to_end.json").write_text(json.dumps(out, indent=1))
print(f"end-to-end: {hit}/{total} reference entities recovered exactly")
for r in rows: print(r["id"], r["type"], r["entity"], "OK" if r["found"] else "MISSED", r["predicted"])
