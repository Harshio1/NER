"""Step 2 - annotation helper.

  suggest : propose [entity|TYPE] markup for each ASR transcript (gazetteer + capitalised-run heuristic).
            You MUST review/correct the suggestions by hand - that manual pass is the gold annotation.
  build   : convert the reviewed markup file into CoNLL (.conll), CSV and JSONL labelled datasets.

  python src/annotate.py suggest --asr data/transcripts_asr_train.tsv --out data/sentences_suggested.tsv
  python src/annotate.py build   --in data/sentences.tsv --prefix data/annotated_train
"""
import argparse, csv, json
from common import *
from augment import POOLS

GAZ = {}
for typ, d in POOLS.items():
    for split in d.values():
        for e in split: GAZ[e.lower()] = typ


def suggest(text):
    toks, out, i = tokenize(text), [], 0
    while i < len(toks):
        hit = None
        for n in range(5, 0, -1):
            if " ".join(toks[i:i + n]).lower() in GAZ:
                hit = (n, GAZ[" ".join(toks[i:i + n]).lower()]); break
        if hit:
            out.append(f"[{' '.join(toks[i:i+hit[0]])}|{hit[1]}]"); i += hit[0]
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)


def build(src, prefix):
    sents = load_sentences(src)
    with open(prefix + ".conll", "w", encoding="utf-8") as f:
        for s in sents:
            f.write(f"# {s['id']}\n" + "".join(f"{t}\t{g}\n" for t, g in zip(s["tokens"], s["tags"])) + "\n")
    with open(prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sentence_id", "token_index", "token", "bio_tag"])
        for s in sents:
            for i, (t, g) in enumerate(zip(s["tokens"], s["tags"])): w.writerow([s["id"], i, t, g])
    with open(prefix + ".jsonl", "w", encoding="utf-8") as f:
        for s in sents: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {prefix}.conll/.csv/.jsonl  ({len(sents)} sentences, {sum(len(s['tokens']) for s in sents)} tokens)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["suggest", "build"])
    ap.add_argument("--asr"); ap.add_argument("--out"); ap.add_argument("--in", dest="src"); ap.add_argument("--prefix")
    a = ap.parse_args()
    if a.cmd == "suggest":
        rows = read_tsv(a.asr)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("id\taudio\tmarked\n")
            for i, r in enumerate(rows, 1): f.write(f"S{i:02d}\taudio/{r['file']}\t{suggest(r['asr_text'])}\n")
        print("wrote", a.out, "- now review it by hand")
    else:
        build(a.src, a.prefix)
