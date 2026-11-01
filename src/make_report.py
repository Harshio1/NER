"""Builds report/REPORT.md (evaluation results + limitations analysis) from results/evaluation_results.json."""
import json
from pathlib import Path
from common import ROOT, load_sentences, spans

r = json.load(open(ROOT / "results/evaluation_results.json"))
_tr = load_sentences(ROOT / "data/sentences.tsv"); _te = load_sentences(ROOT / "data/test_sentences.tsv")
N_TOK = sum(len(s["tokens"]) for s in _tr); N_ENT = sum(len(spans(s["tags"])) for s in _tr); N_TEST_ENT = sum(len(spans(s["tags"])) for s in _te)
c, a = r["config"], r["ablation_mean_std_f1"]
pc = lambda v: f"{v*100:.1f}%"
ms = lambda d: f"{d['mean']*100:.1f}% ± {d['std']*100:.1f}"
L = []
L += ["# Evaluation Report: Named Entity Recognition from Speech with a Recurrent Network", "",
      "## 1. Setup", "",
      f"- Model: bidirectional GRU, hidden size {c['hidden']} per direction, word embedding {c['word_dim']}, suffix embedding {c['suffix_dim']}, plus capitalisation and first-word flags ({c['params']:,} parameters).",
      f"- Training data: {c['n_real_train']} real recorded sentences (repeated 6x) + {c['n_aug_train']} template-augmented sentences. Vocabulary: {c['vocab']} lower-cased types.",
      f"- Optimiser: {c['optimizer']}, {c['epochs']} epochs, batch 32, random word dropout to the unknown token, random removal of capitalisation (50%).",
      f"- Evaluation: {c['n_test']} new recordings (held out; their entities appear nowhere in training or augmentation) and {c['n_dev']} synthetic sentences built from entities not used in training.",
      f"- Metric: strict entity-level precision/recall/F1 (type and exact span must match). The model shown is the best of {c['seeds']} seeds **chosen on the synthetic dev set, never on the test recordings** (seed {c['selected_seed']}).", "",
      "## 2. Results", "",
      "| Evaluation set | Precision | Recall | F1 | Token accuracy |", "|---|---|---|---|---|"]
for k, n in (("test_cased", "Test recordings (original casing)"), ("test_lowercase_asr_style", "Test recordings (lower-case, ASR style)"), ("dev_unseen_entities", "Synthetic dev (unseen entities)")):
    x = r[k]; L.append(f"| {n} | {pc(x['precision'])} | {pc(x['recall'])} | {pc(x['f1'])} | {pc(x['token_accuracy'])} |")
L += ["", "### Per entity type (test recordings, original casing)", "", "| Type | Precision | Recall | F1 | Gold entities |", "|---|---|---|---|---|"]
for t, m in r["test_cased"]["per_type"].items():
    L.append(f"| {t} | {pc(m['precision'])} | {pc(m['recall'])} | {pc(m['f1'])} | {m['support']} |")
L += ["", f"### Ablation: does augmentation matter? (mean ± std F1 over {c['seeds']} seeds)", "",
      "| Training data | Test (cased) | Test (lower-case) | Synthetic dev |", "|---|---|---|---|"]
for k, n in (("real_only_10", "10 real recordings only"), ("real_plus_augmented", "10 real + augmented")):
    L.append(f"| {n} | {ms(a[k]['test_cased'])} | {ms(a[k]['test_lower'])} | {ms(a[k]['dev'])} |")
L += ["", "### Predictions on the new recordings", ""]
errs = []
for p in r["test_predictions"]:
    L += [f"**{p['id']}**", "", "| Token | Gold | Predicted | |", "|---|---|---|---|"]
    for t, g, q in zip(p["tokens"], p["gold"], p["pred"]):
        L.append(f"| {t} | {g} | {q} | {'' if g == q else '✗'} |")
        if g != q: errs.append((p["id"], t, g, q))
    L.append("")
L += ["## 3. Analysis: challenges of a small, self-created audio dataset", ""]
if errs:
    L.append("**Errors observed:** " + "; ".join(f"`{t}` in {i} was gold `{g}` but predicted `{q}`" for i, t, g, q in errs) +
             ". Where this is a name the model has never seen, only the surrounding words and the word ending are available as evidence, so a wrong guess is expected.")
else:
    L.append("The model labelled every test token correctly in this run; with only a handful of test entities this should not be over-interpreted.")
L += ["", f"1. **Tiny data.** Ten sentences contain {N_TOK} tokens and {N_ENT} entities. Trained on them alone the model reaches only {ms(a['real_only_10']['test_cased'])} test F1 with a large spread across seeds; the embeddings and recurrent weights simply have too little to learn from.",
 f"2. **Augmentation is essential but inflates confidence.** Substituting entities into {25} templates lifts test F1 to {ms(a['real_plus_augmented']['test_cased'])}, and synthetic-dev F1 is much higher ({pc(r['dev_unseen_entities']['f1'])}) than test F1 ({pc(r['test_cased']['f1'])}). The synthetic data share a small set of phrasings, so the model partly learns the templates (training loss approaches zero). Performance on real, differently worded speech is the honest number.",
 "3. **Out-of-vocabulary names and ambiguous words.** Every new person, company or place is unknown to the embedding table. Word dropout makes the network lean on context and suffixes, which works when the context is clear and fails when a known word has another role: `Chennai` is a city in training, so `Chennai Super Kings` (a cricket team) is pulled towards LOC. Only one real training sentence names a sports team, so the template data cannot teach this.",
 "4. **Speech recognition changes the problem.** Speech-to-text engines may drop capitals and punctuation, split or misspell rare names (for example `Rameswaram`, `Sriharikota`) and mis-hear homophones. A named entity that is transcribed wrongly cannot be tagged correctly; the NER model inherits the ASR error rate. Casing removal was simulated during training, but acoustic errors were not.",
 "5. **Multi-word entities and boundaries.** Training names are mostly one or two tokens, so a three-token name (`Mahendra Singh Dhoni`) or a two-token name that is unfamiliar (`Smriti Mandhana`) is easily split or cut short. Strict span matching counts one boundary error as both a false positive and a false negative.",
 "6. **Single speaker, quiet room.** All recordings come from one voice. Generalisation to other accents, speeds and background noise is untested.",
 f"7. **Statistical power.** With {N_TEST_ENT} gold entities in the test recordings, a single error changes F1 by roughly {100//N_TEST_ENT} points. The results demonstrate the method; they are not a benchmark.", "",
 "### Improvements that would help", "",
 "- Record many more sentences from several speakers; use a public NER corpus (CoNLL-2003, WikiANN) for pre-training, then fine-tune on the spoken data.",
 "- Use character-level embeddings or sub-word / pretrained embeddings (GloVe, fastText) so unseen names share information with similar known ones.",
 "- Add a CRF layer to enforce valid BIO transitions (no `I-LOC` after `B-PER`).",
 "- Train on ASR output from the same engine used at test time (including its errors) and pass confidence scores or n-best lists downstream.",
 "- Use a transformer encoder (BERT/DistilBERT) fine-tuned for NER, or an end-to-end spoken-NER model, as a stronger baseline."]

# ---------------- extra sections: ASR, ASR-annotated run, end-to-end, Keras ----------------
from common import read_tsv
pc1 = lambda v: f"{v*100:.1f}%"
ref_txt = {Path(s["audio"]).stem: " ".join(s["tokens"]) for s in _tr + _te if s["audio"]}
L += ["", "## 4. Speech-to-text output (Whisper small.en) and word error rate", "",
      "| File | Typed reference | What Whisper heard | WER |", "|---|---|---|---|"]
asr_rows = []
for split in ("train", "test"):
    for row in read_tsv(ROOT / f"data/transcripts_asr_{split}.tsv"):
        stem = row["file"].rsplit(".", 1)[0]; asr_rows.append(float(row["wer_vs_reference"]))
        L.append(f"| {row['file']} | {ref_txt.get(stem, '')} | {row['asr_text']} | {row['wer_vs_reference']} |")
L += ["", f"Mean WER over {len(asr_rows)} recordings: **{sum(asr_rows)/len(asr_rows):.3f}**. Errors concentrate on Indian proper nouns (e.g. a name heard as `Strena Dey`, a place as `Dremesh`, `Wipro` as `the bro`).", "",
      "## 5. Second setting: annotating the Whisper transcripts as heard", "",
      "`data/sentences_asr.tsv` and `data/test_sentences_asr.tsv` hold the Whisper text with no word corrections; entities are marked only where a real name is present (so `the bro` and `Dremesh` stay unmarked).", ""]
ra = json.load(open(ROOT / "results/evaluation_results_asr.json")); aa = ra["ablation_mean_std_f1"]
L += ["| Setting | Test F1 (selected seed) | Test F1, mean ± std of 5 seeds | Gold test entities |", "|---|---|---|---|",
      f"| Typed reference annotation | {pc1(r['test_cased']['f1'])} | {ms(a['real_plus_augmented']['test_cased'])} | {r['test_cased']['support']} |",
      f"| Whisper-transcript annotation | {pc1(ra['test_cased']['f1'])} | {ms(aa['real_plus_augmented']['test_cased'])} | {ra['test_cased']['support']} |", ""]
e = json.load(open(ROOT / "results/end_to_end.json"))
L += ["## 6. End-to-end: speech → text → entities", "",
      f"The model trained on the typed reference data was run on the Whisper transcripts of the test recordings. A reference entity counts as recovered only if both its exact text and its type are predicted. **{e['found']} of {e['total']} entities were recovered.**", "",
      "| Sentence | Reference entity | Type | Recovered | Entities predicted on the Whisper text |", "|---|---|---|---|---|"]
for x in e["rows"]:
    L.append(f"| {x['id']} | {x['entity']} | {x['type']} | {'yes' if x['found'] else 'no'} | {', '.join(x['predicted'])} |")
L += ["", "Most misses trace to a speech-recognition error (a mis-heard or misspelt name), so the speech-to-text step caps the quality of the whole system. The exception is `Chennai Super Kings`: Whisper heard it correctly, but the tagger labelled it as locations because `Chennai` is a known city.", "",
      "## 7. SimpleRNN vs LSTM vs GRU (Keras)", ""]
kj = ROOT / "results/keras_results.json"
if kj.exists():
    k = json.load(open(kj))
    L += ["| Cell | Dev F1 | Test precision | Test recall | Test F1 | Test F1 (lower-case) |", "|---|---|---|---|---|---|"]
    for cell, v in k.items():
        L.append(f"| {cell} | {pc1(v['dev']['f1'])} | {pc1(v['test']['precision'])} | {pc1(v['test']['recall'])} | {pc1(v['test']['f1'])} | {pc1(v['test_lowercase']['f1'])} |")
else:
    L.append((ROOT / "report/keras_table.md").read_text(encoding="utf-8"))
# website data for the ASR section
(ROOT / "website/data/asr.json").write_text(json.dumps({"rows": [
    {"file": r_["file"], "ref": ref_txt.get(r_["file"].rsplit(".", 1)[0], ""), "asr": r_["asr_text"], "wer": float(r_["wer_vs_reference"])}
    for sp in ("train", "test") for r_ in read_tsv(ROOT / f"data/transcripts_asr_{sp}.tsv")], "end_to_end": e,
    "asr_run_f1": ra["test_cased"]["f1"]}))
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report/REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("wrote report/REPORT.md")
