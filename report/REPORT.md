# Evaluation Report: Named Entity Recognition from Speech with a Recurrent Network

## 1. Setup

- Model: bidirectional GRU, hidden size 48 per direction, word embedding 32, suffix embedding 8, plus capitalisation and first-word flags (35,967 parameters).
- Training data: 10 real recorded sentences (repeated 6x) + 700 template-augmented sentences. Vocabulary: 234 lower-cased types.
- Optimiser: Adam lr=0.008, 30 epochs, batch 32, random word dropout to the unknown token, random removal of capitalisation (50%).
- Evaluation: 3 new recordings (held out; their entities appear nowhere in training or augmentation) and 150 synthetic sentences built from entities not used in training.
- Metric: strict entity-level precision/recall/F1 (type and exact span must match). The model shown is the best of 5 seeds **chosen on the synthetic dev set, never on the test recordings** (seed 3).

## 2. Results

| Evaluation set | Precision | Recall | F1 | Token accuracy |
|---|---|---|---|---|
| Test recordings (original casing) | 50.0% | 62.5% | 55.6% | 73.9% |
| Test recordings (lower-case, ASR style) | 44.4% | 50.0% | 47.1% | 69.6% |
| Synthetic dev (unseen entities) | 91.9% | 93.7% | 92.8% | 97.8% |

### Per entity type (test recordings, original casing)

| Type | Precision | Recall | F1 | Gold entities |
|---|---|---|---|---|
| PER | 25.0% | 33.3% | 28.6% | 3 |
| ORG | 100.0% | 66.7% | 80.0% | 3 |
| LOC | 50.0% | 100.0% | 66.7% | 2 |

### Ablation: does augmentation matter? (mean ± std F1 over 5 seeds)

| Training data | Test (cased) | Test (lower-case) | Synthetic dev |
|---|---|---|---|
| 10 real recordings only | 26.8% ± 10.7 | 22.0% ± 6.0 | 32.9% ± 2.6 |
| 10 real + augmented | 42.8% ± 8.6 | 36.8% ± 8.0 | 86.0% ± 5.6 |

### Predictions on the new recordings

**T01**

| Token | Gold | Predicted | |
|---|---|---|---|
| Narendra | B-PER | B-PER |  |
| Modi | I-PER | I-PER |  |
| visited | O | O |  |
| Madurai | B-LOC | B-LOC |  |
| and | O | O |  |
| met | O | O |  |
| officials | O | O |  |
| from | O | O |  |
| Wipro | B-ORG | B-ORG |  |

**T02**

| Token | Gold | Predicted | |
|---|---|---|---|
| Mahendra | B-PER | B-PER |  |
| Singh | I-PER | O | ✗ |
| Dhoni | I-PER | O | ✗ |
| played | O | O |  |
| for | O | O |  |
| Chennai | B-ORG | B-LOC | ✗ |
| Super | I-ORG | I-LOC | ✗ |
| Kings | I-ORG | B-LOC | ✗ |

**T03**

| Token | Gold | Predicted | |
|---|---|---|---|
| Smriti | B-PER | B-PER |  |
| Mandhana | I-PER | B-PER | ✗ |
| joined | O | O |  |
| Deloitte | B-ORG | B-ORG |  |
| in | O | O |  |
| Vishakhapatnam | B-LOC | B-LOC |  |

## 3. Analysis: challenges of a small, self-created audio dataset

**Errors observed:** `Singh` in T02 was gold `I-PER` but predicted `O`; `Dhoni` in T02 was gold `I-PER` but predicted `O`; `Chennai` in T02 was gold `B-ORG` but predicted `B-LOC`; `Super` in T02 was gold `I-ORG` but predicted `I-LOC`; `Kings` in T02 was gold `I-ORG` but predicted `B-LOC`; `Mandhana` in T03 was gold `I-PER` but predicted `B-PER`. Where this is a name the model has never seen, only the surrounding words and the word ending are available as evidence, so a wrong guess is expected.

1. **Tiny data.** Ten sentences contain 75 tokens and 21 entities. Trained on them alone the model reaches only 26.8% ± 10.7 test F1 with a large spread across seeds; the embeddings and recurrent weights simply have too little to learn from.
2. **Augmentation is essential but inflates confidence.** Substituting entities into 25 templates lifts test F1 to 42.8% ± 8.6, and synthetic-dev F1 is much higher (92.8%) than test F1 (55.6%). The synthetic data share a small set of phrasings, so the model partly learns the templates (training loss approaches zero). Performance on real, differently worded speech is the honest number.
3. **Out-of-vocabulary names and ambiguous words.** Every new person, company or place is unknown to the embedding table. Word dropout makes the network lean on context and suffixes, which works when the context is clear and fails when a known word has another role: `Chennai` is a city in training, so `Chennai Super Kings` (a cricket team) is pulled towards LOC. Only one real training sentence names a sports team, so the template data cannot teach this.
4. **Speech recognition changes the problem.** Speech-to-text engines may drop capitals and punctuation, split or misspell rare names (for example `Rameswaram`, `Sriharikota`) and mis-hear homophones. A named entity that is transcribed wrongly cannot be tagged correctly; the NER model inherits the ASR error rate. Casing removal was simulated during training, but acoustic errors were not.
5. **Multi-word entities and boundaries.** Training names are mostly one or two tokens, so a three-token name (`Mahendra Singh Dhoni`) or a two-token name that is unfamiliar (`Smriti Mandhana`) is easily split or cut short. Strict span matching counts one boundary error as both a false positive and a false negative.
6. **Single speaker, quiet room.** All recordings come from one voice. Generalisation to other accents, speeds and background noise is untested.
7. **Statistical power.** With 8 gold entities in the test recordings, a single error changes F1 by roughly 12 points. The results demonstrate the method; they are not a benchmark.

### Improvements that would help

- Record many more sentences from several speakers; use a public NER corpus (CoNLL-2003, WikiANN) for pre-training, then fine-tune on the spoken data.
- Use character-level embeddings or sub-word / pretrained embeddings (GloVe, fastText) so unseen names share information with similar known ones.
- Add a CRF layer to enforce valid BIO transitions (no `I-LOC` after `B-PER`).
- Train on ASR output from the same engine used at test time (including its errors) and pass confidence scores or n-best lists downstream.
- Use a transformer encoder (BERT/DistilBERT) fine-tuned for NER, or an end-to-end spoken-NER model, as a stronger baseline.

## 4. Speech-to-text output (Whisper small.en) and word error rate

| File | Typed reference | What Whisper heard | WER |
|---|---|---|---|
| sentence_01.ogg | Sundar Pichai is the chief executive of Google | Sundar Pichai is the chief executive of Google. | 0.000 |
| sentence_02.ogg | Infosys opened a new office in Bangalore | Infosys opened a new office in Bangalore. | 0.000 |
| sentence_03.ogg | I visited Chennai with my friend Sreenedhi last summer | I visited Chennai with my friend Strena Dey last summer. | 0.222 |
| sentence_04.ogg | Abdul Kalam was born in Rameshwaram | Abdul Kalam was born in Dremesh, with him. | 0.500 |
| sentence_05.ogg | The Indian Space Research Organization launched a rocket from Sriharikota | The Indian Space Research Organization launched a rocket from Sriharikota. | 0.000 |
| sentence_06.ogg | Virat Kohli played for Royal Challengers Bangalore | Virat Kohli played for Royal Challengers Bangalore | 0.000 |
| sentence_07.ogg | Anna University is located in Kandy near Chennai | Anna University is located in Gindi near Chennai. | 0.125 |
| sentence_08.ogg | The Reserve Bank of India is headquartered in Mumbai | The Reserve Bank of India is headquartered in Mumbai. | 0.000 |
| sentence_09.ogg | Kalpana Chawla worked for NASA | Kalpana Chabla have worked for NASA | 0.400 |
| sentence_10.ogg | Tesla opened a showroom in Delhi | Tesla opened the showroom in Delhi. | 0.167 |
| test_01.ogg | Narendra Modi visited Madurai and met officials from Wipro | Narainthra Modi visited Madurai and met officials from the bro. | 0.333 |
| test_02.ogg | Mahendra Singh Dhoni played for Chennai Super Kings | Mahindra Singh Dhoni played for Chennai Super Kings | 0.125 |
| test_03.ogg | Smriti Mandhana joined Deloitte in Vishakhapatnam | Subhriti Mandana joined Deloitte in Vishalakapatnam | 0.500 |

Mean WER over 13 recordings: **0.182**. Errors concentrate on Indian proper nouns (e.g. a name heard as `Strena Dey`, a place as `Dremesh`, `Wipro` as `the bro`).

## 5. Second setting: annotating the Whisper transcripts as heard

`data/sentences_asr.tsv` and `data/test_sentences_asr.tsv` hold the Whisper text with no word corrections; entities are marked only where a real name is present (so `the bro` and `Dremesh` stay unmarked).

| Setting | Test F1 (selected seed) | Test F1, mean ± std of 5 seeds | Gold test entities |
|---|---|---|---|
| Typed reference annotation | 55.6% | 42.8% ± 8.6 | 8 |
| Whisper-transcript annotation | 44.4% | 40.4% ± 6.7 | 7 |

## 6. End-to-end: speech → text → entities

The model trained on the typed reference data was run on the Whisper transcripts of the test recordings. A reference entity counts as recovered only if both its exact text and its type are predicted. **2 of 8 entities were recovered.**

| Sentence | Reference entity | Type | Recovered | Entities predicted on the Whisper text |
|---|---|---|---|---|
| T01 | Narendra Modi | PER | no | PER:Narainthra Modi, LOC:Madurai, ORG:bro |
| T01 | Madurai | LOC | yes | PER:Narainthra Modi, LOC:Madurai, ORG:bro |
| T01 | Wipro | ORG | no | PER:Narainthra Modi, LOC:Madurai, ORG:bro |
| T02 | Mahendra Singh Dhoni | PER | no | PER:Mahindra, LOC:Chennai Super, LOC:Kings |
| T02 | Chennai Super Kings | ORG | no | PER:Mahindra, LOC:Chennai Super, LOC:Kings |
| T03 | Smriti Mandhana | PER | no | PER:Subhriti, PER:Mandana, ORG:Deloitte, LOC:Vishalakapatnam |
| T03 | Deloitte | ORG | yes | PER:Subhriti, PER:Mandana, ORG:Deloitte, LOC:Vishalakapatnam |
| T03 | Vishakhapatnam | LOC | no | PER:Subhriti, PER:Mandana, ORG:Deloitte, LOC:Vishalakapatnam |

Most misses trace to a speech-recognition error (a mis-heard or misspelt name), so the speech-to-text step caps the quality of the whole system. The exception is `Chennai Super Kings`: Whisper heard it correctly, but the tagger labelled it as locations because `Chennai` is a known city.

## 7. SimpleRNN vs LSTM vs GRU (Keras)

| Cell (bidirectional, 48 units) | Dev F1 | Test precision | Test recall | Test F1 | Test F1 (lower-case) |
|---|---|---|---|---|---|
| SimpleRNN | 94.0% | 66.7% | 25.0% | 36.4% | 0.0% |
| LSTM | 97.4% | 66.7% | 25.0% | 36.4% | 18.2% |
| GRU | 90.5% | 25.0% | 12.5% | 16.7% | 0.0% |

*Single training run (seed 0) of `src/train_keras.py --cell all`, executed locally with TensorFlow. With 8 test entities one correct
or wrong entity moves F1 by about 12 points, so differences between cells are within noise. Re-run `train_keras.py` to replace this
table: if `results/keras_results.json` exists, `make_report.py` uses it instead of this file.*
