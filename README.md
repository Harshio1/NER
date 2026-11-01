# Voice NER: Named Entity Recognition from Speech with Recurrent Networks

**Lab Assessment 10.** Record sentences in your own voice → speech-to-text (Whisper) → BIO annotation → RNN tagger
(SimpleRNN / LSTM / GRU) → evaluation on new recordings → analysis of a tiny self-made audio dataset.
The repo also contains a showcase website with an in-browser tagger (type or speak a sentence).

## Deploy in 3 steps (GitHub → Vercel)
1. **Put your recordings in the repo.** Copy your 10 training files to `audio/train/` (`sentence_01.ogg` … `sentence_10.ogg`) and your 3 test files to `audio/test/` (`test_01.ogg` … `test_03.ogg`).
2. **Push to GitHub**
   ```bash
   git init && git add . && git commit -m "Voice NER project"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<your-repo>.git
   git push -u origin main
   ```
3. **Import the repo in Vercel** (vercel.com → Add New → Project). Leave every setting at its default: the root `vercel.json` already
   sets *Framework = Other*, no build step, and publishes the `website/` folder. Click **Deploy**.

Preview locally first with `npx serve website` (the model is loaded with `fetch`, so opening `index.html` as a file will not work).

## Repository layout
```
├── vercel.json                root deploy config (serves ./website)
├── website/                   static site: live tagger, dataset, results, Whisper errors
├── data/
│   ├── sentences.tsv, test_sentences.tsv          typed-reference gold annotation ([Name|PER] markup)
│   ├── sentences_asr.tsv, test_sentences_asr.tsv  annotation of the Whisper text exactly as heard
│   ├── transcripts_asr_train.tsv / _test.tsv      Whisper small.en output + WER
│   └── annotated_*.conll / .csv / .jsonl          BIO-tagged datasets
├── src/                       transcribe, annotate, augment, Keras + NumPy models, metrics, end_to_end, report
├── results/                   evaluation_results.json, evaluation_results_asr.json, end_to_end.json
├── report/REPORT.md           evaluation tables + limitations analysis
├── notebooks/audio_ner_colab.ipynb
├── tests/test_gradcheck.py    numerical gradient check of the hand-written backprop
└── audio/train, audio/test    YOUR RECORDINGS
```

## Reproduce
```bash
pip install -r requirements.txt                       # + ffmpeg for Whisper
python tests/test_gradcheck.py                        # prints PASS
python src/transcribe.py --audio audio/train --out data/transcripts_asr_train.tsv --reference data/sentences.tsv --model small.en
python src/transcribe.py --audio audio/test  --out data/transcripts_asr_test.tsv  --reference data/test_sentences.tsv --model small.en
python src/annotate.py build --in data/sentences.tsv --prefix data/annotated_train
python src/annotate.py build --in data/test_sentences.tsv --prefix data/annotated_test
python src/train_keras.py --cell all                  # SimpleRNN vs LSTM vs GRU (needs TensorFlow)
cd src
python run_experiments.py                             # NumPy BiGRU: ablation, evaluation, website export
python run_experiments.py --train ../data/sentences_asr.tsv --test ../data/test_sentences_asr.tsv --out ../results/evaluation_results_asr.json --no_web
python end_to_end.py
python make_report.py                                 # rebuilds report/REPORT.md and website/data/asr.json
```
Never write ASR output to `data/test_sentences.tsv`: that file holds the gold test annotation.

## Method
| Part | Choice |
|---|---|
| Tags | O, B/I-PER, B/I-ORG, B/I-LOC |
| Input features | lower-cased word embedding (32) + 3-letter-suffix embedding (8) + "capitalised" and "first word" flags |
| Model | Bidirectional recurrent layer (48 units per direction) → softmax per token |
| Robustness | random word → UNK dropout; half of the training sentences lower-cased (ASR output often drops capitals) |
| Augmentation | 25 sentence templates × entity pools; train/dev pools disjoint; no test entity in any pool |
| Evaluation | strict span-level precision/recall/F1, 5 seeds, model selected on a synthetic dev set only |

## Headline results (see `report/REPORT.md`)
- Test recordings, typed-reference annotation: **55.6% entity F1** (selected seed), 42.8% ± 8.6 mean of 5 seeds; 10 recordings alone: 26.8% ± 10.7.
- Annotating the Whisper text as heard: 44.4% F1 (selected), 40.4% ± 6.7 mean.
- End to end (audio → Whisper → tagger): **2 of 8** reference entities recovered exactly; most misses are mis-heard names.
- Test set has only 8 gold entities, so one entity moves F1 by about 12 points. Treat all numbers as a demonstration of method.

## Honest notes
- The NumPy BiGRU is the model behind the website and the main numbers. Its gradients are verified numerically and the website's JavaScript matches Python.
- The Keras SimpleRNN/LSTM/GRU table in the report is from a **single seed** run on the author's machine. If you re-run `train_keras.py`, `results/keras_results.json` is picked up by `make_report.py` automatically.
- Whisper small.en struggles with Indian proper nouns; this is the main finding of the project.
