"""Step 1 - speech to text. Turns every .wav in audio/ into a transcript.

    python src/transcribe.py --audio audio/train --out data/transcripts_asr_train.tsv
Backends: openai-whisper (default, offline) or SpeechRecognition (Google Web Speech API, online).
If data/sentences.tsv holds reference text, word error rate (WER) per file is added.
"""
import argparse, csv
from pathlib import Path
from common import tokenize, load_sentences, ROOT


def wer(ref, hyp):
    r, h = [w.lower() for w in tokenize(ref)], [w.lower() for w in tokenize(hyp)]
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[-1] / max(len(r), 1)


def transcribe_whisper(files, model_name):
    import whisper
    model = whisper.load_model(model_name)
    return [model.transcribe(str(f), language="en", fp16=False)["text"].strip() for f in files]


def transcribe_google(files):
    import speech_recognition as sr
    rec, out = sr.Recognizer(), []
    for f in files:
        with sr.AudioFile(str(f)) as src:
            out.append(rec.recognize_google(rec.record(src), language="en-IN"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(ROOT / "audio/train"))
    ap.add_argument("--out", default=str(ROOT / "data/transcripts_asr_train.tsv"))
    ap.add_argument("--backend", choices=["whisper", "google"], default="whisper")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--reference", default=str(ROOT / "data/sentences.tsv"))
    a = ap.parse_args()
    EXT = {".wav", ".ogg", ".mp3", ".m4a", ".flac", ".opus"}
    files = sorted(f for f in Path(a.audio).iterdir() if f.suffix.lower() in EXT)
    assert files, f"No audio files (.wav/.ogg/.mp3/.m4a/.flac) in {a.audio}. Record your sentences first (see RECORDING_GUIDE.md)."
    texts = transcribe_whisper(files, a.model) if a.backend == "whisper" else transcribe_google(files)
    ref = {Path(s["audio"]).stem: " ".join(s["tokens"]) for s in load_sentences(a.reference) if s["audio"]}
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(["file", "asr_text", "wer_vs_reference"])
        for f, t in zip(files, texts):
            e = f"{wer(ref[f.stem], t):.3f}" if f.stem in ref else ""
            w.writerow([f.name, t, e]); print(f.name, "|", t, "|", e)
