"""Shared utilities: tokenisation, entity-markup parsing, BIO tagging, feature encoding."""
import re, csv
from pathlib import Path

TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.'’-][A-Za-z0-9]+)*")
MARK_RE = re.compile(r"\[([^\]|]+)\|(PER|ORG|LOC)\]")
ROOT = Path(__file__).resolve().parent.parent


def tokenize(text):
    """Speech-to-text style tokenisation: words only, punctuation dropped."""
    return TOKEN_RE.findall(text)


def parse_markup(marked):
    """'[Sundar Pichai|PER] leads [Google|ORG].' -> (tokens, BIO tags)."""
    tokens, tags, pos = [], [], 0
    for m in MARK_RE.finditer(marked):
        for t in tokenize(marked[pos:m.start()]):
            tokens.append(t); tags.append("O")
        for i, t in enumerate(tokenize(m.group(1))):
            tokens.append(t); tags.append(("B-" if i == 0 else "I-") + m.group(2))
        pos = m.end()
    for t in tokenize(marked[pos:]):
        tokens.append(t); tags.append("O")
    return tokens, tags


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_sentences(path):
    out = []
    for r in read_tsv(path):
        toks, tags = parse_markup(r["marked"])
        out.append({"id": r["id"], "audio": r.get("audio", ""), "tokens": toks, "tags": tags})
    return out


def spans(tags):
    """BIO tags -> set of (type, start, end). Lenient: a stray I- starts a span."""
    res, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                res.add((typ, start, i))
                start, typ = None, None
            if t != "O":
                start, typ = i, t[2:]
    return res


# ---------------- feature encoding (mirrored exactly in website/app.js) ----------------
class Vocab:
    def __init__(self, words, suffixes):
        self.words = ["<pad>", "<unk>"] + words
        self.sufs = ["<pad>", "<unk>"] + suffixes
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.s2i = {s: i for i, s in enumerate(self.sufs)}

    @staticmethod
    def suffix(w):
        return w.lower()[-3:]

    @classmethod
    def build(cls, sentences):
        ws, ss = set(), set()
        for s in sentences:
            for t in s["tokens"]:
                ws.add(t.lower()); ss.add(cls.suffix(t))
        return cls(sorted(ws), sorted(ss))

    def encode(self, tokens):
        """-> word ids, suffix ids, cap flags (non-first tokens), first flags"""
        wid = [self.w2i.get(t.lower(), 1) for t in tokens]
        sid = [self.s2i.get(self.suffix(t), 1) for t in tokens]
        cap = [1.0 if (i > 0 and t[0].isupper()) else 0.0 for i, t in enumerate(tokens)]
        first = [1.0 if i == 0 else 0.0 for i in range(len(tokens))]
        return wid, sid, cap, first
