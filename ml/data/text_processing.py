"""Text cleaning and the two text encoders (word vocabulary for LSTM, WordPiece for BERT)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Optional, Protocol

from ml.constants import PAD_TOKEN, UNK_TOKEN

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_RT = re.compile(r"^\s*rt\s+(?=@|\w)", re.IGNORECASE)
_SPACES = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


def clean_text(text: Optional[str], lowercase: bool = False) -> str:
    """Normalise a social-media post.

    NFKC unicode normalisation; control characters removed; a leading ``RT`` dropped;
    URLs -> ``url``; ``@mentions`` -> ``user``; ``#hashtag`` -> ``hashtag`` word kept;
    whitespace collapsed. Missing values become ``""``.
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = "".join(ch for ch in s if ch in "\n\t " or unicodedata.category(ch)[0] != "C")
    s = _RT.sub("", s)
    s = _URL.sub(" url ", s)
    s = _MENTION.sub(" user ", s)
    s = _HASHTAG.sub(r" \1 ", s)
    s = _SPACES.sub(" ", s).strip()
    return s.lower() if lowercase else s


def tokenize_words(text: str) -> list[str]:
    """Lower-cased word tokens (alphanumerics, simple contractions, ``!`` and ``?``)."""
    return _WORD.findall(text.lower())


class TextEncoder(Protocol):
    """Turns a string into fixed-length ``input_ids`` and ``attention_mask`` lists."""
    max_len: int
    def encode(self, text: str) -> tuple[list[int], list[int]]: ...
    def tokens(self, text: str) -> list[str]: ...
    def state(self) -> dict[str, Any]: ...


class WordVocabEncoder:
    """Word-level encoder; the vocabulary must be built from the **training split only**."""

    kind = "word"

    def __init__(self, itos: list[str], max_len: int = 64) -> None:
        if not itos or itos[0] != PAD_TOKEN or itos[1] != UNK_TOKEN:
            raise ValueError("itos must start with [<pad>, <unk>]")
        self.itos, self.max_len = list(itos), max_len
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Iterable[str], max_len: int = 64, max_vocab: int = 20000,
              min_freq: int = 2) -> "WordVocabEncoder":
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(tokenize_words(clean_text(t, lowercase=True)))
        words = [w for w, c in counts.most_common() if c >= min_freq][: max_vocab - 2]
        return cls([PAD_TOKEN, UNK_TOKEN] + words, max_len)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokens(self, text: str) -> list[str]:
        return tokenize_words(clean_text(text, lowercase=True))[: self.max_len]

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        ids = [self.stoi.get(w, 1) for w in self.tokens(text)] or [1]  # empty text -> single <unk>
        mask = [1] * len(ids)
        pad = self.max_len - len(ids)
        return ids + [0] * pad, mask + [0] * pad

    def state(self) -> dict[str, Any]:
        return {"type": "word", "max_len": self.max_len, "itos": self.itos}


class BertTokenizerEncoder:
    """WordPiece encoder backed by a Hugging Face tokenizer (hub name or local directory/vocab)."""

    kind = "bert"

    def __init__(self, name_or_path: str = "bert-base-uncased", max_len: int = 64) -> None:
        from transformers import AutoTokenizer, BertTokenizerFast

        self.name_or_path, self.max_len = name_or_path, max_len
        if name_or_path.endswith(".txt"):
            # Local WordPiece vocabulary file. transformers>=5 takes ``vocab`` (dict), older versions ``vocab_file``.
            import inspect
            from pathlib import Path

            vocab = {w.strip(): i for i, w in enumerate(Path(name_or_path).read_text().splitlines())}
            if "vocab" in inspect.signature(BertTokenizerFast.__init__).parameters:
                self.tokenizer = BertTokenizerFast(vocab=vocab, do_lower_case=True)
            else:
                self.tokenizer = BertTokenizerFast(vocab_file=name_or_path, do_lower_case=True)
            if len(self.tokenizer) != len(vocab):
                raise RuntimeError("Tokenizer vocabulary did not load completely")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        enc = self.tokenizer(clean_text(text), truncation=True, max_length=self.max_len,
                             padding="max_length")
        return list(enc["input_ids"]), list(enc["attention_mask"])

    def tokens(self, text: str) -> list[str]:
        ids, _ = self.encode(text)
        return self.tokenizer.convert_ids_to_tokens(ids)

    def state(self) -> dict[str, Any]:
        return {"type": "bert", "max_len": self.max_len, "name_or_path": self.name_or_path}


def text_encoder_from_state(state: dict[str, Any]) -> WordVocabEncoder | BertTokenizerEncoder:
    """Rebuild an encoder from :meth:`state` (stored beside each checkpoint)."""
    if state["type"] == "word":
        return WordVocabEncoder(state["itos"], state["max_len"])
    if state["type"] == "bert":
        return BertTokenizerEncoder(state["name_or_path"], state["max_len"])
    raise ValueError(f"Unknown text encoder type '{state['type']}'")


def sequence_length_stats(texts: Iterable[str]) -> dict[str, float]:
    """Word-token length statistics for the preprocessing report."""
    import numpy as np

    lengths = np.array([len(tokenize_words(clean_text(t, lowercase=True))) for t in texts])
    if lengths.size == 0:
        return {"count": 0}
    return {"count": int(lengths.size), "mean": float(lengths.mean()), "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max()), "min": int(lengths.min())}
