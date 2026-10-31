import numpy as np
import pytest
import torch
from PIL import Image

from ml.data.image_processing import build_transforms, check_image, denormalize, load_rgb
from ml.data.text_processing import WordVocabEncoder, clean_text, sequence_length_stats, text_encoder_from_state, tokenize_words


def test_clean_text_rules():
    s = "RT @bob: Loving this!! https://t.co/x1 #HappyDay  caf\u00e9\u00a0\u00a0time"
    out = clean_text(s)
    assert "http" not in out and "@bob" not in out and "#" not in out
    assert out.startswith("user") and "HappyDay" in out and "url" in out
    assert "  " not in out
    assert clean_text(None) == "" and clean_text(float("nan")) == ""
    assert clean_text("ＡＢＣ") == "ABC"          # NFKC full-width -> ascii
    assert clean_text("Hi", lowercase=True) == "hi"


def test_tokenizer_keeps_contractions_and_emphasis():
    assert tokenize_words("I can't wait!!") == ["i", "can't", "wait", "!", "!"]


def test_word_encoder_padding_mask_unk_and_roundtrip():
    enc = WordVocabEncoder.build(["good day", "good night", "bad day"], max_len=5, min_freq=2)
    assert enc.itos[:2] == ["<pad>", "<unk>"] and "good" in enc.stoi and "night" not in enc.stoi
    ids, mask = enc.encode("good night")
    assert len(ids) == len(mask) == 5 and mask == [1, 1, 0, 0, 0] and ids[1] == 1
    assert enc.encode("")[1] == [1, 0, 0, 0, 0]               # empty text -> one <unk>
    long_ids, long_mask = enc.encode("good " * 20)
    assert len(long_ids) == 5 and sum(long_mask) == 5          # truncation
    again = text_encoder_from_state(enc.state())
    assert again.encode("good day") == enc.encode("good day")


def test_vocab_built_only_from_given_texts():
    enc = WordVocabEncoder.build(["alpha beta"], min_freq=1)
    assert "gamma" not in enc.stoi


def test_sequence_stats():
    st = sequence_length_stats(["a b c", "a"])
    assert st["count"] == 2 and st["max"] == 3 and st["min"] == 1


def _save(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_check_image_flags(tmp_path):
    rng = np.random.default_rng(0)
    _save(tmp_path / "ok.png", rng.integers(0, 255, (64, 64, 3)))
    _save(tmp_path / "tiny.png", rng.integers(0, 255, (8, 8, 3)))
    _save(tmp_path / "blank.png", np.full((64, 64, 3), 128))
    (tmp_path / "bad.png").write_bytes(b"garbage")
    assert check_image(tmp_path / "ok.png").valid
    assert check_image(tmp_path / "tiny.png").reason.startswith("too_small")
    assert check_image(tmp_path / "blank.png").reason == "blank_image"
    assert check_image(tmp_path / "bad.png").reason.startswith("unreadable")
    assert check_image(tmp_path / "missing.png").reason.startswith("unreadable")


def test_rgb_conversion_and_transform_shape(tmp_path):
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 50), dtype=np.uint8)).save(tmp_path / "g.png")  # grayscale
    Image.new("RGBA", (30, 30), (1, 2, 3, 4)).save(tmp_path / "a.png")
    for n in ("g.png", "a.png"):
        assert load_rgb(tmp_path / n).mode == "RGB"
        t = build_transforms(32)(load_rgb(tmp_path / n))
        assert t.shape == (3, 32, 32) and t.dtype == torch.float32


def test_normalisation_roundtrip():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8))
    back = denormalize(build_transforms(32)(img))
    assert np.abs(back.astype(int) - np.asarray(img).astype(int)).max() <= 2
