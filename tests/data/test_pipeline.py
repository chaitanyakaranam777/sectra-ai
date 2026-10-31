import json

import pandas as pd
import pytest

from ml.data.dataset import SentimentDataset
from ml.data.text_processing import WordVocabEncoder
from pipelines.ingestion.mvsa import build_raw_table, parse_labels, resolve_pair_label
from pipelines.preprocessing.split import assert_no_leakage, assign_groups, stratified_group_split
from pipelines.run_pipeline import run_pipeline


@pytest.mark.parametrize("t,i,expect", [
    ("positive", "positive", ("positive", None)), ("neutral", "negative", ("negative", None)),
    ("positive", "neutral", ("positive", None)), ("neutral", "neutral", ("neutral", None)),
    ("positive", "negative", (None, "text_image_conflict")), (None, "positive", (None, "no_annotator_majority"))])
def test_label_resolution_rules(t, i, expect):
    assert resolve_pair_label(t, i) == expect


def test_multiple_variant_majority_and_ties(tmp_path):
    f = tmp_path / "labelResultAll.txt"
    f.write_text("ID\ttext,image\n"
                 "1\tpositive,positive\tpositive,neutral\tnegative,positive\n"   # text pos (2/3), image pos (2/3)
                 "2\tpositive,positive\tnegative,negative\tneutral,neutral\n"    # three-way tie -> dropped
                 "3\tpositive,positive\n"                                        # wrong column count -> malformed
                 "4\tpositive,happy\tpositive,positive\tpositive,positive\n")    # invalid label -> malformed
    df, info = parse_labels(f, "multiple")
    assert info["malformed_label_rows"] == 2
    assert df.loc[df.id == "1", "label"].item() == "positive"
    assert df.loc[df.id == "2", "drop_reason"].item() == "no_annotator_majority"


def test_missing_dataset_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATASET_IMPORT"):
        build_raw_table(tmp_path, "single")


def test_raw_files_are_not_modified(mvsa_like_dir, tmp_path):
    before = {p: p.stat().st_mtime_ns for p in mvsa_like_dir.rglob("*") if p.is_file()}
    run_pipeline(mvsa_like_dir, "single", tmp_path / "out", seed=1)
    after = {p: p.stat().st_mtime_ns for p in mvsa_like_dir.rglob("*") if p.is_file()}
    assert before == after


def test_pipeline_report_detects_every_injected_defect(processed):
    _, rep = processed
    assert rep["invalid_records"].get("missing_image") == 1
    assert rep["invalid_records"].get("empty_text") == 1
    assert any(k.startswith("image_unreadable") for k in rep["invalid_records"])
    assert rep["dropped_unlabeled"].get("text_image_conflict") == 1
    assert rep["duplicates"]["duplicate_images"] >= 1 and rep["duplicates"]["duplicate_texts"] >= 1
    # 240 clean + 1 (neutral/positive -> positive) + 1 retweet duplicate; 4 defects removed
    assert rep["final_samples"] == 242
    assert sum(rep["class_distribution"].values()) == 242
    assert rep["leakage_check"]["leaking_keys"] == 0


def test_processing_log_and_outputs_exist(processed):
    samples = processed[0]
    assert (samples.parent / "report.json").is_file()
    stages = [json.loads(l)["stage"] for l in (samples.parent / "processing_log.jsonl").read_text().splitlines()]
    assert stages == ["ingestion", "validation", "label_resolution", "duplicate_analysis", "dedupe_ids", "split", "export"]


def test_label_mapping_and_columns(processed):
    df = pd.read_csv(processed[0])
    assert dict(zip(df.label, df.label_id)) == {"negative": 0, "neutral": 1, "positive": 2}
    assert df.text_clean.notna().all() and df.id.is_unique


def test_split_is_stratified_and_sized(processed):
    df = pd.read_csv(processed[0])
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["test"] - 0.1) < 0.04 and abs(frac["val"] - 0.1) < 0.04
    for part in ("train", "val", "test"):
        assert set(df[df.split == part].label) == {"negative", "neutral", "positive"}


def test_retweet_duplicates_share_a_split_and_no_leakage(processed):
    df = pd.read_csv(processed[0])
    twins = df[df.image_md5 == df.loc[df.id == 1, "image_md5"].item()]
    assert len(twins) == 2 and twins.split.nunique() == 1 and twins.group_id.nunique() == 1
    assert_no_leakage(df)


def test_leakage_check_actually_fails_when_leaking(processed):
    df = pd.read_csv(processed[0])
    bad = df.copy()
    idx = bad[bad.image_md5 == bad.loc[bad.id == 1, "image_md5"].item()].index
    bad.loc[idx[0], "split"], bad.loc[idx[1], "split"] = "train", "test"
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(bad)


def test_split_is_reproducible_and_seed_sensitive(processed):
    df = pd.read_csv(processed[0]).drop(columns="split")
    a, b = stratified_group_split(df, seed=5), stratified_group_split(df, seed=5)
    c = stratified_group_split(df, seed=6)
    assert a.equals(b) and not a.equals(c)


def test_assign_groups_links_shared_image_or_text():
    df = pd.DataFrame({"image_md5": ["a", "a", "b", "c"], "text_key": ["x", "y", "y", "z"]})
    g = assign_groups(df)
    assert g[0] == g[1] == g[2] != g[3]       # 0-1 share image, 1-2 share text


def test_dataset_item_shapes_and_fingerprint(processed):
    df = pd.read_csv(processed[0])
    enc = WordVocabEncoder.build(df.text_clean, max_len=12, min_freq=1)
    ds = SentimentDataset(df.head(6), "multimodal", enc, image_size=32)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32) and item["input_ids"].shape == (12,) and item["label"].ndim == 0
    assert SentimentDataset(df.head(6), "text", enc).fingerprint() == ds.fingerprint()
    assert SentimentDataset(df.head(5), "text", enc).fingerprint() != ds.fingerprint()
    with pytest.raises(ValueError):
        SentimentDataset(df.head(2), "text", None)
