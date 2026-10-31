import torch
import pytest

from ml.architectures import build_model
from ml.training.checkpoint import load_checkpoint, load_model_from_checkpoint, next_run_dir, save_checkpoint
from tests.ml.test_architectures import SMALL, make_batch


@pytest.mark.parametrize("name", sorted(SMALL))
def test_checkpoint_roundtrip_reproduces_outputs(tmp_path, name):
    torch.manual_seed(1)
    m = build_model(name, **SMALL[name]).eval()
    batch = make_batch(3)
    ref = m(**m.select_inputs(batch))
    path = save_checkpoint(tmp_path / "m.pt", m, epoch=3, metrics={"val_acc": 0.5}, extra={"k": "v"})
    assert path.is_file()
    loaded, ckpt = load_model_from_checkpoint(path)
    assert ckpt["epoch"] == 3 and ckpt["metrics"]["val_acc"] == 0.5 and ckpt["extra"] == {"k": "v"}
    assert type(loaded) is type(m)
    assert torch.equal(ref, loaded(**loaded.select_inputs(batch)))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_run_dirs_are_versioned(tmp_path):
    a, b = next_run_dir(tmp_path, "cnn", "ds"), next_run_dir(tmp_path, "cnn", "ds")
    assert (a.name, b.name) == ("cnn_ds_v1", "cnn_ds_v2")
    assert next_run_dir(tmp_path, "lstm", "ds").name == "lstm_ds_v1"
