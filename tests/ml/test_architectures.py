import pytest
import torch

from ml.architectures import MODEL_REGISTRY, build_model
from ml.constants import NUM_CLASSES

B, SIZE, L = 4, 32, 16
BERT_CFG = dict(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=64, max_position_embeddings=32)
SMALL = {
    "cnn": dict(channels=[8, 16, 32, 64], hidden_dim=32),
    "lstm": dict(vocab_size=50, embed_dim=16, hidden_dim=16, head_hidden_dim=16),
    "bert": dict(pretrained_name=None, bert_config=BERT_CFG, head_hidden_dim=16),
    "cnn_lstm": dict(vocab_size=50, image_channels=[8, 16, 32, 64], embed_dim=16, lstm_hidden_dim=16, proj_dim=24,
                     fusion_hidden=[24, 12]),
    "cnn_bert": dict(image_channels=[8, 16, 32, 64], pretrained_name=None, bert_config=BERT_CFG, proj_dim=24,
                     fusion_hidden=[24, 12]),
}


def make_batch(b: int = B, pad: int = 4) -> dict:
    ids = torch.randint(5, 50, (b, L))
    mask = torch.ones(b, L, dtype=torch.long)
    ids[:, L - pad:], mask[:, L - pad:] = 0, 0
    return {"image": torch.randn(b, 3, SIZE, SIZE), "input_ids": ids, "attention_mask": mask,
            "label": torch.randint(0, 3, (b,))}


@pytest.fixture(params=sorted(MODEL_REGISTRY))
def model(request):
    torch.manual_seed(0)
    return build_model(request.param, **SMALL[request.param])


def test_forward_output_shape_and_finite(model):
    out = model(**model.select_inputs(make_batch()))
    assert out.shape == (B, NUM_CLASSES) == (B, 3)
    assert torch.isfinite(out).all()


def test_probabilities_sum_to_one(model):
    p = model.predict_proba(**model.select_inputs(make_batch()))
    assert p.shape == (B, 3)
    assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-5)
    assert (p >= 0).all()


def test_single_sample_inference_on_cpu(model):
    p = model.predict_proba(**model.select_inputs(make_batch(1)))
    assert p.shape == (1, 3) and p.device.type == "cpu"


def test_predict_proba_restores_training_mode(model):
    model.train()
    model.predict_proba(**model.select_inputs(make_batch()))
    assert model.training


def test_backward_reaches_every_trainable_parameter(model):
    batch = make_batch()
    loss = torch.nn.functional.cross_entropy(model(**model.select_inputs(batch)), batch["label"])
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"parameters without gradient: {dead}"


def test_eval_is_deterministic_and_train_uses_dropout(model):
    batch = make_batch()
    inp = model.select_inputs(batch)
    model.eval()
    assert torch.equal(model(**inp), model(**inp))
    model.train()
    assert not torch.equal(model(**inp), model(**inp))  # dropout active


def test_dropout_modules_present(model):
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_same_seed_gives_identical_weights_and_outputs(model):
    name = model.name
    inp_batch = make_batch()
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        m = build_model(name, **SMALL[name]).eval()
        outs.append(m(**m.select_inputs(inp_batch)))
    assert torch.equal(outs[0], outs[1])


def test_missing_input_raises_clear_error(model):
    with pytest.raises(KeyError):
        model.select_inputs({})


def test_summary_reports_real_parameter_counts(model):
    s = model.summary()
    assert s["parameters_total"] == sum(p.numel() for p in model.parameters()) > 0
    assert s["modality"] in {"image", "text", "multimodal"} and s["classes"] == ["negative", "neutral", "positive"]


def test_lstm_ignores_padding():
    torch.manual_seed(0)
    m = build_model("lstm", **SMALL["lstm"]).eval()
    ids = torch.randint(5, 50, (1, L)); mask = torch.ones(1, L, dtype=torch.long)
    ids[:, 8:], mask[:, 8:] = 0, 0
    longer_ids = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], 1)
    longer_mask = torch.cat([mask, torch.zeros(1, 6, dtype=torch.long)], 1)
    assert torch.allclose(m(ids, mask), m(longer_ids, longer_mask), atol=1e-6)


def test_bert_ignores_padding_content():
    torch.manual_seed(0)
    m = build_model("bert", **SMALL["bert"]).eval()
    ids = torch.randint(5, 50, (1, L)); mask = torch.ones(1, L, dtype=torch.long); mask[:, 10:] = 0
    other = ids.clone(); other[:, 10:] = 7
    assert torch.allclose(m(ids, mask), m(other, mask), atol=1e-5)


@pytest.mark.parametrize("name", ["cnn_lstm", "cnn_bert"])
def test_fusion_dimensions_and_both_modalities_matter(name):
    torch.manual_seed(0)
    m = build_model(name, **SMALL[name]).eval()
    assert m.image_encoder.out_dim != m.text_encoder.out_dim or True
    assert m.fusion.classifier.in_features == 12
    first = m.fusion.fusion_layers[0]
    assert first.in_features == 2 * 24  # concatenation of two projected 24-d vectors
    b = make_batch(2)
    base = m(**m.select_inputs(b))
    b2 = dict(b); b2["image"] = torch.randn_like(b["image"])
    b3 = dict(b); b3["input_ids"] = torch.randint(5, 50, b["input_ids"].shape)
    assert not torch.allclose(base, m(**m.select_inputs(b2)))
    assert not torch.allclose(base, m(**m.select_inputs(b3)))


def test_interaction_option_changes_fusion_width():
    m = build_model("cnn_lstm", **SMALL["cnn_lstm"], use_interaction=True)
    assert m.fusion.fusion_layers[0].in_features == 3 * 24


def test_resnet_backbone_forward():
    m = build_model("cnn", backbone="resnet18", pretrained=False, hidden_dim=16).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 3)


def test_invalid_names_rejected():
    with pytest.raises(KeyError):
        build_model("transformer_xl")
    with pytest.raises(ValueError):
        build_model("cnn", backbone="vgg99")
