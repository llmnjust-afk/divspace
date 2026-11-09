import torch

from divspace.attribute.attribution import attribution_patching
from divspace.attribute.logit_lens import logit_lens
from divspace.hooks import ablate, capture, inject
from divspace.models import build_tiny


def test_capture_records_every_requested_layer():
    lm = build_tiny(n_layers=4)
    enc = lm.tokenizer("a b c d", return_tensors="pt")
    store: dict[int, torch.Tensor] = {}
    with capture(lm.layers, [0, 2, 3], store, positions="all", pool=None):
        lm.model(**enc)
    assert sorted(store) == [0, 2, 3]
    for tensor in store.values():
        assert tensor.shape[0] == 1
        assert tensor.shape[2] == lm.hidden_size


def test_mean_pooling_matches_manual():
    lm = build_tiny(n_layers=2)
    enc = lm.tokenizer("a b c", return_tensors="pt")
    full: dict[int, torch.Tensor] = {}
    pooled: dict[int, torch.Tensor] = {}
    with capture(lm.layers, [1], full, positions="all", pool=None):
        with capture(lm.layers, [1], pooled, positions="all", pool="mean"):
            lm.model(**enc)
    assert torch.allclose(full[1].mean(dim=1), pooled[1], atol=1e-5)


def test_injection_changes_logits():
    lm = build_tiny(n_layers=4)
    enc = lm.tokenizer("a b c d", return_tensors="pt")
    with torch.no_grad():
        clean = lm.model(**enc).logits
    direction = torch.ones(lm.hidden_size)
    with torch.no_grad(), inject(lm.layers, [2], direction, coef=1.0):
        steered = lm.model(**enc).logits
    assert not torch.allclose(clean, steered)


def test_ablation_changes_logits():
    lm = build_tiny(n_layers=4)
    enc = lm.tokenizer("a b c d", return_tensors="pt")
    direction = torch.randn(lm.hidden_size)
    with torch.no_grad():
        clean = lm.model(**enc).logits
    with torch.no_grad(), ablate(lm.layers, [1], direction):
        ablated = lm.model(**enc).logits
    assert not torch.allclose(clean, ablated)


def test_logit_lens_reports_all_layers():
    lm = build_tiny(n_layers=4)
    report = logit_lens(lm, "a b c", [0, 1, 2, 3], top_k=3)
    assert sorted(report) == [0, 1, 2, 3]
    assert all(len(entry["top_tokens"]) == 3 for entry in report.values())
    assert all(entry["entropy"] >= 0.0 for entry in report.values())


def test_attribution_patching_returns_shapes():
    lm = build_tiny(n_layers=4)
    report = attribution_patching(
        lm,
        prompt="a b c",
        clean_response="d e f",
        corrupt_response="d e",
        layers=[0, 1, 2, 3],
    )
    assert sorted(report["scalars"]) == [0, 1, 2, 3]
    for layer in range(4):
        assert report["vectors"][layer].shape == (lm.hidden_size,)
    assert isinstance(report["delta_metric"], float)
