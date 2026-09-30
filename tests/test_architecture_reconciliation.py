import gc

import pytest
import torch

from data_selection_swap import common
from data_selection_swap import trajectory


@pytest.mark.parametrize("architecture", ["resnet18", "vgg16", "inception"])
def test_cifar_model_classifier_contract_and_checkpoint_round_trip(
        architecture, tmp_path):
    cfg = {
        "architecture": architecture,
        "input_channels": 3,
        "small_images": True,
        "cifar_style": True,
    }
    model = common.build_model(cfg, num_classes=10, device="cpu")
    model.eval()
    classifier = common.classifier_layer(model)
    weight_name, bias_name = common.classifier_parameter_names(model)

    assert classifier.out_features == 10
    assert weight_name in dict(model.named_parameters())
    assert bias_name in dict(model.named_parameters())
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)

    checkpoint = tmp_path / f"{architecture}.pth"
    torch.save({"model": model.state_dict()}, checkpoint)
    clone = common.build_model(cfg, num_classes=10, device="cpu")
    clone.load_state_dict(common.load_model_weights(str(checkpoint)))
    assert torch.equal(
        dict(model.named_parameters())[weight_name],
        dict(clone.named_parameters())[weight_name],
    )
    del model, clone
    gc.collect()


@pytest.mark.parametrize("architecture", ["resnet18", "vgg16", "inception"])
def test_factored_adam_metric_uses_architecture_classifier_names(
        architecture, tmp_path):
    cfg = {
        "architecture": architecture,
        "input_channels": 3,
        "small_images": True,
        "cifar_style": True,
    }
    model = common.build_model(cfg, num_classes=10, device="cpu")
    classifier = common.classifier_layer(model)
    weight_name, bias_name = common.classifier_parameter_names(model)
    record = {
        "model": model.state_dict(),
        "step": 1,
        "betas": (0.9, 0.999),
        "exp_avg_sq": {
            weight_name: torch.ones_like(classifier.weight),
            bias_name: torch.ones_like(classifier.bias),
        },
    }
    path = tmp_path / f"{architecture}_adam.pth"
    torch.save(record, path)
    checkpoint = trajectory.Checkpoint(
        path=str(path), step=1, epoch=1, lr=1e-3,
        weight=1.0, window=(1, 1),
    )
    metric = trajectory.adam_metric_factored(
        checkpoint,
        num_classes=10,
        feat_dim=classifier.in_features,
        include_bias=True,
        weight_name=weight_name,
        bias_name=bias_name,
    )
    assert metric.shape == (10, classifier.in_features + 1)
    assert torch.isfinite(metric).all()
    del model
    gc.collect()


def test_baseline_paths_are_architecture_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    resnet = common.get_save_paths("random", "cifar10", "resnet18", 4500, 42)
    vgg = common.get_save_paths("random", "cifar10", "vgg16", 4500, 42)
    assert resnet["base_dir"] != vgg["base_dir"]
    assert "resnet18" in resnet["base_dir"]
    assert "vgg16" in vgg["base_dir"]
