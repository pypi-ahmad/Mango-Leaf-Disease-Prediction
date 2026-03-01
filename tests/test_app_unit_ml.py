import importlib

import pytest
import torch
from PIL import Image


def _import_app_module():
    import app
    return importlib.reload(app)


@pytest.mark.parametrize(
    "model_name",
    ["EfficientNet-B0", "ResNet18", "MobileNetV3-Large", "DenseNet121"],
)
def test_get_model_supported_architectures_produce_expected_logits_shape(model_name):
    app = _import_app_module()

    num_classes = 8
    model, target_layer = app.get_model(model_name, num_classes)

    assert isinstance(model, torch.nn.Module)
    assert target_layer is not None

    with torch.no_grad():
        output = model(torch.randn(1, 3, app.IMG_SIZE, app.IMG_SIZE))

    assert output.shape == (1, num_classes)


def test_get_model_unknown_name_raises_current_runtime_error():
    app = _import_app_module()

    with pytest.raises(ValueError):
        app.get_model("UnknownModel", 3)


def test_process_image_returns_normalized_batch_tensor():
    app = _import_app_module()

    image = Image.new("RGB", (300, 300), color=(128, 64, 32))
    tensor = app.process_image(image)

    assert tensor.shape == (1, 3, app.IMG_SIZE, app.IMG_SIZE)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_process_image_none_input_raises():
    app = _import_app_module()

    with pytest.raises((AttributeError, TypeError)):
        app.process_image(None)


def test_load_bundle_missing_file_returns_none_and_reports_error(monkeypatch, tmp_path):
    app = _import_app_module()

    missing_file = tmp_path / "missing_bundle.pth"
    monkeypatch.setattr(app, "MODEL_PATH", str(missing_file))

    messages = []
    monkeypatch.setattr(app.st, "error", lambda msg: messages.append(msg))

    result = app.load_bundle()

    assert result is None
    assert messages
    assert str(missing_file) in messages[0]


def test_load_bundle_corrupted_model_file_returns_none(monkeypatch, tmp_path):
    app = _import_app_module()

    corrupted = tmp_path / "corrupted_bundle.pth"
    corrupted.write_bytes(b"not-a-valid-torch-artifact")
    monkeypatch.setattr(app, "MODEL_PATH", str(corrupted))

    result = app.load_bundle()

    assert result is None


def test_inference_pipeline_end_to_end_from_saved_bundle(monkeypatch, tmp_path):
    app = _import_app_module()

    class_names = ["A", "B", "C"]
    base_model, _ = app.get_model("ResNet18", len(class_names))
    model_state = base_model.state_dict()

    bundle = {
        "models": {"ResNet18": model_state},
        "class_names": class_names,
        "metrics": {
            "ResNet18": {
                "Accuracy": 1.0,
                "Precision": 1.0,
                "Recall": 1.0,
                "F1-Score": 1.0,
            }
        },
    }

    bundle_path = tmp_path / "bundle.pth"
    torch.save(bundle, bundle_path)
    monkeypatch.setattr(app, "MODEL_PATH", str(bundle_path))

    loaded = app.load_bundle()

    assert set(loaded.keys()) == {"models", "class_names", "metrics"}

    image = Image.new("RGB", (224, 224), color=(0, 0, 0))
    input_tensor = app.process_image(image)

    model, _ = app.get_model("ResNet18", len(loaded["class_names"]))
    model.load_state_dict(loaded["models"]["ResNet18"])
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    assert logits.shape == (1, len(class_names))
    assert probabilities.shape == (1, len(class_names))
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(1),
        atol=1e-5,
    )
