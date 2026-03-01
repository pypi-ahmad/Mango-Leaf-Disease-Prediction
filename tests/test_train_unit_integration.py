import importlib

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset


def _import_train_module():
    import train
    return importlib.reload(train)


def _write_sample_dataset(root_path):
    dataset_root = root_path / "MangoLeafBD Dataset"
    classes = ["Healthy", "Anthracnose"]
    for class_name in classes:
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            image = Image.new("RGB", (32, 32), color=(index * 10, index * 20, index * 30))
            image.save(class_dir / f"{class_name}_{index}.jpg")
    return dataset_root


def test_setup_data_skips_download_when_data_dir_is_non_empty(monkeypatch, tmp_path):
    train = _import_train_module()

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    sub_dir = data_dir / "Healthy"
    sub_dir.mkdir()
    (sub_dir / "sample.jpg").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(train, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        train.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not download")),
    )

    train.setup_data()

    assert sub_dir.exists()


def test_setup_data_download_error_cleans_created_data_dir(monkeypatch, tmp_path):
    train = _import_train_module()

    data_dir = tmp_path / "data"
    monkeypatch.setattr(train, "DATA_DIR", str(data_dir))

    class _Response:
        status_code = 500
        content = b""

    monkeypatch.setattr(train.requests, "get", lambda *_args, **_kwargs: _Response())

    with pytest.raises(Exception, match="download|setup"):
        train.setup_data()

    assert not data_dir.exists()


def test_get_data_loaders_builds_loaders_and_expected_tensor_shapes(monkeypatch, tmp_path):
    train = _import_train_module()

    _write_sample_dataset(tmp_path)
    monkeypatch.setattr(train, "DATA_DIR", str(tmp_path))

    def _patched_dataloader(dataset, batch_size, shuffle, num_workers):
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    monkeypatch.setattr(train, "DataLoader", _patched_dataloader)

    train_loader, val_loader, test_loader, class_names = train.get_data_loaders()

    assert set(class_names) == {"Healthy", "Anthracnose"}
    train_batch_inputs, train_batch_labels = next(iter(train_loader))
    val_batch_inputs, val_batch_labels = next(iter(val_loader))
    test_batch_inputs, test_batch_labels = next(iter(test_loader))

    assert train_batch_inputs.ndim == 4
    assert val_batch_inputs.ndim == 4
    assert test_batch_inputs.ndim == 4
    assert train_batch_inputs.shape[1:] == (3, train.IMG_SIZE, train.IMG_SIZE)
    assert val_batch_inputs.shape[1:] == (3, train.IMG_SIZE, train.IMG_SIZE)
    assert test_batch_inputs.shape[1:] == (3, train.IMG_SIZE, train.IMG_SIZE)
    assert train_batch_labels.dtype in (torch.int64, torch.long)
    assert val_batch_labels.dtype in (torch.int64, torch.long)
    assert test_batch_labels.dtype in (torch.int64, torch.long)


def test_train_get_model_unknown_architecture_raises_value_error():
    train = _import_train_module()

    with pytest.raises(ValueError):
        train.get_model("UnknownModel", 2)


def test_train_one_model_returns_state_dict_and_metric_contract(monkeypatch):
    train = _import_train_module()

    monkeypatch.setattr(train, "EPOCHS", 1)
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))

    inputs = torch.rand(8, 3, train.IMG_SIZE, train.IMG_SIZE)
    labels = torch.randint(0, 2, (8,))

    train_loader = TorchDataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=True, num_workers=0)
    val_loader = TorchDataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False, num_workers=0)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * train.IMG_SIZE * train.IMG_SIZE, 2),
    )

    state_dict, metrics = train.train_one_model("ToyModel", model, train_loader, val_loader)

    assert isinstance(state_dict, dict)
    assert set(metrics.keys()) == {"Accuracy", "Precision", "Recall", "F1-Score"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_train_one_model_empty_validation_loader_raises(monkeypatch):
    train = _import_train_module()

    monkeypatch.setattr(train, "EPOCHS", 1)
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))

    train_inputs = torch.rand(4, 3, train.IMG_SIZE, train.IMG_SIZE)
    train_labels = torch.randint(0, 2, (4,))
    val_inputs = torch.rand(0, 3, train.IMG_SIZE, train.IMG_SIZE)
    val_labels = torch.randint(0, 2, (0,))

    train_loader = TorchDataLoader(TensorDataset(train_inputs, train_labels), batch_size=2, shuffle=True, num_workers=0)
    val_loader = TorchDataLoader(TensorDataset(val_inputs, val_labels), batch_size=2, shuffle=False, num_workers=0)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * train.IMG_SIZE * train.IMG_SIZE, 2),
    )

    with pytest.raises((ZeroDivisionError, AttributeError, RuntimeError)):
        train.train_one_model("ToyModel", model, train_loader, val_loader)


def test_main_orchestrates_pipeline_and_saves_bundle(monkeypatch):
    train = _import_train_module()

    monkeypatch.setattr(train, "setup_data", lambda: None)
    monkeypatch.setattr(train, "get_data_loaders", lambda: ("train_loader", "val_loader", "test_loader", ["Healthy", "Disease"]))

    class _DummyModel:
        def load_state_dict(self, _state_dict):
            return None

    monkeypatch.setattr(train, "get_model", lambda _name, _n: _DummyModel())

    def _fake_train_one_model(name, _model, _train_loader, _val_loader):
        return ({"weight": torch.tensor([1.0])}, {"Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0})

    monkeypatch.setattr(train, "train_one_model", _fake_train_one_model)
    monkeypatch.setattr(train, "_evaluate_loader_metrics", lambda *_args, **_kwargs: {"Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0})

    captured = {}

    def _fake_save(bundle, path):
        captured["bundle"] = bundle
        captured["path"] = path

    monkeypatch.setattr(train.torch, "save", _fake_save)
    monkeypatch.setattr(train.torch.cuda, "empty_cache", lambda: None)

    train.main()

    assert captured["path"] == train.BUNDLE_PATH
    assert set(captured["bundle"].keys()) == {"models", "class_names", "metrics"}
    assert captured["bundle"]["class_names"] == ["Healthy", "Disease"]
    assert set(captured["bundle"]["models"].keys()) == {
        "EfficientNet-B0",
        "ResNet18",
        "MobileNetV3-Large",
        "DenseNet121",
    }
