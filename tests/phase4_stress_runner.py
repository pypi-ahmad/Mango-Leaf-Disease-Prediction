import io
import json
import os
import sys
import tempfile
import time
import traceback
import types
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def image(self, *args, **kwargs):
        return None


class DummySidebar:
    def __init__(self):
        self.upload_mode = "none"

    def header(self, *args, **kwargs):
        return None

    def multiselect(self, _label, options, default=None):
        return default if default is not None else options

    def file_uploader(self, *args, **kwargs):
        if self.upload_mode == "none":
            return None
        if self.upload_mode == "invalid":
            payload = io.BytesIO(b"not_an_image")
            payload.name = "broken.jpg"
            return payload
        if self.upload_mode == "csv":
            rows = ["a,b,c\n"] + [f"{i},{i+1},{i+2}\n" for i in range(200000)]
            payload = io.BytesIO("".join(rows).encode("utf-8"))
            payload.name = "large.csv"
            return payload
        if self.upload_mode == "video":
            payload = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
            payload.name = "sample.mp4"
            return payload

        image = Image.new("RGB", (1024, 1024), color=(123, 200, 45))
        payload = io.BytesIO()
        image.save(payload, format="JPEG")
        payload.seek(0)
        payload.name = "sample.jpg"
        return payload


class DummyStreamlit:
    def __init__(self):
        self.sidebar = DummySidebar()

    def set_page_config(self, *args, **kwargs):
        return None

    def cache_resource(self, func=None, **kwargs):
        if func is None:
            def _decorator(inner):
                return inner
            return _decorator
        return func

    def error(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def tabs(self, labels):
        return tuple(DummyCtx() for _ in labels)

    def columns(self, spec):
        ncols = spec if isinstance(spec, int) else len(spec)
        return [DummyCtx() for _ in range(ncols)]

    def image(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def bar_chart(self, *args, **kwargs):
        return None

    def selectbox(self, _label, options, *args, **kwargs):
        return options[0] if options else None

    def info(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None


class DummyFigure:
    def add_trace(self, *args, **kwargs):
        return None

    def update_layout(self, *args, **kwargs):
        return None


class DummyScatterPolar:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummyGradCAM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_tensor, targets=None):
        _, _, h, w = input_tensor.shape
        return np.zeros((1, h, w), dtype=np.float32)


class DummyClassifierOutputTarget:
    def __init__(self, idx):
        self.idx = idx


def install_runtime_stubs():
    streamlit = DummyStreamlit()
    sys.modules["streamlit"] = streamlit

    plotly = types.ModuleType("plotly")
    plotly_express = types.ModuleType("plotly.express")
    plotly_graph_objects = types.ModuleType("plotly.graph_objects")
    plotly_graph_objects.Figure = DummyFigure
    plotly_graph_objects.Scatterpolar = DummyScatterPolar
    plotly.express = plotly_express
    plotly.graph_objects = plotly_graph_objects

    sys.modules["plotly"] = plotly
    sys.modules["plotly.express"] = plotly_express
    sys.modules["plotly.graph_objects"] = plotly_graph_objects

    gradcam_pkg = types.ModuleType("pytorch_grad_cam")
    gradcam_pkg.GradCAM = DummyGradCAM

    gradcam_utils = types.ModuleType("pytorch_grad_cam.utils")
    gradcam_utils_image = types.ModuleType("pytorch_grad_cam.utils.image")
    gradcam_utils_targets = types.ModuleType("pytorch_grad_cam.utils.model_targets")

    def _show_cam_on_image(img_norm, grayscale_cam, use_rgb=True):
        return (img_norm * 255).astype(np.uint8)

    gradcam_utils_image.show_cam_on_image = _show_cam_on_image
    gradcam_utils_targets.ClassifierOutputTarget = DummyClassifierOutputTarget

    sys.modules["pytorch_grad_cam"] = gradcam_pkg
    sys.modules["pytorch_grad_cam.utils"] = gradcam_utils
    sys.modules["pytorch_grad_cam.utils.image"] = gradcam_utils_image
    sys.modules["pytorch_grad_cam.utils.model_targets"] = gradcam_utils_targets

    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2


def execute_case(name, category, fn):
    started = time.perf_counter()
    try:
        details = fn()
        elapsed = time.perf_counter() - started
        return {
            "name": name,
            "category": category,
            "status": "ok",
            "duration_sec": round(elapsed, 4),
            "details": details,
            "error_type": None,
            "traceback": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "name": name,
            "category": category,
            "status": "failed",
            "duration_sec": round(elapsed, 4),
            "details": None,
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }


def main():
    install_runtime_stubs()

    import app
    import train

    results = []

    def case_all_tests():
        import pytest
        code = pytest.main(["-q"])
        return {"pytest_exit_code": int(code)}

    def case_large_image_processing():
        image = Image.new("RGB", (8192, 8192), color=(200, 100, 50))
        tensor = app.process_image(image)
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}

    def case_repeated_inference_calls():
        model, _ = app.get_model("ResNet18", 8)
        model.eval()
        input_tensor = torch.randn(1, 3, app.IMG_SIZE, app.IMG_SIZE)
        iterations = 30
        for _ in range(iterations):
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
            if probs.shape != (1, 8):
                raise RuntimeError(f"unexpected probs shape {tuple(probs.shape)}")
        return {"iterations": iterations, "last_shape": list(probs.shape)}

    def case_rapid_ui_interactions():
        app.st.sidebar.upload_mode = "valid_image"
        class_names = [
            "Anthracnose",
            "Bacterial Canker",
            "Cutting Weevil",
            "Die Back",
            "Gall Midge",
            "Healthy",
            "Powdery Mildew",
            "Sooty Mould",
        ]

        model_names = ["DenseNet121", "MobileNetV3-Large"]
        models = {}
        for name in model_names:
            model, _ = app.get_model(name, len(class_names))
            models[name] = model.state_dict()

        bundle = {
            "models": models,
            "class_names": class_names,
            "metrics": {
                name: {
                    "Accuracy": 1.0,
                    "Precision": 1.0,
                    "Recall": 1.0,
                    "F1-Score": 1.0,
                }
                for name in model_names
            },
        }

        original_loader = app.load_bundle
        app.load_bundle = lambda: bundle
        try:
            for _ in range(10):
                app.main()
        finally:
            app.load_bundle = original_loader
            app.st.sidebar.upload_mode = "none"
        return {"interactions": 10}

    def case_csv_upload_stress():
        original_loader = app.load_bundle
        app.st.sidebar.upload_mode = "csv"
        app.load_bundle = lambda: {
            "models": {},
            "class_names": ["A"],
            "metrics": {},
        }
        try:
            app.main()
        finally:
            app.load_bundle = original_loader
            app.st.sidebar.upload_mode = "none"
        return {"unexpected": "no error"}

    def case_video_upload_stress():
        original_loader = app.load_bundle
        app.st.sidebar.upload_mode = "video"
        app.load_bundle = lambda: {
            "models": {},
            "class_names": ["A"],
            "metrics": {},
        }
        try:
            app.main()
        finally:
            app.load_bundle = original_loader
            app.st.sidebar.upload_mode = "none"
        return {"unexpected": "no error"}

    def case_batch_processing_stress():
        model, _ = app.get_model("ResNet18", 8)
        model.eval()
        batch_tensor = torch.randn(64, 3, app.IMG_SIZE, app.IMG_SIZE)
        with torch.no_grad():
            output = model(batch_tensor)
            probs = torch.softmax(output, dim=1)
        return {
            "input_batch_shape": list(batch_tensor.shape),
            "output_shape": list(probs.shape),
        }

    def case_cpu_gpu_switching():
        original_device = train.DEVICE
        outcomes = {}

        train.DEVICE = torch.device("cpu")
        model_cpu = train.get_model("ResNet18", 8)
        outcomes["cpu_device"] = str(next(model_cpu.parameters()).device)

        if torch.cuda.is_available():
            train.DEVICE = torch.device("cuda")
            model_cuda = train.get_model("ResNet18", 8)
            outcomes["cuda_device"] = str(next(model_cuda.parameters()).device)
        else:
            outcomes["cuda_device"] = "skipped (no CUDA)"

        train.DEVICE = original_device
        return outcomes

    def case_missing_model_file():
        original_path = app.MODEL_PATH
        app.MODEL_PATH = "__missing_bundle__.pth"
        try:
            loaded = app.load_bundle()
        finally:
            app.MODEL_PATH = original_path
        return {"load_result_is_none": loaded is None}

    def case_corrupted_model_file():
        original_path = app.MODEL_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = os.path.join(tmpdir, "corrupted.pth")
            with open(bad_file, "wb") as stream:
                stream.write(b"corrupted-binary")
            app.MODEL_PATH = bad_file
            try:
                loaded = app.load_bundle()
            finally:
                app.MODEL_PATH = original_path
        return {"load_result_is_none": loaded is None}

    def case_data_wrong_schema():
        original_data_dir = train.DATA_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            train.DATA_DIR = tmpdir
            try:
                train.get_data_loaders()
            except ValueError as e:
                return {"validation_error": str(e)}
            finally:
                train.DATA_DIR = original_data_dir
        return {"unexpected": "no validation error"}

    def case_data_null_values_schema():
        original_loader = app.load_bundle
        app.st.sidebar.upload_mode = "none"
        app.load_bundle = lambda: {
            "models": {},
            "class_names": None,
            "metrics": {},
        }
        try:
            app.main()
        finally:
            app.load_bundle = original_loader
            app.st.sidebar.upload_mode = "none"
        return {"unexpected": "no error"}

    def case_data_large_dataset_loader():
        original_data_dir = train.DATA_DIR
        original_loader = train.DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "MangoLeafBD Dataset")
            os.makedirs(os.path.join(root, "Healthy"), exist_ok=True)
            os.makedirs(os.path.join(root, "Anthracnose"), exist_ok=True)

            for idx in range(160):
                img = Image.new("RGB", (32, 32), color=(idx % 255, 10, 20))
                img.save(os.path.join(root, "Healthy", f"h_{idx}.jpg"))
                img.save(os.path.join(root, "Anthracnose", f"a_{idx}.jpg"))

            train.DATA_DIR = tmpdir
            train.DataLoader = lambda ds, batch_size, shuffle, num_workers: torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
            )
            train_loader, val_loader, test_loader, class_names = train.get_data_loaders()
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            test_batch = next(iter(test_loader))

        train.DATA_DIR = original_data_dir
        train.DataLoader = original_loader

        return {
            "class_count": len(class_names),
            "train_batch_shape": list(train_batch[0].shape),
            "val_batch_shape": list(val_batch[0].shape),
            "test_batch_shape": list(test_batch[0].shape),
        }

    scenarios = [
        ("large_image_processing", "system", case_large_image_processing),
        ("repeated_inference_calls", "system", case_repeated_inference_calls),
        ("rapid_ui_interactions", "system", case_rapid_ui_interactions),
        ("csv_upload", "system", case_csv_upload_stress),
        ("video_upload", "system", case_video_upload_stress),
        ("batch_processing", "ml", case_batch_processing_stress),
        ("cpu_gpu_switching", "ml", case_cpu_gpu_switching),
        ("missing_model_file", "ml", case_missing_model_file),
        ("corrupted_model_file", "ml", case_corrupted_model_file),
        ("wrong_schema", "data", case_data_wrong_schema),
        ("null_values", "data", case_data_null_values_schema),
        ("large_dataset", "data", case_data_large_dataset_loader),
        ("all_tests", "system", case_all_tests),
    ]

    for name, category, fn in scenarios:
        results.append(execute_case(name, category, fn))

    failed = [item for item in results if item["status"] == "failed"]
    passed = [item for item in results if item["status"] == "ok"]

    summary = {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "results": results,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
