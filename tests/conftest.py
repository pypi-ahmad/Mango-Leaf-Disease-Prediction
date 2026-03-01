import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummySidebar:
    def header(self, *args, **kwargs):
        return None

    def multiselect(self, *args, **kwargs):
        default = kwargs.get("default")
        if default is not None:
            return default
        if len(args) >= 3:
            return args[2]
        return []

    def file_uploader(self, *args, **kwargs):
        return None


class _DummyStreamlit:
    def __init__(self):
        self.sidebar = _DummySidebar()

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
        return tuple(_DummyCtx() for _ in labels)

    def columns(self, spec):
        if isinstance(spec, int):
            ncols = spec
        else:
            ncols = len(spec)
        return [_DummyCtx() for _ in range(ncols)]

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


class _DummyFigure:
    def add_trace(self, *args, **kwargs):
        return None

    def update_layout(self, *args, **kwargs):
        return None


class _DummyScatterPolar:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _DummyGradCAM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input_tensor, targets=None):
        _, _, height, width = input_tensor.shape
        return np.zeros((1, height, width), dtype=np.float32)


class _DummyClassifierOutputTarget:
    def __init__(self, idx):
        self.idx = idx


@pytest.fixture(scope="session", autouse=True)
def install_module_stubs():
    streamlit = _DummyStreamlit()
    sys.modules["streamlit"] = streamlit

    plotly = types.ModuleType("plotly")
    plotly_express = types.ModuleType("plotly.express")
    plotly_graph_objects = types.ModuleType("plotly.graph_objects")
    plotly_graph_objects.Figure = _DummyFigure
    plotly_graph_objects.Scatterpolar = _DummyScatterPolar
    plotly.express = plotly_express
    plotly.graph_objects = plotly_graph_objects

    sys.modules["plotly"] = plotly
    sys.modules["plotly.express"] = plotly_express
    sys.modules["plotly.graph_objects"] = plotly_graph_objects

    gradcam_pkg = types.ModuleType("pytorch_grad_cam")
    gradcam_pkg.GradCAM = _DummyGradCAM

    gradcam_utils = types.ModuleType("pytorch_grad_cam.utils")
    gradcam_utils_image = types.ModuleType("pytorch_grad_cam.utils.image")
    gradcam_utils_targets = types.ModuleType("pytorch_grad_cam.utils.model_targets")

    def _show_cam_on_image(img_norm, grayscale_cam, use_rgb=True):
        return (img_norm * 255).astype(np.uint8)

    gradcam_utils_image.show_cam_on_image = _show_cam_on_image
    gradcam_utils_targets.ClassifierOutputTarget = _DummyClassifierOutputTarget

    sys.modules["pytorch_grad_cam"] = gradcam_pkg
    sys.modules["pytorch_grad_cam.utils"] = gradcam_utils
    sys.modules["pytorch_grad_cam.utils.image"] = gradcam_utils_image
    sys.modules["pytorch_grad_cam.utils.model_targets"] = gradcam_utils_targets

    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2
