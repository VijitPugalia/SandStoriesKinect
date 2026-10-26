"""UI components for the Azure Kinect Multimodal Capture Dashboard."""

from .main_window import MainWindow
from .styles import DARK_THEME_STYLESHEET
from .viewports import DepthViewportWidget, RGBViewportWidget
from .vu_meter import VUMeterWidget

__all__ = [
    "MainWindow",
    "DARK_THEME_STYLESHEET",
    "RGBViewportWidget",
    "DepthViewportWidget",
    "VUMeterWidget",
]
