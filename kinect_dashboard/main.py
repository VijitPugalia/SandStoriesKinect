"""Main entry point for Azure Kinect DK Multimodal Capture Dashboard."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow


from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging to console and persistent log file."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_dir / "kinect_dashboard.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
        )
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


def default_output_dir() -> Path:
    """Desktop\\AzureKinectRecordings on the capture PC; ./recordings elsewhere."""
    home = Path.home()
    desktop = home / "Desktop"
    if desktop.is_dir():
        return desktop / "AzureKinectRecordings"
    return Path("./recordings")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SandStories - Azure Kinect DK Multimodal Capture Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force use of synthetic Mock RGB-D stream (for testing without physical hardware)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Azure Kinect hardware device index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where recorded experimental sessions are saved",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch dashboard in fullscreen mode",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    return parser.parse_args()


def main() -> int:
    """Application entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("kinect_dashboard")
    logger.info("Starting SandStories Azure Kinect Capture Dashboard...")

    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SandStories Kinect Dashboard")
    app.setOrganizationName("SandStories Research")

    # Allow clean Ctrl+C termination from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    window = MainWindow(
        output_dir=args.output_dir,
        use_mock=args.mock,
        device_id=args.device_id,
    )

    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
