# SandStories - Azure Kinect Multimodal Capture Dashboard

Native Windows application for multimodal recording (1080p RGB, uncompressed WFOV depth, 48kHz audio, hand tracking) matching the SandStories iPhone v2_streams format.

## Requirements
- Windows 10/11
- Python 3.12 (`py -3.12`)
- Azure Kinect SDK v1.4.2 (installed at `C:\Program Files\Azure Kinect SDK v1.4.2`)
- NVIDIA GPU or Intel GPU for hardware H.264 encoding via Windows MediaFoundation (`h264_mf`)

## Quick Start
To launch the dashboard:
```powershell
# From this directory:
py -3.12 run_dashboard.py

# Or run the Desktop batch script:
C:\Users\vijit\Desktop\Kinect_Dashboard.bat
```

## Running Tests
Run the test suite directly with pytest:
```powershell
py -3.12 -m pytest tests -v
```

## Structure
- `kinect_dashboard/`
  - `core/`: Kinect stream worker, audio recorder, hand tracker, gesture state machine.
  - `storage/`: Session writer (v2_streams format: `depth.f32`, `rgb.mp4`, `audio.wav`, `v2_frames.jsonl`, `manifest.json`).
  - `ui/`: PyQt6 user interface, controls, and viewports.
  - `main.py`: Application entry point.
- `tests/`: Complete pytest test suite (16 automated tests).
- `run_dashboard.py`: Convenience command-line launcher.
