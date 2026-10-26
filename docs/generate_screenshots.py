"""Generate rich, high-resolution visual screenshots for the README."""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QColor

from kinect_dashboard.ui.main_window import MainWindow
from kinect_dashboard.core.hand_tracker import HandTrackingResult
from kinect_dashboard.core.gesture_engine import GestureStatusOverlay, GestureState

def run_capture():
    app = QApplication(sys.argv)
    out_dir = Path(__file__).resolve().parent / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    win = MainWindow(use_mock=True)
    win.resize(1600, 960)
    win.show()

    # Step 1: Wait for streams to initialize and pump frames
    def step1_overview():
        print("Capturing 01_dashboard_overview.png...")
        win.grab().save(str(out_dir / "01_dashboard_overview.png"))
        QTimer.singleShot(1000, step2_recording_state)

    def step2_recording_state():
        print("Switching UI to active recording state...")
        win._set_controls_recording_state(True)
        win.gesture_engine.set_state(GestureState.RECORDING)
        win._update_status_badge_style(GestureState.RECORDING)
        win.session_writer = type('MockWriter', (), {'elapsed_seconds': lambda self: 4.82})()
        win.lbl_writer_stats.setText("Frames: 144 | Dropped: 0")
        win.status_bar.showMessage(
            "Recording live multimodal session -> session_20261027_011024_H60_Sun_Dry_T1"
        )
        if win.audio_worker is not None:
            win.audio_worker.current_rms_db = -18.5
            win.audio_worker.peak_db = -12.2
        if hasattr(win, "vu_meter"):
            win.vu_meter.set_levels(rms_db=-18.5, peak_db=-12.2)
            if hasattr(win, "lbl_audio_stats"):
                win.lbl_audio_stats.setText("RMS: -18.5 dBFS  |  Peak: -12.2 dBFS")
        
        QTimer.singleShot(1000, step2_capture_recording)

    def step2_capture_recording():
        print("Capturing 02_recording_active.png...")
        win.grab().save(str(out_dir / "02_recording_active.png"))
        QTimer.singleShot(1000, step3_gesture_tracking)

    def step3_gesture_tracking():
        print("Injecting hand tracking & gesture overlay...")
        win.session_writer = type('MockWriter', (), {'elapsed_seconds': lambda self: 12.65})()
        cx, cy = 960, 520
        landmarks_2d = [
            (cx, cy + 130),       # 0: wrist
            (cx - 35, cy + 90),   # 1: thumb cmc
            (cx - 55, cy + 60),   # 2: thumb mcp
            (cx - 58, cy + 30),   # 3: thumb ip
            (cx - 15, cy + 5),    # 4: thumb tip (pinched)
            (cx - 20, cy - 20),   # 5: index mcp
            (cx - 20, cy - 50),   # 6: index pip
            (cx - 18, cy - 70),   # 7: index dip
            (cx - 12, cy + 8),    # 8: index tip (pinched touching thumb)
            (cx + 8, cy - 25),    # 9: middle mcp
            (cx + 12, cy - 65),   # 10: middle pip
            (cx + 15, cy - 90),   # 11: middle dip
            (cx + 18, cy - 110),  # 12: middle tip
            (cx + 32, cy - 20),   # 13: ring mcp
            (cx + 38, cy - 55),   # 14: ring pip
            (cx + 42, cy - 80),   # 15: ring dip
            (cx + 44, cy - 98),   # 16: ring tip
            (cx + 56, cy - 10),   # 17: pinky mcp
            (cx + 64, cy - 40),   # 18: pinky pip
            (cx + 68, cy - 60),   # 19: pinky dip
            (cx + 70, cy - 75),   # 20: pinky tip
        ]
        
        tracking = HandTrackingResult(
            frame_index=380,
            device_timestamp_usec=12650000,
            hand_detected=True,
            handedness="Right",
            landmarks_2d_px=landmarks_2d,
            is_pinch=True,
            pinch_distance_mm=12.5,
            wrist_velocity_mm_s=25.0,
        )
        gesture = GestureStatusOverlay(
            current_state=GestureState.RECORDING,
            active_gesture_name="Pinch",
            target_action="MARK_SEGMENT",
            hold_duration_sec=1.4,
            required_duration_sec=2.0,
            progress_ratio=0.70,
            hud_message="Pinch: hold 0.6s to mark segment",
            is_active=True,
        )
        
        # Inject into worker and engine update methods
        if win.hand_worker:
            win.hand_worker.get_latest_result = lambda: tracking
        win.gesture_engine.update = lambda tr: gesture
        
        QTimer.singleShot(1000, step3_capture_gesture)

    def step3_capture_gesture():
        print("Capturing 03_gesture_hand_tracking.png...")
        win.grab().save(str(out_dir / "03_gesture_hand_tracking.png"))
        QTimer.singleShot(1000, step4_flash_banner)

    def step4_flash_banner():
        print("Flashing segment banner...")
        win.session_writer = type('MockWriter', (), {'elapsed_seconds': lambda self: 14.80})()
        win.rgb_viewport.flash_message("SEGMENT 1 MARKED", duration_sec=10.0)
        gesture_done = GestureStatusOverlay(
            current_state=GestureState.RECORDING,
            active_gesture_name="Pinch",
            target_action="MARK_SEGMENT",
            hold_duration_sec=2.0,
            required_duration_sec=2.0,
            progress_ratio=1.0,
            hud_message="Segment 1 recorded",
            is_active=True,
        )
        win.gesture_engine.update = lambda tr: gesture_done
        win.rgb_viewport.update()
        QTimer.singleShot(1000, step4_capture_banner)

    def step4_capture_banner():
        print("Capturing 04_segment_marked_feedback.png...")
        win.grab().save(str(out_dir / "04_segment_marked_feedback.png"))
        QTimer.singleShot(1000, step5_viewports_detail)

    def step5_viewports_detail():
        print("Capturing 05_depth_viewport_detail.png and 06_color_viewport_detail.png...")
        win.depth_viewport.grab().save(str(out_dir / "05_depth_viewport_detail.png"))
        win.rgb_viewport.grab().save(str(out_dir / "06_color_viewport_detail.png"))
        print("All screenshots generated successfully in docs/screenshots/!")
        app.quit()

    QTimer.singleShot(3000, step1_overview)
    app.exec()

if __name__ == "__main__":
    run_capture()
