"""Main PyQt6 Application Window for Azure Kinect DK Capture Dashboard."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core.audio_recorder import AudioChunk, AudioRecorderWorker
from ..core.gesture_engine import GestureCommand, GestureEngine, GestureState, GestureStatusOverlay
from ..core.hand_tracker import HandTrackerWorker, HandTrackingResult
from ..core.kinect_stream import BaseKinectStream, FramePacket, KinectStreamWorker, create_kinect_stream
from ..storage.raw_mkv_session import RawMkvSessionWriter
from ..storage.session_writer import (
    SessionMetadata,
    SessionWriterWorker,
    generate_session_id,
    sanitize_session_name,
)
from .styles import DARK_THEME_STYLESHEET
from .viewports import DepthViewportWidget, RGBViewportWidget
from .vu_meter import VUMeterWidget

logger = logging.getLogger(__name__)

CAMERA_STALE_AFTER_SECONDS = 3.0


class MainWindow(QMainWindow):
    """Primary GUI window coordinating capture pipelines, viewports, and metadata."""

    def __init__(
        self,
        output_dir: Path = Path("./recordings"),
        use_mock: bool = False,
        device_id: int = 0,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir).resolve()
        self.use_mock = use_mock
        self.device_id = device_id

        self.setWindowTitle("SandStories — Azure Kinect Capture")
        self.resize(1600, 960)
        self.setMinimumSize(1180, 720)
        self.setStyleSheet(DARK_THEME_STYLESHEET)

        # Active worker handles
        self.stream_worker: Optional[BaseKinectStream] = None
        self.audio_worker: Optional[AudioRecorderWorker] = None
        self.hand_worker: Optional[HandTrackerWorker] = None
        self.gesture_engine = GestureEngine()
        self.session_writer: Optional[RawMkvSessionWriter] = None

        # UI components
        self.sidebar_scroll: Optional[QScrollArea] = None
        self.btn_sidebar_toggle: Optional[QPushButton] = None

        # Queues for Main Thread UI updates
        self._ui_frame_queue: Optional[queue.Queue[FramePacket]] = None
        self._session_frame_queue: Optional[queue.Queue[FramePacket]] = None
        self._session_audio_queue: Optional[queue.Queue[AudioChunk]] = None
        self._session_rgb_queue: Optional[queue.Queue[FramePacket]] = None

        # State tracking
        self._session_start_time: float = 0.0
        self._session_elapsed_sec: float = 0.0
        self._flash_state: bool = False
        self._initialization_error: Optional[str] = None
        self._pause_started_at: Optional[float] = None
        self._paused_total_sec: float = 0.0
        self._audio_warning_shown: bool = False
        self._last_gesture_hud: str = ""

        self._init_ui()
        self._init_shortcuts()
        self._init_workers()

        # Master GUI Refresh Timer (~30 FPS display loop)
        self._gui_timer = QTimer(self)
        self._gui_timer.timeout.connect(self._on_render_tick)
        self._gui_timer.start(30)

        # 1-second background telemetry timer
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._on_telemetry_tick)
        self._telemetry_timer.start(1000)

    def _init_ui(self) -> None:
        """Construct a camera-first layout: viewfinder, compact sidebar, bottom controls."""
        central_widget = QWidget(self)
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        root = QVBoxLayout(central_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._create_header_bar())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 10)
        body_layout.setSpacing(12)

        viewport_container = QWidget()
        viewport_layout = QHBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(10)

        self.rgb_viewport = RGBViewportWidget()
        self.depth_viewport = DepthViewportWidget()
        viewport_layout.addWidget(self.rgb_viewport, stretch=1)
        viewport_layout.addWidget(self.depth_viewport, stretch=1)
        body_layout.addWidget(viewport_container, stretch=1)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setObjectName("SetupRail")
        self.sidebar_scroll.setFixedWidth(336)
        self.sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_scroll.setWidget(self._create_sidebar())
        body_layout.addWidget(self.sidebar_scroll)

        root.addWidget(body, stretch=1)
        root.addWidget(self._create_bottom_bar())

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready — press Start or Space. Pinch 2 s marks segment; cover lens 3 s to stop."
        )

    def toggle_sidebar(self) -> None:
        """Toggle sidebar panel visibility to expand viewports across the window."""
        if self.sidebar_scroll is None:
            return
        is_visible = self.sidebar_scroll.isVisible()
        self.sidebar_scroll.setVisible(not is_visible)
        if self.btn_sidebar_toggle is not None:
            if not is_visible:
                self.btn_sidebar_toggle.setText("Hide setup")
                self.btn_sidebar_toggle.setToolTip("Hide sidebar panel (Ctrl+B / F4)")
            else:
                self.btn_sidebar_toggle.setText("Show setup")
                self.btn_sidebar_toggle.setToolTip("Show sidebar panel (Ctrl+B / F4)")

    def _create_header_bar(self) -> QWidget:
        """Compact top bar: brand, sensor status, timer, session state, and panel toggle."""
        header = QWidget()
        header.setObjectName("AppHeader")
        header.setFixedHeight(58)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(12)

        # Brand title & subtitle
        brand_widget = QWidget()
        brand_layout = QHBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(6)

        lbl_title = QLabel("SandStories")
        lbl_title.setObjectName("BrandTitle")
        lbl_sep = QLabel("/")
        lbl_sep.setObjectName("BrandSep")
        lbl_subtitle = QLabel("Azure Kinect")
        lbl_subtitle.setObjectName("BrandSubtitle")

        brand_layout.addWidget(lbl_title)
        brand_layout.addWidget(lbl_sep)
        brand_layout.addWidget(lbl_subtitle)
        layout.addWidget(brand_widget)

        layout.addStretch()

        # Sensor status pill
        self.lbl_hw_status = QLabel("SENSOR  Checking")
        self.lbl_hw_status.setObjectName("SensorPill")
        layout.addWidget(self.lbl_hw_status)

        # Master Clock
        self.lbl_master_timer = QLabel("00:00.000")
        self.lbl_master_timer.setObjectName("MasterTimer")
        self.lbl_master_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_master_timer.setMinimumWidth(115)
        layout.addWidget(self.lbl_master_timer)

        # Status badge
        self.lbl_status_badge = QLabel("IDLE")
        self.lbl_status_badge.setObjectName("StatusBadge")
        self.lbl_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_status_badge_style(GestureState.IDLE)
        layout.addWidget(self.lbl_status_badge)

        # Sidebar Toggle
        self.btn_sidebar_toggle = QPushButton("Hide setup")
        self.btn_sidebar_toggle.setObjectName("BtnSidebarToggle")
        self.btn_sidebar_toggle.setToolTip("Toggle sidebar panel (Ctrl+B / F4)")
        self.btn_sidebar_toggle.clicked.connect(self.toggle_sidebar)
        layout.addWidget(self.btn_sidebar_toggle)

        return header

    def _create_bottom_bar(self) -> QWidget:
        """Camera-style bottom control strip."""
        bar = QWidget()
        bar.setObjectName("BottomBar")
        bar.setFixedHeight(92)
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(14, 8, 14, 8)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.btn_discard = QPushButton("Discard session")
        self.btn_discard.setObjectName("BtnDiscard")
        self.btn_discard.setEnabled(False)
        self.btn_discard.clicked.connect(self._on_btn_discard_clicked)
        row.addWidget(self.btn_discard)

        row.addStretch()

        self.btn_start = QPushButton("Start recording")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.setMinimumWidth(150)
        self.btn_start.clicked.connect(self._on_btn_start_clicked)
        row.addWidget(self.btn_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_btn_pause_clicked)
        self.btn_pause.setVisible(False)

        self.btn_stop = QPushButton("Stop and save")
        self.btn_stop.setObjectName("BtnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_btn_stop_clicked)
        row.addWidget(self.btn_stop)

        row.addStretch()
        outer.addLayout(row)

        hint = QLabel("Space: Start   ·   Pinch 2s: Mark Segment   ·   Cover 3s: Stop & Save   ·   Ctrl+B: Toggle Sidebar")
        hint.setObjectName("GestureHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(hint)

        return bar

    def _create_sidebar(self) -> QWidget:
        """Collapsible sidebar: session setup, audio, diagnostics."""
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(0)

        rail_title = QLabel("CAPTURE SETUP")
        rail_title.setObjectName("RailTitle")
        layout.addWidget(rail_title)

        # ── Session setup ───────────────────────────────────────────────
        setup = QWidget()
        setup_layout = QVBoxLayout(setup)
        setup_layout.setContentsMargins(4, 8, 4, 8)
        setup_layout.setSpacing(10)

        hint = QLabel("Optional name and save location. Condition fields are written to metadata.json.")
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        setup_layout.addWidget(hint)

        form = QGridLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(8)

        form.addWidget(self._field_label("Height"), 0, 0)
        self.combo_height = QComboBox()
        self.combo_height.addItems(["60 cm", "80 cm", "100 cm"])
        form.addWidget(self.combo_height, 0, 1)

        form.addWidget(self._field_label("Lighting"), 1, 0)
        self.combo_lighting = QComboBox()
        self.combo_lighting.addItems([
            "Simulated Harsh Sun",
            "Simulated Dappled Shade",
            "Simulated Overcast",
        ])
        form.addWidget(self.combo_lighting, 1, 1)

        form.addWidget(self._field_label("Surface"), 2, 0)
        self.combo_surface = QComboBox()
        self.combo_surface.addItems(["Dry Loose Sand", "Wet Compacted Sand"])
        form.addWidget(self.combo_surface, 2, 1)

        form.addWidget(self._field_label("Trial"), 3, 0)
        self.combo_trial = QComboBox()
        self.combo_trial.addItems(["Trial 1", "Trial 2"])
        form.addWidget(self.combo_trial, 3, 1)

        form.addWidget(self._field_label("Lux"), 4, 0)
        self.spin_lux = QSpinBox()
        self.spin_lux.setRange(0, 150000)
        self.spin_lux.setValue(8500)
        self.spin_lux.setSingleStep(500)
        form.addWidget(self.spin_lux, 4, 1)

        form.addWidget(self._field_label("Name"), 5, 0)
        self.edit_session_name = QLineEdit()
        self.edit_session_name.setPlaceholderText("optional prefix")
        self.edit_session_name.setMaxLength(60)
        self.edit_session_name.textChanged.connect(self._refresh_session_preview)
        form.addWidget(self.edit_session_name, 5, 1)

        setup_layout.addLayout(form)

        preview_lbl = self._field_label("Folder preview")
        setup_layout.addWidget(preview_lbl)
        self.lbl_session_preview = QLabel("")
        self.lbl_session_preview.setObjectName("PreviewPath")
        self.lbl_session_preview.setWordWrap(True)
        setup_layout.addWidget(self.lbl_session_preview)

        save_row = QHBoxLayout()
        save_row.setSpacing(6)
        self.lbl_output_dir = QLabel(str(self.output_dir))
        self.lbl_output_dir.setObjectName("PathReadout")
        self.lbl_output_dir.setWordWrap(True)
        self.lbl_output_dir.setToolTip(str(self.output_dir))
        save_row.addWidget(self.lbl_output_dir, stretch=1)
        self.btn_output_dir = QPushButton("Change")
        self.btn_output_dir.setObjectName("BtnGhost")
        self.btn_output_dir.setFixedWidth(72)
        self.btn_output_dir.clicked.connect(self._on_choose_output_dir)
        save_row.addWidget(self.btn_output_dir)
        setup_layout.addLayout(save_row)

        self.config_group = setup  # used by _set_controls_recording_state
        self.chk_allow_no_audio = QCheckBox("Allow recording without audio (testing only)")
        self.chk_allow_no_audio.stateChanged.connect(lambda _s: self._refresh_start_availability())
        setup_layout.addWidget(self.chk_allow_no_audio)

        for combo in (self.combo_height, self.combo_lighting, self.combo_surface, self.combo_trial):
            combo.currentIndexChanged.connect(lambda _i: self._refresh_session_preview())

        layout.addWidget(self._section_heading("SESSION METADATA"))
        layout.addWidget(setup)

        # ── Audio ───────────────────────────────────────────────────────
        audio_panel = QWidget()
        audio_layout = QVBoxLayout(audio_panel)
        audio_layout.setContentsMargins(4, 8, 4, 8)
        audio_layout.setSpacing(8)

        self.vu_meter = VUMeterWidget()
        audio_layout.addWidget(self.vu_meter)

        self.lbl_audio_stats = QLabel("—")
        self.lbl_audio_stats.setObjectName("FieldValue")
        self.lbl_audio_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        audio_layout.addWidget(self.lbl_audio_stats)

        self.lbl_audio_device = QLabel("Opening microphone…")
        self.lbl_audio_device.setObjectName("PreviewPath")
        self.lbl_audio_device.setWordWrap(True)
        audio_layout.addWidget(self.lbl_audio_device)

        layout.addWidget(self._section_heading("AUDIO INPUT"))
        layout.addWidget(audio_panel)

        # ── Diagnostics ─────────────────────────────────────────────────
        diag = QWidget()
        diag_layout = QGridLayout(diag)
        diag_layout.setContentsMargins(4, 8, 4, 8)
        diag_layout.setVerticalSpacing(6)
        diag_layout.setHorizontalSpacing(8)

        self._add_diag_row(diag_layout, 0, "Capture FPS", "lbl_fps_capture")
        self._add_diag_row(diag_layout, 1, "Frames", "lbl_frames_total")
        self._add_diag_row(diag_layout, 2, "Dropped", "lbl_frames_dropped")
        self._add_diag_row(diag_layout, 3, "Hand", "lbl_hand_state")
        self._add_diag_row(diag_layout, 4, "Pinch ratio", "lbl_pinch_ratio")
        self._add_diag_row(diag_layout, 5, "Gesture", "lbl_gesture_hud")
        self._add_diag_row(diag_layout, 6, "Writer", "lbl_writer_stats")

        self.lbl_pinch_ratio.setToolTip("Thumb–index distance / palm span. Pinch starts below 0.30.")
        self.lbl_gesture_hud.setObjectName("MonoReadout")

        layout.addWidget(self._section_heading("LIVE DIAGNOSTICS"))
        layout.addWidget(diag)
        layout.addStretch()
        self._refresh_session_preview()
        return sidebar

    def _section_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def _add_diag_row(self, grid: QGridLayout, row: int, label: str, attr: str) -> None:
        grid.addWidget(self._field_label(label), row, 0)
        value = QLabel("—")
        value.setObjectName("MonoReadout" if row == 0 else "FieldValue")
        setattr(self, attr, value)
        grid.addWidget(value, row, 1)

    def _refresh_session_preview(self) -> None:
        """Show the folder name START SESSION will create right now."""
        try:
            meta = self._get_current_metadata(include_intrinsics=False)
            preview = generate_session_id(meta)
        except Exception:
            preview = ""
        if hasattr(self, "lbl_session_preview"):
            self.lbl_session_preview.setText(preview)

    def _on_choose_output_dir(self) -> None:
        """Let the operator pick where session folders are written."""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose folder for Kinect sessions",
            str(self.output_dir),
        )
        if chosen:
            self.output_dir = Path(chosen).resolve()
            self.lbl_output_dir.setText(str(self.output_dir))
            self.lbl_output_dir.setToolTip(str(self.output_dir))
            self.status_bar.showMessage(f"Sessions will be saved under {self.output_dir}", 4000)

    def _audio_ready(self) -> bool:
        """True when audio is capturing, or the operator explicitly allowed no audio."""
        if self.audio_worker is None:
            return self.chk_allow_no_audio.isChecked()
        if self.audio_worker.capture_mode in ("hardware", "synthetic"):
            sample_is_fresh = (
                self.audio_worker.total_samples_captured > 0
                and time.monotonic() - self.audio_worker.last_sample_monotonic < 2.0
            )
            if sample_is_fresh:
                return True
        return self.chk_allow_no_audio.isChecked()

    def _refresh_start_availability(self) -> None:
        """Enable START only when the pipeline can actually record."""
        if self._initialization_error:
            self.btn_start.setEnabled(False)
            return
        if self.gesture_engine.state != GestureState.IDLE:
            return
        self.btn_start.setEnabled(bool(self._camera_ready() and self._audio_ready()))

    def _camera_ready(self) -> bool:
        """Require a live worker and a recent frame, not a stale lifetime counter."""
        worker = self.stream_worker
        if worker is None or not worker.is_alive() or worker.frame_count == 0:
            return False
        if not self.use_mock and not worker.device_started:
            return False
        return worker.last_error is None and worker.has_fresh_frame(CAMERA_STALE_AFTER_SECONDS)

    def _init_shortcuts(self) -> None:
        """Register keyboard accelerators for lab convenience."""
        QShortcut(QKeySequence("Space"), self, self._on_space_pressed)
        QShortcut(QKeySequence("S"), self, self._on_btn_stop_clicked)
        QShortcut(QKeySequence("Escape"), self, self._on_btn_discard_clicked)
        QShortcut(QKeySequence("Ctrl+B"), self, self.toggle_sidebar)
        QShortcut(QKeySequence("F4"), self, self.toggle_sidebar)

    def _init_workers(self) -> None:
        """Initialize producer and processor threads."""
        logger.info("Initializing Azure Kinect capture pipeline...")

        # 1. Kinect Stream
        try:
            self.stream_worker = create_kinect_stream(use_mock=self.use_mock, device_id=self.device_id)
        except RuntimeError as exc:
            self._initialization_error = str(exc)
            self._set_sensor_status("Sensor unavailable", "error")
            self.btn_start.setEnabled(False)
            self.btn_start.setToolTip(self._initialization_error)
            self.status_bar.showMessage(self._initialization_error)
            logger.error("Capture initialization blocked: %s", exc)
            return
        self._ui_frame_queue = self.stream_worker.subscribe(maxsize=5)

        # 2. Audio Recorder
        self.audio_worker = AudioRecorderWorker(
            sample_rate=48000,
            channels=1,
            allow_synthetic=self.use_mock,
            prefer_kinect_wdm=not self.use_mock,
        )

        # 3. Hand Tracker — gesture-only at ~10 fps so drawing does not stall video.
        hand_q = self.stream_worker.subscribe(maxsize=2)
        self.hand_worker = HandTrackerWorker(
            frame_queue=hand_q,
            projection_fn=self.stream_worker.convert_2d_to_3d,
            gesture_only=True,
            target_fps=10.0,
        )

        # 4. Wire Gesture Engine
        self.gesture_engine.on_command_triggered = self._handle_gesture_command
        self.gesture_engine.on_state_change = self._on_gesture_state_changed
        self.gesture_engine.on_audio_beep = self._play_beep

        # Start all workers
        self.stream_worker.start()
        self.audio_worker.start()
        self.hand_worker.start()

        # Update HW label — stay honest until frames actually arrive.
        from ..core.kinect_stream import MockKinectStreamWorker
        if isinstance(self.stream_worker, MockKinectStreamWorker):
            self._set_sensor_status("Demo mode", "warn")
        else:
            self._set_sensor_status("Starting camera…", "warn")
            self.status_bar.showMessage(
                "Opening Azure Kinect… viewfinder stays blank until the first frame arrives."
            )
        # START is enabled by the telemetry tick once frames and audio are ready.
        self.btn_start.setEnabled(self.use_mock)

    def _play_beep(self, beep_type: str) -> None:
        """Provide acoustic feedback on gesture activation."""
        def beep_thread():
            try:
                import platform
                if platform.system() == "Windows":
                    import winsound
                    if beep_type == "confirm":
                        winsound.Beep(1200, 150)
                        winsound.Beep(1600, 180)
                    else:
                        winsound.Beep(1000, 120)
            except Exception:
                pass

        threading.Thread(target=beep_thread, daemon=True).start()

    def _get_current_metadata(self, include_intrinsics: bool = True) -> SessionMetadata:
        """Extract metadata from UI configuration controls."""
        height_cm = int(self.combo_height.currentText().split()[0])
        lighting = self.combo_lighting.currentText()
        surface = self.combo_surface.currentText()
        trial = int(self.combo_trial.currentText().split()[-1])
        lux = int(self.spin_lux.value())
        session_name = sanitize_session_name(self.edit_session_name.text()) if hasattr(self, "edit_session_name") else ""
        allow_no_audio = self.chk_allow_no_audio.isChecked() if hasattr(self, "chk_allow_no_audio") else False

        intrinsics = {"fx": 505.2, "fy": 505.4, "cx": 320.1, "cy": 288.4}
        if include_intrinsics and self.stream_worker is not None:
            try:
                intrinsics = self.stream_worker.get_intrinsics()
            except Exception:
                pass

        audio_expected = True
        if self.audio_worker is not None and self.audio_worker.capture_mode == "failed" and allow_no_audio:
            audio_expected = False

        return SessionMetadata(
            mounting_height_cm=height_cm,
            lighting_type=lighting,
            measured_lux=lux,
            surface_state=surface,
            trial_number=trial,
            depth_intrinsics=intrinsics,
            sensor_model="Synthetic Azure Kinect Demo" if self.use_mock else "Microsoft Azure Kinect DK",
            session_name=session_name,
            audio_expected=audio_expected,
        )

    def _start_recording(self) -> None:
        """Start direct raw MKV, WAV, metadata, and gesture-event capture."""
        if self.session_writer is not None or self.gesture_engine.state == GestureState.SAVING:
            return
        if self._initialization_error or not isinstance(self.stream_worker, KinectStreamWorker):
            msg = self._initialization_error or "Direct MKV recording requires the physical Azure Kinect."
            self.status_bar.showMessage(msg)
            QMessageBox.warning(self, "Cannot start", msg)
            return
        if not self._camera_ready():
            msg = self.stream_worker.last_error or (
                "The Kinect stream is not live. Restart the dashboard and wait for "
                "both viewfinders before recording."
            )
            QMessageBox.warning(self, "Cannot start", msg)
            return
        if not self._audio_ready():
            detail = self.audio_worker.last_error if self.audio_worker else "Audio input unavailable."
            QMessageBox.warning(self, "Cannot start", f"Fresh Kinect microphone samples required.\n\n{detail}")
            return

        metadata = self._get_current_metadata()
        self._session_audio_queue = self.audio_worker.subscribe(maxsize=600)
        writer = RawMkvSessionWriter(self.output_dir, metadata, self._session_audio_queue)
        writer.start()
        writer.ready.wait(timeout=5.0)
        if writer.setup_error:
            self._detach_session_queues()
            QMessageBox.critical(self, "Cannot start", writer.setup_error)
            return
        if not self.stream_worker.start_mkv_recording(str(writer.mkv_path)):
            writer.finalization_error = self.stream_worker.mkv_error
            self._detach_session_queues()
            writer.stop_and_save()
            writer.join(timeout=5.0)
            QMessageBox.critical(self, "Cannot start MKV", self.stream_worker.mkv_error or "Unknown MKV error.")
            return

        self.session_writer = writer
        self._session_start_time = writer.start_monotonic
        self.gesture_engine.set_state(GestureState.RECORDING)
        self._set_controls_recording_state(True)
        self.status_bar.showMessage(f"Raw MKV recording: {writer.session_dir}")
        logger.info("Direct MKV recording started -> %s", writer.session_dir)

    def _pause_recording(self) -> None:
        """Pause active recording."""
        # The gesture engine may already have switched itself to PAUSED before
        # emitting the command, so decide from the writer, not the engine state.
        if self.session_writer is None or self.session_writer.paused.is_set():
            return
        self.session_writer.pause()
        self.session_writer.record_gesture_event("pause")
        self._pause_started_at = time.monotonic()
        self.gesture_engine.set_state(GestureState.PAUSED)
        self.btn_pause.setText("Resume")
        self.status_bar.showMessage("Recording paused. Press Resume or cover the lens 3 s to stop.")

    def _resume_recording(self) -> None:
        """Resume paused recording."""
        if self.session_writer is None or not self.session_writer.paused.is_set():
            return
        self.session_writer.resume()
        self.session_writer.record_gesture_event("resume")
        if self._pause_started_at is not None:
            self._paused_total_sec += time.monotonic() - self._pause_started_at
            self._pause_started_at = None
        self.gesture_engine.set_state(GestureState.RECORDING)
        self.btn_pause.setText("Pause")
        self.status_bar.showMessage("Recording resumed.")

    def _stop_and_save_recording(self, source: str = "button") -> None:
        """Finalize and save session."""
        if self.session_writer is None:
            # The gesture engine may already have moved to SAVING; recover.
            if self.gesture_engine.state != GestureState.IDLE:
                self.gesture_engine.set_state(GestureState.IDLE)
                self._set_controls_recording_state(False)
                self._refresh_start_availability()
            return
        if self._pause_started_at is not None:
            self._paused_total_sec += time.monotonic() - self._pause_started_at
            self._pause_started_at = None
        self.status_bar.showMessage("Finalizing session files. Controls unlock when the files are closed.")
        self.gesture_engine.set_state(GestureState.SAVING)
        self._set_controls_finalizing_state()
        writer = self.session_writer
        self.session_writer = None
        writer.record_gesture_event("stop", source=source)
        writer.stop_source = source
        if isinstance(self.stream_worker, KinectStreamWorker):
            if not self.stream_worker.stop_mkv_recording():
                writer.finalization_error = self.stream_worker.mkv_error
            writer.frames_written = self.stream_worker.mkv_frames_written
        self._detach_session_queues()
        writer.stop_and_save()
        self._poll_writer_finalized(writer, discarded=False)

    def _detach_session_queues(self) -> None:
        """Stop producers from refilling queues while the writer drains them."""
        if self._session_frame_queue is not None and self.stream_worker is not None:
            self.stream_worker.unsubscribe(self._session_frame_queue)
        if self._session_rgb_queue is not None and self.stream_worker is not None:
            self.stream_worker.unsubscribe(self._session_rgb_queue)
        if self._session_audio_queue is not None and self.audio_worker is not None:
            self.audio_worker.unsubscribe(self._session_audio_queue)
        self._session_frame_queue = None
        self._session_rgb_queue = None
        self._session_audio_queue = None

    def _set_controls_finalizing_state(self) -> None:
        self.config_group.setEnabled(False)
        for button in (self.btn_start, self.btn_pause, self.btn_stop, self.btn_discard):
            button.setEnabled(False)

    def _poll_writer_finalized(self, writer: SessionWriterWorker, discarded: bool) -> None:
        """Publish success only after the writer has actually closed its files."""
        if writer.is_alive():
            self.lbl_writer_stats.setText(
                f"finalizing… {writer.frames_written} frames, queue {writer.frame_queue.qsize()}"
            )
            QTimer.singleShot(100, lambda: self._poll_writer_finalized(writer, discarded))
            return
        self.gesture_engine.set_state(GestureState.IDLE)
        self._set_controls_recording_state(False)
        self._refresh_start_availability()
        if discarded:
            self.status_bar.showMessage(f"Session discarded: {writer.session_id}")
            logger.info("Session %s discarded.", writer.session_id)
            return

        summary = (
            f"{writer.frames_written} frames, {writer.elapsed_seconds(writer.end_monotonic):.1f} s, "
            f"{len(writer.segment_marks)} segment(s)"
        )
        if writer.finalization_error:
            self.status_bar.showMessage(f"Saved with problems: {writer.finalization_error}")
            QMessageBox.warning(
                self,
                "Session saved with problems",
                f"{writer.session_dir}\n\n{summary}\n\n{writer.finalization_error}",
            )
        else:
            self.status_bar.showMessage(f"Session saved: {writer.session_dir}  ({summary})")
        logger.info("Session %s saved (%s).", writer.session_id, summary)

    def _discard_recording(self) -> None:
        """Discard current recording session."""
        reply = QMessageBox.question(
            self,
            "Confirm Discard",
            "Are you sure you want to discard this recording session? All captured files will be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.session_writer is not None:
                writer = self.session_writer
                self.session_writer = None
                if isinstance(self.stream_worker, KinectStreamWorker):
                    self.stream_worker.stop_mkv_recording()
                    writer.frames_written = self.stream_worker.mkv_frames_written
                self._detach_session_queues()
                self._set_controls_finalizing_state()
                self.gesture_engine.set_state(GestureState.SAVING)
                writer.discard()
                self._poll_writer_finalized(writer, discarded=True)

    def _set_controls_recording_state(self, is_recording: bool) -> None:
        """Lock configuration controls and toggle button availability."""
        # Lock metadata selectors
        self.config_group.setEnabled(not is_recording)

        self.btn_start.setEnabled(not is_recording)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(is_recording)
        self.btn_discard.setEnabled(is_recording)

        if not is_recording:
            self.btn_pause.setText("Pause")
            self.btn_start.setText("●  Start")
            self.lbl_master_timer.setText("00:00.000")
            self.lbl_writer_stats.setText("—")
        else:
            self.btn_start.setText("●  Recording")

    def _handle_gesture_command(self, cmd: GestureCommand) -> None:
        """Respond to command emitted by GestureEngine."""
        if cmd == GestureCommand.START_RECORDING:
            self._start_recording()
            if self.gesture_engine.state != GestureState.RECORDING:
                # Start was refused (camera/audio). Return engine to IDLE.
                self.gesture_engine.set_state(GestureState.IDLE)
        elif cmd in (GestureCommand.PAUSE_RECORDING, GestureCommand.RESUME_RECORDING):
            return
        elif cmd == GestureCommand.STOP_AND_SAVE:
            self._stop_and_save_recording(source="camera_cover")
        elif cmd == GestureCommand.MARK_SEGMENT:
            self._mark_segment(source="pinch")
        elif cmd == GestureCommand.DISCARD:
            self._discard_recording()

    def _mark_segment(self, source: str = "pinch") -> None:
        """Handle bookmark / mark segment event from pinch gesture."""
        if self.session_writer is not None:
            mark = self.session_writer.mark_segment(source=source)
            self.status_bar.showMessage(
                f"Gesture captured: Segment marked (#{mark['segment_id']} @ {mark['elapsed_seconds']:.1f}s)",
                3000,
            )
            self.rgb_viewport.flash_message(f"SEGMENT #{mark['segment_id']} MARKED")
            logger.info("Segment %d marked at %.2fs", mark["segment_id"], mark["elapsed_seconds"])

    def _on_btn_start_clicked(self) -> None:
        self._start_recording()

    def _on_btn_pause_clicked(self) -> None:
        if self.session_writer is None:
            return
        if self.session_writer.paused.is_set():
            self._resume_recording()
        else:
            self._pause_recording()

    def _on_btn_stop_clicked(self) -> None:
        if self.gesture_engine.state in (GestureState.RECORDING, GestureState.PAUSED):
            self._stop_and_save_recording(source="button")

    def _on_btn_discard_clicked(self) -> None:
        self._discard_recording()

    def _on_space_pressed(self) -> None:
        if self.session_writer is None and self.gesture_engine.state == GestureState.IDLE:
            self._start_recording()

    def _on_gesture_state_changed(self, state: GestureState) -> None:
        self._update_status_badge_style(state)

    def _update_status_badge_style(self, state: GestureState) -> None:
        """Update top status pill appearance."""
        if state == GestureState.IDLE:
            self.lbl_status_badge.setText("IDLE")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #10141D; color: #7D8590; border: 1px solid #1F2535;"
            )
        elif state == GestureState.RECORDING:
            self.lbl_status_badge.setText("REC")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #2D1417; color: #F85149; border: 1px solid #DA3633;"
            )
        elif state == GestureState.PAUSED:
            self.lbl_status_badge.setText("PAUSED")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #241A0D; color: #D29922; border: 1px solid #9E6A03;"
            )
        elif state == GestureState.SAVING:
            self.lbl_status_badge.setText("SAVING")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #0E1D33; color: #58A6FF; border: 1px solid #1F6FEB;"
            )

    def _on_render_tick(self) -> None:
        """30 FPS main UI render tick: update viewports, HUD overlays, timers, and VU meter."""
        # 1. Fetch latest frame
        latest_packet: Optional[FramePacket] = None
        if self._ui_frame_queue is not None:
            while not self._ui_frame_queue.empty():
                try:
                    latest_packet = self._ui_frame_queue.get_nowait()
                except queue.Empty:
                    break

        # 2. Get latest hand tracking result
        tracking_result = self.hand_worker.get_latest_result() if self.hand_worker else None

        # 3. Update Gesture State Machine
        gesture_hud = self.gesture_engine.update(tracking_result)

        # 4. Render RGB Viewport
        if latest_packet is not None:
            self.rgb_viewport.update_frame(
                latest_packet.rgb_image,
                tracking_result=tracking_result,
                gesture_overlay=gesture_hud,
            )
            # Render Depth Viewport
            self.depth_viewport.update_depth_frame(latest_packet.depth_image)

        # 5. Update Master Clock while a session is open. The writer owns the
        #    authoritative clock (it excludes paused time), so read from it.
        if self.gesture_engine.state in (GestureState.RECORDING, GestureState.PAUSED):
            if self.session_writer is not None:
                elapsed = self.session_writer.elapsed_seconds()
            else:
                elapsed = 0.0
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            msecs = int((elapsed * 1000) % 1000)
            self.lbl_master_timer.setText(f"{mins:02d}:{secs:02d}.{msecs:03d}")

        if self.gesture_engine.state == GestureState.RECORDING:
            # Pulsing recording badge
            self._flash_state = not self._flash_state
            if self._flash_state:
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #DA3633; color: #FFFFFF; border: 1px solid #F85149;"
                )
            else:
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #2D1417; color: #F85149; border: 1px solid #DA3633;"
                )

        # 6. Gesture HUD text in the sidebar (mirrors the on-video card)
        hud_text = gesture_hud.hud_message if gesture_hud.is_active else (
            "READY" if gesture_hud.hud_message == "READY" else "-"
        )
        if hud_text != self._last_gesture_hud:
            self.lbl_gesture_hud.setText(hud_text)
            self._last_gesture_hud = hud_text

        # 7. Update Audio VU Meter
        if self.audio_worker is not None:
            rms = self.audio_worker.current_rms_db
            peak = self.audio_worker.peak_db
            self.vu_meter.set_levels(rms, peak)
            if self.audio_worker.capture_mode != "failed":
                self.lbl_audio_stats.setText(f"RMS: {rms:+.1f} dBFS  |  Peak: {peak:+.1f} dBFS")

    def _on_telemetry_tick(self) -> None:
        """1-second update for system statistics."""
        if self.audio_worker is not None:
            mode = self.audio_worker.capture_mode
            if mode == "failed":
                self.lbl_audio_stats.setText("No input")
                self.lbl_audio_device.setText("Microphone unavailable — enable testing mode or fix audio device")
            elif mode == "hardware":
                name = self.audio_worker.active_device_name or "default input"
                self.lbl_audio_device.setText(
                    f"{name} · {self.audio_worker.active_input_channels} ch @ {self.audio_worker.sample_rate} Hz"
                )
            elif mode == "synthetic":
                self.lbl_audio_device.setText("Synthetic demo audio")

            if mode == "failed" and not self._audio_warning_shown:
                self._audio_warning_shown = True
                self.status_bar.showMessage(
                    (self.audio_worker.last_error or "Audio input is unavailable.")
                    + "  Enable 'Allow recording without audio' to test without sound."
                )

        if self.session_writer is not None:
            w = self.session_writer
            self.lbl_writer_stats.setText(
                f"{w.frames_written} written · {len(w.segment_marks)} marks · q={w.frame_queue.qsize()}"
            )
            if w.finalization_error and not w.is_alive():
                # Writer died mid-session: surface it and unlock controls.
                self.status_bar.showMessage(f"Recording stopped: {w.finalization_error}")
                self._stop_and_save_recording(source="writer_error")

        self._refresh_start_availability()

        if self.stream_worker is not None:
            worker = self.stream_worker
            camera_problem: Optional[str] = None
            if not worker.is_alive():
                camera_problem = worker.last_error or "Kinect capture thread stopped."
            elif worker.frame_count == 0:
                self._set_sensor_status("Waiting for frames…", "warn")
            elif not worker.has_fresh_frame(CAMERA_STALE_AFTER_SECONDS):
                camera_problem = worker.last_error or (
                    f"Kinect frame stream stalled for more than "
                    f"{CAMERA_STALE_AFTER_SECONDS:.0f} seconds."
                )
            elif self.use_mock:
                self._set_sensor_status("Demo mode", "warn")
            else:
                self._set_sensor_status("Kinect online", "ok")

            self.lbl_fps_capture.setText(
                f"{worker.fps_measured:.1f}" if camera_problem is None else "0.0"
            )
            self.lbl_frames_total.setText(str(worker.frame_count))
            recording_drops = sum(
                getattr(q, "dropped_frames", 0)
                for q in (self._session_frame_queue, self._session_rgb_queue)
                if q is not None
            )
            self.lbl_frames_dropped.setText(str(recording_drops))
            if isinstance(worker, KinectStreamWorker) and worker.mkv_error and self.session_writer is not None:
                self.session_writer.finalization_error = worker.mkv_error
                self._stop_and_save_recording(source="mkv_error")
            elif camera_problem is not None:
                self._set_sensor_status("Camera stopped", "error")
                self.status_bar.showMessage(camera_problem)
                if self.session_writer is not None:
                    self.session_writer.finalization_error = camera_problem
                    worker.stop()
                    self._stop_and_save_recording(source="camera_error")

        if self.hand_worker is not None:
            res = self.hand_worker.get_latest_result()
            track_fps = self.hand_worker.fps_measured
            if res and res.hand_detected:
                state = f"{res.handedness} hand"
                if res.is_pinch:
                    state += " · pinch"
                self.lbl_hand_state.setText(f"{state} ({track_fps:.0f} fps)")
                self.lbl_pinch_ratio.setText(f"{self.hand_worker.last_pinch_ratio:.2f}")
            else:
                self.lbl_hand_state.setText(f"None ({track_fps:.0f} fps)")
                self.lbl_pinch_ratio.setText("—")

    def _set_sensor_status(self, text: str, tone: str = "neutral") -> None:
        """Update the header sensor pill with consistent styling."""
        styles = {
            "neutral": (
                "color: #7D8590; background-color: #0E121A; border: 1px solid #1F2535;"
                " border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 600;"
            ),
            "ok": (
                "color: #3FB950; background-color: #0F1F14; border: 1px solid #238636;"
                " border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 600;"
            ),
            "warn": (
                "color: #D29922; background-color: #1F1A0B; border: 1px solid #9E6A03;"
                " border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 600;"
            ),
            "error": (
                "color: #F85149; background-color: #2A1214; border: 1px solid #DA3633;"
                " border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 600;"
            ),
        }
        self.lbl_hw_status.setText(f"●  {text}")
        self.lbl_hw_status.setStyleSheet(styles.get(tone, styles["neutral"]))

    def closeEvent(self, event: Any) -> None:
        """Gracefully terminate background threads and close sensors."""
        logger.info("Closing application. Stopping worker threads...")
        if self.session_writer is not None:
            self.session_writer.record_gesture_event("stop", source="window_close")
            self.session_writer.stop_source = "window_close"
            if isinstance(self.stream_worker, KinectStreamWorker):
                self.stream_worker.stop_mkv_recording()
                self.session_writer.frames_written = self.stream_worker.mkv_frames_written
            self._detach_session_queues()
            self.session_writer.stop_and_save()
        if self.stream_worker is not None:
            self.stream_worker.stop()
        if self.audio_worker is not None:
            self.audio_worker.stop()
        if self.hand_worker is not None:
            self.hand_worker.stop()
        for worker in (self.session_writer, self.hand_worker, self.audio_worker, self.stream_worker):
            if worker is not None and worker.is_alive():
                worker.join(timeout=3.0)
        event.accept()
