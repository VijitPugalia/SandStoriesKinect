"""Automated test suite for Azure Kinect Multimodal Capture Dashboard."""

import json
import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from kinect_dashboard.core.audio_recorder import AudioChunk, AudioRecorderWorker
import kinect_dashboard.core.audio_recorder as audio_recorder_module
import kinect_dashboard.core.kinect_stream as kinect_stream_module
from kinect_dashboard.core.gesture_engine import GestureCommand, GestureEngine, GestureState
from kinect_dashboard.core.hand_tracker import (
    HAND_JOINT_NAMES,
    HandTrackerWorker,
    HandTrackingResult,
    Joint3D,
)
from kinect_dashboard.core.kinect_stream import (
    FramePacket,
    KinectStreamWorker,
    MockKinectStreamWorker,
    create_kinect_stream,
)
from kinect_dashboard.storage.raw_mkv_session import RawMkvSessionWriter
from kinect_dashboard.ui.main_window import MainWindow
from kinect_dashboard.storage.session_writer import (
    SessionMetadata,
    SessionWriterWorker,
    generate_session_id,
)


def test_mock_kinect_stream_generation():
    """Verify mock stream generates valid 1080p RGB and 576x640 Depth frames."""
    stream = MockKinectStreamWorker(fps=30.0, height_mm=650.0)
    q = stream.subscribe(maxsize=10)
    stream.start()

    try:
        # Collect 5 frames
        frames = []
        for _ in range(5):
            packet = q.get(timeout=2.0)
            frames.append(packet)

        assert len(frames) == 5
        for f in frames:
            assert isinstance(f, FramePacket)
            assert f.rgb_image.shape == (1080, 1920, 3)
            assert f.rgb_image.dtype == np.uint8
            assert f.depth_image.shape == (576, 640)
            assert f.depth_image.dtype == np.uint16
            assert f.color_timestamp_usec > 0
            assert f.depth_timestamp_usec > 0
            # Test depth projection
            pt3d = stream.convert_2d_to_3d((960, 540), 650.0)
            assert pt3d is not None
            assert len(pt3d) == 3
            assert pt3d[2] == 650.0
    finally:
        stream.stop()
        stream.join(timeout=2.0)


def test_audio_recorder_worker_and_rms(monkeypatch):
    """Verify audio capture worker produces 48 kHz chunks and accurate RMS dB."""
    monkeypatch.setattr(audio_recorder_module, "SOUNDDEVICE_AVAILABLE", False)
    recorder = AudioRecorderWorker(
        sample_rate=48000,
        channels=1,
        block_size=1024,
        allow_synthetic=True,
    )
    q = recorder.subscribe(maxsize=10)
    recorder.start()

    try:
        chunks = []
        for _ in range(3):
            chunk = q.get(timeout=2.0)
            chunks.append(chunk)

        assert len(chunks) == 3
        for c in chunks:
            assert isinstance(c, AudioChunk)
            assert c.sample_rate == 48000
            assert c.samples.dtype == np.int16
            assert -60.0 <= c.rms_db <= 0.0

        # Test RMS dB math
        silence = np.zeros((1024, 1), dtype=np.int16)
        assert recorder._calculate_rms_db(silence) == -60.0

        full_scale = np.full((1024, 1), 32767, dtype=np.int16)
        db_full = recorder._calculate_rms_db(full_scale)
        assert -0.1 <= db_full <= 0.0
    finally:
        recorder.stop()
        recorder.join(timeout=2.0)


def test_hardware_stream_does_not_fall_back_to_mock(monkeypatch):
    """Production mode must surface missing hardware instead of fabricating frames."""
    monkeypatch.setattr(kinect_stream_module, "PYK4A_AVAILABLE", True)

    class _FakePyk4a:
        @staticmethod
        def connected_device_count():
            return 0

    monkeypatch.setattr(kinect_stream_module, "pyk4a", _FakePyk4a(), raising=False)
    with pytest.raises(RuntimeError, match="No Azure Kinect DK"):
        create_kinect_stream(use_mock=False)
    assert isinstance(create_kinect_stream(use_mock=True), MockKinectStreamWorker)


def test_stale_lifetime_frame_count_does_not_enable_recording():
    """Old frames cannot make a stopped or stalled camera look ready."""
    worker = SimpleNamespace(
        frame_count=2032,
        device_started=True,
        last_error=None,
        is_alive=lambda: True,
        has_fresh_frame=lambda max_age: False,
    )
    window = SimpleNamespace(stream_worker=worker, use_mock=False)

    assert MainWindow._camera_ready(window) is False

    worker.has_fresh_frame = lambda max_age: True
    assert MainWindow._camera_ready(window) is True


def test_invalid_mjpg_preview_frame_does_not_kill_capture(monkeypatch):
    """A damaged preview image is skipped and the next synchronized frame survives."""
    monkeypatch.setattr(kinect_stream_module, "PYK4A_AVAILABLE", True)
    worker = KinectStreamWorker(
        color_resolution=0,
        depth_mode=0,
        camera_fps=0,
        lock_exposure=False,
    )

    invalid_capture = SimpleNamespace(
        color=np.array([1, 2, 3], dtype=np.uint8),
        depth=np.full((2, 2), 650, dtype=np.uint16),
        color_timestamp_usec=1,
        depth_timestamp_usec=1,
    )
    valid_capture = SimpleNamespace(
        color=np.zeros((2, 2, 4), dtype=np.uint8),
        depth=np.full((2, 2), 650, dtype=np.uint16),
        color_timestamp_usec=2,
        depth_timestamp_usec=2,
    )

    class _FakeDevice:
        def __init__(self):
            self.captures = iter((invalid_capture, valid_capture))

        def get_capture(self, timeout):
            capture = next(self.captures)
            if capture is valid_capture:
                worker.running.clear()
            return capture

        def stop(self):
            pass

    fake_device = _FakeDevice()
    monkeypatch.setattr(worker, "_initialize_device", lambda: setattr(worker, "k4a", fake_device))

    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert worker.preview_decode_errors == 1
    assert worker.frame_count == 1
    assert worker.last_error is None
    assert worker.frame_queue.get_nowait().color_timestamp_usec == 2


def test_repeated_capture_errors_stop_worker(monkeypatch):
    """A dead sensor cannot leave the capture worker retrying forever."""
    monkeypatch.setattr(kinect_stream_module, "PYK4A_AVAILABLE", True)
    worker = KinectStreamWorker(
        color_resolution=0,
        depth_mode=0,
        camera_fps=0,
        lock_exposure=False,
    )

    class _FailingDevice:
        def get_capture(self, timeout):
            raise RuntimeError("device timeout")

        def stop(self):
            pass

    fake_device = _FailingDevice()
    monkeypatch.setattr(worker, "_initialize_device", lambda: setattr(worker, "k4a", fake_device))

    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert worker.last_error is not None
    assert "5 consecutive" in worker.last_error


def test_audio_requires_explicit_synthetic_mode(monkeypatch):
    """A missing microphone must fail closed unless demo audio was requested."""
    monkeypatch.setattr(audio_recorder_module, "SOUNDDEVICE_AVAILABLE", False)
    recorder = AudioRecorderWorker(allow_synthetic=False)
    recorder.start()
    recorder.join(timeout=1.0)
    assert recorder.capture_mode == "failed"
    assert recorder.total_samples_captured == 0


def test_missing_aligned_depth_is_not_fabricated():
    tracker = HandTrackerWorker(frame_queue=queue.Queue())
    assert tracker._sample_depth(None, 10.0, 10.0) is None
    assert tracker._sample_depth(np.zeros((8, 8), dtype=np.uint16), 4.0, 4.0) is None


def test_hand_tracker_3d_projection_and_velocity():
    """Verify 3D joint landmark calculation and wrist velocity tracking."""
    stream_q = queue.Queue()
    tracker = HandTrackerWorker(frame_queue=stream_q)

    # Synthetic RGB frame and Depth map
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    depth = np.full((576, 640), 700, dtype=np.uint16)

    # Process frame
    res = tracker.process_frame(
        rgb_image=rgb,
        depth_image=depth,
        transformed_depth=None,
        frame_index=1,
        timestamp_usec=1000000,
    )
    assert isinstance(res, HandTrackingResult)
    assert res.frame_index == 1
    assert res.device_timestamp_usec == 1000000

    # Test velocity calculation
    v1 = tracker._compute_wrist_velocity(1.0, (0.0, 0.0, 700.0))
    v2 = tracker._compute_wrist_velocity(1.1, (10.0, 0.0, 700.0))
    # 10 mm in 0.1s = 100 mm/s
    assert abs(v2 - 100.0) < 5.0


def test_gesture_state_machine_transitions():
    """Verify pinch and camera-cover gesture rules."""
    engine = GestureEngine(
        pinch_hold_sec=0.1,
        cover_hold_sec=0.1,
    )

    triggered_commands = []
    engine.on_command_triggered = lambda cmd: triggered_commands.append(cmd)

    assert engine.state == GestureState.IDLE
    engine.set_state(GestureState.RECORDING)

    time.sleep(1.3)  # Cooldown from set_state if any

    # Pinch held for 0.1s in RECORDING -> MARK_SEGMENT
    pinch_result = HandTrackingResult(
        frame_index=1,
        device_timestamp_usec=1000000,
        hand_detected=True,
        is_pinch=True,
        pinch_distance_mm=15.0,
        wrist_velocity_mm_s=5.0,
        mean_luminance=0.5,
        std_luminance=0.1,
    )
    overlay = engine.update(pinch_result)
    assert overlay.is_active
    assert overlay.active_gesture_name == "PINCH"
    time.sleep(0.12)
    pinch_result.frame_index = 2
    pinch_result.device_timestamp_usec = 1_100_000
    engine.update(pinch_result)
    assert GestureCommand.MARK_SEGMENT in triggered_commands

    time.sleep(1.3)  # Cooldown

    # Camera Cover held for 0.1s -> STOP_AND_SAVE
    covered_result = HandTrackingResult(
        frame_index=2,
        device_timestamp_usec=2000000,
        hand_detected=False,
        mean_luminance=0.03,
        std_luminance=0.01,
    )
    overlay = engine.update(covered_result)
    assert overlay.is_active
    assert overlay.active_gesture_name == "COVER CAMERA"
    time.sleep(0.12)
    covered_result.frame_index = 3
    covered_result.device_timestamp_usec = 2_100_000
    engine.update(covered_result)
    assert engine.state == GestureState.SAVING
    assert GestureCommand.STOP_AND_SAVE in triggered_commands


def test_unified_gestures_priority_and_hysteresis():
    """Verify Camera Cover priority over Pinch, and Pinch separation hysteresis."""
    engine = GestureEngine(
        pinch_hold_sec=0.1,
        cover_hold_sec=0.1,
    )
    engine.set_state(GestureState.RECORDING)
    triggered_commands = []
    engine.on_command_triggered = lambda cmd: triggered_commands.append(cmd)

    # Both Camera Cover and Pinch active -> Cover must override
    ambiguous_result = HandTrackingResult(
        frame_index=1,
        device_timestamp_usec=1000000,
        hand_detected=True,
        is_pinch=True,
        mean_luminance=0.02,
        std_luminance=0.01,
        wrist_velocity_mm_s=5.0,
    )
    overlay = engine.update(ambiguous_result)
    assert overlay.active_gesture_name == "COVER CAMERA"
    time.sleep(0.12)
    ambiguous_result.frame_index = 2
    ambiguous_result.device_timestamp_usec = 1_100_000
    engine.update(ambiguous_result)
    assert GestureCommand.STOP_AND_SAVE in triggered_commands
    assert GestureCommand.MARK_SEGMENT not in triggered_commands


def test_pinch_marks_segment_while_hand_moves():
    """Pinch held 2 s must mark even when the hand is moving."""
    engine = GestureEngine(pinch_hold_sec=0.1)
    engine.set_state(GestureState.RECORDING)
    triggered = []
    engine.on_command_triggered = lambda cmd: triggered.append(cmd)

    moving_pinch = HandTrackingResult(
        frame_index=1,
        device_timestamp_usec=1_000_000,
        hand_detected=True,
        is_pinch=True,
        wrist_velocity_mm_s=450.0,
        mean_luminance=0.5,
        std_luminance=0.1,
    )
    overlay = engine.update(moving_pinch)
    assert overlay.is_active and overlay.active_gesture_name == "PINCH"
    time.sleep(0.12)
    moving_pinch.frame_index = 2
    moving_pinch.device_timestamp_usec = 1_100_000
    engine.update(moving_pinch)
    assert triggered == [GestureCommand.MARK_SEGMENT]


def test_frozen_tracking_frame_cannot_complete_hold():
    """Repeated UI reads of one tracking frame must not trigger a gesture."""
    engine = GestureEngine(pinch_hold_sec=0.05, max_tracking_gap_sec=0.02)
    engine.set_state(GestureState.RECORDING)
    triggered = []
    engine.on_command_triggered = lambda cmd: triggered.append(cmd)
    pinch = HandTrackingResult(
        frame_index=1,
        device_timestamp_usec=1_000_000,
        hand_detected=True,
        is_pinch=True,
        mean_luminance=0.5,
        std_luminance=0.1,
    )

    engine.update(pinch)
    time.sleep(0.07)
    engine.update(pinch)

    assert triggered == []


def test_tracking_loss_does_not_rearm_pinch():
    """A marked pinch requires an observed separation before another mark."""
    engine = GestureEngine(pinch_hold_sec=0.02)
    engine.set_state(GestureState.RECORDING)
    triggered = []
    engine.on_command_triggered = lambda cmd: triggered.append(cmd)

    def sample(frame_index: int, is_pinch: bool, hand_detected: bool = True):
        return HandTrackingResult(
            frame_index=frame_index,
            device_timestamp_usec=frame_index * 100_000,
            hand_detected=hand_detected,
            is_pinch=is_pinch,
            mean_luminance=0.5,
            std_luminance=0.1,
        )

    engine.update(sample(1, True))
    time.sleep(0.03)
    engine.update(sample(2, True))
    assert triggered == [GestureCommand.MARK_SEGMENT]

    engine._cooldown_until = 0.0
    engine.update(None)
    engine.update(sample(3, True))
    time.sleep(0.03)
    engine.update(sample(4, True))
    assert triggered == [GestureCommand.MARK_SEGMENT]

    engine.update(sample(5, False))
    engine.update(sample(6, True))
    time.sleep(0.03)
    engine.update(sample(7, True))
    assert triggered == [GestureCommand.MARK_SEGMENT, GestureCommand.MARK_SEGMENT]


def test_pinch_classification_uses_2d_ratio_with_hysteresis():
    """Pinch uses image-space ratio and holds through small openings."""
    tracker = HandTrackerWorker(frame_queue=queue.Queue())
    wrist = (500.0, 500.0)
    middle_mcp = (500.0, 300.0)

    def landmarks(thumb_index_gap_px: float):
        pts = [(0.0, 0.0)] * 21
        pts[0] = wrist
        pts[9] = middle_mcp
        pts[4] = (400.0, 350.0)
        pts[8] = (400.0 + thumb_index_gap_px, 350.0)
        for idx in (5, 13, 17):
            pts[idx] = (480.0 + idx, 320.0)
        for idx in (6, 10, 14, 18):
            pts[idx] = (480.0 + idx, 250.0)
        for idx in (12, 16, 20):
            pts[idx] = (480.0 + idx, 200.0)
        pts[2], pts[3] = (430.0, 420.0), (410.0, 380.0)
        pts[7] = (400.0, 230.0)
        pts[11], pts[15], pts[19] = (500.0, 220.0), (520.0, 220.0), (540.0, 220.0)
        return pts

    _, _, _, _, is_pinch, _ = tracker._classify_gestures(landmarks(40.0), [])
    assert is_pinch
    tracker._last_pinch = True
    _, _, _, _, still_pinch, _ = tracker._classify_gestures(landmarks(72.0), [])
    assert still_pinch
    _, _, _, _, released, _ = tracker._classify_gestures(landmarks(100.0), [])
    assert not released


def test_session_name_prefixes_folder_and_is_sanitized():
    meta = SessionMetadata(
        session_name="Pilot 01 / p03!",
        mounting_height_cm=80,
        lighting_type="Simulated Overcast",
        surface_state="Wet Compacted Sand",
        trial_number=2,
    )
    ts = datetime(2026, 9, 8, 16, 30, 5)
    assert generate_session_id(meta, ts) == "Pilot_01_p03_20260908_163005_H80_Overcast_Wet_T2"
    assert generate_session_id(SessionMetadata(), ts) == "session_20260908_163005_H60_Sun_Dry_T1"


def test_kinect_audio_uses_channel_zero_not_average():
    """Kinect WDM seven-mic capture must use channel 0, not downmix all mics."""
    recorder = AudioRecorderWorker()
    t = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    ch0 = (np.sin(t * 20.0) * 0.4).astype(np.float32)
    others = np.zeros((128, 6), dtype=np.float32)
    data = np.column_stack([ch0, others])

    recorder._active_extract = "kinect_ch0"
    mono = recorder._to_mono_int16(data)
    expected = np.rint(np.clip(ch0, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(-1, 1)
    np.testing.assert_array_equal(mono, expected)

    # When every mic carries the same signal, downmixing still works but Kinect
    # mode must keep channel 0 only (others may be phase-shifted in hardware).
    shifted = np.column_stack([ch0] + [ch0 * 0.25] * 6)
    recorder._active_extract = "kinect_ch0"
    kinect_mono = recorder._to_mono_int16(shifted)
    recorder._active_extract = "mono"
    downmix_mono = recorder._to_mono_int16(shifted)
    assert not np.array_equal(kinect_mono, downmix_mono)


def test_session_writer_video_writes_one_frame_per_capture(tmp_path: Path):
    """Video should write captured frames directly without duplicate gap-fill."""
    writer = SessionWriterWorker(
        output_root=tmp_path,
        metadata=SessionMetadata(fps=30, audio_expected=False),
        frame_queue=queue.Queue(),
        audio_queue=queue.Queue(),
    )
    writer.start()
    assert writer.ready.wait(timeout=5.0)
    base = writer.start_monotonic
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    depth = np.full((576, 640), 650, dtype=np.uint16)
    for i in range(20):
        writer.frame_queue.put(
            FramePacket(
                frame_index=i,
                rgb_image=rgb,
                depth_image=depth,
                color_timestamp_usec=1_000_000 + i * 100_000,
                depth_timestamp_usec=1_000_000 + i * 100_000,
                system_timestamp_monotonic=base + i * 0.1,
            )
        )
    time.sleep(0.3)
    writer.mark_segment()
    writer.stop_and_save()
    writer.join(timeout=10.0)

    assert writer.frames_written == 20
    manifest = json.loads((writer.session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["captureFormat"] == "v2_streams"
    assert manifest["audio"] is None
    assert len(manifest["segments"]) == 1
    assert "monotonicElapsedSeconds" in manifest["segments"][0]
    assert (writer.session_dir / "streams" / "rgb.mp4").is_file()
    assert not (writer.session_dir / "metadata.json").exists()
    assert not (writer.session_dir / "gesture_events.jsonl").exists()


def test_session_writer_fair_export(tmp_path: Path):
    """Verify SessionWriter creates all 5 FAIR files with exact schema & compression."""
    metadata = SessionMetadata(
        mounting_height_cm=60,
        lighting_type="Simulated Harsh Sun",
        measured_lux=8500,
        surface_state="Dry Loose Sand",
        trial_number=1,
    )

    frame_q = queue.Queue()
    audio_q = queue.Queue()
    hand_q = queue.Queue()

    writer = SessionWriterWorker(
        output_root=tmp_path,
        metadata=metadata,
        frame_queue=frame_q,
        audio_queue=audio_q,
        hand_queue=hand_q,
    )
    writer.start()
    assert writer.ready.wait(timeout=5.0)
    base = writer.start_monotonic

    # Feed 10 synthetic frames
    for i in range(10):
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.full((576, 640), 650 + i, dtype=np.uint16)
        packet = FramePacket(
            frame_index=i,
            rgb_image=rgb,
            depth_image=depth,
            color_timestamp_usec=1000000 + i * 33333,
            depth_timestamp_usec=1000000 + i * 33333,
            system_timestamp_monotonic=base + i * (1.0 / 30.0),
        )
        frame_q.put(packet)

        # Feed audio chunk
        audio_chunk = AudioChunk(
            samples=np.full((1600, 1), 100, dtype=np.int16),
            sample_rate=48000,
        )
        audio_q.put(audio_chunk)

        # Feed hand tracking result
        hand_res = HandTrackingResult(
            frame_index=i,
            device_timestamp_usec=1000000 + i * 33333,
            hand_detected=True,
            handedness="Right",
            landmarks_3d_mm=[
                {"joint_id": 0, "name": "WRIST", "x": 12.4, "y": -45.2, "z": 650.1},
                {"joint_id": 4, "name": "THUMB_TIP", "x": 35.1, "y": -20.4, "z": 642.0},
                {"joint_id": 8, "name": "INDEX_FINGER_TIP", "x": 48.0, "y": 10.2, "z": 635.5},
            ],
        )
        hand_q.put(hand_res)

    # Test mark segment
    writer.mark_segment()

    time.sleep(0.3)
    writer.stop_and_save()
    writer.join(timeout=5.0)

    # Verify session directory naming
    session_dir = writer.session_dir
    assert session_dir.is_dir()
    assert "session_" in session_dir.name
    assert "_H60_Sun_Dry_T1" in session_dir.name

    # 1. Verify manifest.json basic metadata and segments
    manifest_path = session_dir / "manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["captureFormat"] == "v2_streams"
    assert manifest_data["captureDeviceFamily"] == "azure_kinect_dk"
    assert manifest_data["deviceModel"] == "Microsoft Azure Kinect DK"
    assert manifest_data["experimentalConditions"]["mountingHeightCm"] == 60
    assert manifest_data["experimentalConditions"]["lightingType"] == "Simulated Harsh Sun"
    assert len(manifest_data["segments"]) == 1
    assert manifest_data["segments"][0]["segmentId"] == 1
    assert "monotonicElapsedSeconds" in manifest_data["segments"][0]

    # 2. Verify streams/depth.f32 (raw uncompressed float32 meters)
    depth_path = session_dir / "streams" / "depth.f32"
    assert depth_path.is_file()
    depth_bytes = depth_path.read_bytes()
    expected_frame_bytes = 576 * 640 * 4
    assert len(depth_bytes) == 10 * expected_frame_bytes
    depth_arr = np.frombuffer(depth_bytes, dtype="<f4").reshape((10, 576, 640))
    # Frame 0 had 650 mm -> 0.650 meters, Frame 9 had 659 mm -> 0.659 meters
    np.testing.assert_allclose(depth_arr[0], 0.650, atol=1e-5)
    np.testing.assert_allclose(depth_arr[9], 0.659, atol=1e-5)

    # 3. Verify streams/rgb.mp4
    mp4_path = session_dir / "streams" / "rgb.mp4"
    assert mp4_path.is_file()
    assert mp4_path.stat().st_size > 0

    # 4. Verify audio.wav
    wav_path = session_dir / "audio.wav"
    assert wav_path.is_file()
    assert wav_path.stat().st_size > 0

    # 5. Verify v2_frames.jsonl
    frames_path = session_dir / "v2_frames.jsonl"
    assert frames_path.is_file()
    frame_records = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(frame_records) == 10
    assert frame_records[0]["frameIndex"] == 1
    assert frame_records[0]["depthByteOffset"] == 0
    assert frame_records[0]["depthByteCount"] == expected_frame_bytes
    assert frame_records[9]["frameIndex"] == 10
    assert frame_records[9]["depthByteOffset"] == 9 * expected_frame_bytes
    assert abs(frame_records[0]["depthMedianMeters"] - 0.650) < 1e-4

    # 6. Verify manifest.json
    manifest_path = session_dir / "manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["captureFormat"] == "v2_streams"
    assert manifest_data["captureDeviceFamily"] == "azure_kinect_dk"
    assert manifest_data["v2Streams"]["depthStreamPath"] == "streams/depth.f32"
    assert manifest_data["v2Streams"]["rgbVideoPath"] == "streams/rgb.mp4"
    assert manifest_data["v2Streams"]["depthFormat"] == "float32_little_endian_meters_contiguous"
    assert manifest_data["depthEncoding"]["format"] == "v2_streams_float32"

    # 7. Verify ro-crate-metadata.json
    rocrate_path = session_dir / "ro-crate-metadata.json"
    assert rocrate_path.is_file()
    with open(rocrate_path, "r", encoding="utf-8") as f:
        rocrate_data = json.load(f)
    assert rocrate_data["@context"] == "https://w3id.org/ro/crate/1.1/context"

    # 8. Verify sidecars (matching iPhone layout)
    assert (session_dir / "boundary_samples.jsonl").is_file()
    assert (session_dir / "objects.jsonl").is_file()

    # 9. Verify NO redundant files are generated (matching iPhone output)
    assert not (session_dir / "metadata.json").exists()
    assert not (session_dir / "hand_trajectories.json").exists()
    assert not (session_dir / "gesture_events.jsonl").exists()
    assert not (session_dir / "strokes.json").exists()


def test_raw_mkv_session_outputs_audio_metadata_and_gestures(tmp_path: Path):
    audio_q = queue.Queue()
    writer = RawMkvSessionWriter(
        tmp_path,
        SessionMetadata(audio_expected=True),
        audio_q,
    )
    writer.start()
    assert writer.ready.wait(timeout=2.0)
    writer.mkv_path.touch()
    audio_q.put(AudioChunk(samples=np.full((4800, 1), 120, dtype=np.int16), sample_rate=48000))
    mark = writer.mark_segment("pinch")
    writer.frames_written = 90
    writer.stop_source = "camera_cover"
    writer.stop_and_save()
    writer.join(timeout=3.0)

    assert not writer.is_alive()
    assert writer.mkv_path.is_file()
    assert writer.audio_path.is_file()
    assert writer.audio_path.stat().st_size > 44
    metadata = json.loads(writer.metadata_path.read_text(encoding="utf-8"))
    assert metadata["captureBackend"] == "pyk4a_direct_mkv"
    assert metadata["capturesWritten"] == 90
    assert metadata["segmentCount"] == 1
    assert metadata["stopSource"] == "camera_cover"
    events = [
        json.loads(line)
        for line in writer.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "mark_segment"
    assert events[0]["segment_id"] == mark["segment_id"]
    assert not (writer.session_dir / "streams" / "rgb.mp4").exists()
    assert not (writer.session_dir / "streams" / "depth.f32").exists()
