"""Session storage worker exporting multimodal captures to standardized v2_streams datasets.

Stores native, uncompressed Azure Kinect captures matching the SandStories iPhone
v2_streams format:
- streams/depth.f32: Contiguous raw Float32 little-endian meters (<f4), zero compression.
- streams/rgb.mp4: Constant frame rate H.264 video.
- v2_frames.jsonl: Per-frame metadata with byte offsets, intrinsics, and depth statistics.
- manifest.json: Authoritative v2_streams dataset manifest.
- ro-crate-metadata.json: FAIR-compliant RO-Crate descriptor.
- audio.wav: 48 kHz mono channel 0 audio.
- metadata.json: Retained for backward compatibility with older tools.
"""

from __future__ import annotations

import json
import logging
import queue
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import scipy.io.wavfile as wavfile

from ..core.audio_recorder import AudioChunk
from ..core.hand_tracker import HandTrackingResult
from ..core.kinect_stream import FramePacket

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-]+")

IDENTITY_4X4_COLUMN_MAJOR = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def sanitize_session_name(name: Optional[str]) -> str:
    """Return a filesystem-safe token for a user supplied session name."""
    if not name:
        return ""
    cleaned = _SAFE_NAME_RE.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned[:60]


@dataclass
class SessionMetadata:
    """Configurable experimental metadata for pre-trial capture."""
    mounting_height_cm: int = 60
    lighting_type: str = "Simulated Harsh Sun"
    measured_lux: Optional[int] = 8500
    surface_state: str = "Dry Loose Sand"
    trial_number: int = 1
    depth_intrinsics: Dict[str, float] = field(
        default_factory=lambda: {"fx": 505.2, "fy": 505.4, "cx": 320.1, "cy": 288.4}
    )
    arena_corners_px: List[List[int]] = field(
        default_factory=lambda: [[120, 80], [540, 85], [535, 490], [115, 485]]
    )
    arena_dimensions_mm: List[float] = field(default_factory=lambda: [650.0, 460.0])
    sensor_model: str = "Microsoft Azure Kinect DK"
    depth_mode: str = "NFOV_UNBINNED_640x576"
    rgb_resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    fps: int = 30
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    session_name: str = ""
    audio_expected: bool = True


def generate_session_id(metadata: SessionMetadata, timestamp: Optional[datetime] = None) -> str:
    """Generate standardized session ID matching FAIR naming convention.

    Without a user name: ``session_<YYYYmmdd_HHMMSS>_H60_Sun_Dry_T1``.
    With a user name:    ``<name>_<YYYYmmdd_HHMMSS>_H60_Sun_Dry_T1``.

    The timestamp is local wall-clock time so folder names match what the
    operator sees on the recording PC; ``manifest.json`` keeps UTC.
    """
    if timestamp is None:
        timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

    lighting_map = {
        "Simulated Harsh Sun": "Sun",
        "Simulated Dappled Shade": "Shade",
        "Simulated Overcast": "Overcast",
    }
    light_token = lighting_map.get(metadata.lighting_type, "Sun")

    surface_map = {
        "Dry Loose Sand": "Dry",
        "Wet Compacted Sand": "Wet",
    }
    surf_token = surface_map.get(metadata.surface_state, "Dry")

    prefix = sanitize_session_name(metadata.session_name) or "session"
    return f"{prefix}_{ts_str}_H{metadata.mounting_height_cm}_{light_token}_{surf_token}_T{metadata.trial_number}"


class SessionWriterWorker(threading.Thread):
    """Consumer worker continuously writing v2_streams dataset: depth.f32, rgb.mp4, audio, metadata."""

    def __init__(
        self,
        output_root: Path,
        metadata: SessionMetadata,
        frame_queue: queue.Queue[FramePacket],
        audio_queue: queue.Queue[AudioChunk],
        rgb_queue: Optional[queue.Queue[FramePacket]] = None,
        hand_queue: Optional[queue.Queue[HandTrackingResult]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.output_root = Path(output_root)
        self.metadata = metadata
        self.frame_queue = frame_queue
        self.audio_queue = audio_queue
        self.rgb_queue = rgb_queue
        self.hand_queue = hand_queue

        self.start_utc = datetime.now(timezone.utc)
        self.start_local = datetime.now()
        self.session_id = generate_session_id(self.metadata, self.start_local)
        self.session_dir = self.output_root / self.session_id
        # Never silently overwrite an existing session folder.
        suffix = 1
        while self.session_dir.exists():
            suffix += 1
            self.session_dir = self.output_root / f"{self.session_id}_{suffix}"
        if suffix > 1:
            self.session_id = self.session_dir.name

        self.streams_dir = self.session_dir / "streams"

        self.running = threading.Event()
        self.paused = threading.Event()
        self.discarded = False
        self.ready = threading.Event()  # set once files are open (or setup failed)

        self.frames_written: int = 0
        self.frames_received: int = 0
        self.depth_bytes_written: int = 0
        self.audio_samples_written: int = 0
        self.audio_sample_rate_actual: int = int(metadata.audio_sample_rate)
        self.start_monotonic: float = 0.0
        self.end_monotonic: float = 0.0
        self.stop_requested_monotonic: float = 0.0
        self.paused_duration: float = 0.0
        self._pause_start_time: float = 0.0
        self._first_frame_monotonic: Optional[float] = None
        self._last_frame_monotonic: Optional[float] = None
        self.first_depth_timestamp_usec: Optional[int] = None
        self.last_depth_timestamp_usec: Optional[int] = None

        # Output handles
        self.depth_f32_fh = None
        self.v2_frames_fh = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_process: Optional[subprocess.Popen[bytes]] = None
        self.video_frames_written: int = 0
        self._last_video_frame: Optional[np.ndarray] = None
        self._rgb_thread: Optional[threading.Thread] = None
        self._first_video_monotonic: Optional[float] = None
        self._gesture_events_fh = None
        self.audio_buffer: List[np.ndarray] = []
        self.hand_trajectories: List[Dict[str, Any]] = []
        self.segment_marks: List[Dict[str, Any]] = []
        self.gesture_events: List[Dict[str, Any]] = []
        self.finalization_error: Optional[str] = None
        self.setup_error: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------
    def elapsed_seconds(self, now: Optional[float] = None) -> float:
        """Recording time excluding pauses, measured from writer start."""
        if self.start_monotonic == 0.0:
            return 0.0
        if now is None:
            now = time.monotonic()
        paused = self.paused_duration
        if self.paused.is_set() and self._pause_start_time > 0.0:
            paused += now - self._pause_start_time
        return max(0.0, (now - self.start_monotonic) - paused)

    # ------------------------------------------------------------------
    # Gesture events
    # ------------------------------------------------------------------
    def _append_gesture_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.gesture_events.append(event)
            fh = self._gesture_events_fh
            if fh is not None:
                try:
                    fh.write(json.dumps(event) + "\n")
                    fh.flush()
                except Exception as e:
                    logger.warning("Failed to write gesture event: %s", e)

    def record_gesture_event(self, event_type: str, **kwargs: Any) -> None:
        """Record an arbitrary gesture state machine trigger."""
        now = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        record = {
            "type": event_type,
            "timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "timestampUnix": round(now_utc.timestamp(), 3),
            "elapsed_seconds": round(self.elapsed_seconds(now), 3),
            "monotonicElapsedSeconds": round(self.elapsed_seconds(now), 3),
            "frame_index": self.frames_written,
            **kwargs,
        }
        self._append_gesture_event(record)

    def mark_segment(self, source: str = "pinch") -> Dict[str, Any]:
        """Record a bookmark/segment marker (pinch gesture or manual)."""
        now = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        elapsed_sec = self.elapsed_seconds(now)
        mark = {
            "segment_id": len(self.segment_marks) + 1,
            "timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "timestampUnix": round(now_utc.timestamp(), 3),
            "elapsed_seconds": round(elapsed_sec, 3),
            "monotonicElapsedSeconds": round(elapsed_sec, 3),
            "frame_index": self.frames_written,
            "source": source,
        }
        with self._lock:
            self.segment_marks.append(mark)
        self._append_gesture_event({"type": "mark_segment", **mark})
        logger.info(
            "Session %s segment %d marked at %.2fs (frame %d)",
            self.session_id, mark["segment_id"], elapsed_sec, self.frames_written,
        )
        return mark

    # ------------------------------------------------------------------
    # File setup
    # ------------------------------------------------------------------
    def _setup_session_files(self) -> None:
        """Create session directory and initialize v2_streams file handles."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.streams_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing session storage directory at: %s", self.session_dir)

        # 1. Uncompressed Float32 depth binary stream (<f4, meters)
        depth_path = self.streams_dir / "depth.f32"
        self.depth_f32_fh = open(depth_path, "wb")

        # 2. v2_frames.jsonl line-by-line frame metadata
        v2_path = self.session_dir / "v2_frames.jsonl"
        self.v2_frames_fh = open(v2_path, "w", encoding="utf-8")

        # 3. RGB video. On Windows use Intel Quick Sync so 1080p encoding
        # does not block depth/audio consumption and collapse capture time.
        video_path = self.streams_dir / "rgb.mp4"
        size = (int(self.metadata.rgb_resolution[0]), int(self.metadata.rgb_resolution[1]))
        ffmpeg = shutil.which("ffmpeg")
        if os.name == "nt" and ffmpeg:
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{size[0]}x{size[1]}",
                "-framerate", str(self.metadata.fps), "-i", "-",
                "-an", "-c:v", "h264_qsv", "-preset", "veryfast",
                "-global_quality", "20", "-movflags", "+faststart",
                str(video_path),
            ]
            self.video_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            logger.info("RGB video encoding uses Intel Quick Sync (h264_qsv) via ffmpeg.")
        else:
            for codec in ("mp4v", "avc1", "XVID"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.video_writer = cv2.VideoWriter(str(video_path), fourcc, float(self.metadata.fps), size)
                if self.video_writer.isOpened():
                    break
                logger.warning("VideoWriter failed to open with %s; trying next codec.", codec)
            if self.video_writer is None or not self.video_writer.isOpened():
                raise RuntimeError("Unable to open an RGB video encoder (mp4v/avc1/XVID).")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Pause active recording."""
        if not self.paused.is_set():
            self.paused.set()
            self._pause_start_time = time.monotonic()
            logger.info("SessionWriter paused.")

    def resume(self) -> None:
        """Resume paused recording."""
        if self.paused.is_set():
            self.paused.clear()
            self.paused_duration += time.monotonic() - self._pause_start_time
            self._pause_start_time = 0.0
            logger.info("SessionWriter resumed.")

    def stop_and_save(self) -> None:
        """Signal recording completion and remember when capture actually stopped."""
        if self.stop_requested_monotonic == 0.0:
            self.stop_requested_monotonic = time.monotonic()
        self.running.clear()

    def discard(self) -> None:
        """Discard recorded session and clean up directory."""
        self.discarded = True
        if self.stop_requested_monotonic == 0.0:
            self.stop_requested_monotonic = time.monotonic()
        self.running.clear()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _encode_rgb_image(self, image: np.ndarray) -> None:
        """Send one image to the active constant-frame-rate encoder."""
        if self.video_process is not None:
            if self.video_process.poll() is not None or self.video_process.stdin is None:
                detail = ""
                if self.video_process.stderr is not None:
                    detail = self.video_process.stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Hardware RGB encoder stopped unexpectedly. {detail}".strip())
            self.video_process.stdin.write(image.tobytes())
        elif self.video_writer is not None:
            self.video_writer.write(image)
        self.video_frames_written += 1

    def _write_video_frame(self, packet: FramePacket) -> None:
        """Encode one real Kinect RGB frame. Never synthesize repeated frames."""
        self._encode_rgb_image(packet.rgb_image)

    def _run_rgb_writer(self) -> None:
        """Consume RGB independently so depth statistics cannot stall video."""
        while self.running.is_set() or (self.rgb_queue is not None and not self.rgb_queue.empty()):
            try:
                packet = self.rgb_queue.get(timeout=0.05)  # type: ignore[union-attr]
            except queue.Empty:
                continue
            if self.paused.is_set():
                continue
            if self._first_video_monotonic is None:
                self._first_video_monotonic = packet.system_timestamp_monotonic
            try:
                self._write_video_frame(packet)
            except Exception as exc:
                self.finalization_error = f"RGB frame write failed: {exc}"
                logger.error(self.finalization_error, exc_info=True)
                self.running.clear()
                break

    def run(self) -> None:
        """Main consumer loop writing frame and audio packets."""
        self.running.set()
        self.start_monotonic = time.monotonic()
        try:
            self._setup_session_files()
        except Exception as exc:
            self.setup_error = f"Could not create session files: {exc}"
            self.finalization_error = self.setup_error
            logger.error(self.setup_error, exc_info=True)
            self.running.clear()
            self.ready.set()
            self._close_handles()
            return
        if self.rgb_queue is not None:
            self._rgb_thread = threading.Thread(
                target=self._run_rgb_writer, name="KinectRGBWriter", daemon=True
            )
            self._rgb_thread.start()
        self.ready.set()

        while self.running.is_set() or not self.frame_queue.empty() or not self.audio_queue.empty():
            # 1. Drain audio queue
            while not self.audio_queue.empty():
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    if not self.paused.is_set():
                        self.audio_buffer.append(audio_chunk.samples)
                        self.audio_samples_written += len(audio_chunk.samples)
                        self.audio_sample_rate_actual = int(audio_chunk.sample_rate)
                except queue.Empty:
                    break

            # 2. Drain hand tracking queue
            if self.hand_queue is not None:
                while not self.hand_queue.empty():
                    try:
                        hand_res = self.hand_queue.get_nowait()
                        if not self.paused.is_set() and hand_res.landmarks_3d_mm:
                            self.hand_trajectories.append({
                                "frame_index": hand_res.frame_index,
                                "device_timestamp_usec": hand_res.device_timestamp_usec,
                                "hand_detected": hand_res.hand_detected,
                                "handedness": hand_res.handedness,
                                "is_pinch": hand_res.is_pinch,
                                "landmarks_3d_mm": hand_res.landmarks_3d_mm,
                            })
                    except queue.Empty:
                        break

            # 3. Read video frame packet
            try:
                packet = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self.frames_received += 1
            if self.paused.is_set():
                continue

            try:
                if self._first_frame_monotonic is None:
                    self._first_frame_monotonic = packet.system_timestamp_monotonic
                self._last_frame_monotonic = packet.system_timestamp_monotonic

                if self.first_depth_timestamp_usec is None:
                    self.first_depth_timestamp_usec = packet.depth_timestamp_usec
                self.last_depth_timestamp_usec = packet.depth_timestamp_usec

                elapsed_sec = self.elapsed_seconds(packet.system_timestamp_monotonic)
                rgb_elapsed_sec = elapsed_sec

                # 1. Native Depth processing - UNCOMPRESSED Float32 meters
                depth_mm = packet.depth_image
                h, w = depth_mm.shape
                valid = depth_mm > 0
                depth_m = depth_mm.astype(np.float32) * 0.001
                depth_m[~valid] = np.nan
                depth_bytes = np.asarray(depth_m, dtype="<f4").tobytes(order="C")

                if self.depth_f32_fh is not None:
                    self.depth_f32_fh.write(depth_bytes)

                valid_values = depth_m[valid]
                valid_ratio = float(valid.mean())
                d_min = float(np.min(valid_values)) if valid_values.size else None
                d_med = float(np.median(valid_values)) if valid_values.size else None
                d_max = float(np.max(valid_values)) if valid_values.size else None

                # Intrinsics
                fx = float(self.metadata.depth_intrinsics.get("fx", 505.2))
                fy = float(self.metadata.depth_intrinsics.get("fy", 505.4))
                cx = float(self.metadata.depth_intrinsics.get("cx", 320.1))
                cy = float(self.metadata.depth_intrinsics.get("cy", 288.4))
                intrinsics_cm = [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]

                # 2. Write v2_frames.jsonl record
                if self.v2_frames_fh is not None:
                    ar_ts = (
                        packet.depth_timestamp_usec / 1_000_000.0
                        if packet.depth_timestamp_usec
                        else elapsed_sec
                    )
                    frame_rec = {
                        "frameIndex": self.frames_written + 1,
                        "monotonicElapsedSeconds": round(elapsed_sec, 6),
                        "arTimestamp": round(ar_ts, 6),
                        "width": w,
                        "height": h,
                        "selectedDepthSource": "azureKinectDepth",
                        "sourceDepthPixelFormat": "K4A_IMAGE_FORMAT_DEPTH16_MILLIMETERS",
                        "storedDepthFormat": "float32_little_endian_meters_contiguous",
                        "depthBytesPerRow": w * 4,
                        "depthByteOffset": self.depth_bytes_written,
                        "depthByteCount": len(depth_bytes),
                        "depthValidPixelRatio": round(valid_ratio, 4),
                        "depthMinMeters": round(d_min, 4) if d_min is not None else None,
                        "depthMedianMeters": round(d_med, 4) if d_med is not None else None,
                        "depthMaxMeters": round(d_max, 4) if d_max is not None else None,
                        "smoothedDepthByteOffset": None,
                        "smoothedDepthByteCount": None,
                        "confidenceByteOffset": None,
                        "confidenceByteCount": None,
                        "confidenceAvailable": False,
                        "rgbPresentationTimeSeconds": round(rgb_elapsed_sec, 6),
                        "cameraTransformColumnMajor": IDENTITY_4X4_COLUMN_MAJOR,
                        "cameraIntrinsicsColumnMajor": intrinsics_cm,
                        "cameraIntrinsicsReference": "azure_kinect_calibration_depth_camera",
                        "rgbCameraIntrinsicsColumnMajor": intrinsics_cm,
                        "cameraImageResolutionWidth": int(self.metadata.rgb_resolution[0]),
                        "cameraImageResolutionHeight": int(self.metadata.rgb_resolution[1]),
                        "enabledFrameSemantics": ["depth", "color_aligned_to_depth"],
                        "captureProfile": "azure_kinect_dk",
                    }
                    self.v2_frames_fh.write(json.dumps(frame_rec, separators=(",", ":"), ensure_ascii=False) + "\n")

                self.depth_bytes_written += len(depth_bytes)

                if self.rgb_queue is None:
                    self._write_video_frame(packet)

                self.frames_written += 1
            except Exception as exc:
                self.finalization_error = f"Frame write failed: {exc}"
                logger.error(self.finalization_error, exc_info=True)
                self.running.clear()
                break

        self.end_monotonic = time.monotonic()
        if self._rgb_thread is not None:
            self._rgb_thread.join(timeout=60.0)
            if self._rgb_thread.is_alive():
                self.finalization_error = "RGB writer did not finish draining its queue."
        self._finalize_session()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _close_handles(self) -> None:
        if self.depth_f32_fh is not None:
            try:
                self.depth_f32_fh.flush()
                self.depth_f32_fh.close()
            except Exception as e:
                logger.error("Error closing depth.f32 file: %s", e)
            self.depth_f32_fh = None

        if self.v2_frames_fh is not None:
            try:
                self.v2_frames_fh.flush()
                self.v2_frames_fh.close()
            except Exception as e:
                logger.error("Error closing v2_frames.jsonl file: %s", e)
            self.v2_frames_fh = None

        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception as e:
                logger.error("Error releasing VideoWriter: %s", e)
            self.video_writer = None

        if self.video_process is not None:
            try:
                if self.video_process.stdin is not None:
                    self.video_process.stdin.close()
                return_code = self.video_process.wait(timeout=30)
                if return_code != 0:
                    detail = ""
                    if self.video_process.stderr is not None:
                        detail = self.video_process.stderr.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg exited with code {return_code}: {detail}")
            except Exception as e:
                logger.error("Error closing hardware RGB encoder: %s", e)
                if self.finalization_error is None:
                    self.finalization_error = f"RGB video finalization failed: {e}"
            self.video_process = None

        with self._lock:
            if self._gesture_events_fh is not None:
                try:
                    self._gesture_events_fh.close()
                except Exception:
                    pass
                self._gesture_events_fh = None

    def _finalize_session(self) -> None:
        """Close open files and write manifest.json / ro-crate-metadata.json / audio.wav."""
        logger.info("Finalizing session %s (Written: %d frames)", self.session_id, self.frames_written)
        self._close_handles()

        if self.discarded:
            logger.info("Session %s discarded. Deleting directory.", self.session_id)
            try:
                shutil.rmtree(self.session_dir, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to remove discarded session dir: %s", e)
            return

        # 1. Write audio.wav
        wav_path = self.session_dir / "audio.wav"
        audio_written = False
        audio_duration_sec = 0.0
        if self.audio_buffer:
            all_audio = np.concatenate(self.audio_buffer, axis=0)
            if all_audio.ndim == 2 and all_audio.shape[1] == 1:
                all_audio = all_audio.reshape(-1)
            wavfile.write(str(wav_path), self.audio_sample_rate_actual, all_audio)
            audio_written = True
            audio_duration_sec = len(all_audio) / float(self.audio_sample_rate_actual)
        elif self.metadata.audio_expected:
            self.finalization_error = "No audio samples were captured; audio.wav was not created."
            logger.error(self.finalization_error)
        else:
            logger.warning("Session %s recorded without audio (explicitly allowed).", self.session_id)

        # 2. Write sidecars: objects.jsonl, boundary_samples.jsonl
        (self.session_dir / "objects.jsonl").touch(exist_ok=True)
        (self.session_dir / "boundary_samples.jsonl").touch(exist_ok=True)

        # 4. Timing calculations
        capture_end = self.stop_requested_monotonic or self.end_monotonic
        total_duration = max(0.0, (capture_end - self.start_monotonic) - self.paused_duration)
        if self._first_frame_monotonic is not None and self._last_frame_monotonic is not None:
            frame_span = max(0.0, self._last_frame_monotonic - self._first_frame_monotonic)
        else:
            frame_span = 0.0
        effective_fps = (self.frames_written / frame_span) if frame_span > 0.5 else 0.0
        video_duration = (
            self.video_frames_written / float(self.metadata.fps) if self.metadata.fps else 0.0
        )

        if self.frames_written == 0 and self.finalization_error is None:
            self.finalization_error = "No RGB-D frames were written; the camera stream never delivered frames."
            logger.error(self.finalization_error)

        # 5. Build manifest.json (v2_streams format matching iPhone session schema)
        manifest_segments = [
            {
                "segmentId": s["segment_id"],
                "timestampUnix": s.get("timestampUnix", s.get("elapsed_seconds", 0.0)),
                "monotonicElapsedSeconds": s["monotonicElapsedSeconds"],
                "source": s.get("source", "pinch"),
            }
            for s in self.segment_marks
        ]

        first_frame_ts = (
            self.first_depth_timestamp_usec / 1_000_000.0
            if self.first_depth_timestamp_usec is not None
            else 0.0
        )

        manifest_dict = {
            "schemaVersion": 1,
            "sessionId": self.session_id,
            "appVersion": "1.0.0",
            "deviceModel": self.metadata.sensor_model,
            "captureDeviceFamily": "azure_kinect_dk",
            "captureFormat": "v2_streams",
            "wallClockStartUnix": self.start_utc.timestamp(),
            "wallClockEndUnix": datetime.now(timezone.utc).timestamp(),
            "monotonicSessionStart": {
                "description": "ContinuousClock session origin"
            },
            "monotonicSessionEndElapsedSeconds": round(total_duration, 3),
            "arKitTimeReference": {
                "firstFrameTimestamp": round(first_frame_ts, 3),
                "note": "ar_timestamp matches depth timestamp seconds"
            },
            "storage": {
                "destination": "windows_filesystem",
                "sessionFolderRelative": self.session_dir.name,
                "securityScopedExternal": False
            },
            "v2Streams": {
                "formatVersion": 2,
                "frameMetadataFile": "v2_frames.jsonl",
                "rgbVideoPath": "streams/rgb.mp4",
                "rgbVideoCodec": "h264_mp4",
                "depthStreamPath": "streams/depth.f32",
                "depthFormat": "float32_little_endian_meters_contiguous",
                "smoothedDepthStreamPath": None,
                "confidenceStreamPath": None,
                "note": "Depth is native uncompressed Azure Kinect metric depth stored as contiguous Float32 little-endian meters."
            },
            "depthEncoding": {
                "format": "v2_streams_float32",
                "maxDepthMeters": 5.46,
                "deltaMeanAbsThresholdU16": 0,
                "note": "streams/depth.f32 stores raw contiguous Float32 little-endian meters; zero/unmeasured samples are stored as NaN"
            },
            "audio": {
                "channelCount": 1,
                "sampleRate": self.audio_sample_rate_actual,
                "format": "wav",
                "relativePath": "audio.wav",
                "durationSeconds": round(audio_duration_sec, 3) if audio_written else 0.0,
                "startMonotonicElapsedSeconds": 0.0
            } if audio_written else None,
            "segments": manifest_segments,
            "trayBoundary": {
                "status": "pending",
                "arenaCornersPx": self.metadata.arena_corners_px,
                "arenaDimensionsMm": self.metadata.arena_dimensions_mm
            },
            "objectDetectionsFile": "objects.jsonl",
            "objectDetectionProvenance": "postprocessed_yolo_world_v1",
            "boundarySamplesFile": "boundary_samples.jsonl",
            "strokesFile": "strokes.json",
            "experimentalConditions": {
                "mountingHeightCm": self.metadata.mounting_height_cm,
                "lightingType": self.metadata.lighting_type,
                "measuredLux": self.metadata.measured_lux,
                "surfaceState": self.metadata.surface_state,
                "trialNumber": self.metadata.trial_number,
                "sessionName": self.metadata.session_name
            },
            "depthFrames": [],
            "rgbFrames": []
        }

        manifest_path = self.session_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)

        # 6. Build ro-crate-metadata.json
        has_part = [
            {"@id": "manifest.json"},
            {"@id": "v2_frames.jsonl"},
            {"@id": "boundary_samples.jsonl"},
            {"@id": "objects.jsonl"},
            {"@id": "streams/depth.f32"},
            {"@id": "streams/rgb.mp4"},
        ]
        if audio_written:
            has_part.append({"@id": "audio.wav"})

        ro_crate = {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {
                    "@id": "ro-crate-metadata.json",
                    "@type": "CreativeWork",
                    "about": {"@id": "./"},
                    "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"}
                },
                {
                    "@id": "./",
                    "@type": "Dataset",
                    "name": f"SandStories Azure Kinect Session - {self.session_id}",
                    "description": "Standardized multimodal storytelling capture dataset package.",
                    "creator": {
                        "@type": "Person",
                        "name": "SandStories Operator"
                    },
                    "datePublished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "license": "https://creativecommons.org/licenses/by/4.0/",
                    "modalities": ["depth", "rgb", "audio", "gestures", "boundary"],
                    "hasPart": has_part
                }
            ]
        }
        for item in has_part:
            ro_crate["@graph"].append({
                "@id": item["@id"],
                "@type": "File",
                "name": item["@id"].split("/")[-1],
            })

        ro_crate_path = self.session_dir / "ro-crate-metadata.json"
        with open(ro_crate_path, "w", encoding="utf-8") as f:
            json.dump(ro_crate, f, indent=2)

        logger.info(
            "Session %s saved in clean v2_streams format: %d frames, %.1fs, to %s",
            self.session_id, self.frames_written, total_duration, self.session_dir,
        )
