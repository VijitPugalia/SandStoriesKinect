"""Azure Kinect DK sensor capture worker and mock stream generator."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_CAPTURE_ERRORS = 5
MAX_CONSECUTIVE_EMPTY_CAPTURES = 30
MAX_CONSECUTIVE_PREVIEW_ERRORS = 30

# Importing the SDK is optional for explicit demo mode, but production capture
# must never silently substitute synthetic data.
try:
    import pyk4a
    from pyk4a import (
        CalibrationType,
        ColorControlCommand,
        ColorControlMode,
        ColorResolution,
        Config,
        DepthMode,
        FPS,
        ImageFormat,
        PyK4A,
        PyK4ACapture,
        PyK4ARecord,
    )
    PYK4A_AVAILABLE = True
except (ImportError, OSError, Exception) as exc:
    logger.warning("pyk4a library or k4a.dll unavailable (%s).", exc)
    PYK4A_AVAILABLE = False
    CalibrationType = None  # type: ignore
    PyK4ARecord = None  # type: ignore


@dataclass
class FramePacket:
    """Thread-safe packet carrying synchronized multi-modal frame data."""
    frame_index: int
    rgb_image: np.ndarray  # [1080, 1920, 3] BGR uint8
    depth_image: np.ndarray  # [576, 640] uint16 (raw depth in mm)
    transformed_depth: Optional[np.ndarray] = None  # [1080, 1920] uint16 (aligned to color)
    color_timestamp_usec: int = 0
    depth_timestamp_usec: int = 0
    system_timestamp_monotonic: float = field(default_factory=time.monotonic)


def decode_preview_color(color_data: Any) -> Optional[np.ndarray]:
    """Decode a colour image for the viewfinder without changing raw capture data."""
    import cv2

    color = np.asarray(color_data)
    if color.ndim == 1:
        return cv2.imdecode(color.reshape(-1), cv2.IMREAD_COLOR)
    if color.ndim == 3 and color.shape[2] == 4:
        return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    return color.copy()


class BaseKinectStream(threading.Thread):
    """Abstract base class for Kinect DK capture streams."""

    def __init__(self, queue_maxsize: int = 30) -> None:
        super().__init__(daemon=True)
        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_maxsize)
        self.subscribers: List[queue.Queue[FramePacket]] = []
        self._subscriber_lock = threading.Lock()
        self.running = threading.Event()
        self.frame_count: int = 0
        self.dropped_frames: int = 0
        self.fps_measured: float = 0.0
        self._fps_window: List[float] = []
        self.last_frame_monotonic: float = 0.0
        self.last_error: Optional[str] = None
        self.device_started: bool = False

    def subscribe(self, maxsize: int = 15) -> queue.Queue[FramePacket]:
        """Subscribe a dedicated queue for a downstream consumer thread."""
        q: queue.Queue[FramePacket] = queue.Queue(maxsize=maxsize)
        q.dropped_frames = 0  # type: ignore[attr-defined]
        with self._subscriber_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[FramePacket]) -> None:
        """Remove a subscriber queue."""
        with self._subscriber_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _broadcast(self, packet: FramePacket) -> None:
        """Broadcast frame packet to primary queue and all subscribers."""
        # Update FPS measurement
        now = time.monotonic()
        self.last_frame_monotonic = now
        self._fps_window.append(now)
        while self._fps_window and (now - self._fps_window[0]) > 1.0:
            self._fps_window.pop(0)
        self.fps_measured = float(len(self._fps_window))

        # Push to primary queue (drop oldest if full)
        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(packet)
                self.dropped_frames += 1
            except (queue.Empty, queue.Full):
                pass

        # Push to subscriber queues
        with self._subscriber_lock:
            for sub_q in self.subscribers:
                try:
                    sub_q.put_nowait(packet)
                except queue.Full:
                    try:
                        _ = sub_q.get_nowait()
                        sub_q.put_nowait(packet)
                        sub_q.dropped_frames += 1  # type: ignore[attr-defined]
                    except (queue.Empty, queue.Full):
                        pass

    def has_fresh_frame(self, max_age_seconds: float = 3.0) -> bool:
        """Return whether a frame was broadcast recently enough for safe recording."""
        return (
            self.last_frame_monotonic > 0.0
            and time.monotonic() - self.last_frame_monotonic <= max_age_seconds
        )

    def get_intrinsics(self) -> Dict[str, Any]:
        """Return camera intrinsics dictionary."""
        raise NotImplementedError

    def convert_2d_to_3d(
        self,
        coordinates: Tuple[float, float],
        depth_mm: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Project pixel coordinate (u, v) + depth in mm to camera 3D (X, Y, Z) in mm."""
        raise NotImplementedError

    def stop(self) -> None:
        """Signal thread to stop."""
        self.running.clear()


class KinectStreamWorker(BaseKinectStream):
    """Hardware capture worker interfacing with Microsoft Azure Kinect DK via pyk4a."""

    def __init__(
        self,
        device_id: int = 0,
        color_resolution: int = ColorResolution.RES_1080P if PYK4A_AVAILABLE else 0,
        depth_mode: int = DepthMode.NFOV_UNBINNED if PYK4A_AVAILABLE else 0,
        camera_fps: int = FPS.FPS_30 if PYK4A_AVAILABLE else 0,
        lock_exposure: bool = True,
        queue_maxsize: int = 30,
    ) -> None:
        if not PYK4A_AVAILABLE:
            raise RuntimeError("pyk4a is not available on this system.")
        super().__init__(queue_maxsize=queue_maxsize)
        self.device_id = device_id
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.camera_fps = camera_fps
        self.lock_exposure = lock_exposure
        self.k4a: Optional[PyK4A] = None
        self._calibration = None
        self._config = None
        self._mkv_record = None
        self._mkv_start_path = None
        self._mkv_stop_requested = False
        self._mkv_command_lock = threading.Lock()
        self.mkv_ready = threading.Event()
        self.mkv_finalized = threading.Event()
        self.mkv_error: Optional[str] = None
        self.mkv_frames_written = 0
        self.preview_decode_errors = 0

    def _initialize_device(self) -> None:
        """Initialize PyK4A device with 1080p RGB and NFOV Unbinned depth."""
        config = Config(
            color_format=ImageFormat.COLOR_MJPG,
            color_resolution=self.color_resolution,
            depth_mode=self.depth_mode,
            camera_fps=self.camera_fps,
            synchronized_images_only=True,
        )
        self._config = config
        self.k4a = PyK4A(config=config, device_id=self.device_id)
        self.k4a.start()
        logger.info("Azure Kinect DK opened on device %d", self.device_id)

        # Retrieve calibration
        self._calibration = self.k4a.calibration

        # Sensor warmup & manual parameter locking
        if self.lock_exposure:
            logger.info("Sensor warming up for parameter locking...")
            for _ in range(30):
                try:
                    _ = self.k4a.get_capture(timeout=1000)
                except Exception:
                    pass
            # Lock manual parameters to prevent auto-exposure / whitebalance hunting
            try:
                # Read current exposure or set robust fixed values
                self.k4a._set_color_control(
                    ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
                    33330,  # ~1/30s in microseconds
                    mode=ColorControlMode.MANUAL,
                )
                self.k4a._set_color_control(
                    ColorControlCommand.WHITEBALANCE,
                    4500,  # 4500K neutral
                    mode=ColorControlMode.MANUAL,
                )
                logger.info("Azure Kinect manual exposure and white balance locked.")
            except Exception as e:
                logger.warning("Could not set manual color controls: %s", e)

    def start_mkv_recording(self, path: str, timeout: float = 5.0) -> bool:
        """Request direct MKV recording on the sensor capture thread."""
        self.mkv_ready.clear()
        self.mkv_finalized.clear()
        self.mkv_error = None
        self.mkv_frames_written = 0
        with self._mkv_command_lock:
            self._mkv_start_path = path
            self._mkv_stop_requested = False
        if not self.mkv_ready.wait(timeout):
            self.mkv_error = "Timed out starting Azure Kinect MKV recorder."
            return False
        return self.mkv_error is None

    def stop_mkv_recording(self, timeout: float = 15.0) -> bool:
        """Flush and close active MKV on the sensor capture thread."""
        with self._mkv_command_lock:
            self._mkv_stop_requested = True
        if not self.mkv_finalized.wait(timeout):
            self.mkv_error = "Timed out finalizing Azure Kinect MKV."
            return False
        return self.mkv_error is None

    def _service_mkv_commands(self) -> None:
        with self._mkv_command_lock:
            start_path = self._mkv_start_path
            self._mkv_start_path = None
            stop_requested = self._mkv_stop_requested
            if stop_requested:
                self._mkv_stop_requested = False
        if start_path is not None:
            try:
                if self.k4a is None or self._config is None:
                    raise RuntimeError("Kinect device is not ready.")
                record = PyK4ARecord(
                    device=self.k4a, config=self._config, path=start_path
                )
                record.create()
                self._mkv_record = record
            except Exception as exc:
                self.mkv_error = f"Could not start MKV recording: {exc}"
            finally:
                self.mkv_ready.set()
        if stop_requested:
            self._close_mkv_recording()
            self.mkv_finalized.set()

    def _close_mkv_recording(self) -> None:
        record = self._mkv_record
        self._mkv_record = None
        if record is None:
            return
        try:
            if record.created:
                try:
                    record.flush()
                finally:
                    record.close()
        except Exception as exc:
            self.mkv_error = f"Could not finalize MKV recording: {exc}"
            logger.error(self.mkv_error, exc_info=True)

    def run(self) -> None:
        """Producer loop: capture synchronized frames at 30 FPS."""
        self.running.set()
        empty_sync_streak = 0
        capture_error_streak = 0
        preview_error_streak = 0
        try:
            self._initialize_device()
            self.device_started = True
            logger.info("Kinect capture loop running; waiting for colour+depth frames.")
            while self.running.is_set():
                self._service_mkv_commands()
                if self.k4a is None:
                    break
                try:
                    capture: PyK4ACapture = self.k4a.get_capture(timeout=1000)
                except Exception as e:
                    if not self.running.is_set():
                        break
                    capture_error_streak += 1
                    self.last_error = f"Kinect capture error: {e}"
                    logger.warning("Kinect capture timeout or error: %s", e)
                    if capture_error_streak >= MAX_CONSECUTIVE_CAPTURE_ERRORS:
                        raise RuntimeError(
                            f"Kinect capture failed {capture_error_streak} consecutive times: {e}"
                        ) from e
                    continue

                capture_error_streak = 0

                if capture.color is None or capture.depth is None:
                    empty_sync_streak += 1
                    if empty_sync_streak >= MAX_CONSECUTIVE_EMPTY_CAPTURES:
                        self.last_error = (
                            "Device open, but colour/depth frames are missing. "
                            "Close k4aviewer and check USB 3 / power."
                        )
                        logger.error(self.last_error)
                        raise RuntimeError(self.last_error)
                    continue

                empty_sync_streak = 0
                self.last_error = None

                # Write original synchronized capture before preview conversion.
                if self._mkv_record is not None:
                    try:
                        self._mkv_record.write_capture(capture)
                        self.mkv_frames_written += 1
                    except Exception as exc:
                        self.mkv_error = f"MKV capture write failed: {exc}"
                        logger.error(self.mkv_error, exc_info=True)
                        self._close_mkv_recording()
                        self.mkv_finalized.set()

                # Decode only a preview copy. The raw MKV already contains this
                # capture, so a damaged MJPG preview must not stop recording.
                try:
                    rgb = decode_preview_color(capture.color)
                except Exception as exc:
                    rgb = None
                    preview_detail = str(exc)
                else:
                    preview_detail = "decoder returned no image"
                if rgb is None:
                    self.preview_decode_errors += 1
                    preview_error_streak += 1
                    logger.warning(
                        "Skipping Kinect MJPG preview frame %d (%s); raw MKV capture was preserved.",
                        self.preview_decode_errors,
                        preview_detail,
                    )
                    if preview_error_streak >= MAX_CONSECUTIVE_PREVIEW_ERRORS:
                        raise RuntimeError(
                            "Kinect MJPG preview failed "
                            f"{preview_error_streak} consecutive times."
                        )
                    continue
                preview_error_streak = 0

                depth = capture.depth.copy()
                transformed_depth = None

                self.frame_count += 1
                if self.frame_count == 1:
                    logger.info(
                        "First Kinect frame received: rgb=%s depth=%s",
                        getattr(rgb, "shape", None),
                        getattr(depth, "shape", None),
                    )
                packet = FramePacket(
                    frame_index=self.frame_count,
                    rgb_image=rgb,
                    depth_image=depth,
                    transformed_depth=transformed_depth,
                    color_timestamp_usec=int(capture.color_timestamp_usec),
                    depth_timestamp_usec=int(capture.depth_timestamp_usec),
                    system_timestamp_monotonic=time.monotonic(),
                )
                self._broadcast(packet)

        except Exception as e:
            self.last_error = f"Kinect stream failed: {e}"
            logger.error("Error in KinectStreamWorker: %s", e, exc_info=True)
        finally:
            self._close_mkv_recording()
            self.mkv_finalized.set()
            self.device_started = False
            self._cleanup_device()

    def _cleanup_device(self) -> None:
        """Gracefully stop and close Kinect device."""
        if self.k4a is not None:
            try:
                self.k4a.stop()
                logger.info("Azure Kinect DK stream stopped.")
            except Exception as e:
                logger.warning("Error stopping PyK4A: %s", e)
            finally:
                self.k4a = None

    def get_intrinsics(self) -> Dict[str, Any]:
        """Return camera calibration intrinsics."""
        if self._calibration is not None and CalibrationType is not None:
            try:
                mat = self._calibration.get_camera_matrix(CalibrationType.DEPTH)
                return {
                    "fx": float(mat[0, 0]),
                    "fy": float(mat[1, 1]),
                    "cx": float(mat[0, 2]),
                    "cy": float(mat[1, 2]),
                }
            except Exception:
                pass
        return {"fx": 505.2, "fy": 505.4, "cx": 320.1, "cy": 288.4}

    def convert_2d_to_3d(
        self,
        coordinates: Tuple[float, float],
        depth_mm: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Project (u, v) + depth into 3D metric camera coordinates (X, Y, Z) in mm."""
        if self._calibration is not None and CalibrationType is not None and depth_mm > 0:
            try:
                pt3d = self._calibration.convert_2d_to_3d(
                    coordinates,
                    depth_mm,
                    CalibrationType.COLOR,
                    CalibrationType.COLOR,
                )
                if pt3d is not None:
                    return (float(pt3d[0]), float(pt3d[1]), float(pt3d[2]))
            except Exception:
                pass

        # Fallback pinhole intrinsic projection
        intr = self.get_intrinsics()
        if depth_mm <= 0:
            return None
        u, v = coordinates
        x = (u - intr["cx"]) * depth_mm / intr["fx"]
        y = (v - intr["cy"]) * depth_mm / intr["fy"]
        return (float(x), float(y), float(depth_mm))


class MockKinectStreamWorker(BaseKinectStream):
    """High-fidelity synthetic RGB-D stream generator for testing and offline development."""

    def __init__(
        self,
        fps: float = 30.0,
        height_mm: float = 650.0,
        queue_maxsize: int = 30,
    ) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.height_mm = height_mm

        # Fixed calibration intrinsics for NFOV 640x576 and 1080p color
        self._intrinsics = {
            "fx": 505.2,
            "fy": 505.4,
            "cx": 320.1,
            "cy": 288.4,
        }

        # Arena fiducial corners on RGB image (650 x 460 mm arena boundary)
        self.arena_corners_px = [[240, 140], [1680, 140], [1680, 940], [240, 940]]

    def run(self) -> None:
        """Generate synthetic RGB and depth frames at steady 30 FPS."""
        self.running.set()
        start_time = time.monotonic()
        sim_time_usec = 1000000

        # Pre-create base depth topography with subtle sand ripples
        depth_h, depth_w = 576, 640
        y_grid, x_grid = np.mgrid[0:depth_h, 0:depth_w].astype(np.float32)
        base_depth = self.height_mm + 15.0 * np.sin(x_grid / 35.0) * np.cos(y_grid / 45.0)

        while self.running.is_set():
            t_loop_start = time.monotonic()
            self.frame_count += 1
            t_sec = t_loop_start - start_time
            sim_time_usec += int(self.frame_interval * 1e6)

            # Synthesize realistic depth frame (uint16 in mm)
            # Add subtle dynamic hand depth silhouette moving across tray
            hand_cx = int(320 + 120 * np.sin(t_sec * 1.2))
            hand_cy = int(288 + 80 * np.cos(t_sec * 0.9))
            dist_to_hand = np.sqrt((x_grid - hand_cx) ** 2 + (y_grid - hand_cy) ** 2)

            curr_depth = base_depth.copy()
            # Hand hovering 150mm above sand
            hand_mask = dist_to_hand < 60
            curr_depth[hand_mask] = self.height_mm - 150.0 + (dist_to_hand[hand_mask] * 0.5)

            # Sand drawing grooves (2-8 mm relief)
            groove_mask = (dist_to_hand > 70) & (dist_to_hand < 85) & (y_grid > 200)
            curr_depth[groove_mask] -= 5.0  # 5mm deep groove

            # Add minor Gaussian noise (+/- 0.8 mm)
            noise = np.random.normal(0, 0.8, (depth_h, depth_w)).astype(np.float32)
            depth_u16 = np.clip(curr_depth + noise, 0, 65535).astype(np.uint16)

            # Synthesize RGB frame (1920x1080 BGR uint8)
            rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Sand color background
            rgb[:, :] = [140, 190, 220]  # Light sand BGR

            # Draw sand tray border
            import cv2
            pts = np.array(self.arena_corners_px, np.int32).reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts], isClosed=True, color=(60, 90, 130), thickness=6)

            # Draw fiducial marker targets at the 4 corners
            for pt in self.arena_corners_px:
                cv2.circle(rgb, (pt[0], pt[1]), 12, (20, 40, 200), -1)
                cv2.circle(rgb, (pt[0], pt[1]), 16, (255, 255, 255), 2)

            # Draw simulated sand drawings (grooves)
            cv2.ellipse(rgb, (960, 540), (350, 220), 0, 0, 360, (100, 150, 180), 8)
            cv2.putText(
                rgb,
                "SANDSTORIES EXPERIMENT TRAY [650x460mm]",
                (260, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 60, 90),
                2,
                cv2.LINE_AA,
            )

            # Draw simulated hand on RGB
            rgb_hand_x = int(960 + 360 * np.sin(t_sec * 1.2))
            rgb_hand_y = int(540 + 200 * np.cos(t_sec * 0.9))
            cv2.circle(rgb, (rgb_hand_x, rgb_hand_y), 50, (180, 210, 240), -1)
            # Simulated finger tips
            for angle in np.linspace(-0.6, 0.6, 5):
                fx = int(rgb_hand_x + 90 * np.sin(angle))
                fy = int(rgb_hand_y - 90 * np.cos(angle))
                cv2.line(rgb, (rgb_hand_x, rgb_hand_y), (fx, fy), (180, 210, 240), 16)
                cv2.circle(rgb, (fx, fy), 14, (160, 190, 230), -1)

            # Add timestamp watermark
            cv2.putText(
                rgb,
                f"MOCK STREAM | Frame: {self.frame_count:06d} | FPS: {self.fps_measured:.1f}",
                (30, 1040),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

            # Transformed depth (scaled to RGB size)
            transformed_depth = cv2.resize(depth_u16, (1920, 1080), interpolation=cv2.INTER_NEAREST)

            packet = FramePacket(
                frame_index=self.frame_count,
                rgb_image=rgb,
                depth_image=depth_u16,
                transformed_depth=transformed_depth,
                color_timestamp_usec=sim_time_usec,
                depth_timestamp_usec=sim_time_usec,
                system_timestamp_monotonic=t_loop_start,
            )
            self._broadcast(packet)

            # Maintain strict 30 FPS pacing
            elapsed = time.monotonic() - t_loop_start
            sleep_time = max(0.0, self.frame_interval - elapsed)
            time.sleep(sleep_time)

    def get_intrinsics(self) -> Dict[str, Any]:
        return self._intrinsics.copy()

    def convert_2d_to_3d(
        self,
        coordinates: Tuple[float, float],
        depth_mm: float,
    ) -> Optional[Tuple[float, float, float]]:
        if depth_mm <= 0:
            return None
        u, v = coordinates
        intr = self._intrinsics
        # Normalize from RGB 1080p to depth plane if needed
        x = (u - (intr["cx"] * (1920.0 / 640.0))) * depth_mm / (intr["fx"] * (1920.0 / 640.0))
        y = (v - (intr["cy"] * (1080.0 / 576.0))) * depth_mm / (intr["fy"] * (1080.0 / 576.0))
        return (float(x), float(y), float(depth_mm))


def create_kinect_stream(
    use_mock: bool = False,
    device_id: int = 0,
    queue_maxsize: int = 30,
) -> BaseKinectStream:
    """Create the explicitly requested stream; production mode is fail-closed."""
    if use_mock:
        logger.warning("DEMO MODE: creating a synthetic Kinect stream.")
        return MockKinectStreamWorker(queue_maxsize=queue_maxsize)
    if not PYK4A_AVAILABLE:
        raise RuntimeError(
            "Azure Kinect SDK/pyk4a is unavailable. Connect and configure the "
            "sensor, or restart with --mock for explicitly synthetic demo data."
        )
    try:
        device_count = pyk4a.connected_device_count()
    except Exception as exc:
        raise RuntimeError(f"Unable to probe Azure Kinect devices: {exc}") from exc
    if device_count <= 0:
        raise RuntimeError(
            "No Azure Kinect DK is connected. Recording is blocked; use --mock "
            "only for an explicitly synthetic demo."
        )
    if device_id >= device_count:
        raise RuntimeError(
            f"Azure Kinect device {device_id} was requested, but only "
            f"{device_count} device(s) are connected."
        )
    logger.info("Found %d Azure Kinect DK device(s).", device_count)
    return KinectStreamWorker(device_id=device_id, queue_maxsize=queue_maxsize)
