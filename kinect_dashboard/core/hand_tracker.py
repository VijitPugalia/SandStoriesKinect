"""MediaPipe 3D Hand Pose Estimation and Metric 2D-to-3D Projection Worker."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Joint names mapped to MediaPipe 21 Hand Landmarks
HAND_JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

# Skeletal connections for rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (17, 0),                                # Palm base
]


@dataclass
class Joint3D:
    """Individual 3D joint landmark."""
    joint_id: int
    name: str
    x: float  # mm
    y: float  # mm
    z: float  # mm
    u_px: float = 0.0
    v_px: float = 0.0
    confidence: float = 1.0


@dataclass
class HandTrackingResult:
    """Tracking output for a single video frame."""
    frame_index: int
    device_timestamp_usec: int
    hand_detected: bool
    handedness: str = "Unknown"
    landmarks_3d_mm: List[Dict[str, Any]] = field(default_factory=list)
    landmarks_2d_px: List[Tuple[float, float]] = field(default_factory=list)
    bounding_box_px: Optional[Tuple[int, int, int, int]] = None  # (min_x, min_y, max_x, max_y)
    wrist_velocity_mm_s: float = 0.0
    is_open_palm: bool = False
    is_raised_flat_hand: bool = False
    is_closed_fist: bool = False
    is_yo_gesture: bool = False
    is_pinch: bool = False
    pinch_distance_mm: float = 0.0
    mean_luminance: float = 1.0
    std_luminance: float = 0.0


class HandTrackerWorker(threading.Thread):
    """Consumer worker running MediaPipe Hand Tracking on incoming frames."""

    def __init__(
        self,
        frame_queue: queue.Queue,
        projection_fn: Optional[Callable[[Tuple[float, float], float], Optional[Tuple[float, float, float]]]] = None,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        inference_width: int = 640,
        pinch_enter_ratio: float = 0.30,
        pinch_exit_ratio: float = 0.42,
        gesture_only: bool = True,
        target_fps: float = 10.0,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.projection_fn = projection_fn
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.inference_width = max(160, int(inference_width))
        self.pinch_enter_ratio = pinch_enter_ratio
        self.pinch_exit_ratio = pinch_exit_ratio
        self.gesture_only = gesture_only
        self.min_mp_interval = 1.0 / max(1.0, float(target_fps))
        self.last_pinch_ratio: float = 9.0
        self.processed_frames: int = 0
        self.fps_measured: float = 0.0
        self._fps_window: List[float] = []
        self._last_mp_monotonic: float = 0.0
        self._cached_hand_detected: bool = False
        self._cached_handedness: str = "Unknown"
        self._cached_landmarks_2d: List[Tuple[float, float]] = []
        self._cached_bbox: Optional[Tuple[int, int, int, int]] = None

        self.running = threading.Event()
        self.latest_result: Optional[HandTrackingResult] = None
        self._result_lock = threading.Lock()
        self.subscribers: List[queue.Queue[HandTrackingResult]] = []
        self._sub_lock = threading.Lock()

        # Wrist position history for velocity calculation: List of (t_sec, (x, y, z))
        self._wrist_history: List[Tuple[float, Tuple[float, float, float]]] = []

        # Landmark bridging and pose cache for up to 0.2s dropout grace
        self._last_valid_pose_time: float = 0.0
        self._last_yo: bool = False
        self._last_pinch: bool = False

        # Initialize MediaPipe Hands solution
        self._hands_solution = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Safely initialize MediaPipe Hands solution."""
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._hands_solution = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    model_complexity=1,
                )
                logger.info("MediaPipe Hands solution initialized successfully.")
            else:
                logger.warning("mp.solutions.hands not found; running in heuristic tracking mode.")
        except Exception as e:
            logger.warning("Failed to initialize MediaPipe Hands: %s", e)

    def subscribe(self, maxsize: int = 30) -> queue.Queue[HandTrackingResult]:
        """Subscribe a queue to receive per-frame HandTrackingResult."""
        q: queue.Queue[HandTrackingResult] = queue.Queue(maxsize=maxsize)
        with self._sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[HandTrackingResult]) -> None:
        with self._sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def get_latest_result(self) -> Optional[HandTrackingResult]:
        with self._result_lock:
            return self.latest_result

    def _sample_depth(
        self,
        depth_img: Optional[np.ndarray],
        u: float,
        v: float,
    ) -> Optional[float]:
        """Return a valid local depth sample, or ``None`` when depth is unknown."""
        if depth_img is None or depth_img.size == 0:
            return None
        h, w = depth_img.shape[:2]
        ix, iy = int(round(u)), int(round(v))
        if 0 <= ix < w and 0 <= iy < h:
            val = depth_img[iy, ix]
            if val > 100:  # valid depth > 100 mm
                return float(val)

            # Sample 3x3 neighborhood
            x_min, x_max = max(0, ix - 2), min(w, ix + 3)
            y_min, y_max = max(0, iy - 2), min(h, iy + 3)
            patch = depth_img[y_min:y_max, x_min:x_max]
            valid = patch[patch > 100]
            if len(valid) > 0:
                return float(np.median(valid))

        return None

    def _compute_wrist_velocity(self, t_sec: float, wrist_pos_3d: Tuple[float, float, float]) -> float:
        """Compute smoothed wrist velocity in mm/s."""
        self._wrist_history.append((t_sec, wrist_pos_3d))
        # Keep last 0.3s of history
        while self._wrist_history and (t_sec - self._wrist_history[0][0]) > 0.3:
            self._wrist_history.pop(0)

        if len(self._wrist_history) < 2:
            return 0.0

        t_old, pos_old = self._wrist_history[0]
        dt = t_sec - t_old
        if dt < 1e-4:
            return 0.0

        dx = wrist_pos_3d[0] - pos_old[0]
        dy = wrist_pos_3d[1] - pos_old[1]
        dz = wrist_pos_3d[2] - pos_old[2]
        dist_mm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return float(dist_mm / dt)

    def _classify_gestures(
        self,
        landmarks_2d: List[Tuple[float, float]],
        landmarks_3d: List[Joint3D],
    ) -> Tuple[bool, bool, bool, bool, bool, float]:
        """Classify Open Palm, Raised Flat Hand, Closed Fist, Yo, and Pinch."""
        if len(landmarks_2d) < 21:
            return False, False, False, False, False, 0.0

        # Landmark indices
        # Wrist = 0
        # Thumb: MCP=2, IP=3, TIP=4
        # Index: MCP=5, PIP=6, TIP=8
        # Middle: MCP=9, PIP=10, TIP=12
        # Ring: MCP=13, PIP=14, TIP=16
        # Pinky: MCP=17, PIP=18, TIP=20

        wrist_2d = np.array(landmarks_2d[0])

        def is_finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip_dist = np.linalg.norm(np.array(landmarks_2d[tip_idx]) - wrist_2d)
            pip_dist = np.linalg.norm(np.array(landmarks_2d[pip_idx]) - wrist_2d)
            mcp_dist = np.linalg.norm(np.array(landmarks_2d[mcp_idx]) - wrist_2d)
            return bool(tip_dist > pip_dist and tip_dist > mcp_dist)

        def is_finger_folded(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip_dist = np.linalg.norm(np.array(landmarks_2d[tip_idx]) - wrist_2d)
            mcp_dist = np.linalg.norm(np.array(landmarks_2d[mcp_idx]) - wrist_2d)
            return bool(tip_dist < mcp_dist * 1.15)

        thumb_extended = is_finger_extended(4, 3, 2)
        index_extended = is_finger_extended(8, 6, 5)
        middle_extended = is_finger_extended(12, 10, 9)
        ring_extended = is_finger_extended(16, 14, 13)
        pinky_extended = is_finger_extended(20, 18, 17)

        thumb_folded = is_finger_folded(4, 3, 2)
        index_folded = is_finger_folded(8, 6, 5)
        middle_folded = is_finger_folded(12, 10, 9)
        ring_folded = is_finger_folded(16, 14, 13)
        pinky_folded = is_finger_folded(20, 18, 17)

        # Pinch detection (Thumb TIP to Index TIP proximity).
        # The 2D image-space ratio is the primary signal: fingertip depth
        # samples from the ToF sensor frequently land on the sand behind the
        # finger, which made the old 3D-only test flicker. Hysteresis keeps a
        # pinch active through small openings so a 2 s hold is achievable.
        palm_span = float(np.linalg.norm(np.array(landmarks_2d[9]) - wrist_2d))
        t_tip = np.array(landmarks_2d[4])
        i_tip = np.array(landmarks_2d[8])
        dist_2d = float(np.linalg.norm(t_tip - i_tip))
        norm_dist = dist_2d / max(palm_span, 1e-4)
        self.last_pinch_ratio = norm_dist

        enter_ratio = self.pinch_enter_ratio
        exit_ratio = self.pinch_exit_ratio
        if self._last_pinch:
            is_pinch = norm_dist < exit_ratio
        else:
            is_pinch = norm_dist < enter_ratio

        # Optional metric distance for logging when both fingertips have depth.
        thumb_3d = next((j for j in landmarks_3d if j.joint_id == 4), None)
        index_3d = next((j for j in landmarks_3d if j.joint_id == 8), None)
        pinch_dist_mm = norm_dist * 100.0
        if thumb_3d is not None and index_3d is not None:
            dx = thumb_3d.x - index_3d.x
            dy = thumb_3d.y - index_3d.y
            dz = thumb_3d.z - index_3d.z
            metric = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            # Only trust metric distance when it is plausible for two fingertips.
            if metric < 250.0:
                pinch_dist_mm = metric
                if not is_pinch and metric <= 30.0:
                    is_pinch = True

        # Yo gesture: Thumb, Index, Pinky extended; Middle, Ring folded; no active pinch
        is_yo = (
            thumb_extended
            and index_extended
            and pinky_extended
            and middle_folded
            and ring_folded
            and not is_pinch
        )

        # Legacy gestures
        is_open_palm = (
            thumb_extended
            and index_extended
            and middle_extended
            and ring_extended
            and pinky_extended
        )
        is_raised_flat = (
            not thumb_extended
            and index_extended
            and middle_extended
            and ring_extended
            and pinky_extended
        )
        is_closed_fist = (
            index_folded
            and middle_folded
            and ring_folded
            and pinky_folded
        )

        return is_open_palm, is_raised_flat, is_closed_fist, is_yo, is_pinch, pinch_dist_mm

    def process_frame(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        transformed_depth: Optional[np.ndarray],
        frame_index: int,
        timestamp_usec: int,
    ) -> HandTrackingResult:
        """Run single-frame hand tracking, 3D projection, and gesture extraction."""
        h, w = rgb_image.shape[:2]
        t_sec = timestamp_usec / 1e6

        # 1. Sample lightweight 16x16 grid from RGB frame for camera cover detection
        mean_lum = 1.0
        std_lum = 0.0
        try:
            if rgb_image is not None and rgb_image.size > 0:
                # Fast subsample (stride 16) before grayscale and 16x16 resize
                # Eliminates 2,000,000 float conversions every frame with identical detection accuracy
                sub = rgb_image[::16, ::16]
                gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                grid_16x16 = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
                mean_lum = float(np.mean(grid_16x16)) / 255.0
                std_lum = float(np.std(grid_16x16)) / 255.0
        except Exception as e:
            logger.debug("Error computing 16x16 luminance grid: %s", e)

        hand_detected = False
        handedness = "Unknown"
        landmarks_3d_dicts: List[Dict[str, Any]] = []
        landmarks_2d_px: List[Tuple[float, float]] = []
        bbox = None
        wrist_vel = 0.0
        is_open = False
        is_raised_flat = False
        is_fist = False
        is_yo = False
        is_pinch = False
        pinch_dist_mm = 0.0

        now_mono = time.monotonic()
        run_mediapipe = (
            self._hands_solution is not None
            and (now_mono - self._last_mp_monotonic) >= self.min_mp_interval
        )

        if run_mediapipe:
            self._last_mp_monotonic = now_mono
            if w > self.inference_width:
                scale = self.inference_width / float(w)
                small = cv2.resize(
                    rgb_image,
                    (self.inference_width, max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = rgb_image
            rgb_converted = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_converted.flags.writeable = False
            mp_results = self._hands_solution.process(rgb_converted)

            if mp_results.multi_hand_landmarks:
                hand_detected = True
                hand_lms = mp_results.multi_hand_landmarks[0]
                if mp_results.multi_handedness:
                    handedness = mp_results.multi_handedness[0].classification[0].label

                landmarks_2d_px = []
                xs, ys = [], []
                for lm in hand_lms.landmark:
                    px_x = float(lm.x * w)
                    px_y = float(lm.y * h)
                    landmarks_2d_px.append((px_x, px_y))
                    xs.append(px_x)
                    ys.append(px_y)

                min_x, max_x = max(0, int(min(xs) - 15)), min(w, int(max(xs) + 15))
                min_y, max_y = max(0, int(min(ys) - 15)), min(h, int(max(ys) + 15))
                bbox = (min_x, min_y, max_x, max_y)

                joints_3d: List[Joint3D] = []
                if not self.gesture_only:
                    depth_source = transformed_depth
                    for j_id, ((u, v), name) in enumerate(zip(landmarks_2d_px, HAND_JOINT_NAMES)):
                        d_mm = self._sample_depth(depth_source, u, v)
                        if d_mm is None:
                            continue
                        pos_3d = None
                        if self.projection_fn is not None:
                            pos_3d = self.projection_fn((u, v), d_mm)
                        if pos_3d is None:
                            fx, fy, cx, cy = 505.2 * (w / 640.0), 505.4 * (h / 576.0), w / 2.0, h / 2.0
                            pos_3d = (
                                float((u - cx) * d_mm / fx),
                                float((v - cy) * d_mm / fy),
                                float(d_mm),
                            )
                        j3d = Joint3D(
                            joint_id=j_id,
                            name=name,
                            x=round(pos_3d[0], 2),
                            y=round(pos_3d[1], 2),
                            z=round(pos_3d[2], 2),
                            u_px=u,
                            v_px=v,
                        )
                        joints_3d.append(j3d)
                        landmarks_3d_dicts.append({
                            "joint_id": j_id,
                            "name": name,
                            "x": j3d.x,
                            "y": j3d.y,
                            "z": j3d.z,
                        })

                    wrist_joint = next((joint for joint in joints_3d if joint.joint_id == 0), None)
                    if wrist_joint is not None:
                        wrist_vel = self._compute_wrist_velocity(
                            t_sec, (wrist_joint.x, wrist_joint.y, wrist_joint.z)
                        )

                (
                    is_open,
                    is_raised_flat,
                    is_fist,
                    is_yo,
                    is_pinch,
                    pinch_dist_mm,
                ) = self._classify_gestures(landmarks_2d_px, joints_3d)

                self._last_valid_pose_time = t_sec
                self._last_yo = is_yo
                self._last_pinch = is_pinch
                self._cached_hand_detected = hand_detected
                self._cached_handedness = handedness
                self._cached_landmarks_2d = landmarks_2d_px
                self._cached_bbox = bbox
            else:
                if (t_sec - self._last_valid_pose_time) <= 0.2:
                    hand_detected = self._cached_hand_detected
                    handedness = self._cached_handedness
                    landmarks_2d_px = list(self._cached_landmarks_2d)
                    bbox = self._cached_bbox
                    is_yo = self._last_yo
                    is_pinch = self._last_pinch
                else:
                    self._last_yo = False
                    self._last_pinch = False
                    self._cached_hand_detected = False
                    self._cached_landmarks_2d = []
                    self._cached_bbox = None
        else:
            hand_detected = self._cached_hand_detected
            handedness = self._cached_handedness
            landmarks_2d_px = list(self._cached_landmarks_2d)
            bbox = self._cached_bbox
            is_yo = self._last_yo
            is_pinch = self._last_pinch

        if self._hands_solution is None:
            pass  # gesture fields remain default/false

        # Build final frame result
        res = HandTrackingResult(
            frame_index=frame_index,
            device_timestamp_usec=timestamp_usec,
            hand_detected=hand_detected or is_yo or is_pinch,
            handedness=handedness,
            landmarks_3d_mm=landmarks_3d_dicts,
            landmarks_2d_px=landmarks_2d_px,
            bounding_box_px=bbox,
            wrist_velocity_mm_s=wrist_vel,
            is_open_palm=is_open,
            is_raised_flat_hand=is_raised_flat,
            is_closed_fist=is_fist,
            is_yo_gesture=is_yo,
            is_pinch=is_pinch,
            pinch_distance_mm=pinch_dist_mm,
            mean_luminance=mean_lum,
            std_luminance=std_lum,
        )

        with self._result_lock:
            self.latest_result = res

        return res

    def run(self) -> None:
        """Consumer worker loop processing frame packets from stream."""
        self.running.set()
        while self.running.is_set():
            try:
                packet = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Always work on the newest frame so gesture latency stays low.
            while True:
                try:
                    packet = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            now = time.monotonic()
            self._fps_window.append(now)
            while self._fps_window and (now - self._fps_window[0]) > 1.0:
                self._fps_window.pop(0)
            self.fps_measured = float(len(self._fps_window))
            self.processed_frames += 1

            result = self.process_frame(
                rgb_image=packet.rgb_image,
                depth_image=packet.depth_image,
                transformed_depth=packet.transformed_depth,
                frame_index=packet.frame_index,
                timestamp_usec=packet.color_timestamp_usec,
            )

            # Broadcast to subscribers
            with self._sub_lock:
                for sub_q in self.subscribers:
                    try:
                        sub_q.put_nowait(result)
                    except queue.Full:
                        try:
                            _ = sub_q.get_nowait()
                            sub_q.put_nowait(result)
                        except (queue.Empty, queue.Full):
                            pass

    def stop(self) -> None:
        self.running.clear()
        if self._hands_solution is not None:
            try:
                self._hands_solution.close()
            except Exception:
                pass
