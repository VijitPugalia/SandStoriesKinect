"""Gesture recognition state machine for hands-free session control."""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .hand_tracker import HandTrackingResult

logger = logging.getLogger(__name__)


class GestureState(enum.Enum):
    """Finite states of the multimodal capture session."""
    IDLE = "STATE_IDLE"
    RECORDING = "STATE_RECORDING"
    PAUSED = "STATE_PAUSED"
    SAVING = "STATE_SAVING"


class GestureCommand(enum.Enum):
    """Actions emitted by gesture engine or manual UI triggers."""
    START_RECORDING = "START_RECORDING"
    PAUSE_RECORDING = "PAUSE_RECORDING"
    RESUME_RECORDING = "RESUME_RECORDING"
    STOP_AND_SAVE = "STOP_AND_SAVE"
    MARK_SEGMENT = "MARK_SEGMENT"
    DISCARD = "DISCARD"


@dataclass
class GestureStatusOverlay:
    """Status information formatted for GUI on-screen overlay HUD."""
    current_state: GestureState
    active_gesture_name: Optional[str] = None
    target_action: Optional[str] = None
    hold_duration_sec: float = 0.0
    required_duration_sec: float = 0.0
    progress_ratio: float = 0.0  # 0.0 to 1.0
    hud_message: str = ""
    is_active: bool = False


class GestureEngine:
    """State machine evaluating pinch and camera-cover gesture triggers."""

    def __init__(
        self,
        pinch_hold_sec: float = 2.0,
        cover_hold_sec: float = 3.0,
        require_static_hand: bool = False,
        max_wrist_velocity_mm_s: float = 30.0,
        max_tracking_gap_sec: float = 0.5,
    ) -> None:
        self.pinch_hold_sec = pinch_hold_sec
        self.cover_hold_sec = cover_hold_sec
        self.require_static_hand = require_static_hand
        self.max_wrist_velocity = max_wrist_velocity_mm_s
        self.max_tracking_gap_sec = max_tracking_gap_sec

        self.state: GestureState = GestureState.IDLE
        self._state_lock = threading.Lock()

        self._cooldown_until: float = 0.0

        # Camera cover tracking
        self._baseline_start_time: float = 0.0
        self._baseline_samples: List[float] = []
        self._baseline_luminance: float = 0.4
        self._cover_start_time: float = 0.0

        # Pinch gesture tracking
        self._pinch_start_time: float = 0.0
        self._pinch_needs_separation: bool = False
        self._last_tracking_key: Optional[tuple[int, int]] = None
        self._last_fresh_tracking_time: float = 0.0

        # Callbacks
        self.on_state_change: Optional[Callable[[GestureState], None]] = None
        self.on_command_triggered: Optional[Callable[[GestureCommand], None]] = None
        self.on_audio_beep: Optional[Callable[[str], None]] = None

    def set_state(self, new_state: GestureState) -> None:
        """Manually transition state (e.g. from UI buttons)."""
        with self._state_lock:
            old_state = self.state
            if old_state != new_state:
                self.state = new_state
                logger.info("GestureEngine state: %s -> %s", old_state.value, new_state.value)
                now = time.monotonic()
                self._cover_start_time = 0.0
                self._pinch_start_time = 0.0
                self._pinch_needs_separation = False
                self._last_tracking_key = None
                self._last_fresh_tracking_time = 0.0
                if new_state == GestureState.RECORDING:
                    self._baseline_start_time = now
                    self._baseline_samples = []
                    self._baseline_luminance = 0.4
                if self.on_state_change:
                    self.on_state_change(new_state)

    def _trigger_command(self, cmd: GestureCommand, beep_type: Optional[str] = "success") -> None:
        """Trigger command and optional sound notification."""
        self._cooldown_until = time.monotonic() + 1.2
        self._cover_start_time = 0.0
        self._pinch_start_time = 0.0

        if beep_type and self.on_audio_beep:
            try:
                self.on_audio_beep(beep_type)
            except Exception:
                pass

        if self.on_command_triggered:
            try:
                self.on_command_triggered(cmd)
            except Exception as e:
                logger.error("Error in on_command_triggered: %s", e)

    def update(self, tracking_result: Optional[HandTrackingResult]) -> GestureStatusOverlay:
        """Evaluate tracking result and return HUD overlay state."""
        now = time.monotonic()
        overlay = GestureStatusOverlay(current_state=self.state)

        # The UI refreshes faster than hand tracking. Only a newly processed
        # tracking frame may advance a hold, so a frozen result cannot fire a
        # gesture by itself.
        if tracking_result is None:
            self._cover_start_time = 0.0
            self._pinch_start_time = 0.0
            return overlay

        tracking_key = (tracking_result.frame_index, tracking_result.device_timestamp_usec)
        if tracking_key == self._last_tracking_key:
            if (
                self._last_fresh_tracking_time > 0.0
                and now - self._last_fresh_tracking_time > self.max_tracking_gap_sec
            ):
                self._cover_start_time = 0.0
                self._pinch_start_time = 0.0
            return overlay

        if (
            self._last_fresh_tracking_time > 0.0
            and now - self._last_fresh_tracking_time > self.max_tracking_gap_sec
        ):
            self._cover_start_time = 0.0
            self._pinch_start_time = 0.0
        self._last_tracking_key = tracking_key
        self._last_fresh_tracking_time = now

        if now < self._cooldown_until:
            overlay.hud_message = "READY"
            return overlay

        mean_lum = tracking_result.mean_luminance if tracking_result is not None else 1.0
        std_lum = tracking_result.std_luminance if tracking_result is not None else 0.0

        is_hand = tracking_result is not None and tracking_result.hand_detected
        if self.require_static_hand:
            is_static = is_hand and (tracking_result.wrist_velocity_mm_s <= self.max_wrist_velocity)
        else:
            is_static = is_hand

        # Baseline luminance estimation during the first 1.0s of recording
        if self.state == GestureState.RECORDING:
            if self._baseline_start_time == 0.0:
                self._baseline_start_time = now
            if (now - self._baseline_start_time) <= 1.0:
                if mean_lum > 0.05:
                    self._baseline_samples.append(mean_lum)
                    import numpy as np
                    self._baseline_luminance = float(np.mean(self._baseline_samples))

        # Priority: 1. Camera Cover -> 2. Pinch

        # 1. Camera Cover: Hold 3 seconds -> Stop and Save
        if self.state in (GestureState.RECORDING, GestureState.PAUSED):
            is_cover_candidate = (
                mean_lum < 0.12
                and mean_lum < (0.35 * self._baseline_luminance)
                and std_lum < 0.08
            )
            if is_cover_candidate:
                if self._cover_start_time == 0.0:
                    self._cover_start_time = now
                hold_duration = now - self._cover_start_time
                progress = min(1.0, hold_duration / max(0.1, self.cover_hold_sec))

                overlay.active_gesture_name = "COVER CAMERA"
                overlay.target_action = "STOP & SAVE"
                overlay.hold_duration_sec = hold_duration
                overlay.required_duration_sec = self.cover_hold_sec
                overlay.progress_ratio = progress
                overlay.is_active = True
                overlay.hud_message = f"COVERING CAMERA: {hold_duration:.1f}s / {self.cover_hold_sec:.1f}s"

                if hold_duration >= self.cover_hold_sec:
                    logger.info("Camera covered for %.2fs! Triggering STOP_AND_SAVE", hold_duration)
                    self.set_state(GestureState.SAVING)
                    self._trigger_command(GestureCommand.STOP_AND_SAVE, beep_type="confirm")
                return overlay
            self._cover_start_time = 0.0
        else:
            self._cover_start_time = 0.0

        # 2. Thumb-Index Pinch: Hold 2 seconds -> Mark Segment
        is_pinch_detected = is_static and tracking_result is not None and tracking_result.is_pinch
        if self._pinch_needs_separation:
            if tracking_result.hand_detected and not tracking_result.is_pinch:
                self._pinch_needs_separation = False

        if is_pinch_detected and not self._pinch_needs_separation and self.state in (GestureState.RECORDING, GestureState.PAUSED):
            if self._pinch_start_time == 0.0:
                self._pinch_start_time = now
            hold_duration = now - self._pinch_start_time
            progress = min(1.0, hold_duration / max(0.1, self.pinch_hold_sec))

            overlay.active_gesture_name = "PINCH"
            overlay.target_action = "MARK SEGMENT"
            overlay.hold_duration_sec = hold_duration
            overlay.required_duration_sec = self.pinch_hold_sec
            overlay.progress_ratio = progress
            overlay.is_active = True
            overlay.hud_message = f"HOLD PINCH: {hold_duration:.1f}s / {self.pinch_hold_sec:.1f}s"

            if hold_duration >= self.pinch_hold_sec:
                logger.info("Pinch held for %.2fs! Triggering MARK_SEGMENT", hold_duration)
                self._pinch_start_time = 0.0
                self._pinch_needs_separation = True
                self._trigger_command(GestureCommand.MARK_SEGMENT, beep_type=None)
            return overlay
        self._pinch_start_time = 0.0

        return overlay
