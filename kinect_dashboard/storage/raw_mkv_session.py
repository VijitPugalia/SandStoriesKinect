"""Lightweight raw MKV session coordinator for dashboard capture."""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.io.wavfile as wavfile

from ..core.audio_recorder import AudioChunk
from .session_writer import SessionMetadata, generate_session_id


class RawMkvSessionWriter(threading.Thread):
    """Write WAV and JSON while Kinect worker writes authoritative MKV."""

    def __init__(self, output_root: Path, metadata: SessionMetadata, audio_queue: queue.Queue[AudioChunk]):
        super().__init__(daemon=True)
        self.metadata = metadata
        self.audio_queue = audio_queue
        self.start_utc = datetime.now(timezone.utc)
        self.start_monotonic = 0.0
        self.end_monotonic = 0.0
        self.stop_requested_monotonic = 0.0
        self.session_id = generate_session_id(metadata)
        self.session_dir = Path(output_root) / self.session_id
        suffix = 1
        while self.session_dir.exists():
            suffix += 1
            self.session_dir = Path(output_root) / f"{self.session_id}_{suffix}"
        self.session_id = self.session_dir.name
        self.source_dir = self.session_dir / "source"
        self.mkv_path = self.source_dir / "azure_kinect.mkv"
        self.audio_path = self.session_dir / "audio.wav"
        self.metadata_path = self.session_dir / "recording_metadata.json"
        self.events_path = self.session_dir / "gesture_events.jsonl"
        self.running = threading.Event()
        self.ready = threading.Event()
        self.paused = threading.Event()
        self.discarded = False
        self.audio_buffer: list[np.ndarray] = []
        self.audio_samples_written = 0
        self.audio_sample_rate_actual = metadata.audio_sample_rate
        self.frames_written = 0
        self.segment_marks: list[dict[str, Any]] = []
        self.finalization_error: Optional[str] = None
        self.setup_error: Optional[str] = None
        self.frame_queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self.stop_source = "button"

    def elapsed_seconds(self, now: Optional[float] = None) -> float:
        if not self.start_monotonic:
            return 0.0
        return max(0.0, (now or time.monotonic()) - self.start_monotonic)

    def run(self) -> None:
        self.running.set()
        self.start_monotonic = time.monotonic()
        try:
            self.source_dir.mkdir(parents=True, exist_ok=False)
            self.events_path.write_text("", encoding="utf-8")
        except Exception as exc:
            self.setup_error = f"Could not create raw session: {exc}"
            self.finalization_error = self.setup_error
            self.running.clear()
        self.ready.set()
        while self.running.is_set() or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self.audio_buffer.append(chunk.samples)
            self.audio_samples_written += len(chunk.samples)
            self.audio_sample_rate_actual = chunk.sample_rate
        self.end_monotonic = time.monotonic()
        if self.discarded:
            shutil.rmtree(self.session_dir, ignore_errors=True)
            return
        self._finalize()

    def record_gesture_event(self, event_type: str, **kwargs: Any) -> None:
        event = {
            "event": event_type,
            "monotonicElapsedSeconds": round(self.elapsed_seconds(), 6),
            "timestampUnix": time.time(),
            **kwargs,
        }
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    def mark_segment(self, source: str = "pinch") -> dict[str, Any]:
        mark = {
            "segment_id": len(self.segment_marks) + 1,
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "monotonicElapsedSeconds": round(self.elapsed_seconds(), 6),
            "source": source,
        }
        self.segment_marks.append(mark)
        self.record_gesture_event("mark_segment", **mark)
        return mark

    def stop_and_save(self) -> None:
        self.stop_requested_monotonic = time.monotonic()
        self.running.clear()

    def discard(self) -> None:
        self.discarded = True
        self.running.clear()

    def _finalize(self) -> None:
        if not self.mkv_path.exists() or self.mkv_path.stat().st_size == 0:
            self.finalization_error = self.finalization_error or "Azure Kinect MKV was not created."
        audio_duration = 0.0
        if self.audio_buffer:
            audio = np.concatenate(self.audio_buffer, axis=0).reshape(-1)
            wavfile.write(str(self.audio_path), self.audio_sample_rate_actual, audio)
            audio_duration = len(audio) / float(self.audio_sample_rate_actual)
        elif self.metadata.audio_expected:
            self.finalization_error = "No audio samples captured; audio.wav not created."
        payload = {
            "schemaVersion": 1,
            "sessionId": self.session_id,
            "captureBackend": "pyk4a_direct_mkv",
            "captureDeviceFamily": "azure_kinect_dk",
            "kinectStartedUtc": self.start_utc.isoformat(),
            "endedUtc": datetime.now(timezone.utc).isoformat(),
            "durationSeconds": round(self.elapsed_seconds(self.stop_requested_monotonic or self.end_monotonic), 3),
            "mkvPath": "source/azure_kinect.mkv",
            "audioPath": "audio.wav" if self.audio_path.exists() else None,
            "gestureEventsPath": "gesture_events.jsonl",
            "capturesWritten": self.frames_written,
            "audioSampleRate": self.audio_sample_rate_actual,
            "audioDurationSeconds": round(audio_duration, 3),
            "segmentCount": len(self.segment_marks),
            "stopSource": self.stop_source,
            "experimentalConditions": {
                "mountingHeightCm": self.metadata.mounting_height_cm,
                "lightingType": self.metadata.lighting_type,
                "measuredLux": self.metadata.measured_lux,
                "surfaceState": self.metadata.surface_state,
                "trialNumber": self.metadata.trial_number,
                "sessionName": self.metadata.session_name,
            },
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
