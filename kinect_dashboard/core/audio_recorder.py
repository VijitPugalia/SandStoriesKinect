"""Multichannel / mono audio capture worker with non-blocking ring buffer and VU metering."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError, Exception) as exc:
    logger.warning("sounddevice is unavailable (%s).", exc)
    SOUNDDEVICE_AVAILABLE = False
    sd = None  # type: ignore


@dataclass
class AudioChunk:
    """Thread-safe packet carrying a block of audio samples."""
    samples: np.ndarray  # int16 mono column vector [frames, 1]
    sample_rate: int
    timestamp_monotonic: float = field(default_factory=time.monotonic)
    rms_db: float = -60.0


@dataclass
class _StreamAttempt:
    device: Optional[int | str]
    name: str
    channels: int
    samplerate: int
    dtype: str = "int16"
    extract: str = "mono"  # mono | kinect_ch0


class AudioRecorderWorker(threading.Thread):
    """Producer worker capturing 48 kHz 16-bit PCM audio continuously."""

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        block_size: int = 1024,
        device: Optional[int | str] = None,
        queue_maxsize: int = 200,
        allow_synthetic: bool = False,
        prefer_kinect_wdm: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device
        self.allow_synthetic = allow_synthetic
        self.prefer_kinect_wdm = prefer_kinect_wdm
        self.capture_mode = "pending"
        self.last_error: Optional[str] = None

        self.audio_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=queue_maxsize)
        self.subscribers: List[queue.Queue[AudioChunk]] = []
        self._subscriber_lock = threading.Lock()

        self.active_device_name: str = ""
        self.active_input_channels: int = channels
        self._active_extract: str = "mono"

        self.running = threading.Event()
        self.stream_start_monotonic: float = 0.0
        self.total_samples_captured: int = 0
        self.last_sample_monotonic: float = 0.0
        self.callback_failure_count: int = 0
        self.current_rms_db: float = -60.0
        self.peak_db: float = -60.0
        self._peak_decay_time: float = time.monotonic()

    @staticmethod
    def find_kinect_wdm_device() -> Optional[int]:
        """Return the Azure Kinect seven-channel Windows WDM-KS input device index."""
        if not SOUNDDEVICE_AVAILABLE:
            return None
        try:
            host_apis = sd.query_hostapis()
            for index, device in enumerate(sd.query_devices()):
                host_name = host_apis[device["hostapi"]]["name"]
                if (
                    "Azure Kinect" in device["name"]
                    and host_name == "Windows WDM-KS"
                    and int(device["max_input_channels"]) >= 7
                ):
                    return index
        except Exception as exc:
            logger.warning("Could not locate Kinect WDM-KS microphone: %s", exc)
        return None

    def subscribe(self, maxsize: int = 100) -> queue.Queue[AudioChunk]:
        """Subscribe a dedicated queue (e.g. for session disk recording)."""
        q: queue.Queue[AudioChunk] = queue.Queue(maxsize=maxsize)
        with self._subscriber_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[AudioChunk]) -> None:
        with self._subscriber_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _calculate_rms_db(self, data_int16: np.ndarray) -> float:
        """Calculate RMS level in decibels relative to full scale (0 dBFS = 32768)."""
        if data_int16.size == 0:
            return -60.0
        flat = data_int16.reshape(-1).astype(np.float32)
        rms = float(np.sqrt(np.mean(flat ** 2)))
        if rms <= 1e-5:
            return -60.0
        db = 20.0 * np.log10(rms / 32768.0)
        return float(np.clip(db, -60.0, 0.0))

    def _to_mono_int16(self, indata: np.ndarray) -> np.ndarray:
        """Convert PortAudio input to mono int16 samples shaped [frames, 1]."""
        data = np.asarray(indata)
        if data.size == 0:
            return np.zeros((0, 1), dtype=np.int16)

        if self._active_extract == "kinect_ch0" and data.ndim == 2 and data.shape[1] >= 1:
            # Averaging the seven Kinect capsules can cancel speech because the
            # capsules receive phase-offset versions of the same source.
            if np.issubdtype(data.dtype, np.floating):
                mono = np.clip(data[:, 0], -1.0, 1.0)
                pcm = np.rint(mono * 32767.0).astype(np.int16)
            else:
                pcm = data[:, 0].astype(np.int16, copy=False)
            return pcm.reshape(-1, 1)

        if data.ndim == 1:
            return data.astype(np.int16, copy=False).reshape(-1, 1)

        if np.issubdtype(data.dtype, np.floating):
            if data.shape[1] == 1:
                mono = np.clip(data[:, 0], -1.0, 1.0)
            else:
                mono = np.clip(np.mean(data, axis=1), -1.0, 1.0)
            pcm = np.rint(mono * 32767.0).astype(np.int16)
            return pcm.reshape(-1, 1)

        if data.shape[1] == 1:
            return data.astype(np.int16, copy=False)
        mono = np.mean(data.astype(np.float32), axis=1)
        return np.rint(np.clip(mono, -32768, 32767)).astype(np.int16).reshape(-1, 1)

    def _process_audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Non-blocking low-latency callback called by PortAudio in high-priority thread."""
        if not self.running.is_set():
            return
        if status:
            logger.debug("Sounddevice input status: %s", status)

        t_now = time.monotonic()
        data_copy = self._to_mono_int16(indata)
        rms_db = self._calculate_rms_db(data_copy)
        self.current_rms_db = rms_db

        if rms_db > self.peak_db or (t_now - self._peak_decay_time) > 1.0:
            self.peak_db = rms_db
            self._peak_decay_time = t_now
        else:
            self.peak_db = max(-60.0, self.peak_db - 0.5)

        self.total_samples_captured += int(data_copy.shape[0])
        chunk = AudioChunk(
            samples=data_copy,
            sample_rate=self.sample_rate,
            timestamp_monotonic=t_now,
            rms_db=rms_db,
        )

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                _ = self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

        with self._subscriber_lock:
            for sub_q in self.subscribers:
                try:
                    sub_q.put_nowait(chunk)
                except queue.Full:
                    try:
                        _ = sub_q.get_nowait()
                        sub_q.put_nowait(chunk)
                    except (queue.Empty, queue.Full):
                        pass

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Keep callback failures visible so the stream watchdog can recover."""
        try:
            self._process_audio_callback(indata, frames, time_info, status)
            self.last_sample_monotonic = time.monotonic()
        except Exception as exc:
            self.callback_failure_count += 1
            self.last_error = f"Audio callback failed: {exc}"
            logger.exception(self.last_error)
            raise

    def _candidate_stream_configs(self) -> List[_StreamAttempt]:
        """Build an ordered list of input configurations to try."""
        candidates: List[_StreamAttempt] = []
        seen = set()

        def add(
            device: Optional[int | str],
            name: str,
            channels: int,
            samplerate: int,
            dtype: str = "int16",
            extract: str = "mono",
        ) -> None:
            key = (str(device), int(channels), int(samplerate), dtype, extract)
            if key in seen or channels <= 0 or samplerate <= 0:
                return
            seen.add(key)
            candidates.append(_StreamAttempt(device, name, channels, samplerate, dtype, extract))

        if self.prefer_kinect_wdm:
            kinect_idx = self.find_kinect_wdm_device()
            if kinect_idx is not None:
                try:
                    info = sd.query_devices(kinect_idx, "input")
                    name = str(info.get("name", f"device {kinect_idx}"))
                except Exception:
                    name = f"Azure Kinect device {kinect_idx}"
                add(kinect_idx, name, 7, self.sample_rate, "float32", "kinect_ch0")

        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
        except Exception as exc:
            logger.warning("Could not enumerate audio devices: %s", exc)
            devices = []
            host_apis = []

        kinect_first: List[tuple] = []
        others: List[tuple] = []
        for idx, dev in enumerate(devices):
            try:
                max_ch = int(dev.get("max_input_channels", 0))
            except Exception:
                max_ch = 0
            if max_ch <= 0:
                continue
            name = str(dev.get("name", f"device {idx}"))
            rate = float(dev.get("default_samplerate", self.sample_rate))
            host_name = host_apis[dev["hostapi"]]["name"] if host_apis else ""
            entry = (idx, name, max_ch, rate, host_name)
            if "kinect" in name.lower():
                kinect_first.append(entry)
            else:
                others.append(entry)

        for idx, name, max_ch, rate, host_name in kinect_first + others:
            if "Azure Kinect" in name and host_name == "Windows WDM-KS" and max_ch >= 7:
                add(idx, name, 7, self.sample_rate, "float32", "kinect_ch0")
            add(idx, name, self.channels, self.sample_rate, "int16", "mono")
            add(idx, name, max_ch, self.sample_rate, "int16", "mono")
            add(idx, name, self.channels, int(rate), "int16", "mono")
            add(idx, name, max_ch, int(rate), "int16", "mono")

        if self.device is not None:
            try:
                info = sd.query_devices(self.device, "input")
                name = str(info.get("name", str(self.device)))
                max_ch = int(info.get("max_input_channels", self.channels))
                rate = int(float(info.get("default_samplerate", self.sample_rate)))
                add(self.device, name, self.channels, self.sample_rate)
                add(self.device, name, max_ch, self.sample_rate)
                add(self.device, name, self.channels, rate)
                add(self.device, name, max_ch, rate)
            except Exception:
                add(self.device, str(self.device), self.channels, self.sample_rate)

        if not candidates:
            add(None, "default input", self.channels, self.sample_rate)
        return candidates

    def run(self) -> None:
        """Run audio capture, using synthetic samples only when explicitly allowed."""
        self.running.set()
        self.stream_start_monotonic = time.monotonic()

        if SOUNDDEVICE_AVAILABLE:
            errors: List[str] = []
            for attempt in self._candidate_stream_configs():
                self._active_extract = attempt.extract
                try:
                    self.capture_mode = "pending"
                    samples_before_open = self.total_samples_captured
                    opened_at = time.monotonic()
                    with sd.InputStream(
                        samplerate=attempt.samplerate,
                        channels=attempt.channels,
                        dtype=attempt.dtype,
                        blocksize=self.block_size,
                        device=attempt.device,
                        callback=self._audio_callback,
                    ):
                        self.capture_mode = "hardware"
                        self.last_error = None
                        self.sample_rate = int(attempt.samplerate)
                        self.active_device_name = attempt.name
                        self.active_input_channels = int(attempt.channels)
                        logger.info(
                            "Audio capture active on '%s' at %d Hz, %d input ch (%s/%s)",
                            attempt.name,
                            self.sample_rate,
                            attempt.channels,
                            attempt.dtype,
                            attempt.extract,
                        )
                        while self.running.is_set():
                            time.sleep(0.05)
                            now = time.monotonic()
                            no_first_callback = (
                                self.total_samples_captured == samples_before_open
                                and now - opened_at > 2.0
                            )
                            callback_stalled = (
                                self.last_sample_monotonic > opened_at
                                and now - self.last_sample_monotonic > 2.0
                            )
                            if no_first_callback or callback_stalled:
                                raise RuntimeError("audio callbacks stopped for more than 2 seconds")
                        return
                except Exception as exc:
                    msg = (
                        f"{attempt.name} ({attempt.channels}ch @ {attempt.samplerate} Hz "
                        f"{attempt.dtype} {attempt.extract}): {exc}"
                    )
                    errors.append(msg)
                    logger.warning("Audio input attempt failed: %s", msg)
                    if not self.running.is_set():
                        return
            self.last_error = "Failed to open any audio input. Tried: " + " | ".join(errors[:4])
            logger.error(self.last_error)

        if not self.allow_synthetic:
            self.capture_mode = "failed"
            if self.last_error is None:
                self.last_error = "Audio input is unavailable."
                logger.error(self.last_error)
            self.running.clear()
            return

        self.capture_mode = "synthetic"
        logger.warning("DEMO MODE: audio recorder is generating synthetic noise.")
        block_duration = self.block_size / self.sample_rate
        while self.running.is_set():
            t_start = time.monotonic()
            noise = np.random.normal(0, 150, (self.block_size, 1)).astype(np.int16)
            self._audio_callback(noise, self.block_size, None, None)
            elapsed = time.monotonic() - t_start
            time.sleep(max(0.0, block_duration - elapsed))

    def stop(self) -> None:
        """Stop audio capture."""
        self.running.clear()
