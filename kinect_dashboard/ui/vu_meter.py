"""Smooth animated Audio VU Meter widget with dB calibration and peak indicators.

Minimalist Viewfinder design:
- Sleek 1px hairline border with deep dark background.
- Clean color transitions from calm green (-60 to -18 dB) to warning amber and peak coral.
- Subtle calibrated tick marks and monospace live dB level readout.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget


class VUMeterWidget(QWidget):
    """Custom scientific VU Meter displaying RMS dB level and peak indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(32)
        self.setMinimumWidth(180)

        self._rms_db: float = -60.0
        self._peak_db: float = -60.0
        self._target_rms_db: float = -60.0
        self._target_peak_db: float = -60.0

        # Animation timer (30 Hz refresh for fluid visual needle/bar)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate_step)
        self._anim_timer.start(33)

    def set_levels(self, rms_db: float, peak_db: float) -> None:
        """Update target dB values (clamped between -60 dB and 0 dB)."""
        self._target_rms_db = max(-60.0, min(0.0, float(rms_db)))
        self._target_peak_db = max(-60.0, min(0.0, float(peak_db)))

    def _animate_step(self) -> None:
        """Smoothly interpolate toward target RMS and decay peak level."""
        # Attack is fast (snappy response), release is slightly smoothed
        if self._target_rms_db > self._rms_db:
            self._rms_db += (self._target_rms_db - self._rms_db) * 0.6
        else:
            self._rms_db += (self._target_rms_db - self._rms_db) * 0.25

        if self._target_peak_db > self._peak_db:
            self._peak_db = self._target_peak_db
        else:
            self._peak_db = max(-60.0, self._peak_db - 0.4)

        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background container
        painter.setPen(QPen(QColor("#27272A"), 1))
        painter.setBrush(QColor("#09090B"))
        painter.drawRoundedRect(0, 0, w, h, 6, 6)

        # Usable meter area
        padding_x = 5
        padding_y = 5
        meter_w = w - 2 * padding_x
        meter_h = h - 2 * padding_y

        if meter_w <= 0 or meter_h <= 0:
            return

        # Normalize dB from [-60, 0] to [0.0, 1.0]
        rms_norm = max(0.0, min(1.0, (self._rms_db + 60.0) / 60.0))
        peak_norm = max(0.0, min(1.0, (self._peak_db + 60.0) / 60.0))

        bar_w = int(meter_w * rms_norm)

        # Gradient bar
        if bar_w > 0:
            grad = QLinearGradient(padding_x, 0, padding_x + meter_w, 0)
            # -60 to -18 dB (0.0 to 0.70) -> Calm Green
            grad.setColorAt(0.0, QColor("#16A34A"))
            grad.setColorAt(0.65, QColor("#22C55E"))
            # -18 to -6 dB (0.70 to 0.90) -> Amber
            grad.setColorAt(0.75, QColor("#F59E0B"))
            # -6 to 0 dB (0.90 to 1.0) -> Coral Red
            grad.setColorAt(0.92, QColor("#EF4444"))
            grad.setColorAt(1.0, QColor("#DC2626"))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRoundedRect(padding_x, padding_y, bar_w, meter_h, 3, 3)

        # Draw subtle dB tick marks (-48, -36, -24, -12, -6, 0)
        tick_dbs = [-48, -36, -24, -12, -6, 0]
        painter.setPen(QPen(QColor("#27272A"), 1))
        for db in tick_dbs:
            pos_x = padding_x + int(meter_w * ((db + 60.0) / 60.0))
            painter.drawLine(pos_x, padding_y, pos_x, padding_y + meter_h)

        # Peak indicator line
        peak_x = padding_x + int(meter_w * peak_norm)
        if peak_x > padding_x:
            peak_color = QColor("#F87171") if self._peak_db > -6.0 else QColor("#FAFAFA")
            painter.setPen(QPen(peak_color, 1.5))
            painter.drawLine(peak_x, padding_y, peak_x, padding_y + meter_h)

        # Subtle dB text readout
        if self._rms_db > -54.0:
            painter.setPen(QColor("#A1A1AA"))
            font = QFont("Cascadia Code", 8, QFont.Weight.DemiBold)
            font.setStyleHint(QFont.StyleHint.Monospace)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self._rms_db:+.1f} dB")
