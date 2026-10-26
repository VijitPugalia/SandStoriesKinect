"""Dual live viewports for RGB stream with HUD overlays and false-color Depth topography.

Minimalist Viewfinder design:
- Sleek 1px hairline container borders and deep background.
- Translucent monospace micro chips for stream identification.
- Clean, uncluttered hand skeletal lines and pinch highlight.
- Non-intrusive floating HUD countdown capsule and segment mark banners.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..core.gesture_engine import GestureStatusOverlay
from ..core.hand_tracker import HAND_CONNECTIONS, HandTrackingResult


def _draw_viewport_chip(painter: QPainter, x: float, y: float, label: str) -> None:
    """Draw a minimalist translucent label chip on a viewport."""
    font = QFont("Cascadia Code", 9, QFont.Weight.DemiBold)
    font.setStyleHint(QFont.StyleHint.Monospace)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    chip_w = metrics.horizontalAdvance(label) + 18
    chip_h = 24
    chip_rect = QRectF(x, y, chip_w, chip_h)

    # Hairline border and dark surface
    painter.setPen(QPen(QColor(39, 39, 42, 230), 1))
    painter.setBrush(QColor(9, 9, 11, 220))
    painter.drawRoundedRect(chip_rect, 6, 6)

    painter.setPen(QColor("#FAFAFA"))
    painter.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, label)


class RGBViewportWidget(QWidget):
    """Left Viewport: 1080p RGB feed with skeletal overlay, arena boundary, and gesture HUD."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)
        self.setStyleSheet(
            "background-color: #09090B; border: 1px solid #27272A; border-radius: 8px;"
        )

        self._current_pixmap: Optional[QPixmap] = None
        self._tracking_result: Optional[HandTrackingResult] = None
        self._gesture_overlay: Optional[GestureStatusOverlay] = None
        self.arena_corners_px: List[List[int]] = [[240, 140], [1680, 140], [1680, 940], [240, 940]]
        self._flash_text: str = ""
        self._flash_until: float = 0.0

    def flash_message(self, text: str, duration_sec: float = 2.0) -> None:
        """Show a short confirmation banner (e.g. after an accepted gesture)."""
        import time as _time
        self._flash_text = text
        self._flash_until = _time.monotonic() + duration_sec
        self.update()

    def update_frame(
        self,
        bgr_image: np.ndarray,
        tracking_result: Optional[HandTrackingResult] = None,
        gesture_overlay: Optional[GestureStatusOverlay] = None,
    ) -> None:
        """Update viewport with latest BGR frame and detection metadata."""
        self._tracking_result = tracking_result
        self._gesture_overlay = gesture_overlay

        h, w, ch = bgr_image.shape
        self._source_w = w
        self._source_h = h

        # For preview display on a ~600x340 widget, downsample to max 960 width
        # to avoid pushing 186 MB/s of uncompressed 1080p frames through Qt's GUI thread.
        # This keeps the UI responsive at 60 FPS without affecting recording quality.
        if w > 960:
            scale = 960.0 / w
            preview_w = 960
            preview_h = max(1, int(round(h * scale)))
            disp_bgr = cv2.resize(bgr_image, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
        else:
            disp_bgr = bgr_image
            preview_w, preview_h = w, h

        bytes_per_line = ch * preview_w
        rgb_img = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_img.data, preview_w, preview_h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q_img)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_w, widget_h = self.width(), self.height()

        # Fill background
        painter.fillRect(0, 0, widget_w, widget_h, QColor("#080A0E"))

        if self._current_pixmap is None:
            painter.setPen(QColor("#484F58"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for color stream…")
            return

        # Scale maintaining 16:9 aspect ratio
        pm_w, pm_h = self._current_pixmap.width(), self._current_pixmap.height()
        aspect = pm_w / float(pm_h)

        if widget_w / float(widget_h) > aspect:
            draw_h = widget_h
            draw_w = int(widget_h * aspect)
            offset_x = (widget_w - draw_w) // 2
            offset_y = 0
        else:
            draw_w = widget_w
            draw_h = int(widget_w / aspect)
            offset_x = 0
            offset_y = (widget_h - draw_h) // 2

        orig_w = float(getattr(self, "_source_w", pm_w))
        orig_h = float(getattr(self, "_source_h", pm_h))
        scale_x = draw_w / orig_w
        scale_y = draw_h / orig_h

        # Draw video frame
        target_rect = QRectF(offset_x, offset_y, draw_w, draw_h)
        painter.drawPixmap(target_rect.toRect(), self._current_pixmap)

        # 1. Draw Sand Arena boundary & Fiducial corners
        if self.arena_corners_px and len(self.arena_corners_px) == 4:
            pts = []
            for px, py in self.arena_corners_px:
                sx = offset_x + px * scale_x
                sy = offset_y + py * scale_y
                pts.append(QPointF(sx, sy))

            # Arena boundary rectangle (subtle hairline dashed)
            painter.setPen(QPen(QColor(88, 166, 255, 90), 1.2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(4):
                painter.drawLine(pts[i], pts[(i + 1) % 4])

            # Corner fiducial markers
            for pt in pts:
                painter.setPen(QPen(QColor(240, 246, 252, 200), 1.5))
                painter.setBrush(QColor(248, 81, 73, 190))
                painter.drawEllipse(pt, 4.0, 4.0)

        # 2. Minimal hand overlay for gesture feedback
        if self._tracking_result is not None and self._tracking_result.hand_detected:
            lms_2d = self._tracking_result.landmarks_2d_px
            if len(lms_2d) == 21:
                screen_pts = [
                    QPointF(offset_x + u * scale_x, offset_y + v * scale_y)
                    for (u, v) in lms_2d
                ]

                # Hand skeleton connections
                painter.setPen(QPen(QColor(88, 166, 255, 160), 1.2))
                for idx1, idx2 in HAND_CONNECTIONS:
                    if idx1 < len(screen_pts) and idx2 < len(screen_pts):
                        painter.drawLine(screen_pts[idx1], screen_pts[idx2])

                # Joints
                for i, pt in enumerate(screen_pts):
                    if i in (4, 8):  # Thumb & index tips
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QColor(88, 166, 255, 210))
                        painter.drawEllipse(pt, 3.5, 3.5)
                    elif i == 0:  # Wrist root
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QColor(240, 246, 252, 190))
                        painter.drawEllipse(pt, 4.0, 4.0)

                # Active pinch indicator line between thumb and index
                if self._tracking_result.is_pinch:
                    painter.setPen(QPen(QColor(46, 160, 67, 240), 2.0))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    t_pt = screen_pts[4]
                    i_pt = screen_pts[8]
                    painter.drawLine(t_pt, i_pt)

        # 3. Draw Gesture Status & Visual Countdown Overlay HUD (Minimal capsule)
        if self._gesture_overlay is not None and self._gesture_overlay.is_active:
            overlay_w = min(280, int(draw_w * 0.60))
            overlay_h = 42
            overlay_x = offset_x + (draw_w - overlay_w) // 2
            overlay_y = offset_y + draw_h - overlay_h - 14

            hud_rect = QRectF(overlay_x, overlay_y, overlay_w, overlay_h)
            painter.setPen(QPen(QColor(31, 37, 53, 220), 1))
            painter.setBrush(QColor(8, 10, 14, 220))
            painter.drawRoundedRect(hud_rect, 10, 10)

            painter.setPen(QColor("#E6EDF3"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            painter.drawText(
                QRectF(overlay_x + 12, overlay_y + 6, overlay_w - 24, 16),
                Qt.AlignmentFlag.AlignLeft,
                self._gesture_overlay.hud_message,
            )

            bar_x = overlay_x + 12
            bar_y = overlay_y + 26
            bar_w = overlay_w - 24
            bar_h = 4

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(22, 27, 36))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

            fill_w = int(bar_w * self._gesture_overlay.progress_ratio)
            if fill_w > 0:
                painter.setBrush(QColor("#58A6FF"))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2, 2)

        # Viewport chip label
        _draw_viewport_chip(painter, offset_x + 10, offset_y + 10, "COLOR")

        # Visual Flash Banner (e.g. SEGMENT MARKED)
        if self._flash_text:
            import time as _time
            if _time.monotonic() <= self._flash_until:
                banner_w = min(360, int(draw_w * 0.75))
                banner_h = 36
                bx = offset_x + (draw_w - banner_w) // 2
                by = offset_y + 12
                b_rect = QRectF(bx, by, banner_w, banner_h)

                painter.setPen(QPen(QColor(46, 160, 67, 220), 1))
                painter.setBrush(QColor(8, 10, 14, 230))
                painter.drawRoundedRect(b_rect, 8, 8)

                painter.setPen(QColor("#3FB950"))
                font = QFont("Cascadia Code", 11, QFont.Weight.Bold)
                font.setStyleHint(QFont.StyleHint.Monospace)
                painter.setFont(font)
                painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, self._flash_text)
            else:
                self._flash_text = ""


class DepthViewportWidget(QWidget):
    """Right Viewport: False-color Depth Topography with sleek scale bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)
        self.setStyleSheet(
            "background-color: #09090B; border: 1px solid #27272A; border-radius: 8px;"
        )
        self.setMouseTracking(True)

        self._current_pixmap: Optional[QPixmap] = None
        self._raw_depth: Optional[np.ndarray] = None
        self.min_depth_mm: float = 400.0  # 0.4 m
        self.max_depth_mm: float = 1200.0  # 1.2 m
        self.hover_depth_mm: Optional[float] = None
        self.hover_pos_px: Optional[Tuple[int, int]] = None

    def update_depth_frame(self, depth_raw: np.ndarray) -> None:
        """Process raw 16-bit depth map into false-color Turbo topography."""
        self._raw_depth = depth_raw
        h, w = depth_raw.shape[:2]

        # Fast in-place clipping and scaling on uint16 (no float32 array allocations)
        min_d = int(self.min_depth_mm)
        max_d = int(self.max_depth_mm)
        span = max(1.0, float(max_d - min_d))
        clamped = np.clip(depth_raw, min_d, max_d)
        depth_norm = ((clamped - min_d) * (255.0 / span)).astype(np.uint8)

        # Apply TURBO colormap
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        depth_color[depth_raw == 0] = [0, 0, 0]

        # Convert to QImage
        bytes_per_line = 3 * w
        rgb_depth = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_depth.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q_img)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Sample depth value under mouse pointer for live inspection."""
        if self._raw_depth is None or self._current_pixmap is None:
            return

        widget_w, widget_h = self.width(), self.height()
        pm_w, pm_h = self._current_pixmap.width(), self._current_pixmap.height()
        aspect = pm_w / float(pm_h)

        if widget_w / float(widget_h) > aspect:
            draw_h = widget_h
            draw_w = int(widget_h * aspect)
            offset_x = (widget_w - draw_w) // 2
            offset_y = 0
        else:
            draw_w = widget_w
            draw_h = int(widget_w / aspect)
            offset_x = 0
            offset_y = (widget_h - draw_h) // 2

        pos = event.position()
        mx, my = pos.x() - offset_x, pos.y() - offset_y
        if 0 <= mx < draw_w and 0 <= my < draw_h:
            img_x = int(mx * pm_w / draw_w)
            img_y = int(my * pm_h / draw_h)
            h, w = self._raw_depth.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.hover_depth_mm = float(self._raw_depth[img_y, img_x])
                self.hover_pos_px = (int(pos.x()), int(pos.y()))
        else:
            self.hover_depth_mm = None
            self.hover_pos_px = None
        self.update()

    def leaveEvent(self, event: Any) -> None:
        self.hover_depth_mm = None
        self.hover_pos_px = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_w, widget_h = self.width(), self.height()
        painter.fillRect(0, 0, widget_w, widget_h, QColor("#080A0E"))

        if self._current_pixmap is None:
            painter.setPen(QColor("#484F58"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for depth stream…")
            return

        pm_w, pm_h = self._current_pixmap.width(), self._current_pixmap.height()
        aspect = pm_w / float(pm_h)

        if widget_w / float(widget_h) > aspect:
            draw_h = widget_h
            draw_w = int(widget_h * aspect)
            offset_x = (widget_w - draw_w) // 2
            offset_y = 0
        else:
            draw_w = widget_w
            draw_h = int(widget_w / aspect)
            offset_x = 0
            offset_y = (widget_h - draw_h) // 2

        target_rect = QRectF(offset_x, offset_y, draw_w, draw_h)
        painter.drawPixmap(target_rect.toRect(), self._current_pixmap)

        # 1. Depth Scale Bar (Right Edge - sleek 12px width)
        scale_w = 12
        scale_h = max(90, int(draw_h * 0.65))
        scale_x = offset_x + draw_w - scale_w - 20
        scale_y = offset_y + (draw_h - scale_h) // 2

        scale_grad = QLinearGradient(0, scale_y, 0, scale_y + scale_h)
        scale_grad.setColorAt(0.0, QColor("#30123B"))  # 0.4 m (Near)
        scale_grad.setColorAt(0.35, QColor("#28BBEC")) # 0.68 m
        scale_grad.setColorAt(0.7, QColor("#A2FC3C"))  # 0.96 m
        scale_grad.setColorAt(1.0, QColor("#7A0403"))  # 1.2 m (Far)

        painter.setPen(QPen(QColor("#1F2535"), 1))
        painter.setBrush(scale_grad)
        painter.drawRoundedRect(QRectF(scale_x, scale_y, scale_w, scale_h), 3, 3)

        # Scale tick labels in monospace
        painter.setPen(QColor("#8B949E"))
        font_scale = QFont("Cascadia Code", 8)
        font_scale.setStyleHint(QFont.StyleHint.Monospace)
        painter.setFont(font_scale)
        painter.drawText(scale_x - 44, scale_y + 10, "0.40m")
        painter.drawText(scale_x - 44, scale_y + scale_h // 2 + 4, "0.80m")
        painter.drawText(scale_x - 44, scale_y + scale_h, "1.20m")

        # Depth probe tooltip
        if self.hover_depth_mm is not None and self.hover_pos_px is not None:
            hx, hy = self.hover_pos_px
            probe_text = f"{self.hover_depth_mm:.0f} mm" if self.hover_depth_mm > 0 else "—"
            probe_w, probe_h = 68, 22
            p_rect = QRectF(hx + 12, hy - 24, probe_w, probe_h)
            painter.setPen(QPen(QColor(31, 37, 53, 220), 1))
            painter.setBrush(QColor(8, 10, 14, 220))
            painter.drawRoundedRect(p_rect, 4, 4)
            painter.setPen(QColor("#58A6FF"))
            p_font = QFont("Cascadia Code", 9, QFont.Weight.DemiBold)
            p_font.setStyleHint(QFont.StyleHint.Monospace)
            painter.setFont(p_font)
            painter.drawText(p_rect, Qt.AlignmentFlag.AlignCenter, probe_text)

        _draw_viewport_chip(painter, offset_x + 10, offset_y + 10, "DEPTH")
