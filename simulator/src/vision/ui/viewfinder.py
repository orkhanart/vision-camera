"""Minimal viewfinder widget for 1024x1024 display."""

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from ..camera.capture import CameraFrame


class ViewfinderWidget(QWidget):
    """
    Minimal live camera viewfinder.

    Features:
    - Real-time camera preview
    - Subtle grid overlay
    - Touch/click to capture
    """

    frame_ready = Signal(object)
    touch_focus = Signal(int, int)
    capture_requested = Signal()

    def __init__(
        self,
        display_width: int = 1024,
        display_height: int = 1024,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.display_width = display_width
        self.display_height = display_height

        self._current_frame: Optional[CameraFrame] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._show_grid = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize UI."""
        self.setMinimumSize(512, 512)
        self.setStyleSheet("background-color: #0a0a0a;")

        self._image_label = QLabel(self)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background-color: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label)

    def update_frame(self, frame: CameraFrame) -> None:
        """Update with new frame."""
        self._current_frame = frame

        # Convert BGR to RGB and ensure contiguous memory
        rgb = np.ascontiguousarray(frame.to_rgb())

        # Create QImage
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._current_pixmap = QPixmap.fromImage(qimg)

        # Get target size - use display size as fallback
        target_size = self._image_label.size()
        if target_size.width() < 100 or target_size.height() < 100:
            target_size = self.size()
        if target_size.width() < 100 or target_size.height() < 100:
            target_size = QSize(self.display_width, self.display_height)

        # Scale to fit
        scaled = self._current_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self._image_label.setPixmap(scaled)
        self.frame_ready.emit(frame)

    def paintEvent(self, event) -> None:
        """Draw overlays."""
        super().paintEvent(event)

        if not self._show_grid or self._current_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self._image_label.geometry()

        # Subtle grid
        pen = QPen(QColor(255, 255, 255, 30))
        pen.setWidth(1)
        painter.setPen(pen)

        w, h = rect.width(), rect.height()
        x, y = rect.x(), rect.y()

        # Rule of thirds
        painter.drawLine(x + w // 3, y, x + w // 3, y + h)
        painter.drawLine(x + 2 * w // 3, y, x + 2 * w // 3, y + h)
        painter.drawLine(x, y + h // 3, x + w, y + h // 3)
        painter.drawLine(x, y + 2 * h // 3, x + w, y + 2 * h // 3)

        painter.end()

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click to capture."""
        if event.button() == Qt.LeftButton:
            self.capture_requested.emit()

    def set_show_grid(self, show: bool) -> None:
        """Toggle grid."""
        self._show_grid = show
        self.update()

    def get_current_frame(self) -> Optional[CameraFrame]:
        """Get current frame."""
        return self._current_frame
