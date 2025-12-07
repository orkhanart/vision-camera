"""Camera capture module for VISION simulator - Qt thread-safe version."""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal, Slot


@dataclass
class CameraFrame:
    """A captured camera frame with metadata."""

    image: np.ndarray  # BGR format from OpenCV
    timestamp: float
    frame_number: int
    width: int
    height: int

    def to_rgb(self) -> np.ndarray:
        """Convert to RGB format."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image (RGB)."""
        return Image.fromarray(self.to_rgb())


class CameraWorker(QObject):
    """
    Camera capture worker that runs in a separate thread.
    Emits frames via Qt signal for thread-safe UI updates.
    """

    frame_captured = Signal(object)  # Emits CameraFrame
    error = Signal(str)
    started = Signal()
    stopped = Signal()

    def __init__(self, camera_id: int = 0, target_fps: int = 30):
        super().__init__()
        self.camera_id = camera_id
        self.target_fps = target_fps
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0

    @Slot()
    def run(self) -> None:
        """Main capture loop - runs in worker thread."""
        self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            self.error.emit(f"Cannot open camera {self.camera_id}")
            return

        # Get actual resolution
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {w}x{h}")

        self._running = True
        self.started.emit()

        frame_interval = 1.0 / self.target_fps

        while self._running:
            start_time = time.time()

            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_count += 1

            # Create frame with COPY of image data (critical for thread safety)
            camera_frame = CameraFrame(
                image=frame.copy(),  # Deep copy!
                timestamp=time.time(),
                frame_number=self._frame_count,
                width=frame.shape[1],
                height=frame.shape[0],
            )

            # Emit signal (Qt handles thread marshalling)
            self.frame_captured.emit(camera_frame)

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        self._cap.release()
        self._cap = None
        self.stopped.emit()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False

    @property
    def frame_count(self) -> int:
        return self._frame_count


class CameraCapture(QObject):
    """
    Camera capture manager with Qt thread support.

    Usage:
        camera = CameraCapture()
        camera.frame_ready.connect(my_handler)
        camera.start()
    """

    frame_ready = Signal(object)  # Re-emits CameraFrame to UI

    def __init__(self, camera_id: int = 0, target_fps: int = 30, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.target_fps = target_fps

        self._thread: Optional[QThread] = None
        self._worker: Optional[CameraWorker] = None
        self._is_running = False
        self._last_frame: Optional[CameraFrame] = None

    def start(self) -> bool:
        """Start camera capture in background thread."""
        if self._is_running:
            return True

        # Create thread and worker
        self._thread = QThread()
        self._worker = CameraWorker(self.camera_id, self.target_fps)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.frame_captured.connect(self._on_frame)
        self._worker.started.connect(self._on_started)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)

        # Start thread
        self._thread.start()

        # Wait for camera to open (up to 3 seconds)
        for _ in range(30):
            if self._is_running:
                return True
            time.sleep(0.1)

        return self._is_running

    def stop(self) -> None:
        """Stop camera capture."""
        if self._worker:
            self._worker.stop()

        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
            self._worker = None

        self._is_running = False

    def _on_frame(self, frame: CameraFrame) -> None:
        """Handle frame from worker - runs in main thread."""
        self._last_frame = frame
        self.frame_ready.emit(frame)

    def _on_started(self) -> None:
        self._is_running = True

    def _on_stopped(self) -> None:
        self._is_running = False

    def _on_error(self, msg: str) -> None:
        print(f"Camera error: {msg}")
        self._is_running = False

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the most recent frame."""
        return self._last_frame

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def frame_count(self) -> int:
        return self._worker.frame_count if self._worker else 0
