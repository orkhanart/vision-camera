"""Main window for VISION simulator - Minimalist 1024x1024 design."""

import asyncio
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QLabel,
    QFrame,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSlider,
    QGraphicsDropShadowEffect,
)
from PIL import Image

from .viewfinder import ViewfinderWidget
from .loading_screen import LoadingScreen
from ..camera.capture import CameraCapture, CameraFrame
from ..inference.engine import InferenceEngine, InferenceRequest, InferenceResult
from ..inference.models import ModelManager
from ..inference.styles import STYLE_PRESETS, get_style_categories
from ..hardware.simulator import HardwareSimulator
from ..hardware.profiles import JetsonOrinProfile


# Global stylesheet with JetBrains Mono
STYLE = """
* {
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', monospace;
}

QMainWindow {
    background-color: #0a0a0a;
}

QLabel {
    color: #e0e0e0;
}

QPushButton {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 11px;
}

QPushButton:hover {
    background-color: #252525;
    border-color: #444;
}

QPushButton:pressed {
    background-color: #333;
}

QPushButton:checked {
    background-color: #2563eb;
    border-color: #2563eb;
    color: white;
}

QPushButton:disabled {
    background-color: #111;
    color: #555;
    border-color: #222;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #333;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #2563eb;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 2px;
}
"""


class InferenceWorker(QObject):
    """Worker for running inference in background thread."""

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int)

    def __init__(self, engine: InferenceEngine):
        super().__init__()
        self.engine = engine
        self._request: Optional[InferenceRequest] = None

    def set_request(self, request: InferenceRequest) -> None:
        self._request = request

    @Slot()
    def run(self) -> None:
        if self._request is None:
            self.error.emit("No request set")
            return
        try:
            print(f"[DEBUG] Starting inference with model: {self._request.model_id}")
            print(f"[DEBUG] Image size: {self._request.image.size}")
            print(f"[DEBUG] Style: {self._request.style_id}, Steps: {self._request.steps}")

            self.engine.set_progress_callback(
                lambda step, total: self.progress.emit(step, total)
            )
            result = self.engine.run_inference_sync(self._request)
            print(f"[DEBUG] Inference complete! Output size: {result.image.size}")
            self.finished.emit(result)
        except Exception as e:
            import traceback
            print(f"[ERROR] Inference failed: {e}")
            traceback.print_exc()
            self.error.emit(str(e))


class StyleChip(QPushButton):
    """Minimal style selection chip."""

    def __init__(self, style_id: str, name: str, parent=None):
        super().__init__(name, parent)
        self.style_id = style_id
        self.setCheckable(True)
        self.setFixedHeight(28)
        self.setMinimumWidth(60)


class VisionMainWindow(QMainWindow):
    """
    Minimalist 1024x1024 VISION simulator.

    Clean, focused interface for AI art generation.
    """

    def __init__(
        self,
        camera_id: int = 0,
        models_dir: Path = Path("models"),
    ):
        super().__init__()

        self.camera_id = camera_id
        self.models_dir = models_dir

        self.setWindowTitle("VISION")
        self.setFixedSize(1024, 1024)

        # Root stacked widget to switch between loading and main app
        self._root_stack = QStackedWidget()
        self.setCentralWidget(self._root_stack)

        # Create and show loading screen first
        self._loading_screen = LoadingScreen()
        self._loading_screen.loading_complete.connect(self._on_loading_complete)
        self._root_stack.addWidget(self._loading_screen)
        self._root_stack.setCurrentIndex(0)

        # Main app container (will be setup after loading)
        self._main_container = QWidget()
        self._root_stack.addWidget(self._main_container)

        # Start loading
        QTimer.singleShot(100, self._loading_screen.start_loading)

    def _on_loading_complete(self, results: dict):
        """Handle loading completion and setup main app."""
        import sys
        print(f"[DEBUG] _on_loading_complete called with results: {results.keys() if results else 'None'}", flush=True)
        sys.stdout.flush()
        self._loading_results = results
        self._init_components(results)
        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()
        self._start_camera()

        # Status update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_status)
        self._update_timer.start(500)

        # Switch to main app
        self._root_stack.setCurrentIndex(1)

    def _init_components(self, loading_results: dict = None) -> None:
        """Initialize core components with pre-loaded model if available."""
        self.hardware_sim = HardwareSimulator(JetsonOrinProfile)
        self.model_manager = ModelManager(
            models_dir=self.models_dir,
            max_memory_gb=32.0,
        )

        # Get selected model from loading results
        default_model = "sdxl-turbo"
        if loading_results and loading_results.get("model_id"):
            default_model = loading_results["model_id"]

        self.inference_engine = InferenceEngine(
            model_manager=self.model_manager,
            hardware_sim=self.hardware_sim,
            default_model=default_model,
        )

        # If we have a pre-loaded pipeline, inject it into the model manager
        if loading_results and loading_results.get("pipeline"):
            pipeline = loading_results["pipeline"]
            model_id = loading_results["model_id"]
            model_info = self.model_manager.get_model_info(model_id)
            print(f"[DEBUG] Injecting pre-loaded pipeline: {model_id}")
            print(f"[DEBUG] Pipeline type: {type(pipeline)}")
            if model_info:
                self.model_manager._loaded_models[model_id] = pipeline
                self.model_manager._model_memory[model_id] = model_info.memory_gb
                self.model_manager._current_memory_gb += model_info.memory_gb
                print(f"[DEBUG] Model injected successfully. Loaded models: {self.model_manager.get_loaded_models()}")
            else:
                print(f"[ERROR] Model info not found for {model_id}")
        else:
            print(f"[DEBUG] No pre-loaded pipeline. Results: {loading_results}")

        self.camera = CameraCapture(
            camera_id=self.camera_id,
            target_fps=30,
        )

        self._inference_thread: Optional[QThread] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._last_result: Optional[InferenceResult] = None
        self._captured_frame: Optional[CameraFrame] = None
        self._camera_available = False
        self._loaded_image: Optional[Image.Image] = None
        self._current_style = "cinematic"
        self._selected_model_id = default_model

    def _setup_ui(self) -> None:
        """Setup minimalist UI."""
        self.setStyleSheet(STYLE)

        # Use the main container (already added to root stack)
        central = self._main_container

        # Main layout - full bleed
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # === TOP BAR (status) ===
        top_bar = QWidget()
        top_bar.setFixedHeight(32)
        top_bar.setStyleSheet("background-color: rgba(0,0,0,0.8);")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 0, 16, 0)

        self._status_label = QLabel("VISION")
        self._status_label.setStyleSheet("color: #666; font-size: 11px; font-weight: 600;")
        top_layout.addWidget(self._status_label)

        top_layout.addStretch()

        # Navigation buttons
        nav_btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                color: #666;
                font-size: 10px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                color: #e0e0e0;
                background-color: rgba(255,255,255,0.1);
                border-radius: 4px;
            }
            QPushButton:checked {
                color: #2563eb;
            }
        """

        self._camera_btn = QPushButton("Camera")
        self._camera_btn.setStyleSheet(nav_btn_style)
        self._camera_btn.setCheckable(True)
        self._camera_btn.setChecked(True)
        self._camera_btn.clicked.connect(self._show_camera)
        top_layout.addWidget(self._camera_btn)

        self._gallery_btn = QPushButton("Gallery")
        self._gallery_btn.setStyleSheet(nav_btn_style)
        self._gallery_btn.setCheckable(True)
        self._gallery_btn.clicked.connect(self._show_gallery)
        top_layout.addWidget(self._gallery_btn)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setStyleSheet(nav_btn_style)
        self._settings_btn.setCheckable(True)
        self._settings_btn.clicked.connect(self._show_settings)
        top_layout.addWidget(self._settings_btn)

        top_layout.addSpacing(16)

        self._hw_label = QLabel("--")
        self._hw_label.setStyleSheet("color: #444; font-size: 10px;")
        top_layout.addWidget(self._hw_label)

        layout.addWidget(top_bar)

        # === MAIN DISPLAY AREA ===
        self._display_stack = QStackedWidget()
        self._display_stack.setStyleSheet("background-color: #0a0a0a;")

        # Viewfinder (index 0)
        self._viewfinder = ViewfinderWidget(
            display_width=1024,
            display_height=1024,
        )
        self._display_stack.addWidget(self._viewfinder)

        # Result display (index 1)
        self._result_display = QLabel()
        self._result_display.setAlignment(Qt.AlignCenter)
        self._result_display.setStyleSheet("background-color: #0a0a0a;")
        self._display_stack.addWidget(self._result_display)

        # No camera / image preview (index 2)
        no_cam = QWidget()
        no_cam.setStyleSheet("background-color: #0a0a0a;")
        no_cam_layout = QVBoxLayout(no_cam)
        no_cam_layout.setAlignment(Qt.AlignCenter)

        self._preview_label = QLabel("No camera detected")
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(512, 512)
        self._preview_label.setStyleSheet("color: #444; font-size: 12px;")
        no_cam_layout.addWidget(self._preview_label)

        self._display_stack.addWidget(no_cam)

        # Gallery view (index 3)
        gallery = QWidget()
        gallery.setStyleSheet("background-color: #0a0a0a;")
        gallery_layout = QVBoxLayout(gallery)
        gallery_layout.setContentsMargins(16, 16, 16, 16)

        gallery_title = QLabel("Gallery")
        gallery_title.setStyleSheet("color: #e0e0e0; font-size: 14px; font-weight: 600;")
        gallery_layout.addWidget(gallery_title)

        self._gallery_grid = QWidget()
        self._gallery_grid.setStyleSheet("background-color: #111;")
        gallery_layout.addWidget(self._gallery_grid, stretch=1)

        gallery_hint = QLabel("Generated images will appear here")
        gallery_hint.setStyleSheet("color: #444; font-size: 10px;")
        gallery_hint.setAlignment(Qt.AlignCenter)
        gallery_layout.addWidget(gallery_hint)

        self._display_stack.addWidget(gallery)

        # Settings view (index 4)
        settings = QWidget()
        settings.setStyleSheet("background-color: #0a0a0a;")
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(16, 16, 16, 16)
        settings_layout.setSpacing(16)

        settings_title = QLabel("Settings")
        settings_title.setStyleSheet("color: #e0e0e0; font-size: 14px; font-weight: 600;")
        settings_layout.addWidget(settings_title)

        # Model selection
        model_section = QWidget()
        model_layout = QHBoxLayout(model_section)
        model_layout.setContentsMargins(0, 0, 0, 0)

        model_label = QLabel("Model")
        model_label.setStyleSheet("color: #666; font-size: 11px;")
        model_label.setFixedWidth(100)
        model_layout.addWidget(model_label)

        self._model_btn_turbo = QPushButton("SDXL-Turbo")
        self._model_btn_turbo.setCheckable(True)
        self._model_btn_turbo.setChecked(True)
        model_layout.addWidget(self._model_btn_turbo)

        self._model_btn_sd = QPushButton("SD 1.5")
        self._model_btn_sd.setCheckable(True)
        model_layout.addWidget(self._model_btn_sd)

        model_layout.addStretch()
        settings_layout.addWidget(model_section)

        # Hardware info
        hw_section = QWidget()
        hw_layout = QVBoxLayout(hw_section)
        hw_layout.setContentsMargins(0, 0, 0, 0)

        hw_title = QLabel("Hardware Simulation")
        hw_title.setStyleSheet("color: #666; font-size: 11px;")
        hw_layout.addWidget(hw_title)

        self._hw_info = QLabel("NVIDIA Jetson AGX Orin 64GB\n275 TOPS | 64GB LPDDR5")
        self._hw_info.setStyleSheet("color: #444; font-size: 10px; padding: 8px; background-color: #111; border-radius: 4px;")
        hw_layout.addWidget(self._hw_info)

        settings_layout.addWidget(hw_section)

        settings_layout.addStretch()

        self._display_stack.addWidget(settings)

        layout.addWidget(self._display_stack, stretch=1)

        # === BOTTOM CONTROLS ===
        controls = QWidget()
        controls.setFixedHeight(140)
        controls.setStyleSheet("background-color: rgba(0,0,0,0.9);")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(12)

        # Style selector row
        style_row = QHBoxLayout()
        style_row.setSpacing(6)

        style_label = QLabel("Style")
        style_label.setStyleSheet("color: #666; font-size: 10px;")
        style_label.setFixedWidth(40)
        style_row.addWidget(style_label)

        self._style_chips: dict[str, StyleChip] = {}

        # Add key styles only (minimal)
        key_styles = ["cinematic", "anime", "watercolor", "cyberpunk", "noir", "vintage"]
        for sid in key_styles:
            if sid in STYLE_PRESETS:
                style = STYLE_PRESETS[sid]
                chip = StyleChip(sid, style.name)
                chip.clicked.connect(lambda checked, s=sid: self._select_style(s))
                style_row.addWidget(chip)
                self._style_chips[sid] = chip

        style_row.addStretch()
        controls_layout.addLayout(style_row)

        # Parameters row
        params_row = QHBoxLayout()
        params_row.setSpacing(20)

        # Strength
        str_label = QLabel("Strength")
        str_label.setStyleSheet("color: #666; font-size: 10px;")
        str_label.setFixedWidth(50)
        params_row.addWidget(str_label)

        self._strength_slider = QSlider(Qt.Horizontal)
        self._strength_slider.setRange(30, 100)
        self._strength_slider.setValue(75)
        self._strength_slider.setFixedWidth(120)
        params_row.addWidget(self._strength_slider)

        self._strength_val = QLabel("0.75")
        self._strength_val.setStyleSheet("color: #2563eb; font-size: 10px;")
        self._strength_val.setFixedWidth(30)
        self._strength_slider.valueChanged.connect(
            lambda v: self._strength_val.setText(f"{v/100:.2f}")
        )
        params_row.addWidget(self._strength_val)

        params_row.addSpacing(20)

        # Steps
        step_label = QLabel("Steps")
        step_label.setStyleSheet("color: #666; font-size: 10px;")
        step_label.setFixedWidth(35)
        params_row.addWidget(step_label)

        self._steps_slider = QSlider(Qt.Horizontal)
        self._steps_slider.setRange(1, 20)
        self._steps_slider.setValue(4)
        self._steps_slider.setFixedWidth(80)
        params_row.addWidget(self._steps_slider)

        self._steps_val = QLabel("4")
        self._steps_val.setStyleSheet("color: #2563eb; font-size: 10px;")
        self._steps_val.setFixedWidth(20)
        self._steps_slider.valueChanged.connect(
            lambda v: self._steps_val.setText(str(v))
        )
        params_row.addWidget(self._steps_val)

        params_row.addStretch()

        # Media button style
        media_btn_style = """
            QPushButton {
                background-color: #1a1a1a;
                border: 1px solid #333;
                color: #e0e0e0;
                font-weight: 500;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #252525;
                border-color: #444;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """

        # Capture button
        self._capture_btn = QPushButton("Capture")
        self._capture_btn.setFixedSize(80, 36)
        self._capture_btn.setStyleSheet(media_btn_style)
        self._capture_btn.clicked.connect(self._on_capture)
        params_row.addWidget(self._capture_btn)

        params_row.addSpacing(4)

        # Load button (next to Capture)
        self._load_btn = QPushButton("Load")
        self._load_btn.setFixedSize(60, 36)
        self._load_btn.setStyleSheet(media_btn_style)
        self._load_btn.clicked.connect(self._load_image_file)
        params_row.addWidget(self._load_btn)

        params_row.addSpacing(8)

        # Generate button (hidden until media captured/loaded)
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedSize(100, 36)
        self._generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                border: none;
                color: white;
                font-weight: 600;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #1e3a5f;
                color: #666;
            }
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        self._generate_btn.hide()  # Hidden initially
        params_row.addWidget(self._generate_btn)

        controls_layout.addLayout(params_row)

        # Info row
        info_row = QHBoxLayout()

        self._info_label = QLabel("Ready")
        self._info_label.setStyleSheet("color: #444; font-size: 10px;")
        info_row.addWidget(self._info_label)

        info_row.addStretch()

        shortcuts = QLabel("L: load  G: generate  V: view  S: save  Esc: quit")
        shortcuts.setStyleSheet("color: #333; font-size: 9px;")
        info_row.addWidget(shortcuts)

        controls_layout.addLayout(info_row)

        layout.addWidget(controls)

        # Select default style
        self._select_style("cinematic")

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence(Qt.Key_Space), self, self._on_capture)
        QShortcut(QKeySequence(Qt.Key_G), self, self._on_generate)
        QShortcut(QKeySequence(Qt.Key_V), self, self._show_viewfinder)
        QShortcut(QKeySequence(Qt.Key_S), self, self._save_result)
        QShortcut(QKeySequence(Qt.Key_L), self, self._load_image_file)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self._viewfinder.capture_requested.connect(self._on_capture)

    def _start_camera(self) -> None:
        """Start camera capture."""
        import sys
        print("[DEBUG] Starting camera...", flush=True)

        # Connect signal before starting (Qt signals are thread-safe)
        self.camera.frame_ready.connect(self._on_camera_frame)

        started = self.camera.start()
        print(f"[DEBUG] camera.start() returned: {started}", flush=True)

        if not started:
            self._camera_available = False
            self._display_stack.setCurrentIndex(2)
            self._info_label.setText("No camera - load an image")
            print("[DEBUG] Camera not available, showing no-camera view", flush=True)
            return

        self._camera_available = True
        self._display_stack.setCurrentIndex(0)
        print(f"[DEBUG] Camera started. _camera_available = {self._camera_available}", flush=True)

    def _load_image_file(self) -> None:
        """Load image file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Images (*.png *.jpg *.jpeg *.webp);;All (*)",
        )
        if path:
            try:
                self._loaded_image = Image.open(path).convert("RGB")

                # Show preview
                preview = self._loaded_image.copy()
                preview.thumbnail((512, 512), Image.Resampling.LANCZOS)

                data = preview.tobytes("raw", "RGB")
                qimg = QImage(data, preview.width, preview.height,
                             3 * preview.width, QImage.Format_RGB888)
                self._preview_label.setPixmap(QPixmap.fromImage(qimg))

                self._generate_btn.show()
                self._info_label.setText(f"Loaded: {Path(path).name} - press Generate")

            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _select_style(self, style_id: str) -> None:
        """Select a style."""
        for sid, chip in self._style_chips.items():
            chip.setChecked(sid == style_id)
        self._current_style = style_id

        # Update recommended params
        if style_id in STYLE_PRESETS:
            style = STYLE_PRESETS[style_id]
            self._strength_slider.setValue(int(style.recommended_strength * 100))

    @Slot(object)
    def _on_camera_frame(self, frame: CameraFrame) -> None:
        """Handle camera frame - runs in main thread via Qt signal."""
        self._viewfinder.update_frame(frame)

    def _on_capture(self) -> None:
        """Capture current frame and show preview."""
        if self._captured_frame is not None:
            # Already have a capture, retake - go back to live view
            self._captured_frame = None
            self._capture_btn.setText("Capture")
            self._info_label.setText("Live view")
            self._display_stack.setCurrentIndex(0)
            return

        frame = self._viewfinder.get_current_frame()
        if frame is None:
            self._info_label.setText("No frame to capture")
            return

        self._captured_frame = frame

        # Show captured frame in result display
        pil_image = frame.to_pil()
        display_size = 1024 - 32 - 140
        pil_image.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)

        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(data, pil_image.width, pil_image.height,
                     3 * pil_image.width, QImage.Format_RGB888)
        self._result_display.setPixmap(QPixmap.fromImage(qimg))
        self._display_stack.setCurrentIndex(1)

        # Update UI
        self._capture_btn.setText("Retake")
        self._generate_btn.show()
        self._info_label.setText("Captured - press Generate or Retake")

    def _on_generate(self) -> None:
        """Generate AI art."""
        import sys
        print(f"[DEBUG] Generate clicked. Camera available: {self._camera_available}", flush=True)
        sys.stdout.flush()

        input_image: Optional[Image.Image] = None

        if self._camera_available:
            frame = self._captured_frame or self._viewfinder.get_current_frame()
            print(f"[DEBUG] Captured frame: {self._captured_frame is not None}, Current frame: {self._viewfinder.get_current_frame() is not None}", flush=True)
            if frame:
                input_image = frame.to_pil()
                print(f"[DEBUG] Got image from camera: {input_image.size}", flush=True)
        else:
            input_image = self._loaded_image
            if input_image:
                print(f"[DEBUG] Using loaded image: {input_image.size}", flush=True)

        if input_image is None:
            self._info_label.setText("No image - capture or load one first")
            print("[DEBUG] No input image available!", flush=True)
            return

        print(f"[DEBUG] Model manager loaded: {self.model_manager.get_loaded_models()}", flush=True)
        print(f"[DEBUG] Selected model: {self._selected_model_id}", flush=True)
        print(f"[DEBUG] Style: {self._current_style}", flush=True)

        request = InferenceRequest(
            image=input_image,
            style_id=self._current_style,
            strength=self._strength_slider.value() / 100,
            steps=self._steps_slider.value(),
            model_id=self._selected_model_id,
        )

        self._generate_btn.setEnabled(False)
        self._generate_btn.setText("...")
        self._run_inference(request)

    def _run_inference(self, request: InferenceRequest) -> None:
        """Run inference in background."""
        self._inference_thread = QThread()
        self._inference_worker = InferenceWorker(self.inference_engine)
        self._inference_worker.set_request(request)
        self._inference_worker.moveToThread(self._inference_thread)

        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.finished.connect(self._on_inference_complete)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.progress.connect(self._on_inference_progress)
        self._inference_worker.finished.connect(self._inference_thread.quit)
        self._inference_worker.error.connect(self._inference_thread.quit)
        self._inference_thread.finished.connect(self._inference_thread.deleteLater)

        self._inference_thread.start()

    @Slot(object)
    def _on_inference_complete(self, result: InferenceResult) -> None:
        """Handle inference completion."""
        self._last_result = result
        self._generate_btn.setEnabled(True)
        self._generate_btn.setText("Generate")

        self._show_result(result.image)

        self._info_label.setText(
            f"{result.inference_time_ms:.0f}ms | "
            f"Est. Jetson: {result.estimated_device_time_ms:.0f}ms"
        )

    @Slot(str)
    def _on_inference_error(self, error: str) -> None:
        """Handle inference error."""
        self._generate_btn.setEnabled(True)
        self._generate_btn.setText("Generate")
        self._info_label.setText(f"Error: {error[:50]}")

    @Slot(int, int)
    def _on_inference_progress(self, step: int, total: int) -> None:
        """Handle progress."""
        self._info_label.setText(f"Step {step}/{total}")

    def _show_result(self, image: Image.Image) -> None:
        """Show result image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Scale to fit display
        display_size = 1024 - 32 - 140  # minus top bar and controls
        image = image.copy()
        image.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)

        data = image.tobytes("raw", "RGB")
        qimg = QImage(data, image.width, image.height,
                     3 * image.width, QImage.Format_RGB888)
        self._result_display.setPixmap(QPixmap.fromImage(qimg))
        self._display_stack.setCurrentIndex(1)

    def _show_viewfinder(self) -> None:
        """Return to viewfinder."""
        if self._camera_available:
            self._display_stack.setCurrentIndex(0)
        else:
            self._display_stack.setCurrentIndex(2)
        self._captured_frame = None
        self._capture_btn.setText("Capture")
        self._generate_btn.hide()
        self._info_label.setText("Live view")

    def _update_nav_buttons(self, active: str) -> None:
        """Update navigation button states."""
        self._camera_btn.setChecked(active == "camera")
        self._gallery_btn.setChecked(active == "gallery")
        self._settings_btn.setChecked(active == "settings")

    def _show_camera(self) -> None:
        """Show camera view."""
        self._update_nav_buttons("camera")
        self._captured_frame = None
        self._capture_btn.setText("Capture")
        self._generate_btn.hide()
        if self._camera_available:
            self._display_stack.setCurrentIndex(0)
            self._info_label.setText("Live view")
        else:
            self._display_stack.setCurrentIndex(2)
            self._info_label.setText("No camera - load an image")

    def _show_gallery(self) -> None:
        """Show gallery view."""
        self._update_nav_buttons("gallery")
        self._display_stack.setCurrentIndex(3)

    def _show_settings(self) -> None:
        """Show settings view."""
        self._update_nav_buttons("settings")
        self._display_stack.setCurrentIndex(4)

    def _save_result(self) -> None:
        """Save result."""
        if self._last_result is None:
            self._info_label.setText("Nothing to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save", "vision_output.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            self._last_result.image.save(path)
            self._info_label.setText(f"Saved: {Path(path).name}")

    def _update_status(self) -> None:
        """Update hardware status."""
        if not self.inference_engine.is_processing:
            self.hardware_sim.update_idle()

        status = self.hardware_sim.get_status_dict()
        mem = status["memory"]["percent"]
        temp = status["thermal"]["temperature_c"]
        bat = status["power"]["battery_percent"]

        self._hw_label.setText(
            f"MEM {mem:.0f}%  TEMP {temp:.0f}°C  BAT {bat:.0f}%"
        )

    def closeEvent(self, event) -> None:
        """Handle close."""
        self.camera.stop()
        self._update_timer.stop()
        if self._inference_thread and self._inference_thread.isRunning():
            self._inference_thread.quit()
            self._inference_thread.wait(2000)
        event.accept()
