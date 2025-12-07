"""ASCII terminal-style loading screen for VISION with model selection."""

import platform
import sys
import time
from pathlib import Path
from typing import Optional, Callable

import torch
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, Slot
from PySide6.QtGui import QFont, QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QStackedWidget, QFrame, QButtonGroup
)


VISION_ASCII = """
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
"""

# Available models for selection
MODEL_OPTIONS = [
    {
        "id": "sdxl-turbo",
        "name": "SDXL Turbo",
        "description": "Fast generation, good quality",
        "size": "~7 GB",
        "speed": "Fast (4 steps)",
        "repo_id": "stabilityai/sdxl-turbo",
    },
    {
        "id": "sd-1.5",
        "name": "Stable Diffusion 1.5",
        "description": "Classic model, lower memory",
        "size": "~4 GB",
        "speed": "Medium (20 steps)",
        "repo_id": "runwayml/stable-diffusion-v1-5",
    },
    {
        "id": "sdxl",
        "name": "Stable Diffusion XL",
        "description": "High quality, more details",
        "size": "~7 GB",
        "speed": "Slow (25 steps)",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    {
        "id": "flux-schnell",
        "name": "FLUX.1 Schnell",
        "description": "Latest tech, excellent quality",
        "size": "~12 GB",
        "speed": "Fast (4 steps)",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
    },
]

LOADING_STYLE = """
QWidget {
    background-color: #0a0a0a;
}

QLabel {
    color: #00ff00;
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
}

QTextEdit {
    background-color: #0a0a0a;
    color: #00ff00;
    border: none;
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 11px;
}

QPushButton {
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
}
"""

MODEL_SELECT_STYLE = """
QWidget {
    background-color: #0a0a0a;
}

QLabel {
    color: #e0e0e0;
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
}

QPushButton#model_card {
    background-color: #1a1a1a;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 16px;
    text-align: left;
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
}

QPushButton#model_card:hover {
    background-color: #252525;
    border-color: #444;
}

QPushButton#model_card:checked {
    background-color: #1e3a5f;
    border-color: #2563eb;
}

QPushButton#start_btn {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px 32px;
    font-size: 13px;
    font-weight: bold;
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
}

QPushButton#start_btn:hover {
    background-color: #3b82f6;
}

QPushButton#start_btn:disabled {
    background-color: #1e3a5f;
    color: #666;
}
"""


class SystemCheckWorker(QObject):
    """Worker thread for system checks and model loading."""

    log_message = Signal(str)
    progress = Signal(int, str)
    finished = Signal(dict)

    def __init__(self, selected_model_id: str = "sdxl-turbo", models_dir: Path = Path("models")):
        super().__init__()
        self.selected_model_id = selected_model_id
        self.models_dir = models_dir

    @Slot()
    def run(self):
        """Run system checks and download selected model."""
        results = {}

        # Find selected model info
        selected_model = None
        for m in MODEL_OPTIONS:
            if m["id"] == self.selected_model_id:
                selected_model = m
                break

        if not selected_model:
            selected_model = MODEL_OPTIONS[0]  # Default to sdxl-turbo

        # Phase 1: System detection
        self.log_message.emit("")
        self.log_message.emit("[SYS] Initializing VISION AI Camera System...")
        self.log_message.emit(f"[SYS] Platform: {platform.system()} {platform.release()}")
        self.log_message.emit(f"[SYS] Python: {sys.version.split()[0]}")
        time.sleep(0.3)

        # Phase 2: PyTorch/Device detection
        self.progress.emit(10, "Detecting compute device...")
        self.log_message.emit("")
        self.log_message.emit("[GPU] Scanning compute devices...")

        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.log_message.emit(f"[GPU] NVIDIA CUDA detected: {device_name}")
            self.log_message.emit(f"[GPU] VRAM: {memory_gb:.1f} GB")
            results["device"] = device
            results["device_name"] = device_name
            results["memory_gb"] = memory_gb
        elif torch.backends.mps.is_available():
            device = "mps"
            import subprocess
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True
                )
                mem_bytes = int(result.stdout.strip())
                memory_gb = mem_bytes / (1024**3)
            except:
                memory_gb = 8.0
            self.log_message.emit(f"[GPU] Apple Metal Performance Shaders (MPS) detected")
            self.log_message.emit(f"[GPU] Unified Memory: {memory_gb:.0f} GB")
            results["device"] = device
            results["device_name"] = "Apple Silicon"
            results["memory_gb"] = memory_gb
        else:
            device = "cpu"
            self.log_message.emit("[GPU] No GPU detected, using CPU")
            self.log_message.emit("[GPU] WARNING: Inference will be slow!")
            results["device"] = device
            results["device_name"] = "CPU"
            results["memory_gb"] = 0

        time.sleep(0.3)

        # Phase 3: PyTorch info
        self.progress.emit(20, "Loading PyTorch...")
        self.log_message.emit("")
        self.log_message.emit(f"[LIB] PyTorch version: {torch.__version__}")

        # Check for diffusers
        try:
            import diffusers
            self.log_message.emit(f"[LIB] Diffusers version: {diffusers.__version__}")
        except ImportError:
            self.log_message.emit("[LIB] ERROR: diffusers not installed!")
            self.log_message.emit("[LIB] Run: pip install diffusers transformers accelerate")

        time.sleep(0.2)

        # Phase 4: Download/Load selected model
        self.progress.emit(30, f"Loading {selected_model['name']}...")
        self.log_message.emit("")
        self.log_message.emit(f"[MDL] Selected model: {selected_model['name']}")
        self.log_message.emit(f"[MDL] Repository: {selected_model['repo_id']}")
        self.log_message.emit(f"[MDL] Size: {selected_model['size']}")
        self.log_message.emit("")
        self.log_message.emit("[MDL] Checking local cache...")

        time.sleep(0.3)

        # Try to load the model
        try:
            self.log_message.emit("[MDL] Downloading model from HuggingFace Hub...")
            self.log_message.emit("[MDL] (First run will download, subsequent runs use cache)")
            self.log_message.emit("")

            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

            self.progress.emit(40, "Downloading model components...")

            # Import diffusers pipelines
            if selected_model["id"] in ("sd-1.5",):
                from diffusers import StableDiffusionImg2ImgPipeline
                self.log_message.emit("[MDL] Loading Stable Diffusion 1.5 pipeline...")

                self.progress.emit(50, "Downloading UNet...")
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    selected_model["repo_id"],
                    torch_dtype=dtype,
                    variant="fp16",
                    safety_checker=None,
                )

            elif selected_model["id"] in ("sdxl", "sdxl-turbo"):
                from diffusers import StableDiffusionXLImg2ImgPipeline
                self.log_message.emit("[MDL] Loading SDXL pipeline...")

                self.progress.emit(50, "Downloading UNet...")
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    selected_model["repo_id"],
                    torch_dtype=dtype,
                    variant="fp16" if selected_model["id"] == "sdxl" else None,
                )

            elif selected_model["id"] == "flux-schnell":
                from diffusers import FluxPipeline
                self.log_message.emit("[MDL] Loading FLUX pipeline...")

                self.progress.emit(50, "Downloading transformer...")
                pipeline = FluxPipeline.from_pretrained(
                    selected_model["repo_id"],
                    torch_dtype=dtype,
                )
            else:
                raise ValueError(f"Unknown model: {selected_model['id']}")

            self.progress.emit(70, "Moving model to device...")
            self.log_message.emit(f"[MDL] Moving model to {device}...")
            pipeline = pipeline.to(device)

            # Enable memory optimizations
            if device == "cuda":
                pipeline.enable_attention_slicing()
                self.log_message.emit("[MDL] Enabled attention slicing for memory optimization")

            self.progress.emit(80, "Model loaded successfully!")
            self.log_message.emit("[MDL] Model loaded successfully!")

            results["pipeline"] = pipeline
            results["model_id"] = selected_model["id"]
            results["model_loaded"] = True

        except Exception as e:
            self.log_message.emit(f"[MDL] ERROR: Failed to load model: {str(e)}")
            self.log_message.emit("[MDL] The app will work but inference won't be available")
            results["model_loaded"] = False
            results["model_error"] = str(e)

        # Phase 5: Hardware simulation
        self.progress.emit(90, "Initializing hardware simulation...")
        self.log_message.emit("")
        self.log_message.emit("[SIM] Target hardware: NVIDIA Jetson AGX Orin 64GB")
        self.log_message.emit("[SIM] Compute: 275 TOPS INT8 | 138 TFLOPS FP16")
        self.log_message.emit("[SIM] Memory: 64 GB LPDDR5 @ 204.8 GB/s")
        time.sleep(0.2)

        # Phase 6: Ready
        self.progress.emit(100, "System ready")
        self.log_message.emit("")
        self.log_message.emit("[SYS] =============================================")
        self.log_message.emit("[SYS] VISION AI Camera System initialized")
        self.log_message.emit(f"[SYS] Device: {results.get('device_name', 'Unknown')}")
        self.log_message.emit(f"[SYS] Model: {selected_model['name']}")
        self.log_message.emit("[SYS] Status: READY")
        self.log_message.emit("[SYS] =============================================")

        time.sleep(0.3)
        self.finished.emit(results)


class ModelSelectScreen(QWidget):
    """Model selection screen shown before loading."""

    model_selected = Signal(str)  # Emits model_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_model = "sdxl-turbo"
        self._setup_ui()

    def _setup_ui(self):
        """Setup model selection UI."""
        self.setStyleSheet(MODEL_SELECT_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        # ASCII logo
        logo = QLabel(VISION_ASCII)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("color: #2563eb; font-size: 12px;")
        layout.addWidget(logo)

        # Title
        title = QLabel("Select AI Model")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #e0e0e0; font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Choose which model to load (will download on first use)")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        # Model cards container
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setSpacing(12)

        self._model_buttons = {}
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        for model in MODEL_OPTIONS:
            card = QPushButton()
            card.setObjectName("model_card")
            card.setCheckable(True)
            card.setMinimumHeight(70)

            # Card content layout
            card_text = f"""
            <div style="text-align: left;">
                <span style="color: #e0e0e0; font-size: 13px; font-weight: bold;">{model['name']}</span><br>
                <span style="color: #888; font-size: 10px;">{model['description']}</span><br>
                <span style="color: #2563eb; font-size: 10px;">Size: {model['size']} | {model['speed']}</span>
            </div>
            """
            card.setText("")  # We'll use a label inside instead

            # Create a label for the card content
            card_label = QLabel(card_text)
            card_label.setStyleSheet("background: transparent; padding: 8px;")
            card_label.setAttribute(Qt.WA_TransparentForMouseEvents)

            card_inner_layout = QVBoxLayout(card)
            card_inner_layout.setContentsMargins(0, 0, 0, 0)
            card_inner_layout.addWidget(card_label)

            # Connect click
            card.clicked.connect(lambda checked, m=model["id"]: self._on_model_selected(m))

            self._button_group.addButton(card)
            self._model_buttons[model["id"]] = card
            cards_layout.addWidget(card)

        # Select default
        self._model_buttons["sdxl-turbo"].setChecked(True)

        layout.addWidget(cards_container)

        layout.addStretch()

        # Start button
        self._start_btn = QPushButton("Load Model & Start")
        self._start_btn.setObjectName("start_btn")
        self._start_btn.setFixedHeight(44)
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        # Hint
        hint = QLabel("Models are cached after first download")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #444; font-size: 10px; margin-top: 8px;")
        layout.addWidget(hint)

    def _on_model_selected(self, model_id: str):
        """Handle model card click."""
        self._selected_model = model_id
        for mid, btn in self._model_buttons.items():
            btn.setChecked(mid == model_id)

    def _on_start(self):
        """Handle start button click."""
        self.model_selected.emit(self._selected_model)


class LoadingScreen(QWidget):
    """Combined model selection + loading screen."""

    loading_complete = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_model_id = "sdxl-turbo"
        self._worker = None
        self._thread = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the combined UI with stacked screens."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stack for model select -> loading screens
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Screen 0: Model selection
        self._model_select = ModelSelectScreen()
        self._model_select.model_selected.connect(self._on_model_chosen)
        self._stack.addWidget(self._model_select)

        # Screen 1: Loading terminal
        self._loading_widget = QWidget()
        self._loading_widget.setStyleSheet(LOADING_STYLE)
        self._setup_loading_ui()
        self._stack.addWidget(self._loading_widget)

        # Start on model selection
        self._stack.setCurrentIndex(0)

    def _setup_loading_ui(self):
        """Setup the loading screen UI."""
        layout = QVBoxLayout(self._loading_widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(10)

        # ASCII logo
        self._logo = QLabel(VISION_ASCII)
        self._logo.setAlignment(Qt.AlignCenter)
        self._logo.setStyleSheet("color: #2563eb; font-size: 12px;")
        layout.addWidget(self._logo)

        # Subtitle
        subtitle = QLabel("AI-Powered Camera System")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        # Terminal output
        self._terminal = QTextEdit()
        self._terminal.setReadOnly(True)
        self._terminal.setMinimumHeight(400)
        layout.addWidget(self._terminal, stretch=1)

        # Progress bar (ASCII style)
        self._progress_label = QLabel("[                              ] 0%")
        self._progress_label.setAlignment(Qt.AlignCenter)
        self._progress_label.setStyleSheet("color: #00ff00; font-size: 11px;")
        layout.addWidget(self._progress_label)

        # Status
        self._status = QLabel("Initializing...")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._status)

    def start_loading(self):
        """Show model selection screen (called from main window)."""
        # Just show the model selection - actual loading starts after selection
        self._stack.setCurrentIndex(0)

    def _on_model_chosen(self, model_id: str):
        """Handle model selection and start loading."""
        self._selected_model_id = model_id
        self._stack.setCurrentIndex(1)  # Switch to loading screen

        self._log("VISION AI Camera System v1.0")
        self._log("=" * 50)

        # Start worker thread with selected model
        self._thread = QThread()
        self._worker = SystemCheckWorker(selected_model_id=model_id)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self._log)
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._on_complete)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    def _log(self, message: str):
        """Add a log message to the terminal."""
        cursor = self._terminal.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Color code based on prefix
        if message.startswith("[GPU]"):
            color = "#00ffff"  # Cyan
        elif message.startswith("[MDL]"):
            color = "#ff00ff"  # Magenta
        elif message.startswith("[SIM]"):
            color = "#ffff00"  # Yellow
        elif message.startswith("[SYS]"):
            color = "#00ff00"  # Green
        elif message.startswith("[LIB]"):
            color = "#ff8800"  # Orange
        elif "ERROR" in message or "WARNING" in message:
            color = "#ff0000"  # Red
        else:
            color = "#00ff00"

        self._terminal.append(f'<span style="color: {color}">{message}</span>')
        self._terminal.verticalScrollBar().setValue(
            self._terminal.verticalScrollBar().maximum()
        )

    def _update_progress(self, percent: int, status: str):
        """Update progress bar."""
        filled = int(percent / 100 * 30)
        empty = 30 - filled
        bar = "█" * filled + "░" * empty
        self._progress_label.setText(f"[{bar}] {percent}%")
        self._status.setText(status)

    def _on_complete(self, results: dict):
        """Handle loading completion."""
        print(f"[DEBUG] Loading complete. Results keys: {results.keys()}")
        print(f"[DEBUG] Model loaded: {results.get('model_loaded')}")
        print(f"[DEBUG] Model ID: {results.get('model_id')}")
        print(f"[DEBUG] Pipeline type: {type(results.get('pipeline'))}")
        self.loading_complete.emit(results)
