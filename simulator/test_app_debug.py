#!/usr/bin/env python3
"""Debug test for the full app flow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

def run_debug():
    app = QApplication(sys.argv)

    from vision.ui.main_window import VisionMainWindow

    window = VisionMainWindow(camera_id=0, models_dir=Path("models"))
    window.show()

    # Add debug after loading completes
    def debug_state():
        print("\n" + "=" * 50)
        print("DEBUG: App State")
        print("=" * 50)
        if hasattr(window, 'model_manager'):
            print(f"Model manager loaded models: {window.model_manager.get_loaded_models()}")
            print(f"Memory used: {window.model_manager.get_memory_usage()} GB")
        if hasattr(window, '_camera_available'):
            print(f"Camera available: {window._camera_available}")
        if hasattr(window, '_selected_model_id'):
            print(f"Selected model: {window._selected_model_id}")
        if hasattr(window, '_current_style'):
            print(f"Current style: {window._current_style}")
        if hasattr(window, '_captured_frame'):
            print(f"Captured frame: {window._captured_frame}")
        if hasattr(window, '_loaded_image'):
            print(f"Loaded image: {window._loaded_image}")
        print("=" * 50 + "\n")

    # Run debug after 10 seconds (after model loads)
    QTimer.singleShot(10000, debug_state)
    # Run again after 20 seconds
    QTimer.singleShot(20000, debug_state)

    return app.exec()

if __name__ == "__main__":
    sys.exit(run_debug())
