"""VISION Simulator application entry point."""

import sys
import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VISION AI Camera Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vision-sim                    # Run with default camera
  vision-sim --camera 1         # Use camera index 1
  vision-sim --models ./models  # Use custom models directory
        """,
    )

    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )

    parser.add_argument(
        "--models", "-m",
        type=Path,
        default=Path("models"),
        help="Directory for AI models (default: ./models)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Window width (default: 1200)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Window height (default: 900)",
    )

    parser.add_argument(
        "--fullscreen", "-f",
        action="store_true",
        help="Start in fullscreen mode",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create models directory if needed
    args.models.mkdir(parents=True, exist_ok=True)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("VISION Simulator")
    app.setOrganizationName("VISION")

    # Enable high DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Import main window (after QApplication is created)
    from .ui.main_window import VisionMainWindow

    # Create and show main window
    window = VisionMainWindow(
        camera_id=args.camera,
        models_dir=args.models,
    )

    if args.fullscreen:
        window.showFullScreen()
    else:
        window.resize(args.width, args.height)
        window.show()

    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
