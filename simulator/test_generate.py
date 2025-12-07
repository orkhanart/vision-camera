#!/usr/bin/env python3
"""Test the generate flow directly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PIL import Image
import numpy as np

from vision.inference.models import ModelManager
from vision.inference.engine import InferenceEngine, InferenceRequest
from vision.hardware.simulator import HardwareSimulator
from vision.hardware.profiles import JetsonOrinProfile


def test_inference():
    """Test inference outside of UI."""
    print("=" * 50)
    print("Testing VISION Inference Pipeline")
    print("=" * 50)

    # Create test image
    print("\n1. Creating test image...")
    test_img = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    print(f"   Test image: {test_img.size}")

    # Setup components
    print("\n2. Setting up components...")
    hw = HardwareSimulator(JetsonOrinProfile)
    print(f"   Hardware sim: {hw.profile.name}")

    mm = ModelManager()
    print(f"   Model manager: {len(mm._available_models)} models available")

    engine = InferenceEngine(mm, hw, default_model="sdxl-turbo")
    print(f"   Inference engine ready")

    # Create request
    print("\n3. Creating inference request...")
    request = InferenceRequest(
        image=test_img,
        style_id="cinematic",
        strength=0.75,
        steps=4,
        model_id="sdxl-turbo",
    )
    print(f"   Style: {request.style_id}")
    print(f"   Model: {request.model_id}")
    print(f"   Steps: {request.steps}")

    # Run inference
    print("\n4. Running inference...")
    print("   (Loading model if not cached...)")

    try:
        result = engine.run_inference_sync(request)
        print(f"\n SUCCESS!")
        print(f"   Output size: {result.image.size}")
        print(f"   Inference time: {result.inference_time_ms:.0f}ms")
        print(f"   Jetson estimate: {result.estimated_device_time_ms:.0f}ms")

        # Save result
        output_path = Path(__file__).parent / "test_output.png"
        result.image.save(output_path)
        print(f"   Saved to: {output_path}")

    except Exception as e:
        print(f"\n FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
