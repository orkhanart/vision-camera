#!/usr/bin/env python3
"""Download additional AI models for VISION simulator."""

import torch
import sys
import os

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype(device):
    return torch.float16 if device in ("cuda", "mps") else torch.float32

# ============================================================
# MODELS TO DOWNLOAD
# ============================================================

CONTROLNET_MODELS = [
    {
        "id": "controlnet-canny",
        "name": "ControlNet Canny (Edge Detection)",
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
    },
    {
        "id": "controlnet-depth",
        "name": "ControlNet Depth",
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
    },
    {
        "id": "controlnet-openpose",
        "name": "ControlNet OpenPose (Body Pose)",
        "repo_id": "thibaud/controlnet-openpose-sdxl-1.0",
    },
]

SEGMENTATION_MODELS = [
    {
        "id": "sam-vit-huge",
        "name": "Segment Anything (SAM) ViT-H",
        "repo_id": "facebook/sam-vit-huge",
        "type": "sam",
    },
    {
        "id": "mobilesam",
        "name": "MobileSAM (Fast)",
        "repo_id": "dhkim2810/MobileSAM",
        "type": "mobilesam",
    },
]

DEPTH_MODELS = [
    {
        "id": "midas-large",
        "name": "MiDaS Large (Depth Estimation)",
        "repo_id": "Intel/dpt-large",
        "type": "dpt",
    },
    {
        "id": "depth-anything",
        "name": "Depth Anything V2",
        "repo_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "type": "depth-anything",
    },
]

UPSCALER_MODELS = [
    {
        "id": "real-esrgan",
        "name": "Real-ESRGAN x4",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "type": "esrgan",
    },
]

IP_ADAPTER_MODELS = [
    {
        "id": "ip-adapter-sdxl",
        "name": "IP-Adapter SDXL",
        "repo_id": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "type": "ip-adapter",
    },
]

FACE_MODELS = [
    {
        "id": "codeformer",
        "name": "CodeFormer (Face Restoration)",
        "repo_id": "sczhou/codeformer",
        "type": "codeformer",
    },
]

DETECTION_MODELS = [
    {
        "id": "yolov8",
        "name": "YOLOv8 Medium",
        "type": "yolo",
    },
]

def download_controlnet(model_info, dtype):
    """Download ControlNet model."""
    print(f"\n  Loading {model_info['name']}...")
    from diffusers import ControlNetModel

    controlnet = ControlNetModel.from_pretrained(
        model_info["repo_id"],
        torch_dtype=dtype,
    )
    print(f"  ✓ {model_info['name']} cached")
    del controlnet
    return True

def download_sam(model_info):
    """Download SAM model."""
    print(f"\n  Loading {model_info['name']}...")
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(model_info["repo_id"])
    model = SamModel.from_pretrained(model_info["repo_id"])
    print(f"  ✓ {model_info['name']} cached")
    del processor, model
    return True

def download_depth(model_info, dtype):
    """Download depth estimation model."""
    print(f"\n  Loading {model_info['name']}...")
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_info["repo_id"])
    model = AutoModelForDepthEstimation.from_pretrained(model_info["repo_id"])
    print(f"  ✓ {model_info['name']} cached")
    del processor, model
    return True

def download_esrgan(model_info):
    """Download Real-ESRGAN model."""
    print(f"\n  Loading {model_info['name']}...")
    import urllib.request
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "vision" / "esrgan"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        print(f"  Downloading from {model_info['url']}...")
        urllib.request.urlretrieve(model_info["url"], model_path)

    print(f"  ✓ {model_info['name']} cached at {model_path}")
    return True

def download_ip_adapter(model_info, dtype):
    """Download IP-Adapter."""
    print(f"\n  Loading {model_info['name']}...")
    from huggingface_hub import hf_hub_download

    # Download the SDXL IP-Adapter
    hf_hub_download(
        repo_id=model_info["repo_id"],
        filename="sdxl_models/ip-adapter_sdxl.bin",
    )
    hf_hub_download(
        repo_id=model_info["repo_id"],
        filename="sdxl_models/image_encoder/config.json",
    )
    print(f"  ✓ {model_info['name']} cached")
    return True

def download_codeformer(model_info):
    """Download CodeFormer."""
    print(f"\n  Loading {model_info['name']}...")
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=model_info["repo_id"],
        filename="codeformer.pth",
    )
    print(f"  ✓ {model_info['name']} cached")
    return True

def download_yolo(model_info):
    """Download YOLOv8."""
    print(f"\n  Loading {model_info['name']}...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8m.pt")  # Downloads automatically
        print(f"  ✓ {model_info['name']} cached")
        del model
        return True
    except ImportError:
        print("  ⚠ ultralytics not installed. Run: pip install ultralytics")
        return False

def main():
    print("=" * 60)
    print("VISION Additional Models Downloader")
    print("=" * 60)

    device = get_device()
    dtype = get_dtype(device)
    print(f"\nDevice: {device} | Dtype: {dtype}")

    results = {}

    # ControlNet
    print("\n" + "=" * 60)
    print("CONTROLNET MODELS")
    print("=" * 60)
    for model in CONTROLNET_MODELS:
        try:
            results[model["id"]] = download_controlnet(model, dtype)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # Clear memory
    if device == "cuda":
        torch.cuda.empty_cache()

    # Segmentation (SAM)
    print("\n" + "=" * 60)
    print("SEGMENTATION MODELS")
    print("=" * 60)
    for model in SEGMENTATION_MODELS:
        try:
            results[model["id"]] = download_sam(model)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # Depth
    print("\n" + "=" * 60)
    print("DEPTH ESTIMATION MODELS")
    print("=" * 60)
    for model in DEPTH_MODELS:
        try:
            results[model["id"]] = download_depth(model, dtype)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # Upscaler
    print("\n" + "=" * 60)
    print("UPSCALER MODELS")
    print("=" * 60)
    for model in UPSCALER_MODELS:
        try:
            results[model["id"]] = download_esrgan(model)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # IP-Adapter
    print("\n" + "=" * 60)
    print("IP-ADAPTER MODELS")
    print("=" * 60)
    for model in IP_ADAPTER_MODELS:
        try:
            results[model["id"]] = download_ip_adapter(model, dtype)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # Face Restoration
    print("\n" + "=" * 60)
    print("FACE RESTORATION MODELS")
    print("=" * 60)
    for model in FACE_MODELS:
        try:
            results[model["id"]] = download_codeformer(model)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # Object Detection
    print("\n" + "=" * 60)
    print("OBJECT DETECTION MODELS")
    print("=" * 60)
    for model in DETECTION_MODELS:
        try:
            results[model["id"]] = download_yolo(model)
        except Exception as e:
            print(f"  ✗ {model['name']}: {e}")
            results[model["id"]] = False

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    success = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for model_id, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {model_id}")

    print(f"\nTotal: {success} succeeded, {failed} failed")

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
