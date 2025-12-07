#!/usr/bin/env python3
"""Download all AI models for VISION simulator."""

import torch
import sys

def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype(device):
    """Get appropriate dtype for device."""
    return torch.float16 if device in ("cuda", "mps") else torch.float32

MODELS = [
    {
        "id": "sd-1.5",
        "name": "Stable Diffusion 1.5",
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionImg2ImgPipeline",
        "variant": "fp16",
    },
    {
        "id": "sdxl",
        "name": "Stable Diffusion XL",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLImg2ImgPipeline",
        "variant": "fp16",
    },
    {
        "id": "sdxl-turbo",
        "name": "SDXL Turbo",
        "repo_id": "stabilityai/sdxl-turbo",
        "pipeline": "StableDiffusionXLImg2ImgPipeline",
        "variant": None,
    },
    {
        "id": "flux-schnell",
        "name": "FLUX.1 Schnell",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "variant": None,
    },
    {
        "id": "flux-dev",
        "name": "FLUX.1 Dev",
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "variant": None,
    },
]

def download_model(model_info, device, dtype):
    """Download a single model."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_info['name']}")
    print(f"Repository: {model_info['repo_id']}")
    print(f"{'='*60}")

    try:
        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            StableDiffusionXLImg2ImgPipeline,
            FluxPipeline,
        )

        pipeline_class = {
            "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
            "StableDiffusionXLImg2ImgPipeline": StableDiffusionXLImg2ImgPipeline,
            "FluxPipeline": FluxPipeline,
        }[model_info["pipeline"]]

        kwargs = {
            "torch_dtype": dtype,
        }
        if model_info["variant"]:
            kwargs["variant"] = model_info["variant"]

        if model_info["pipeline"] == "StableDiffusionImg2ImgPipeline":
            kwargs["safety_checker"] = None

        print(f"Loading pipeline...")
        pipeline = pipeline_class.from_pretrained(
            model_info["repo_id"],
            **kwargs
        )

        print(f"SUCCESS: {model_info['name']} downloaded and cached!")

        # Clean up to free memory
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"ERROR downloading {model_info['name']}: {e}")
        return False

def main():
    print("="*60)
    print("VISION AI Model Downloader")
    print("="*60)

    device = get_device()
    dtype = get_dtype(device)

    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"\nModels to download: {len(MODELS)}")
    print("This will download models to HuggingFace cache (~50GB total)")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")

    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    results = {}
    for model in MODELS:
        success = download_model(model, device, dtype)
        results[model["id"]] = success

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for model_id, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model_id}: {status}")

    failed = sum(1 for s in results.values() if not s)
    if failed:
        print(f"\n{failed} model(s) failed to download.")
        sys.exit(1)
    else:
        print(f"\nAll {len(MODELS)} models downloaded successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
