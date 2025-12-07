"""Model management for AI inference."""

import gc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch


class ModelType(Enum):
    """Types of AI models supported."""

    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "sdxl"
    FLUX = "flux"
    STYLE_TRANSFER = "style_transfer"
    UPSCALER = "upscaler"


@dataclass
class ModelInfo:
    """Information about an AI model."""

    id: str
    name: str
    type: ModelType
    repo_id: str  # HuggingFace repo ID
    variant: Optional[str] = None  # e.g., "fp16"
    revision: Optional[str] = None

    # Resource requirements
    memory_gb: float = 4.0
    compute_tops: float = 10.0  # TOPS required per inference

    # Model characteristics
    default_steps: int = 20
    default_guidance: float = 7.5
    supports_img2img: bool = True
    native_resolution: int = 512  # Native inference resolution

    # Status
    is_downloaded: bool = False
    local_path: Optional[Path] = None


# Pre-configured models for VISION
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    # Stable Diffusion 1.5 - Fast, lower memory
    "sd-1.5": ModelInfo(
        id="sd-1.5",
        name="Stable Diffusion 1.5",
        type=ModelType.STABLE_DIFFUSION,
        repo_id="runwayml/stable-diffusion-v1-5",
        variant="fp16",
        memory_gb=4.0,
        compute_tops=8.0,
        default_steps=20,
        native_resolution=512,
    ),
    # Stable Diffusion XL - Higher quality
    "sdxl": ModelInfo(
        id="sdxl",
        name="Stable Diffusion XL",
        type=ModelType.STABLE_DIFFUSION_XL,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        memory_gb=8.0,
        compute_tops=25.0,
        default_steps=25,
        default_guidance=7.0,
        native_resolution=1024,
    ),
    # SDXL Turbo - Faster SDXL
    "sdxl-turbo": ModelInfo(
        id="sdxl-turbo",
        name="SDXL Turbo",
        type=ModelType.STABLE_DIFFUSION_XL,
        repo_id="stabilityai/sdxl-turbo",
        memory_gb=8.0,
        compute_tops=20.0,
        default_steps=4,  # Turbo needs fewer steps
        default_guidance=0.0,  # No CFG for turbo
        native_resolution=512,
    ),
    # FLUX Schnell - Fast high quality
    "flux-schnell": ModelInfo(
        id="flux-schnell",
        name="FLUX.1 Schnell",
        type=ModelType.FLUX,
        repo_id="black-forest-labs/FLUX.1-schnell",
        memory_gb=12.0,
        compute_tops=20.0,
        default_steps=4,
        default_guidance=0.0,
        native_resolution=1024,
    ),
    # FLUX Dev - Higher quality
    "flux-dev": ModelInfo(
        id="flux-dev",
        name="FLUX.1 Dev",
        type=ModelType.FLUX,
        repo_id="black-forest-labs/FLUX.1-dev",
        memory_gb=16.0,
        compute_tops=35.0,
        default_steps=20,
        default_guidance=3.5,
        native_resolution=1024,
    ),
}


class ModelManager:
    """
    Manages AI model lifecycle - loading, caching, and memory management.

    Designed to work within VISION's 64GB memory budget, handling model
    swapping and memory pressure.
    """

    def __init__(
        self,
        models_dir: Path = Path("models"),
        max_memory_gb: float = 32.0,  # Model cache budget
        device: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_gb = max_memory_gb
        self.device = device or self._detect_device()

        # Loaded model cache
        self._loaded_models: dict[str, Any] = {}
        self._model_memory: dict[str, float] = {}  # Memory used per model
        self._current_memory_gb: float = 0.0

        # Available models registry
        self._available_models = AVAILABLE_MODELS.copy()

    def _detect_device(self) -> str:
        """Detect best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"

    def get_available_models(self) -> list[ModelInfo]:
        """Get list of all available models."""
        return list(self._available_models.values())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info for a specific model."""
        return self._available_models.get(model_id)

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        return model_id in self._loaded_models

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return list(self._loaded_models.keys())

    def get_memory_usage(self) -> float:
        """Get current model memory usage in GB."""
        return self._current_memory_gb

    async def load_model(self, model_id: str, force_reload: bool = False) -> Any:
        """
        Load a model into memory.

        Handles memory management, unloading other models if needed.

        Args:
            model_id: ID of model to load
            force_reload: Force reload even if already loaded

        Returns:
            Loaded model pipeline
        """
        if model_id in self._loaded_models and not force_reload:
            return self._loaded_models[model_id]

        model_info = self._available_models.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        # Check if we need to free memory
        if self._current_memory_gb + model_info.memory_gb > self.max_memory_gb:
            await self._free_memory(model_info.memory_gb)

        # Load based on model type
        pipeline = await self._load_pipeline(model_info)

        # Track in cache
        self._loaded_models[model_id] = pipeline
        self._model_memory[model_id] = model_info.memory_gb
        self._current_memory_gb += model_info.memory_gb

        return pipeline

    def load_model_sync(self, model_id: str, force_reload: bool = False) -> Any:
        """Synchronous version of load_model."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.load_model(model_id, force_reload))
        finally:
            loop.close()

    async def _load_pipeline(self, model_info: ModelInfo) -> Any:
        """Load the appropriate pipeline for a model."""
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            FluxPipeline,
        )

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        if model_info.type == ModelType.STABLE_DIFFUSION:
            # Load both txt2img and img2img pipelines
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_info.repo_id,
                torch_dtype=dtype,
                variant=model_info.variant,
                safety_checker=None,  # Disable for speed
            )
            pipeline = pipeline.to(self.device)

            # Enable memory optimizations
            if self.device == "cuda":
                pipeline.enable_attention_slicing()

        elif model_info.type == ModelType.STABLE_DIFFUSION_XL:
            pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_info.repo_id,
                torch_dtype=dtype,
                variant=model_info.variant,
            )
            pipeline = pipeline.to(self.device)

            if self.device == "cuda":
                pipeline.enable_attention_slicing()

        elif model_info.type == ModelType.FLUX:
            pipeline = FluxPipeline.from_pretrained(
                model_info.repo_id,
                torch_dtype=dtype,
            )
            pipeline = pipeline.to(self.device)

        else:
            raise ValueError(f"Unsupported model type: {model_info.type}")

        return pipeline

    async def _free_memory(self, required_gb: float) -> None:
        """Free memory by unloading models until we have enough space."""
        # Simple FIFO eviction - could be improved with LRU
        models_to_unload = []
        freed = 0.0

        for model_id in list(self._loaded_models.keys()):
            if freed >= required_gb:
                break
            models_to_unload.append(model_id)
            freed += self._model_memory.get(model_id, 0)

        for model_id in models_to_unload:
            await self.unload_model(model_id)

    async def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id not in self._loaded_models:
            return

        # Delete pipeline
        del self._loaded_models[model_id]

        # Update memory tracking
        memory = self._model_memory.pop(model_id, 0)
        self._current_memory_gb = max(0, self._current_memory_gb - memory)

        # Force garbage collection
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def unload_model_sync(self, model_id: str) -> None:
        """Synchronous version of unload_model."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.unload_model(model_id))
        finally:
            loop.close()

    async def unload_all(self) -> None:
        """Unload all models."""
        for model_id in list(self._loaded_models.keys()):
            await self.unload_model(model_id)

    def get_status(self) -> dict:
        """Get model manager status."""
        return {
            "device": self.device,
            "memory_used_gb": round(self._current_memory_gb, 2),
            "memory_max_gb": self.max_memory_gb,
            "loaded_models": list(self._loaded_models.keys()),
            "available_models": list(self._available_models.keys()),
        }
