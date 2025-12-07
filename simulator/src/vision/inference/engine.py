"""AI inference engine for VISION camera."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from .models import ModelManager, ModelInfo, ModelType
from .styles import StylePreset, STYLE_PRESETS, get_style_by_id
from ..hardware.simulator import HardwareSimulator


@dataclass
class InferenceRequest:
    """Request for AI inference."""

    image: Image.Image
    style_id: str
    prompt: str = ""
    negative_prompt: str = ""
    steps: Optional[int] = None
    guidance: Optional[float] = None
    strength: float = 0.75  # How much to transform (0=none, 1=full)
    seed: Optional[int] = None
    model_id: str = "sdxl-turbo"  # Default to fast model


@dataclass
class InferenceResult:
    """Result of AI inference."""

    image: Image.Image
    inference_time_ms: float
    model_used: str
    style_applied: str
    steps_used: int
    seed_used: int
    estimated_device_time_ms: float  # What it would take on Jetson


class InferenceEngine:
    """
    AI inference engine for VISION camera.

    Handles image-to-image transformation using diffusion models,
    with style presets and hardware simulation.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        hardware_sim: Optional[HardwareSimulator] = None,
        default_model: str = "sdxl-turbo",
    ):
        self.model_manager = model_manager
        self.hardware_sim = hardware_sim
        self.default_model = default_model

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_request: Optional[InferenceRequest] = None
        self._is_processing = False

        # Callbacks
        self._on_progress: Optional[Callable[[int, int], None]] = None
        self._on_complete: Optional[Callable[[InferenceResult], None]] = None

    @property
    def is_processing(self) -> bool:
        """Check if currently processing an inference."""
        return self._is_processing

    def set_progress_callback(
        self, callback: Optional[Callable[[int, int], None]]
    ) -> None:
        """Set callback for progress updates (step, total_steps)."""
        self._on_progress = callback

    def set_complete_callback(
        self, callback: Optional[Callable[[InferenceResult], None]]
    ) -> None:
        """Set callback for completion."""
        self._on_complete = callback

    async def run_inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Run AI inference on an image.

        Args:
            request: Inference request with image, style, and parameters

        Returns:
            InferenceResult with transformed image and metrics
        """
        self._is_processing = True
        self._current_request = request

        try:
            # Get style preset
            style = get_style_by_id(request.style_id)
            if style is None:
                style = STYLE_PRESETS.get("cinematic")  # Default fallback

            # Get model info
            model_info = self.model_manager.get_model_info(request.model_id)
            if model_info is None:
                model_info = self.model_manager.get_model_info(self.default_model)

            # Determine parameters
            steps = request.steps or style.recommended_steps
            guidance = request.guidance if request.guidance is not None else style.recommended_guidance
            strength = request.strength or style.recommended_strength

            # Handle turbo models (fewer steps, no CFG)
            if "turbo" in request.model_id.lower():
                steps = min(steps, 4)
                guidance = 0.0
                # Turbo models work best with higher strength
                strength = max(strength, 0.5)

            # Generate seed
            seed = request.seed if request.seed is not None else int(time.time() * 1000) % (2**32)

            # Build prompt
            prompt = style.build_prompt(request.prompt)
            negative_prompt = request.negative_prompt or style.negative_prompt

            print(f"[DEBUG] Built prompt: {prompt}", flush=True)
            print(f"[DEBUG] Negative prompt: {negative_prompt}", flush=True)
            print(f"[DEBUG] Strength: {strength}, Guidance: {guidance}, Steps: {steps}", flush=True)

            # Estimate Jetson inference time
            estimated_device_time = 0.0
            if self.hardware_sim:
                estimated_device_time = self.hardware_sim.estimate_inference_time(
                    request.model_id,
                    steps=steps,
                )

            # Load model if needed
            pipeline = await self.model_manager.load_model(request.model_id)

            # Prepare image
            input_image = self._prepare_image(request.image, model_info)

            # Run inference
            start_time = time.time()
            result_image = await self._run_pipeline(
                pipeline=pipeline,
                model_info=model_info,
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                strength=strength,
                seed=seed,
            )
            inference_time = (time.time() - start_time) * 1000

            # Update hardware simulation
            if self.hardware_sim:
                self.hardware_sim.simulate_inference(
                    request.model_id,
                    inference_time,
                )

            result = InferenceResult(
                image=result_image,
                inference_time_ms=inference_time,
                model_used=request.model_id,
                style_applied=request.style_id,
                steps_used=steps,
                seed_used=seed,
                estimated_device_time_ms=estimated_device_time,
            )

            if self._on_complete:
                self._on_complete(result)

            return result

        finally:
            self._is_processing = False
            self._current_request = None

    def run_inference_sync(self, request: InferenceRequest) -> InferenceResult:
        """Synchronous version of run_inference."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_inference(request))
        finally:
            loop.close()

    def _prepare_image(
        self,
        image: Image.Image,
        model_info: ModelInfo,
    ) -> Image.Image:
        """Prepare image for model input."""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model's native resolution
        target_size = model_info.native_resolution

        # Maintain aspect ratio with padding
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        # Pad to square if needed
        if image.size[0] != target_size or image.size[1] != target_size:
            padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            offset = (
                (target_size - image.size[0]) // 2,
                (target_size - image.size[1]) // 2,
            )
            padded.paste(image, offset)
            image = padded

        return image

    async def _run_pipeline(
        self,
        pipeline,
        model_info: ModelInfo,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
    ) -> Image.Image:
        """Run the diffusion pipeline."""
        generator = torch.Generator(device=self.model_manager.device)
        generator.manual_seed(seed)

        # Progress callback wrapper
        def progress_callback(pipe, step, timestep, callback_kwargs):
            if self._on_progress:
                self._on_progress(step + 1, steps)
            return callback_kwargs

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        if model_info.type in (ModelType.STABLE_DIFFUSION, ModelType.STABLE_DIFFUSION_XL):
            # img2img pipeline
            result = await loop.run_in_executor(
                self._executor,
                lambda: pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=progress_callback,
                ).images[0],
            )
        elif model_info.type == ModelType.FLUX:
            # FLUX uses different API
            result = await loop.run_in_executor(
                self._executor,
                lambda: pipeline(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    height=model_info.native_resolution,
                    width=model_info.native_resolution,
                ).images[0],
            )
        else:
            raise ValueError(f"Unsupported model type: {model_info.type}")

        return result

    def cancel(self) -> None:
        """Attempt to cancel current inference."""
        # Note: Actual cancellation is complex with diffusers
        # This sets a flag that could be checked
        self._current_request = None

    def get_available_styles(self) -> list[StylePreset]:
        """Get all available style presets."""
        return list(STYLE_PRESETS.values())

    def get_style_categories(self) -> dict[str, list[StylePreset]]:
        """Get styles organized by category."""
        from .styles import get_style_categories

        categories = get_style_categories()
        result = {}
        for category, style_ids in categories.items():
            result[category] = [
                STYLE_PRESETS[sid] for sid in style_ids if sid in STYLE_PRESETS
            ]
        return result

    def estimate_time(
        self,
        model_id: str,
        steps: int,
        on_device: bool = True,
    ) -> float:
        """
        Estimate inference time in milliseconds.

        Args:
            model_id: Model to estimate for
            steps: Number of inference steps
            on_device: If True, estimate for Jetson; if False, for host machine

        Returns:
            Estimated time in milliseconds
        """
        if on_device and self.hardware_sim:
            return self.hardware_sim.estimate_inference_time(model_id, steps)

        # Rough host estimates (varies wildly by hardware)
        model_info = self.model_manager.get_model_info(model_id)
        if model_info is None:
            return 5000.0  # Default 5 seconds

        base_time_per_step = {
            ModelType.STABLE_DIFFUSION: 50,      # ms per step
            ModelType.STABLE_DIFFUSION_XL: 200,  # ms per step
            ModelType.FLUX: 250,                  # ms per step
        }.get(model_info.type, 100)

        return base_time_per_step * steps

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "is_processing": self._is_processing,
            "default_model": self.default_model,
            "available_styles": len(STYLE_PRESETS),
            "model_manager": self.model_manager.get_status(),
        }
