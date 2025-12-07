"""Style presets for AI art generation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StylePreset:
    """A predefined style configuration for AI art generation."""

    id: str
    name: str
    description: str
    prompt_prefix: str
    prompt_suffix: str
    negative_prompt: str
    recommended_steps: int = 20
    recommended_guidance: float = 7.5
    recommended_strength: float = 0.75  # For img2img
    thumbnail: Optional[str] = None  # Path to preview image

    def build_prompt(self, base_prompt: str = "") -> str:
        """Build complete prompt with style modifiers."""
        parts = []
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)
        if base_prompt:
            parts.append(base_prompt)
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)
        return ", ".join(parts)


# Pre-defined style presets for VISION camera
STYLE_PRESETS: dict[str, StylePreset] = {
    # Artistic Styles
    "impressionist": StylePreset(
        id="impressionist",
        name="Impressionist",
        description="Soft brushstrokes, light play, Monet-inspired",
        prompt_prefix="impressionist painting style",
        prompt_suffix="soft brushstrokes, dappled light, oil painting, artistic",
        negative_prompt="sharp, digital, photorealistic, modern",
        recommended_steps=25,
        recommended_guidance=7.0,
        recommended_strength=0.7,
    ),
    "watercolor": StylePreset(
        id="watercolor",
        name="Watercolor",
        description="Soft, flowing watercolor painting effect",
        prompt_prefix="watercolor painting",
        prompt_suffix="soft edges, flowing colors, paper texture, artistic watercolor",
        negative_prompt="sharp edges, digital, photorealistic, oil paint",
        recommended_steps=22,
        recommended_guidance=6.5,
        recommended_strength=0.72,
    ),
    "oil_painting": StylePreset(
        id="oil_painting",
        name="Oil Painting",
        description="Rich, textured oil painting style",
        prompt_prefix="oil painting masterpiece",
        prompt_suffix="visible brushstrokes, rich colors, canvas texture, classical art",
        negative_prompt="digital art, smooth, photographic, flat",
        recommended_steps=25,
        recommended_guidance=7.5,
        recommended_strength=0.75,
    ),
    "pencil_sketch": StylePreset(
        id="pencil_sketch",
        name="Pencil Sketch",
        description="Detailed pencil drawing style",
        prompt_prefix="detailed pencil sketch",
        prompt_suffix="graphite drawing, hatching, paper texture, artistic sketch",
        negative_prompt="color, painted, digital, photograph",
        recommended_steps=20,
        recommended_guidance=7.0,
        recommended_strength=0.68,
    ),
    "pop_art": StylePreset(
        id="pop_art",
        name="Pop Art",
        description="Bold colors, Andy Warhol inspired",
        prompt_prefix="pop art style",
        prompt_suffix="bold colors, halftone dots, high contrast, Warhol inspired",
        negative_prompt="muted colors, realistic, subtle, photographic",
        recommended_steps=20,
        recommended_guidance=8.0,
        recommended_strength=0.78,
    ),

    # Digital/Modern Styles
    "cyberpunk": StylePreset(
        id="cyberpunk",
        name="Cyberpunk",
        description="Neon-lit futuristic aesthetic",
        prompt_prefix="cyberpunk style",
        prompt_suffix="neon lights, rain, futuristic city, blade runner aesthetic, high tech",
        negative_prompt="nature, rural, historical, bright daylight, warm colors",
        recommended_steps=25,
        recommended_guidance=8.0,
        recommended_strength=0.8,
    ),
    "anime": StylePreset(
        id="anime",
        name="Anime",
        description="Japanese anime art style",
        prompt_prefix="anime style artwork",
        prompt_suffix="cel shading, vibrant colors, anime aesthetic, detailed eyes",
        negative_prompt="photorealistic, western art, 3d render, photograph",
        recommended_steps=22,
        recommended_guidance=7.5,
        recommended_strength=0.75,
    ),
    "pixel_art": StylePreset(
        id="pixel_art",
        name="Pixel Art",
        description="Retro 8-bit pixel art style",
        prompt_prefix="pixel art",
        prompt_suffix="8-bit style, retro gaming aesthetic, limited color palette, pixelated",
        negative_prompt="smooth, high resolution, photorealistic, anti-aliased",
        recommended_steps=18,
        recommended_guidance=8.5,
        recommended_strength=0.82,
    ),
    "vaporwave": StylePreset(
        id="vaporwave",
        name="Vaporwave",
        description="80s retro aesthetic with pink/purple tones",
        prompt_prefix="vaporwave aesthetic",
        prompt_suffix="pink and purple gradients, retro 80s, palm trees, sunset, synthwave",
        negative_prompt="modern, realistic colors, natural lighting",
        recommended_steps=22,
        recommended_guidance=7.5,
        recommended_strength=0.75,
    ),
    "low_poly": StylePreset(
        id="low_poly",
        name="Low Poly",
        description="Geometric low-polygon 3D art style",
        prompt_prefix="low poly art style",
        prompt_suffix="geometric shapes, triangular facets, 3d render, minimalist",
        negative_prompt="smooth, organic, photorealistic, detailed textures",
        recommended_steps=20,
        recommended_guidance=8.0,
        recommended_strength=0.78,
    ),

    # Photography Enhancement Styles
    "cinematic": StylePreset(
        id="cinematic",
        name="Cinematic",
        description="Movie-like dramatic lighting and color grading",
        prompt_prefix="cinematic photography",
        prompt_suffix="dramatic lighting, film grain, anamorphic, movie still, color graded",
        negative_prompt="flat lighting, amateur, oversaturated, snapshot",
        recommended_steps=20,
        recommended_guidance=6.5,
        recommended_strength=0.6,
    ),
    "noir": StylePreset(
        id="noir",
        name="Film Noir",
        description="Black and white with dramatic shadows",
        prompt_prefix="film noir style",
        prompt_suffix="black and white, high contrast, dramatic shadows, moody lighting",
        negative_prompt="color, bright, flat lighting, cheerful",
        recommended_steps=20,
        recommended_guidance=7.0,
        recommended_strength=0.7,
    ),
    "vintage": StylePreset(
        id="vintage",
        name="Vintage",
        description="Retro film photography look",
        prompt_prefix="vintage photograph",
        prompt_suffix="film grain, faded colors, light leaks, retro, 1970s aesthetic",
        negative_prompt="modern, digital, sharp, oversaturated",
        recommended_steps=18,
        recommended_guidance=6.0,
        recommended_strength=0.55,
    ),
    "hdr": StylePreset(
        id="hdr",
        name="HDR",
        description="High dynamic range with enhanced details",
        prompt_prefix="HDR photography",
        prompt_suffix="high dynamic range, enhanced details, vibrant colors, sharp",
        negative_prompt="flat, low contrast, muted, blurry",
        recommended_steps=18,
        recommended_guidance=6.0,
        recommended_strength=0.5,
    ),

    # Abstract/Artistic
    "abstract": StylePreset(
        id="abstract",
        name="Abstract",
        description="Non-representational abstract art",
        prompt_prefix="abstract art",
        prompt_suffix="non-representational, bold shapes, expressive colors, modern art",
        negative_prompt="realistic, figurative, photographic, detailed",
        recommended_steps=25,
        recommended_guidance=9.0,
        recommended_strength=0.85,
    ),
    "surreal": StylePreset(
        id="surreal",
        name="Surreal",
        description="Dreamlike surrealist imagery",
        prompt_prefix="surrealist art style",
        prompt_suffix="dreamlike, Dali inspired, impossible geometry, ethereal",
        negative_prompt="realistic, ordinary, mundane, photographic",
        recommended_steps=28,
        recommended_guidance=8.5,
        recommended_strength=0.8,
    ),
    "psychedelic": StylePreset(
        id="psychedelic",
        name="Psychedelic",
        description="Vibrant, trippy visual patterns",
        prompt_prefix="psychedelic art",
        prompt_suffix="vibrant colors, fractals, flowing patterns, trippy, kaleidoscopic",
        negative_prompt="muted colors, realistic, plain, simple",
        recommended_steps=25,
        recommended_guidance=9.0,
        recommended_strength=0.85,
    ),

    # Special Effects
    "miniature": StylePreset(
        id="miniature",
        name="Miniature",
        description="Tilt-shift miniature world effect",
        prompt_prefix="tilt-shift photography",
        prompt_suffix="miniature effect, selective focus, toy-like, diorama aesthetic",
        negative_prompt="full focus, realistic scale, sharp everywhere",
        recommended_steps=18,
        recommended_guidance=6.5,
        recommended_strength=0.6,
    ),
    "double_exposure": StylePreset(
        id="double_exposure",
        name="Double Exposure",
        description="Layered double exposure effect",
        prompt_prefix="double exposure photography",
        prompt_suffix="layered images, silhouette blend, artistic photography, ethereal",
        negative_prompt="single image, simple, clear separation",
        recommended_steps=22,
        recommended_guidance=7.5,
        recommended_strength=0.72,
    ),
    "glitch": StylePreset(
        id="glitch",
        name="Glitch Art",
        description="Digital glitch and distortion effects",
        prompt_prefix="glitch art style",
        prompt_suffix="digital artifacts, RGB shift, corrupted data, datamosh aesthetic",
        negative_prompt="clean, smooth, perfect, undistorted",
        recommended_steps=20,
        recommended_guidance=8.0,
        recommended_strength=0.78,
    ),
}


def get_style_categories() -> dict[str, list[str]]:
    """Get styles organized by category."""
    return {
        "Artistic": ["impressionist", "watercolor", "oil_painting", "pencil_sketch", "pop_art"],
        "Digital": ["cyberpunk", "anime", "pixel_art", "vaporwave", "low_poly"],
        "Photography": ["cinematic", "noir", "vintage", "hdr"],
        "Abstract": ["abstract", "surreal", "psychedelic"],
        "Effects": ["miniature", "double_exposure", "glitch"],
    }


def get_style_by_id(style_id: str) -> Optional[StylePreset]:
    """Get a style preset by ID."""
    return STYLE_PRESETS.get(style_id)
