"""Hardware profiles for simulation targets."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class PowerMode(Enum):
    """Jetson power modes."""

    MAX_PERFORMANCE = "MAXN"      # 60W, 275 TOPS
    BALANCED = "30W"              # 30W, ~200 TOPS
    POWER_SAVE = "15W"            # 15W, ~100 TOPS


@dataclass
class ThermalProfile:
    """Thermal characteristics."""

    ambient_temp_c: float = 25.0
    max_temp_c: float = 95.0
    throttle_temp_c: float = 85.0
    thermal_resistance: float = 0.5  # °C/W - heat sink efficiency
    thermal_mass: float = 50.0       # J/°C - thermal capacitance


@dataclass
class MemoryProfile:
    """Memory specifications."""

    total_gb: float = 64.0
    bandwidth_gbps: float = 204.0
    type: str = "LPDDR5"

    # Reserved allocations
    os_reserved_gb: float = 6.0
    model_cache_gb: float = 32.0
    inference_buffer_gb: float = 12.0
    camera_pipeline_gb: float = 6.0
    ui_cache_gb: float = 8.0

    @property
    def available_gb(self) -> float:
        """Memory available for dynamic allocation."""
        reserved = (
            self.os_reserved_gb
            + self.model_cache_gb
            + self.inference_buffer_gb
            + self.camera_pipeline_gb
            + self.ui_cache_gb
        )
        return self.total_gb - reserved


@dataclass
class ComputeProfile:
    """Compute specifications."""

    tops: float = 275.0              # INT8 TOPS
    fp16_tflops: float = 138.0       # FP16 TFLOPS
    fp32_tflops: float = 69.0        # FP32 TFLOPS
    cuda_cores: int = 2048
    tensor_cores: int = 64
    cpu_cores: int = 12
    cpu_type: str = "ARM Cortex-A78AE"


@dataclass
class StorageProfile:
    """Storage specifications."""

    internal_gb: float = 512.0
    internal_type: str = "UFS 3.1"
    read_speed_mbps: float = 2100.0
    write_speed_mbps: float = 1200.0

    # Partitioning
    os_partition_gb: float = 32.0
    models_partition_gb: float = 100.0
    cache_partition_gb: float = 50.0
    user_partition_gb: float = 50.0


@dataclass
class PowerProfile:
    """Power specifications."""

    battery_wh: float = 150.0
    max_draw_w: float = 60.0
    typical_draw_w: float = 40.0

    # Component power breakdown (watts)
    ai_processor_w: float = 30.0
    display_w: float = 3.0
    camera_sensor_w: float = 1.0
    stabilization_w: float = 2.0
    connectivity_w: float = 2.0
    system_logic_w: float = 1.0
    cooling_w: float = 1.0


@dataclass
class DisplayProfile:
    """Display specifications."""

    width_px: int = 1080
    height_px: int = 810
    diagonal_inch: float = 3.0
    type: str = "IPS LCD"
    brightness_nits: int = 1000
    touch_points: int = 10
    refresh_hz: int = 60


@dataclass
class CameraProfile:
    """Camera sensor specifications."""

    sensor_model: str = "Sony IMX989"
    megapixels: float = 50.3
    width_px: int = 8192
    height_px: int = 6144
    sensor_size_mm: tuple = (13.2, 8.8)  # 1-inch sensor
    pixel_size_um: float = 1.6
    max_fps: int = 30
    hdr_support: bool = True
    phase_detect_af: bool = True


@dataclass
class HardwareProfile:
    """Complete hardware profile for simulation."""

    name: str = "Generic"
    thermal: ThermalProfile = field(default_factory=ThermalProfile)
    memory: MemoryProfile = field(default_factory=MemoryProfile)
    compute: ComputeProfile = field(default_factory=ComputeProfile)
    storage: StorageProfile = field(default_factory=StorageProfile)
    power: PowerProfile = field(default_factory=PowerProfile)
    display: DisplayProfile = field(default_factory=DisplayProfile)
    camera: CameraProfile = field(default_factory=CameraProfile)
    power_mode: PowerMode = PowerMode.BALANCED

    def get_effective_tops(self) -> float:
        """Get effective TOPS based on power mode."""
        mode_factors = {
            PowerMode.MAX_PERFORMANCE: 1.0,
            PowerMode.BALANCED: 0.73,
            PowerMode.POWER_SAVE: 0.36,
        }
        return self.compute.tops * mode_factors[self.power_mode]

    def estimate_inference_time_ms(self, model_tops_required: float) -> float:
        """Estimate inference time for a model."""
        effective_tops = self.get_effective_tops()
        # Rough estimate: time = ops / (tops * efficiency)
        efficiency = 0.7  # Account for memory bandwidth, etc.
        return (model_tops_required / (effective_tops * efficiency)) * 1000


# Pre-configured Jetson Orin profile
JetsonOrinProfile = HardwareProfile(
    name="NVIDIA Jetson AGX Orin 64GB",
    thermal=ThermalProfile(
        ambient_temp_c=25.0,
        max_temp_c=95.0,
        throttle_temp_c=85.0,
        thermal_resistance=0.4,
        thermal_mass=60.0,
    ),
    memory=MemoryProfile(
        total_gb=64.0,
        bandwidth_gbps=204.0,
        type="LPDDR5",
        os_reserved_gb=6.0,
        model_cache_gb=32.0,
        inference_buffer_gb=12.0,
        camera_pipeline_gb=6.0,
        ui_cache_gb=8.0,
    ),
    compute=ComputeProfile(
        tops=275.0,
        fp16_tflops=138.0,
        fp32_tflops=69.0,
        cuda_cores=2048,
        tensor_cores=64,
        cpu_cores=12,
        cpu_type="ARM Cortex-A78AE",
    ),
    storage=StorageProfile(
        internal_gb=512.0,
        internal_type="UFS 3.1",
        read_speed_mbps=2100.0,
        write_speed_mbps=1200.0,
    ),
    power=PowerProfile(
        battery_wh=150.0,
        max_draw_w=60.0,
        typical_draw_w=40.0,
        ai_processor_w=30.0,
    ),
    display=DisplayProfile(
        width_px=1080,
        height_px=810,
        diagonal_inch=3.0,
        type="IPS LCD",
        brightness_nits=1000,
    ),
    camera=CameraProfile(
        sensor_model="Sony IMX989",
        megapixels=50.3,
        width_px=8192,
        height_px=6144,
    ),
    power_mode=PowerMode.BALANCED,
)


# Model compute requirements (rough TOPS estimates for single inference)
MODEL_COMPUTE_REQUIREMENTS: Dict[str, float] = {
    "sd-1.5": 8.0,           # Stable Diffusion 1.5 (512x512)
    "sdxl": 25.0,            # Stable Diffusion XL (1024x1024)
    "flux-schnell": 20.0,    # FLUX.1 Schnell
    "flux-dev": 35.0,        # FLUX.1 Dev
    "style-transfer": 2.0,   # Real-time style transfer
    "upscale-2x": 5.0,       # AI upscaling
}
