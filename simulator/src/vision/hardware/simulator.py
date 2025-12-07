"""Hardware constraint simulator for VISION camera."""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import psutil

from .profiles import HardwareProfile, JetsonOrinProfile, MODEL_COMPUTE_REQUIREMENTS


@dataclass
class SimulationState:
    """Current state of hardware simulation."""

    # Memory tracking (GB)
    memory_used_gb: float = 0.0
    memory_model_cache_gb: float = 0.0
    memory_inference_gb: float = 0.0

    # Thermal state
    temperature_c: float = 25.0
    is_throttling: bool = False

    # Power state
    power_draw_w: float = 10.0
    battery_remaining_wh: float = 150.0

    # Performance metrics
    last_inference_ms: float = 0.0
    total_inferences: int = 0
    avg_inference_ms: float = 0.0

    # GPU utilization (simulated)
    gpu_utilization: float = 0.0


class HardwareSimulator:
    """Simulates Jetson Orin hardware constraints for development."""

    def __init__(
        self,
        profile: HardwareProfile = JetsonOrinProfile,
        enforce_limits: bool = True,
    ):
        self.profile = profile
        self.enforce_limits = enforce_limits
        self.state = SimulationState(
            battery_remaining_wh=profile.power.battery_wh
        )
        self._lock = Lock()
        self._last_update = time.time()

        # Track loaded models
        self._loaded_models: dict[str, float] = {}  # model_id -> memory_gb

    def allocate_memory(self, size_gb: float, category: str = "inference") -> bool:
        """
        Attempt to allocate memory.

        Returns True if allocation succeeded, False if would exceed limits.
        """
        with self._lock:
            available = self.profile.memory.total_gb - self.state.memory_used_gb

            if self.enforce_limits and size_gb > available:
                return False

            self.state.memory_used_gb += size_gb

            if category == "model":
                self.state.memory_model_cache_gb += size_gb
            elif category == "inference":
                self.state.memory_inference_gb += size_gb

            return True

    def free_memory(self, size_gb: float, category: str = "inference") -> None:
        """Free allocated memory."""
        with self._lock:
            self.state.memory_used_gb = max(0, self.state.memory_used_gb - size_gb)

            if category == "model":
                self.state.memory_model_cache_gb = max(
                    0, self.state.memory_model_cache_gb - size_gb
                )
            elif category == "inference":
                self.state.memory_inference_gb = max(
                    0, self.state.memory_inference_gb - size_gb
                )

    def load_model(self, model_id: str, memory_gb: float) -> bool:
        """
        Simulate loading a model into memory.

        Returns True if model was loaded, False if insufficient memory.
        """
        if model_id in self._loaded_models:
            return True  # Already loaded

        if self.allocate_memory(memory_gb, "model"):
            self._loaded_models[model_id] = memory_gb
            return True
        return False

    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id in self._loaded_models:
            memory_gb = self._loaded_models.pop(model_id)
            self.free_memory(memory_gb, "model")

    def estimate_inference_time(
        self,
        model_id: str,
        steps: int = 20,
        batch_size: int = 1,
    ) -> float:
        """
        Estimate inference time in milliseconds for a model.

        Accounts for:
        - Model compute requirements
        - Current power mode / available TOPS
        - Thermal throttling
        - Memory bandwidth constraints
        """
        base_tops = MODEL_COMPUTE_REQUIREMENTS.get(model_id, 10.0)

        # Adjust for steps and batch
        total_compute = base_tops * steps * batch_size

        # Get effective compute power
        effective_tops = self.profile.get_effective_tops()

        # Apply thermal throttling
        if self.state.is_throttling:
            effective_tops *= 0.7

        # Base inference time
        efficiency = 0.65  # Real-world efficiency factor
        inference_ms = (total_compute / (effective_tops * efficiency)) * 1000

        # Add memory transfer overhead
        # Assume ~2GB transfer per inference at bandwidth limit
        memory_overhead_ms = (2.0 / self.profile.memory.bandwidth_gbps) * 1000 * steps

        return inference_ms + memory_overhead_ms

    def simulate_inference(
        self,
        model_id: str,
        actual_time_ms: float,
        power_multiplier: float = 1.0,
    ) -> None:
        """
        Update simulation state after an inference.

        Call this after running actual inference to update thermal/power state.
        """
        with self._lock:
            # Update inference metrics
            self.state.last_inference_ms = actual_time_ms
            self.state.total_inferences += 1

            # Running average
            n = self.state.total_inferences
            self.state.avg_inference_ms = (
                (self.state.avg_inference_ms * (n - 1) + actual_time_ms) / n
            )

            # Update thermal state
            self._update_thermal(actual_time_ms, power_multiplier)

            # Update power state
            self._update_power(actual_time_ms, power_multiplier)

    def _update_thermal(self, inference_ms: float, power_mult: float) -> None:
        """Update thermal simulation."""
        now = time.time()
        dt = now - self._last_update

        # Heat generated during inference
        power_w = self.profile.power.ai_processor_w * power_mult
        heat_j = power_w * (inference_ms / 1000)

        # Temperature rise
        temp_rise = heat_j / self.profile.thermal.thermal_mass

        # Cooling (passive dissipation)
        temp_diff = self.state.temperature_c - self.profile.thermal.ambient_temp_c
        cooling = temp_diff * self.profile.thermal.thermal_resistance * dt

        # Update temperature
        self.state.temperature_c = max(
            self.profile.thermal.ambient_temp_c,
            self.state.temperature_c + temp_rise - cooling
        )

        # Check throttling
        self.state.is_throttling = (
            self.state.temperature_c >= self.profile.thermal.throttle_temp_c
        )

        self._last_update = now

    def _update_power(self, inference_ms: float, power_mult: float) -> None:
        """Update power/battery simulation."""
        # Total power during inference
        power_w = self.profile.power.typical_draw_w * power_mult
        self.state.power_draw_w = power_w

        # Energy consumed (Wh)
        energy_wh = power_w * (inference_ms / 1000) / 3600
        self.state.battery_remaining_wh = max(
            0, self.state.battery_remaining_wh - energy_wh
        )

    def update_idle(self) -> None:
        """Update simulation during idle time (cooling, etc.)."""
        with self._lock:
            now = time.time()
            dt = now - self._last_update

            # Idle power draw
            idle_power_w = (
                self.profile.power.display_w
                + self.profile.power.camera_sensor_w
                + self.profile.power.system_logic_w
            )
            self.state.power_draw_w = idle_power_w

            # Passive cooling
            temp_diff = self.state.temperature_c - self.profile.thermal.ambient_temp_c
            cooling = temp_diff * self.profile.thermal.thermal_resistance * dt * 0.5
            self.state.temperature_c = max(
                self.profile.thermal.ambient_temp_c,
                self.state.temperature_c - cooling
            )

            # Clear throttling if cooled down
            if self.state.temperature_c < self.profile.thermal.throttle_temp_c - 5:
                self.state.is_throttling = False

            # Idle battery drain
            energy_wh = idle_power_w * dt / 3600
            self.state.battery_remaining_wh = max(
                0, self.state.battery_remaining_wh - energy_wh
            )

            # Reset GPU utilization
            self.state.gpu_utilization = 0.0

            self._last_update = now

    def get_battery_percent(self) -> float:
        """Get remaining battery percentage."""
        return (self.state.battery_remaining_wh / self.profile.power.battery_wh) * 100

    def get_memory_percent(self) -> float:
        """Get memory usage percentage."""
        return (self.state.memory_used_gb / self.profile.memory.total_gb) * 100

    def get_estimated_runtime_hours(self) -> float:
        """Estimate remaining runtime based on current power draw."""
        if self.state.power_draw_w <= 0:
            return float("inf")
        return self.state.battery_remaining_wh / self.state.power_draw_w

    def reset(self) -> None:
        """Reset simulation to initial state."""
        with self._lock:
            self.state = SimulationState(
                battery_remaining_wh=self.profile.power.battery_wh
            )
            self._loaded_models.clear()
            self._last_update = time.time()

    def get_host_stats(self) -> dict:
        """Get actual host machine stats for comparison."""
        return {
            "host_memory_percent": psutil.virtual_memory().percent,
            "host_cpu_percent": psutil.cpu_percent(interval=0.1),
            "host_memory_gb": psutil.virtual_memory().used / (1024**3),
        }

    def get_status_dict(self) -> dict:
        """Get complete status as dictionary for UI display."""
        return {
            "memory": {
                "used_gb": round(self.state.memory_used_gb, 2),
                "total_gb": self.profile.memory.total_gb,
                "percent": round(self.get_memory_percent(), 1),
                "model_cache_gb": round(self.state.memory_model_cache_gb, 2),
                "inference_gb": round(self.state.memory_inference_gb, 2),
            },
            "thermal": {
                "temperature_c": round(self.state.temperature_c, 1),
                "max_temp_c": self.profile.thermal.max_temp_c,
                "throttle_temp_c": self.profile.thermal.throttle_temp_c,
                "is_throttling": self.state.is_throttling,
            },
            "power": {
                "draw_w": round(self.state.power_draw_w, 1),
                "battery_percent": round(self.get_battery_percent(), 1),
                "runtime_hours": round(self.get_estimated_runtime_hours(), 2),
            },
            "compute": {
                "effective_tops": round(self.profile.get_effective_tops(), 1),
                "gpu_utilization": round(self.state.gpu_utilization, 1),
                "power_mode": self.profile.power_mode.value,
            },
            "inference": {
                "last_ms": round(self.state.last_inference_ms, 1),
                "avg_ms": round(self.state.avg_inference_ms, 1),
                "total_count": self.state.total_inferences,
            },
        }
