"""Tests for hardware simulation module."""

import pytest
from vision.hardware.profiles import (
    HardwareProfile,
    JetsonOrinProfile,
    PowerMode,
    MODEL_COMPUTE_REQUIREMENTS,
)
from vision.hardware.simulator import HardwareSimulator


class TestHardwareProfile:
    """Tests for HardwareProfile."""

    def test_jetson_profile_defaults(self):
        """Test Jetson Orin profile has correct defaults."""
        profile = JetsonOrinProfile

        assert profile.name == "NVIDIA Jetson AGX Orin 64GB"
        assert profile.compute.tops == 275.0
        assert profile.memory.total_gb == 64.0
        assert profile.display.width_px == 1080
        assert profile.display.height_px == 810

    def test_effective_tops_by_power_mode(self):
        """Test effective TOPS calculation by power mode."""
        profile = JetsonOrinProfile

        # Max performance
        profile.power_mode = PowerMode.MAX_PERFORMANCE
        assert profile.get_effective_tops() == 275.0

        # Balanced
        profile.power_mode = PowerMode.BALANCED
        assert profile.get_effective_tops() == pytest.approx(200.75, rel=0.01)

        # Power save
        profile.power_mode = PowerMode.POWER_SAVE
        assert profile.get_effective_tops() == pytest.approx(99.0, rel=0.01)

    def test_inference_time_estimation(self):
        """Test inference time estimation."""
        profile = JetsonOrinProfile
        profile.power_mode = PowerMode.BALANCED

        # Should return reasonable time in ms
        time_ms = profile.estimate_inference_time_ms(20.0)
        assert time_ms > 0
        assert time_ms < 10000  # Less than 10 seconds


class TestHardwareSimulator:
    """Tests for HardwareSimulator."""

    def test_memory_allocation(self):
        """Test memory allocation tracking."""
        sim = HardwareSimulator()

        # Initial state
        assert sim.state.memory_used_gb == 0.0

        # Allocate memory
        assert sim.allocate_memory(8.0) is True
        assert sim.state.memory_used_gb == 8.0

        # Free memory
        sim.free_memory(4.0)
        assert sim.state.memory_used_gb == 4.0

    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced."""
        sim = HardwareSimulator(enforce_limits=True)

        # Try to allocate more than available
        assert sim.allocate_memory(100.0) is False
        assert sim.state.memory_used_gb == 0.0

    def test_model_loading(self):
        """Test model loading simulation."""
        sim = HardwareSimulator()

        # Load a model
        assert sim.load_model("test-model", 8.0) is True
        assert "test-model" in sim._loaded_models
        assert sim.state.memory_model_cache_gb == 8.0

        # Load same model again (should skip)
        assert sim.load_model("test-model", 8.0) is True

        # Unload model
        sim.unload_model("test-model")
        assert "test-model" not in sim._loaded_models
        assert sim.state.memory_model_cache_gb == 0.0

    def test_inference_time_estimation(self):
        """Test inference time estimation."""
        sim = HardwareSimulator()

        time_ms = sim.estimate_inference_time("sdxl-turbo", steps=4)
        assert time_ms > 0
        assert time_ms < 5000  # Should be under 5 seconds

    def test_thermal_simulation(self):
        """Test thermal simulation updates."""
        sim = HardwareSimulator()
        initial_temp = sim.state.temperature_c

        # Simulate inference (should heat up)
        sim.simulate_inference("sdxl", 1000.0, power_multiplier=1.5)
        assert sim.state.temperature_c > initial_temp

    def test_battery_simulation(self):
        """Test battery drain simulation."""
        sim = HardwareSimulator()
        initial_battery = sim.state.battery_remaining_wh

        # Simulate inference (should drain battery)
        sim.simulate_inference("sdxl", 2000.0)
        assert sim.state.battery_remaining_wh < initial_battery

    def test_status_dict(self):
        """Test status dictionary generation."""
        sim = HardwareSimulator()
        status = sim.get_status_dict()

        assert "memory" in status
        assert "thermal" in status
        assert "power" in status
        assert "compute" in status
        assert "inference" in status

        assert status["memory"]["total_gb"] == 64.0
        assert status["compute"]["effective_tops"] > 0

    def test_reset(self):
        """Test simulator reset."""
        sim = HardwareSimulator()

        # Make some changes
        sim.allocate_memory(10.0)
        sim.load_model("test", 5.0)
        sim.simulate_inference("sd-1.5", 500.0)

        # Reset
        sim.reset()

        assert sim.state.memory_used_gb == 0.0
        assert len(sim._loaded_models) == 0
        assert sim.state.total_inferences == 0
