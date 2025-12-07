# VISION Simulator

Desktop simulation app for the VISION AI Camera. Runs real diffusion models with simulated hardware constraints matching the Jetson AGX Orin 64GB.

## Features

- **Live Camera Input**: Uses laptop webcam as sensor simulation
- **Real AI Models**: Runs Stable Diffusion, FLUX.1, and style transfer models
- **Hardware Simulation**: Simulates Jetson Orin constraints (275 TOPS, 64GB memory, thermal limits)
- **Camera UI**: Mimics the 3" 1080x810 touch interface
- **Performance Profiling**: Tracks inference time, memory usage, thermal estimates

## Architecture

```
simulator/
├── src/
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── app.py              # Main application entry
│   │   ├── camera/             # Camera capture module
│   │   │   ├── __init__.py
│   │   │   └── capture.py      # Webcam/video input
│   │   ├── inference/          # AI inference pipeline
│   │   │   ├── __init__.py
│   │   │   ├── engine.py       # Inference orchestrator
│   │   │   ├── models.py       # Model loading/management
│   │   │   └── styles.py       # Style presets
│   │   ├── hardware/           # Hardware simulation
│   │   │   ├── __init__.py
│   │   │   ├── simulator.py    # Resource constraints
│   │   │   └── profiles.py     # Jetson Orin specs
│   │   └── ui/                 # User interface
│   │       ├── __init__.py
│   │       ├── main_window.py  # Main display window
│   │       ├── viewfinder.py   # Live camera view
│   │       ├── controls.py     # Touch controls
│   │       └── styles.qss      # UI styling
│   └── tests/
├── models/                     # Downloaded AI models
├── config/                     # Configuration files
├── pyproject.toml
└── README.md
```

## Hardware Specs Simulated

| Component | Spec | Simulation |
|-----------|------|------------|
| AI Processor | 275 TOPS | Inference timing throttle |
| Memory | 64GB LPDDR5 | Memory budget enforcement |
| Memory Bandwidth | 204 GB/s | Transfer time simulation |
| GPU Cores | 2048 CUDA | Parallel processing limits |
| Power Budget | 30W typical | Thermal accumulation model |

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the simulator
vision-sim

# Or with specific camera
vision-sim --camera 0
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format src/
```
