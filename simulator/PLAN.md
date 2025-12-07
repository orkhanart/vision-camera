# VISION Simulator - Fix Plan

## Issues to Fix (from Audit)

### Medium Severity

#### 1. Bare `except` clause in `loading_screen.py:195`
**File**: `src/vision/ui/loading_screen.py`
**Line**: 195
**Fix**: Replace bare `except:` with `except Exception:`
```python
# Before
except:
    memory_gb = 8.0

# After
except Exception:
    memory_gb = 8.0
```

#### 2. Deprecated Qt Attributes in `app.py:74-75`
**File**: `src/vision/app.py`
**Lines**: 74-75
**Fix**: Remove deprecated High DPI attributes (no-op in Qt6)
```python
# Remove these lines:
app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
```

#### 3. Missing `sdxl-turbo` in MODEL_COMPUTE_REQUIREMENTS
**File**: `src/vision/hardware/profiles.py`
**Line**: ~220
**Fix**: Add sdxl-turbo entry to the dict
```python
MODEL_COMPUTE_REQUIREMENTS: Dict[str, float] = {
    "sd-1.5": 8.0,
    "sdxl": 25.0,
    "sdxl-turbo": 20.0,  # Add this
    "flux-schnell": 20.0,
    "flux-dev": 35.0,
    "style-transfer": 2.0,
    "upscale-2x": 5.0,
}
```

#### 4. Race Condition Risk in Camera Start
**File**: `src/vision/camera/capture.py`
**Lines**: 154-159
**Fix**: Use QEventLoop with timeout instead of blocking sleep
```python
# Before
for _ in range(10):
    if self._is_running:
        return True
    time.sleep(0.1)

# After
from PySide6.QtCore import QEventLoop, QTimer

loop = QEventLoop()
QTimer.singleShot(1000, loop.quit)  # 1 second timeout
self._worker.started.connect(loop.quit)
loop.exec()
return self._is_running
```

### Low Severity

#### 5. Direct Modification of Private Attributes
**File**: `src/vision/inference/models.py`
**Fix**: Add public method `register_preloaded_model()`
```python
def register_preloaded_model(self, model_id: str, pipeline: Any) -> None:
    """Register a pre-loaded pipeline (e.g., from loading screen)."""
    model_info = self.get_model_info(model_id)
    if model_info is None:
        raise ValueError(f"Unknown model: {model_id}")

    self._loaded_models[model_id] = pipeline
    self._model_memory[model_id] = model_info.memory_gb
    self._current_memory_gb += model_info.memory_gb
```

**File**: `src/vision/ui/main_window.py`
**Lines**: 220-222
**Fix**: Use the new public method
```python
# Before
self.model_manager._loaded_models[model_id] = pipeline
self.model_manager._model_memory[model_id] = model_info.memory_gb
self.model_manager._current_memory_gb += model_info.memory_gb

# After
self.model_manager.register_preloaded_model(model_id, pipeline)
```

#### 6. Print statements -> Logging
**Files**: `src/vision/camera/capture.py`
**Fix**: Replace print() with logging
```python
import logging

logger = logging.getLogger(__name__)

# Line 64
logger.info(f"Camera opened: {w}x{h}")

# Line 186
logger.error(f"Camera error: {msg}")
```

#### 7. Unused Import
**File**: `src/vision/ui/main_window.py`
**Line**: 21
**Fix**: Remove `QGraphicsDropShadowEffect` from imports

#### 8. Hardcoded Window Size Conflict
**File**: `src/vision/ui/main_window.py`
**Line**: 159
**Fix**: Accept size parameters and use them
```python
def __init__(
    self,
    camera_id: int = 0,
    models_dir: Path = Path("models"),
    width: int = 1024,
    height: int = 1024,
    fixed_size: bool = True,
):
    ...
    if fixed_size:
        self.setFixedSize(width, height)
    else:
        self.resize(width, height)
```

**File**: `src/vision/app.py`
**Fix**: Pass dimensions to window
```python
window = VisionMainWindow(
    camera_id=args.camera,
    models_dir=args.models,
    width=args.width,
    height=args.height,
    fixed_size=not args.fullscreen,
)
```

#### 9. Magic Numbers -> Named Constants
**File**: `src/vision/ui/main_window.py`
**Fix**: Add constants at module level
```python
# UI Layout Constants
TOP_BAR_HEIGHT = 32
BOTTOM_CONTROLS_HEIGHT = 140
WINDOW_SIZE = 1024
```

**File**: `src/vision/hardware/simulator.py`
**Fix**: Add constants
```python
# Simulation Constants
THROTTLE_EFFICIENCY_FACTOR = 0.7
REAL_WORLD_EFFICIENCY = 0.65
THROTTLE_HYSTERESIS_C = 5.0
```

## Implementation Order

1. **Quick fixes** (5 min):
   - Remove deprecated Qt attributes
   - Fix bare except clause
   - Add sdxl-turbo to compute requirements
   - Remove unused import

2. **Encapsulation fix** (10 min):
   - Add `register_preloaded_model()` to ModelManager
   - Update main_window.py to use it

3. **Logging setup** (10 min):
   - Add logging to camera/capture.py
   - Replace print statements

4. **Camera start improvement** (15 min):
   - Refactor to use QEventLoop

5. **Window size fix** (10 min):
   - Add parameters to VisionMainWindow
   - Update app.py to pass them

6. **Constants extraction** (10 min):
   - Extract magic numbers to named constants

## Total Estimated Changes

- **Files modified**: 6
- **Lines changed**: ~50
