---
name: yolo-detection-2026-coral-tpu
description: "Google Coral Edge TPU — real-time object detection with LiteRT"
version: 2.0.0
icon: assets/icon.png
entry: scripts/detect.py
deploy: deploy.sh

requirements:
  python: ">=3.9"
  ai-edge-litert: ">=2.1.0"
  system: "libedgetpu"
  platforms: ["linux", "macos", "windows"]

parameters:
  - name: auto_start
    label: "Auto Start"
    type: boolean
    default: false
    description: "Start this skill automatically when Aegis launches"
    group: Lifecycle

  - name: confidence
    label: "Confidence Threshold"
    type: number
    min: 0.1
    max: 1.0
    default: 0.5
    description: "Minimum detection confidence — lower than GPU models due to INT8 quantization"
    group: Model

  - name: classes
    label: "Detect Classes"
    type: string
    default: "person,car,dog,cat"
    description: "Comma-separated COCO class names (80 classes available)"
    group: Model

  - name: fps
    label: "Processing FPS"
    type: select
    options: [0.2, 0.5, 1, 3, 5, 15]
    default: 5
    description: "Frames per second — Edge TPU handles 15+ FPS easily"
    group: Performance

  - name: input_size
    label: "Input Resolution"
    type: select
    options: [320, 640]
    default: 320
    description: "320 fits fully on TPU (~4ms), 640 partially on CPU (~20ms)"
    group: Performance

  - name: tpu_device
    label: "TPU Device"
    type: select
    options: ["auto", "0", "1", "2", "3"]
    default: "auto"
    description: "Which Edge TPU to use — auto selects first available"
    group: Performance

  - name: clock_speed
    label: "TPU Clock Speed"
    type: select
    options: ["standard", "max"]
    default: "standard"
    description: "Max is faster but runs hotter — needs active cooling for sustained use"
    group: Performance

capabilities:
  live_detection:
    script: scripts/detect.py
    description: "Real-time object detection on live camera frames via Edge TPU"

category: detection
mutex: detection
---

# Coral TPU Object Detection

Real-time object detection using Google Coral Edge TPU accelerator. Uses [ai-edge-litert](https://pypi.org/project/ai-edge-litert/) (LiteRT — the modern successor to TFLite/pycoral) with the `libedgetpu` delegate for hardware acceleration. Detects 80 COCO classes with ~4ms inference on 320×320 input.

## Requirements

- **Google Coral USB Accelerator** (USB 3.0 port recommended)
- **Python 3.9+** (3.9–3.13 supported via ai-edge-litert)
- **libedgetpu** system library (installed per platform, see below)
- Adequate cooling for sustained inference

## How It Works

1. `deploy.sh` creates a Python venv, installs `ai-edge-litert`, and checks for `libedgetpu`
2. Aegis writes camera frame JPEG to shared `/tmp/aegis_detection/` directory
3. Sends `frame` event via stdin JSONL to `detect.py`
4. `detect.py` loads the Edge TPU delegate via `litert.load_delegate('libedgetpu')`
5. Returns `detections` event via stdout JSONL
6. Same protocol as `yolo-detection-2026` — Aegis sees no difference

## Platform Setup

### Linux
```bash
# 1. Install libedgetpu (deploy.sh auto-installs on Debian/Ubuntu)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install libedgetpu1-std

# 2. Connect Coral USB Accelerator, then:
./deploy.sh
```

### macOS
```bash
# 1. Install libedgetpu
curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
unzip edgetpu_runtime_20221024.zip && cd edgetpu_runtime && sudo bash install.sh

# 2. Connect Coral USB Accelerator, then:
./deploy.sh
```

### Windows
```powershell
# 1. Install libedgetpu
# Download edgetpu_runtime_20221024.zip from:
#   https://github.com/google-coral/libedgetpu/releases
# Extract and run install.bat

# 2. Connect Coral USB Accelerator, then:
.\deploy.bat
```

## Model

Ships with pre-compiled `yolo26n_edgetpu.tflite` (YOLO 2026 nano, INT8 quantized, 320×320). To compile custom models:

```bash
python scripts/compile_model.py --model yolo26s --size 320
```

## Performance

| Input Size | Inference | On-chip | Notes |
|-----------|-----------|---------|-------|
| 320×320 | ~4ms | 100% | Fully on TPU, best for real-time |
| 640×640 | ~20ms | Partial | Some layers on CPU (model segmented) |

> **Cooling**: The USB Accelerator aluminum case acts as a heatsink. If too hot to touch during continuous inference, it will thermal-throttle. Consider active cooling or `clock_speed: standard`.

## Protocol

Same JSONL as `yolo-detection-2026`:

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "yolo26n_edgetpu", "device": "coral", "format": "edgetpu_tflite", "runtime": "ai-edge-litert", "tpu_count": 1, "classes": 80}
{"event": "detections", "frame_id": 42, "camera_id": "front_door", "objects": [{"class": "person", "confidence": 0.85, "bbox": [100, 50, 300, 400]}]}
{"event": "perf_stats", "total_frames": 50, "timings_ms": {"inference": {"avg": 4.1, "p50": 3.9, "p95": 5.2}}}
```

### Bounding Box Format
`[x_min, y_min, x_max, y_max]` — pixel coordinates (xyxy).

## Installation

```bash
./deploy.sh
```

The deployer creates a Python venv, installs `ai-edge-litert` and dependencies, checks for the `libedgetpu` system library, probes for Edge TPU devices, and sets the native `run_command`.

### Docker (Optional)

A `Dockerfile` is provided for environments where native installation is impractical. Note: Docker requires USB/IP passthrough for Edge TPU access on macOS and Windows.
