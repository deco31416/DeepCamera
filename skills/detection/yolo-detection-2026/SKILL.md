---
name: yolo-detection-2026
description: "State-of-the-art real-time object detection using YOLO"
version: 1.0.0
icon: assets/icon.png

parameters:
  - name: model
    label: "Model"
    type: select
    options: ["yolov11n", "yolov11s", "yolov11m", "yolov10n", "yolov10s", "yolov8n"]
    default: "yolov11n"
    group: Model

  - name: confidence
    label: "Confidence Threshold"
    type: number
    min: 0.1
    max: 1.0
    default: 0.5
    group: Model

  - name: classes
    label: "Detect Classes"
    type: string
    default: "person,car,dog,cat"
    description: "Comma-separated COCO class names (80 classes available)"
    group: Model

  - name: fps
    label: "Processing FPS"
    type: number
    min: 1
    max: 30
    default: 5
    group: Performance

  - name: device
    label: "Inference Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    description: "auto = GPU if available, else CPU"
    group: Performance

capabilities:
  live_detection:
    script: scripts/detect.py
    description: "Real-time object detection on live camera frames"
---

# YOLO Object Detection (2026)

Real-time object detection using state-of-the-art YOLO models. Detects 80+ COCO object classes including people, vehicles, animals, and everyday objects. Outputs bounding boxes with labels and confidence scores that SharpAI Aegis renders as overlays on the live camera feed.

## What You Get

When installed in SharpAI Aegis, this skill unlocks:
- **Live detection overlays** on camera feeds — bounding boxes around detected objects
- **Smart alert triggers** — configure alerts when specific objects are detected
- **Detection history** — searchable log of all detections

## Models

| Model | Size | Speed (FPS) | Accuracy (mAP) | Best For |
|-------|------|-------------|-----------------|----------|
| YOLOv11n | 6 MB | 30+ | 39.5 | Real-time on CPU |
| YOLOv11s | 22 MB | 20+ | 47.0 | Balanced |
| YOLOv11m | 68 MB | 12+ | 51.5 | High accuracy |
| YOLOv10n | 7 MB | 28+ | 38.5 | Ultra-fast |
| YOLOv10s | 24 MB | 18+ | 46.3 | Balanced (v10) |
| YOLOv8n | 6 MB | 30+ | 37.3 | Legacy compatible |

## Setup

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download model weights (automatic on first run, or manually):
   ```bash
   python scripts/download_models.py --model yolov11n
   ```

## Protocol

This skill communicates with SharpAI Aegis via **JSON lines** over stdin/stdout.

### Aegis → Skill (stdin): frames to process

```jsonl
{"event": "frame", "camera_id": "front_door", "timestamp": "2026-03-01T14:30:00Z", "frame_path": "/tmp/frame_001.jpg", "width": 1920, "height": 1080}
```

### Skill → Aegis (stdout): detection results

```jsonl
{"event": "ready", "model": "yolov11n", "device": "mps", "classes": 80}
{"event": "detections", "camera_id": "front_door", "timestamp": "2026-03-01T14:30:00Z", "objects": [
  {"class": "person", "confidence": 0.92, "bbox": [100, 50, 300, 400]},
  {"class": "car",    "confidence": 0.87, "bbox": [500, 200, 900, 500]}
]}
```

### Bounding Box Format

`[x_min, y_min, x_max, y_max]` in pixel coordinates.

## Hardware Requirements

| Device | Performance |
|--------|------------|
| Apple Silicon (M1+) | 20-30 FPS with MPS acceleration |
| NVIDIA GPU | 25-60 FPS with CUDA |
| CPU (modern x86) | 5-15 FPS |
| Raspberry Pi 5 | 2-5 FPS |

## Contributing

This skill is part of the [DeepCamera](https://github.com/SharpAI/DeepCamera) open-source project. Contributions welcome — see [Contributions.md](../../Contributions.md).
