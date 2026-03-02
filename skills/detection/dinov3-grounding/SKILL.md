---
name: dinov3-grounding
description: "Open-vocabulary object detection using DINOv3 visual grounding"
version: 1.0.0

parameters:
  - name: model
    label: "Model"
    type: select
    options: ["dinov3-base", "dinov3-large"]
    default: "dinov3-base"
    group: Model

  - name: prompt
    label: "Detection Prompt"
    type: string
    default: "person . car . dog . cat"
    description: "Dot-separated object names to detect (open vocabulary)"
    group: Model

  - name: box_threshold
    label: "Box Threshold"
    type: number
    min: 0.1
    max: 1.0
    default: 0.3
    group: Model

  - name: text_threshold
    label: "Text Threshold"
    type: number
    min: 0.1
    max: 1.0
    default: 0.25
    group: Model

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    group: Performance

capabilities:
  live_detection:
    script: scripts/ground.py
    description: "Open-vocabulary grounding detection on live frames"
  feature_extraction:
    script: scripts/extract_features.py
    description: "Patch-level DINOv3 feature extraction for similarity search"
---

# DINOv3 Visual Grounding

Open-vocabulary object detection — describe what you want to find in natural language, and DINOv3 locates it. Unlike YOLO (fixed 80 classes), DINOv3 can detect **anything you describe**.

## Use Cases

- "person carrying a package" → bounding box around delivery driver
- "red car" → detects only red cars, ignores others
- "dog . cat . bird" → multi-class open-vocabulary detection
- Annotation assistance — click a region, get patch-level feature similarity

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "camera_id": "front_door", "frame_path": "/tmp/frame.jpg", "timestamp": "..."}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "dinov3-base", "device": "mps"}
{"event": "detections", "camera_id": "front_door", "objects": [
  {"class": "person carrying package", "confidence": 0.87, "bbox": [100, 50, 300, 400]}
]}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
