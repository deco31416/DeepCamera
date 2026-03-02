---
name: person-recognition
description: "Person re-identification — track and recognize individuals across cameras"
version: 1.0.0

parameters:
  - name: model
    label: "ReID Model"
    type: select
    options: ["mgn-r50", "osnet-ain", "bot-r50"]
    default: "mgn-r50"
    group: Model

  - name: similarity_threshold
    label: "Match Threshold"
    type: number
    min: 0.3
    max: 0.95
    default: 0.7
    group: Model

  - name: gallery_size
    label: "Max Gallery Size"
    type: number
    min: 10
    max: 1000
    default: 100
    description: "Maximum number of known identities to track"
    group: Model

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    group: Performance

capabilities:
  live_detection:
    script: scripts/detect.py
    description: "Person re-identification on detected crops"
---

# Person Recognition (ReID)

Tracks and re-identifies individuals across cameras and over time. When a person is detected, this skill extracts an appearance embedding and matches it against a gallery of known identities.

## What You Get

- **Cross-camera tracking** — recognize the same person across different cameras
- **Identity gallery** — save and label known individuals
- **Re-appearance alerts** — "Person X was last seen 2 hours ago at the front door"

## Wraps

This skill builds on DeepCamera's original `src/yolov7_reid/` module.

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "camera_id": "...", "frame_path": "/tmp/frame.jpg", "detections": [{"class": "person", "bbox": [100, 50, 300, 400]}]}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "mgn-r50", "gallery_size": 0}
{"event": "detections", "camera_id": "...", "objects": [
  {"class": "person", "bbox": [100, 50, 300, 400], "identity": "delivery_driver", "confidence": 0.85, "track_id": "p1"}
]}
{"event": "new_identity", "identity_id": "unknown_003", "snapshot_path": "/tmp/crop.jpg"}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
