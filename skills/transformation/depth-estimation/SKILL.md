---
name: depth-estimation
description: "Real-time depth map estimation using Depth Anything v2"
version: 1.0.0

parameters:
  - name: model
    label: "Depth Model"
    type: select
    options: ["depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large", "midas-small"]
    default: "depth-anything-v2-small"
    group: Model

  - name: blend_mode
    label: "Display Mode"
    type: select
    options: ["overlay", "side_by_side", "depth_only"]
    default: "overlay"
    group: Display

  - name: opacity
    label: "Overlay Opacity"
    type: number
    min: 0.0
    max: 1.0
    default: 0.5
    group: Display

  - name: colormap
    label: "Depth Colormap"
    type: select
    options: ["inferno", "viridis", "plasma", "magma", "jet"]
    default: "inferno"
    group: Display

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    group: Performance

capabilities:
  live_transform:
    script: scripts/transform.py
    description: "Real-time depth estimation overlay on live feed"
---

# Depth Estimation

Real-time monocular depth estimation using Depth Anything v2. Transforms camera feeds with colorized depth maps — near objects appear warm, far objects appear cool.

## What You Get

- **Depth overlays** on live camera feeds
- **Distance estimation** — approximate distance to detected objects
- **3D scene understanding** — spatial layout of the scene

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "camera_id": "front_door", "frame_path": "/tmp/frame.jpg", "timestamp": "..."}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "depth-anything-v2-small", "device": "mps"}
{"event": "transformed_frame", "camera_id": "front_door", "frame_path": "/tmp/depth_001.jpg", "metadata": {"min_depth": 0.2, "max_depth": 15.0}}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
