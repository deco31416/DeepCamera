---
name: depth-estimation-cuda
description: "GPU-accelerated depth maps — TensorRT FP16 or PyTorch CUDA with Depth Anything v2"
version: 1.0.0
category: privacy

parameters:
  - name: model
    label: "Depth Model"
    type: select
    options: ["depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large"]
    default: "depth-anything-v2-small"
    group: Model

  - name: blend_mode
    label: "Display Mode"
    type: select
    options: ["depth_only", "overlay", "side_by_side"]
    default: "depth_only"
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
    options: ["inferno", "viridis", "plasma", "magma", "jet", "turbo", "hot", "cool"]
    default: "inferno"
    group: Display

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda"]
    default: "auto"
    group: Performance

capabilities:
  live_transform:
    script: scripts/transform.py
    description: "Real-time depth estimation overlay on live feed"
---

# 3D Depth Vision (CUDA)

GPU-accelerated monocular depth estimation using Depth Anything v2. Transforms camera feeds with colorized depth maps — near objects appear warm, far objects appear cool.

## Hardware Backends

| Priority | Backend | Runtime | Speed |
|----------|---------|---------|-------|
| 1 | TensorRT FP16 | NVIDIA TensorRT | ~7x faster |
| 2 | PyTorch CUDA | torch.cuda | Baseline |
| 3 | PyTorch CPU | torch (CPU) | Slow fallback |

## What You Get

- **Privacy anonymization** — depth-only mode hides all visual identity
- **Depth overlays** on live camera feeds
- **3D scene understanding** — spatial layout of the scene
- **TensorRT acceleration** — auto-builds FP16 engine for ~7x speedup

## Interface: TransformSkillBase

Implements the JSONL stdin/stdout protocol via `TransformSkillBase`.

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "frame_id": "cam1_1710001", "camera_id": "front_door", "frame_path": "/tmp/frame.jpg", "timestamp": "..."}
{"command": "config-update", "config": {"opacity": 0.8, "blend_mode": "overlay"}}
{"command": "stop"}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "depth-anything-v2-small", "device": "cuda", "backend": "tensorrt"}
{"event": "transform", "frame_id": "cam1_1710001", "camera_id": "front_door", "transform_data": "<base64 JPEG>"}
{"event": "perf_stats", "total_frames": 50, "timings_ms": {"transform": {"avg": 12.5, ...}}}
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```
