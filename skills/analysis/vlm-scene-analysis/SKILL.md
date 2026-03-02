---
name: vlm-scene-analysis
description: "Offline scene understanding of recorded clips using vision language models"
version: 1.0.0

parameters:
  - name: model
    label: "VLM Model"
    type: select
    options: ["smolvlm2-500m", "qwen2.5-vl-3b", "gemma-3-4b", "llava-1.6-7b"]
    default: "smolvlm2-500m"
    group: Model

  - name: prompt
    label: "Analysis Prompt"
    type: string
    default: "Describe what is happening in this security camera footage. Focus on people, vehicles, and any unusual activity."
    group: Model

  - name: auto_analyze
    label: "Auto-analyze New Clips"
    type: boolean
    default: true
    group: Behavior

  - name: max_frames
    label: "Frames per Clip"
    type: number
    min: 1
    max: 16
    default: 4
    description: "Number of frames sampled from each clip for analysis"
    group: Performance

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    group: Performance

capabilities:
  offline_analysis:
    script: scripts/analyze.py
    description: "VLM analysis of recorded video clips"
---

# VLM Scene Analysis

Offline scene understanding using vision language models. Analyzes recorded clips to generate natural language descriptions of what happened — people, vehicles, activities, and anomalies.

## What You Get

- **Clip descriptions** — "Delivery driver places package at front door, rings doorbell, walks back to van"
- **Searchable** — search your clips by description: "find clips with dogs in the backyard"
- **Timeline badges** — analysis results shown on the timeline in SharpAI Aegis

## Models

| Model | Size | Speed | Quality | VRAM |
|-------|------|-------|---------|------|
| SmolVLM2 500M | 1 GB | ⚡ Fast | Good | 2 GB |
| Qwen2.5-VL 3B | 6 GB | Medium | Very Good | 6 GB |
| Gemma-3 4B | 8 GB | Medium | Very Good | 8 GB |
| LLaVA 1.6 7B | 14 GB | Slow | Excellent | 12 GB |

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "clip_ready", "clip_id": "blink_403785_1709312400", "video_path": "/path/to/clip.mp4", "camera_id": "front_door", "camera_name": "Front Door", "duration_seconds": 15}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "smolvlm2-500m", "device": "mps"}
{"event": "analysis_result", "clip_id": "blink_403785_1709312400", "description": "A delivery driver in a brown uniform approaches the front door carrying a medium-sized package. They place it on the porch, ring the doorbell, and return to their van parked in the driveway.", "objects": ["person", "package", "van"], "confidence": 0.9}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py --model smolvlm2-500m
```
