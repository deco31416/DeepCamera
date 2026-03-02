# Skill Development Guide

This guide explains how to create a new skill for the DeepCamera skill catalog.

## What is a Skill?

A skill is a self-contained folder that provides an AI capability to [SharpAI Aegis](https://sharpai.org). Skills communicate with Aegis via **JSON lines** over stdin/stdout.

## Skill Structure

```
skills/<category>/<skill-name>/
├── SKILL.md              # Manifest + setup instructions
├── requirements.txt      # Python dependencies
├── scripts/
│   └── main.py           # Entry point
├── assets/
│   └── icon.png          # 64×64 icon (optional)
└── tests/
    └── test_main.py      # Tests (optional)
```

## SKILL.md Format

The `SKILL.md` file has two parts:
1. **YAML frontmatter** — machine-readable parameters and capabilities
2. **Markdown body** — human/LLM-readable setup instructions

```yaml
---
name: my-skill
description: "What this skill does"
version: 1.0.0

parameters:
  - name: model
    label: "Model"
    type: select
    options: ["option1", "option2"]
    default: "option1"
    group: Model

capabilities:
  my_capability:
    script: scripts/main.py
    description: "What this capability does"
---

# My Skill

Description of the skill.

## Setup

Step-by-step setup instructions that SharpAI Aegis's
LLM agent can read and execute.
```

## Parameter Types

| Type | Renders As | Example |
|------|-----------|---------|
| `string` | Text input | Email, URL, API key |
| `password` | Masked input | Passwords, tokens |
| `number` | Number input with min/max | Confidence threshold |
| `boolean` | Toggle switch | Enable/disable feature |
| `select` | Dropdown | Model selection |
| `url` | URL input with validation | Server address |
| `camera_select` | Camera picker | Target cameras |

## JSON Lines Protocol

Scripts communicate with Aegis via stdin/stdout. Each line is a JSON object.

### Script → Aegis (stdout)

```jsonl
{"event": "ready", "model": "...", "device": "..."}
{"event": "detections", "camera_id": "...", "objects": [...]}
{"event": "error", "message": "...", "retriable": true}
```

### Aegis → Script (stdin)

```jsonl
{"event": "frame", "camera_id": "...", "frame_path": "...", "timestamp": "..."}
{"command": "stop"}
```

## Categories

| Category | Directory | Use For |
|----------|-----------|---------|
| `detection` | `skills/detection/` | Object detection, person recognition |
| `analysis` | `skills/analysis/` | VLM scene understanding, offline analysis |
| `transformation` | `skills/transformation/` | Depth estimation, style transfer |
| `annotation` | `skills/annotation/` | Dataset labeling, COCO export |
| `camera-providers` | `skills/camera-providers/` | Blink, Eufy, Ring, Reolink, Tapo |
| `streaming` | `skills/streaming/` | RTSP/WebRTC via go2rtc |
| `channels` | `skills/channels/` | Messaging: Matrix, LINE, Signal |
| `automation` | `skills/automation/` | MQTT, webhooks, HA triggers |
| `integrations` | `skills/integrations/` | Home Assistant bridge |

## Testing Locally

```bash
# Test your skill without Aegis by piping JSON:
echo '{"event": "frame", "camera_id": "test", "frame_path": "/tmp/test.jpg"}' | python scripts/main.py
```

## Reference

See [`skills/detection/yolo-detection-2026/`](../skills/detection/yolo-detection-2026/) for a complete working example.
