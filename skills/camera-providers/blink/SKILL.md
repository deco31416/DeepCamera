---
name: camera-provider-blink
description: "Amazon Blink camera integration — clip feed, snapshots, arm/disarm"
version: 1.0.0

parameters:
  - name: email
    label: "Blink Account Email"
    type: string
    required: true
    group: Account

  - name: password
    label: "Blink Account Password"
    type: password
    required: true
    group: Account

  - name: poll_interval
    label: "Poll Interval (seconds)"
    type: number
    min: 30
    max: 3600
    default: 300
    group: Sync

capabilities:
  clip_feed:
    script: scripts/feed.py
    description: "Continuous clip download from Blink cameras"
  discover_cameras:
    script: scripts/discover.py
    description: "List available Blink cameras"
  snapshot:
    script: scripts/snapshot.py
    description: "Capture current frame"
  arm_disarm:
    script: scripts/arm.py
    description: "Arm/disarm Blink system"
---

# Blink Camera Provider

Downloads motion clips from Amazon Blink cameras into SharpAI Aegis for AI analysis. Supports multi-camera systems with automatic deduplication.

## Protocol

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "cameras": [{"id": "403785", "name": "Front Door", "type": "mini"}]}
{"event": "clip", "camera_id": "403785", "camera_name": "Front Door", "clip_id": "blink_403785_1709312400", "timestamp": "2026-03-01T14:00:00Z", "duration_seconds": 30, "video_path": "/tmp/clip.mp4", "thumbnail_path": "/tmp/thumb.jpg"}
```

### Aegis → Skill (stdin)
```jsonl
{"event": "query_existing", "camera_id": "403785", "since": "2026-03-01T00:00:00Z"}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
