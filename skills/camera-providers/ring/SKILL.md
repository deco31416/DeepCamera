---
name: camera-provider-ring
description: "Ring camera integration — event clips and live view"
version: 1.0.0

parameters:
  - name: refresh_token
    label: "Ring Refresh Token"
    type: password
    required: true
    description: "Obtain via ring-client-api authentication flow"
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
    description: "Download Ring event clips"
  discover_cameras:
    script: scripts/discover.py
    description: "List Ring cameras and doorbells"
  live_stream:
    script: scripts/stream.py
    description: "Ring live view stream URL"
---

# Ring Camera Provider

Downloads event clips from Ring cameras and doorbells into SharpAI Aegis.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
