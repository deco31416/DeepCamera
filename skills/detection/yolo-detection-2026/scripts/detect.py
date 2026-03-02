#!/usr/bin/env python3
"""
YOLO Detection Skill — Real-time object detection for SharpAI Aegis.

Communicates via JSON lines over stdin/stdout:
  stdin:  {"event": "frame", "camera_id": "...", "frame_path": "...", ...}
  stdout: {"event": "detections", "camera_id": "...", "objects": [...]}

Usage:
  python detect.py --config config.json
  python detect.py --model yolov11n --confidence 0.5 --device auto
"""

import sys
import json
import argparse
import signal
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Detection Skill")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--model", type=str, default="yolov11n", 
                        choices=["yolov11n", "yolov11s", "yolov11m", "yolov10n", "yolov10s", "yolov8n"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--classes", type=str, default="person,car,dog,cat")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def load_config(args):
    """Load config from JSON file or CLI args."""
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return {
        "model": args.model,
        "confidence": args.confidence,
        "classes": args.classes.split(","),
        "device": args.device,
        "fps": args.fps,
    }


def select_device(preference: str) -> str:
    """Select the best available inference device."""
    if preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def emit(event: dict):
    """Write a JSON line to stdout."""
    print(json.dumps(event), flush=True)


def main():
    args = parse_args()
    config = load_config(args)

    # Select device
    device = select_device(config.get("device", "auto"))
    model_name = config.get("model", "yolov11n")
    confidence = config.get("confidence", 0.5)
    target_classes = config.get("classes", ["person", "car", "dog", "cat"])

    # Load YOLO model
    try:
        from ultralytics import YOLO
        model = YOLO(f"{model_name}.pt")
        model.to(device)
        emit({
            "event": "ready",
            "model": model_name,
            "device": device,
            "classes": len(model.names),
        })
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load model: {e}", "retriable": False})
        sys.exit(1)

    # Graceful shutdown
    running = True
    def handle_signal(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main loop: read frames from stdin, output detections to stdout
    for line in sys.stdin:
        if not running:
            break

        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("command") == "stop":
            break

        if msg.get("event") == "frame":
            frame_path = msg.get("frame_path")
            camera_id = msg.get("camera_id", "unknown")
            timestamp = msg.get("timestamp", "")

            if not frame_path or not Path(frame_path).exists():
                emit({"event": "error", "message": f"Frame not found: {frame_path}", "retriable": True})
                continue

            # Run inference
            try:
                results = model(frame_path, conf=confidence, verbose=False)
                objects = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        if cls_name in target_classes or not target_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            objects.append({
                                "class": cls_name,
                                "confidence": round(float(box.conf[0]), 3),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            })

                emit({
                    "event": "detections",
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "objects": objects,
                })
            except Exception as e:
                emit({"event": "error", "message": f"Inference error: {e}", "retriable": True})


if __name__ == "__main__":
    main()
