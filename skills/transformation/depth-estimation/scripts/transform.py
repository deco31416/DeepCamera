#!/usr/bin/env python3
"""
Depth Estimation Privacy Skill — Monocular depth maps via Depth Anything v2.

Implements the TransformSkillBase interface to provide real-time depth map
overlays on camera feeds. When used as a privacy skill, the depth-only mode
anonymizes the scene while preserving spatial layout and activity recognition.

Usage:
  python transform.py --model depth-anything-v2-small --device auto
  python transform.py --config config.json
"""

import sys
import argparse
from pathlib import Path

# Import the base class from the same directory
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from transform_base import TransformSkillBase, _log  # noqa: E402


COLORMAP_MAP = {
    "inferno": 1,   # cv2.COLORMAP_INFERNO
    "viridis": 16,  # cv2.COLORMAP_VIRIDIS
    "plasma": 13,   # cv2.COLORMAP_PLASMA
    "magma": 12,    # cv2.COLORMAP_MAGMA
    "jet": 2,       # cv2.COLORMAP_JET
}


class DepthEstimationSkill(TransformSkillBase):
    """
    Depth estimation using Depth Anything v2.

    Produces colorized depth maps that can be blended with the original frame
    (overlay mode), shown side-by-side, or displayed as depth-only anonymized view.
    """

    def __init__(self):
        super().__init__()
        self._tag = "DepthEstimation"
        self.model = None
        self.colormap_id = 1
        self.opacity = 0.5
        self.blend_mode = "depth_only"  # Default for privacy: depth_only anonymizes

    def parse_extra_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--model", type=str, default="depth-anything-v2-small",
                            choices=["depth-anything-v2-small", "depth-anything-v2-base",
                                     "depth-anything-v2-large", "midas-small"])
        parser.add_argument("--colormap", type=str, default="inferno",
                            choices=list(COLORMAP_MAP.keys()))
        parser.add_argument("--blend-mode", type=str, default="depth_only",
                            choices=["overlay", "side_by_side", "depth_only"])
        parser.add_argument("--opacity", type=float, default=0.5)

    def load_model(self, config: dict) -> dict:
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        model_name = config.get("model", "depth-anything-v2-small")
        self.colormap_id = COLORMAP_MAP.get(config.get("colormap", "inferno"), 1)
        self.opacity = config.get("opacity", 0.5)
        self.blend_mode = config.get("blend_mode", "depth_only")

        _log(f"Loading {model_name} on {self.device}", self._tag)

        # Model configs: encoder name, features, HF repo, weight filename
        MODEL_CONFIGS = {
            "depth-anything-v2-small": {
                "encoder": "vits", "features": 64,
                "out_channels": [48, 96, 192, 384],
                "repo": "depth-anything/Depth-Anything-V2-Small",
                "filename": "depth_anything_v2_vits.pth",
            },
            "depth-anything-v2-base": {
                "encoder": "vitb", "features": 128,
                "out_channels": [96, 192, 384, 768],
                "repo": "depth-anything/Depth-Anything-V2-Base",
                "filename": "depth_anything_v2_vitb.pth",
            },
            "depth-anything-v2-large": {
                "encoder": "vitl", "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "repo": "depth-anything/Depth-Anything-V2-Large",
                "filename": "depth_anything_v2_vitl.pth",
            },
        }

        cfg = MODEL_CONFIGS.get(model_name)
        if not cfg:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")

        # Download weights from HuggingFace Hub (cached after first download)
        _log(f"Downloading weights from HF: {cfg['repo']}", self._tag)
        weights_path = hf_hub_download(cfg["repo"], cfg["filename"])

        # Build model from pip package
        self.model = DepthAnythingV2(
            encoder=cfg["encoder"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        _log(f"Model loaded: {model_name} on {self.device}", self._tag)

        return {
            "model": model_name,
            "device": self.device,
            "blend_mode": self.blend_mode,
            "colormap": config.get("colormap", "inferno"),
        }

    def transform_frame(self, image, metadata: dict):
        import torch
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth = self.model.infer_image(rgb)

        # Normalize depth to 0-255
        d_min, d_max = depth.min(), depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, self.colormap_id)

        if self.blend_mode == "overlay":
            output = cv2.addWeighted(image, 1 - self.opacity, depth_colored, self.opacity, 0)
        elif self.blend_mode == "side_by_side":
            output = np.hstack([image, depth_colored])
        else:  # depth_only — full anonymization
            output = depth_colored

        return output

    def on_config_update(self, config: dict):
        """Handle live config updates from Aegis."""
        if "colormap" in config:
            self.colormap_id = COLORMAP_MAP.get(config["colormap"], self.colormap_id)
            _log(f"Colormap updated: {config['colormap']}", self._tag)
        if "opacity" in config:
            self.opacity = float(config["opacity"])
            _log(f"Opacity updated: {self.opacity}", self._tag)
        if "blend_mode" in config:
            self.blend_mode = config["blend_mode"]
            _log(f"Blend mode updated: {self.blend_mode}", self._tag)

    def get_output_mode(self) -> str:
        """Use base64 for privacy transforms — avoids temp file cleanup issues."""
        return "base64"


if __name__ == "__main__":
    DepthEstimationSkill().run()
