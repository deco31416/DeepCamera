#!/usr/bin/env python3
"""
3D Depth Vision (CUDA) — GPU-accelerated depth maps via Depth Anything v2.

Backend selection (priority order):
  1. TensorRT FP16  — fastest (~7x over PyTorch), auto-builds engine
  2. PyTorch CUDA   — baseline GPU inference
  3. PyTorch CPU    — fallback if no GPU

Uses vendored depth_anything_v2 model code (no pip install needed).
Implements TransformSkillBase for real-time JSONL protocol.

Usage:
  python transform.py --model depth-anything-v2-small --device auto
  python transform.py --config config.json
"""

import sys
import os
import argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_skill_dir = _script_dir.parent

# Add skill root to path so vendored depth_anything_v2 package is importable
sys.path.insert(0, str(_skill_dir))
# Add scripts dir for transform_base
sys.path.insert(0, str(_script_dir))

from transform_base import TransformSkillBase, _log  # noqa: E402


COLORMAP_MAP = {
    "inferno": 1,   # cv2.COLORMAP_INFERNO
    "viridis": 16,  # cv2.COLORMAP_VIRIDIS
    "plasma": 13,   # cv2.COLORMAP_PLASMA
    "magma": 12,    # cv2.COLORMAP_MAGMA
    "jet": 2,       # cv2.COLORMAP_JET
    "turbo": 18,    # cv2.COLORMAP_TURBO
    "hot": 11,      # cv2.COLORMAP_HOT
    "cool": 8,      # cv2.COLORMAP_COOL
}

# PyTorch model configs
PYTORCH_CONFIGS = {
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

# Where Aegis stores models
MODELS_DIR = Path.home() / ".aegis-ai" / "models" / "feature-extraction"
TRT_CACHE_DIR = MODELS_DIR / "trt_engines"


class DepthEstimationSkill(TransformSkillBase):
    """
    CUDA-accelerated depth estimation using Depth Anything v2.

    Produces colorized depth maps for privacy anonymization
    or depth overlay on camera feeds.
    """

    def __init__(self):
        super().__init__()
        self._tag = "DepthEstimation"
        self.model = None
        self.backend = None  # "tensorrt" or "pytorch"
        self.colormap_id = 1
        self.opacity = 0.5
        self.blend_mode = "depth_only"
        # TensorRT state
        self._trt_context = None
        self._trt_input_name = None
        self._trt_output_name = None
        self._trt_input_tensor = None
        self._trt_output_tensor = None
        self._trt_stream = None

    def parse_extra_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--model", type=str, default="depth-anything-v2-small",
                            choices=list(PYTORCH_CONFIGS.keys()))
        parser.add_argument("--colormap", type=str, default="inferno",
                            choices=list(COLORMAP_MAP.keys()))
        parser.add_argument("--blend-mode", type=str, default="depth_only",
                            choices=["overlay", "side_by_side", "depth_only"])
        parser.add_argument("--opacity", type=float, default=0.5)

    def load_model(self, config: dict) -> dict:
        model_name = config.get("model", "depth-anything-v2-small")
        self.colormap_id = COLORMAP_MAP.get(config.get("colormap", "inferno"), 1)
        self.opacity = config.get("opacity", 0.5)
        self.blend_mode = config.get("blend_mode", "depth_only")

        # Try TensorRT first (fails fast if not installed)
        try:
            info = self._load_tensorrt(model_name, config)
            return info
        except Exception as e:
            _log(f"TensorRT unavailable ({e}), falling back to PyTorch", self._tag)

        # Fallback: PyTorch CUDA or CPU
        return self._load_pytorch(model_name, config)

    # ── TensorRT backend ──────────────────────────────────────────────

    def _load_tensorrt(self, model_name: str, config: dict) -> dict:
        """Load or build a TensorRT FP16 engine for fastest NVIDIA inference."""
        import torch
        import tensorrt as trt

        _log(f"Attempting TensorRT FP16 for {model_name}", self._tag)

        cfg = PYTORCH_CONFIGS.get(model_name)
        if not cfg:
            raise ValueError(f"Unknown model: {model_name}")

        gpu_tag = torch.cuda.get_device_name(0).replace(" ", "_").lower()
        engine_path = TRT_CACHE_DIR / f"{cfg['filename'].replace('.pth', '')}_fp16_{gpu_tag}.trt"

        if engine_path.exists():
            _log(f"Loading cached TRT engine: {engine_path}", self._tag)
            engine = self._deserialize_engine(engine_path)
        else:
            _log("No cached engine — building from ONNX (30-120s)...", self._tag)
            engine = self._build_trt_engine(cfg, engine_path)

        if engine is None:
            raise RuntimeError("TensorRT engine build/load failed")

        self._trt_context = engine.create_execution_context()
        self._trt_input_name = engine.get_tensor_name(0)
        self._trt_output_name = engine.get_tensor_name(1)

        input_shape = engine.get_tensor_shape(self._trt_input_name)
        fixed_shape = tuple(1 if d == -1 else d for d in input_shape)
        self._trt_context.set_input_shape(self._trt_input_name, fixed_shape)

        self._trt_input_tensor = torch.zeros(fixed_shape, dtype=torch.float32, device="cuda")
        actual_out_shape = self._trt_context.get_tensor_shape(self._trt_output_name)
        self._trt_output_tensor = torch.empty(list(actual_out_shape), dtype=torch.float32, device="cuda")

        self._trt_context.set_tensor_address(self._trt_input_name, self._trt_input_tensor.data_ptr())
        self._trt_context.set_tensor_address(self._trt_output_name, self._trt_output_tensor.data_ptr())
        self._trt_stream = torch.cuda.current_stream().cuda_stream

        self.backend = "tensorrt"
        _log(f"TensorRT FP16 engine ready: {engine_path.name}", self._tag)
        return {
            "model": model_name,
            "device": "cuda",
            "blend_mode": self.blend_mode,
            "colormap": config.get("colormap", "inferno"),
            "backend": "tensorrt",
        }

    def _build_trt_engine(self, cfg: dict, engine_path: Path):
        """Export PyTorch → ONNX → build TRT FP16 engine → serialize to disk."""
        import torch
        import tensorrt as trt
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(cfg["repo"], cfg["filename"])
        pt_model = DepthAnythingV2(
            encoder=cfg["encoder"], features=cfg["features"],
            out_channels=cfg["out_channels"],
        )
        pt_model.load_state_dict(torch.load(weights_path, map_location="cuda", weights_only=True))
        pt_model.to("cuda").eval()

        dummy = torch.randn(1, 3, 518, 518, device="cuda")
        onnx_path = TRT_CACHE_DIR / f"{cfg['filename'].replace('.pth', '')}.onnx"
        TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        _log(f"Exporting ONNX: {onnx_path.name}", self._tag)
        torch.onnx.export(
            pt_model, dummy, str(onnx_path),
            input_names=["input"], output_names=["depth"],
            dynamic_axes={"input": {0: "batch"}, "depth": {0: "batch"}},
            opset_version=17,
        )
        del pt_model
        torch.cuda.empty_cache()

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        _log("Parsing ONNX for TensorRT...", self._tag)
        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    _log(f"  ONNX parse error: {parser.get_error(i)}", self._tag)
                return None

        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        inp = network.get_input(0)
        if any(d == -1 for d in inp.shape):
            profile = builder.create_optimization_profile()
            fixed = tuple(1 if d == -1 else d for d in inp.shape)
            profile.set_shape(inp.name, fixed, fixed, fixed)
            trt_config.add_optimization_profile(profile)

        trt_config.set_flag(trt.BuilderFlag.FP16)

        _log("Building TRT FP16 engine (30-120s)...", self._tag)
        serialized = builder.build_serialized_network(network, trt_config)
        if serialized is None:
            _log("TRT engine build failed!", self._tag)
            return None

        engine_bytes = bytes(serialized)
        with open(str(engine_path), "wb") as f:
            f.write(engine_bytes)
        _log(f"Engine cached: {engine_path} ({len(engine_bytes) / 1e6:.1f} MB)", self._tag)

        try:
            onnx_path.unlink()
        except OSError:
            pass

        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engine_bytes)

    @staticmethod
    def _deserialize_engine(engine_path: Path):
        """Load a previously serialized TRT engine from disk."""
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(str(engine_path), "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    # ── PyTorch backend (fallback) ────────────────────────────────────

    def _load_pytorch(self, model_name: str, config: dict) -> dict:
        """Load PyTorch model — CUDA or CPU fallback."""
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        _log(f"Loading {model_name} on {self.device} (PyTorch)", self._tag)

        cfg = PYTORCH_CONFIGS.get(model_name)
        if not cfg:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(PYTORCH_CONFIGS.keys())}")

        # Download weights from HuggingFace Hub (cached after first download)
        _log(f"Downloading weights from HF: {cfg['repo']}", self._tag)
        weights_path = hf_hub_download(cfg["repo"], cfg["filename"])

        # Build model from vendored architecture
        self.model = DepthAnythingV2(
            encoder=cfg["encoder"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.backend = "pytorch"

        _log(f"PyTorch model loaded: {model_name} on {self.device}", self._tag)
        return {
            "model": model_name,
            "device": self.device,
            "blend_mode": self.blend_mode,
            "colormap": config.get("colormap", "inferno"),
            "backend": "pytorch",
        }

    # ── Frame transform ───────────────────────────────────────────────

    def transform_frame(self, image, metadata: dict):
        import cv2
        import numpy as np

        if self.backend == "tensorrt":
            depth_colored = self._infer_tensorrt(image)
        else:
            depth_colored = self._infer_pytorch(image)

        if self.blend_mode == "overlay":
            output = cv2.addWeighted(image, 1 - self.opacity, depth_colored, self.opacity, 0)
        elif self.blend_mode == "side_by_side":
            output = np.hstack([image, depth_colored])
        else:  # depth_only — full anonymization
            output = depth_colored

        return output

    def _infer_pytorch(self, image):
        """Run PyTorch inference and return colorized depth map (BGR, original size)."""
        import torch
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth = self.model.infer_image(rgb)

        d_min, d_max = depth.min(), depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, self.colormap_id)

        return depth_colored

    def _infer_tensorrt(self, image):
        """Run TensorRT FP16 inference and return colorized depth map."""
        import torch
        import cv2
        import numpy as np

        original_h, original_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std
        img_nchw = np.transpose(img_float, (2, 0, 1))[np.newaxis]

        self._trt_input_tensor.copy_(torch.from_numpy(img_nchw))
        self._trt_context.execute_async_v3(self._trt_stream)
        torch.cuda.synchronize()

        depth = self._trt_output_tensor.cpu().numpy()
        depth = np.squeeze(depth)

        d_min, d_max = depth.min(), depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, self.colormap_id)
        depth_colored = cv2.resize(depth_colored, (original_w, original_h))

        return depth_colored

    # ── Config updates ────────────────────────────────────────────────

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
        """Use base64 for privacy transforms."""
        return "base64"


if __name__ == "__main__":
    DepthEstimationSkill().run()
