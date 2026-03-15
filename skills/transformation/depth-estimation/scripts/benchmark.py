#!/usr/bin/env python3
"""
Depth Estimation Benchmark — Cross-platform performance measurement.

Measures inference latency for depth-estimation models across CoreML (macOS)
and PyTorch (CUDA/MPS/CPU). Outputs JSON results compatible with
Aegis DepthVisionStudio's benchmark UI.

Usage:
  python benchmark.py --variant depth_anything_v2_vits --runs 10 --device auto
  python benchmark.py --variant DepthAnythingV2SmallF16 --runs 5 --compute-units cpu_and_ne

The --compute-units flag is macOS/CoreML only (maps to coremltools.ComputeUnit).
On other platforms, use --device to select cuda/cpu/mps.
"""

import sys
import os
import json
import time
import argparse
import tempfile
import statistics
from pathlib import Path

# Add parent for transform imports
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))


def _log(msg):
    """Log to stderr (not captured by Aegis JSON parser)."""
    print(f"[Benchmark] {msg}", file=sys.stderr, flush=True)


def _download_test_image(url, dest_path):
    """Download a test image from URL."""
    try:
        from urllib.request import urlretrieve
        _log(f"Downloading test image: {url}")
        urlretrieve(url, dest_path)
        return True
    except Exception as e:
        _log(f"Download failed: {e}")
        return False


def _get_test_image(test_image_url):
    """Get or download a test image, returns path or None."""
    # Check for cached test image
    cache_dir = Path.home() / ".aegis-ai" / "tmp" / "benchmark"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "test_image.jpg"

    if cached.exists():
        return str(cached)

    if test_image_url:
        if _download_test_image(test_image_url, str(cached)):
            return str(cached)

    # Generate a synthetic test image as fallback
    try:
        import numpy as np
        import cv2
        _log("Generating synthetic 640x480 test image")
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(cached), img)
        return str(cached)
    except ImportError:
        _log("ERROR: cv2/numpy not available — cannot generate test image")
        return None


def _resolve_device(device_pref, compute_units):
    """
    Resolve the compute device from CLI args.

    Priority:
      1. Explicit --device (cuda, cpu, mps)
      2. Map --compute-units to device (CoreML-specific, macOS only)
      3. Auto-detect
    """
    import platform

    if device_pref and device_pref != "auto":
        return device_pref

    # Map compute_units → device hint (for Aegis UI compatibility)
    if compute_units and compute_units != "all":
        cu_map = {
            "gpu": "cuda",           # Aegis UI "GPU" → CUDA
            "cpu": "cpu",
            "npu": "mps",            # On macOS, NPU maps to Neural Engine via CoreML
            "cpu_npu": "mps",        # CPU + Neural Engine
            "cpu_and_ne": "mps",     # CoreML compute units string
        }
        mapped = cu_map.get(compute_units)
        if mapped:
            return mapped

    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    if platform.system() == "Darwin":
        return "mps"

    return "cpu"


def run_benchmark(args):
    """Run the depth estimation benchmark and output JSON results."""
    import cv2
    import numpy as np

    variant_id = args.variant
    num_runs = args.runs
    colormap_name = args.colormap
    device = _resolve_device(args.device, args.compute_units)
    test_image_url = args.test_image_url

    _log(f"Benchmark: variant={variant_id}, runs={num_runs}, device={device}, colormap={colormap_name}")

    # Colormap lookup
    colormap_map = {
        "inferno": 1, "viridis": 16, "plasma": 13, "magma": 12,
        "jet": 2, "turbo": 18, "hot": 11, "cool": 8,
    }
    colormap_id = colormap_map.get(colormap_name, 16)

    # Load test image
    test_image_path = _get_test_image(test_image_url)
    if not test_image_path:
        result = {"error": "Could not obtain test image"}
        print(json.dumps(result))
        return

    image = cv2.imread(test_image_path)
    if image is None:
        result = {"error": f"Failed to read test image: {test_image_path}"}
        print(json.dumps(result))
        return

    _log(f"Test image: {image.shape[1]}x{image.shape[0]}")

    # Determine backend and load model
    import platform as plat
    is_mac = plat.system() == "Darwin"
    backend = None
    model = None
    model_load_ms = 0.0

    # Try CoreML first on macOS (if variant looks like a CoreML model)
    coreml_variants = {"DepthAnythingV2SmallF16", "DepthAnythingV2SmallF16INT8", "DepthAnythingV2SmallF32"}
    if is_mac and variant_id in coreml_variants:
        try:
            import coremltools as ct
            from PIL import Image

            models_dir = Path.home() / ".aegis-ai" / "models" / "feature-extraction"
            model_path = models_dir / f"{variant_id}.mlpackage"

            if model_path.exists():
                _log(f"Loading CoreML model: {model_path}")
                t0 = time.perf_counter()

                # Map compute_units string to coremltools enum
                cu_str = args.compute_units or "all"
                cu_enum_map = {
                    "all": ct.ComputeUnit.ALL,
                    "cpu": ct.ComputeUnit.CPU_ONLY,
                    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
                    "npu": ct.ComputeUnit.CPU_AND_NE,
                    "cpu_npu": ct.ComputeUnit.CPU_AND_NE,
                    "gpu": ct.ComputeUnit.CPU_AND_GPU,
                }
                cu_enum = cu_enum_map.get(cu_str, ct.ComputeUnit.ALL)

                model = ct.models.MLModel(str(model_path), compute_units=cu_enum)
                model_load_ms = (time.perf_counter() - t0) * 1000
                backend = "coreml"
                _log(f"CoreML model loaded in {model_load_ms:.0f}ms (compute_units={cu_str})")
            else:
                _log(f"CoreML model not found at {model_path}, falling back to PyTorch")
        except Exception as e:
            _log(f"CoreML load failed: {e}, trying PyTorch")

    # PyTorch fallback
    if model is None:
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
            from huggingface_hub import hf_hub_download

            # Map variant_id to PyTorch config
            pytorch_configs = {
                "depth_anything_v2_vits": {
                    "encoder": "vits", "features": 64,
                    "out_channels": [48, 96, 192, 384],
                    "repo": "depth-anything/Depth-Anything-V2-Small",
                    "filename": "depth_anything_v2_vits.pth",
                },
                "depth_anything_v2_vitb": {
                    "encoder": "vitb", "features": 128,
                    "out_channels": [96, 192, 384, 768],
                    "repo": "depth-anything/Depth-Anything-V2-Base",
                    "filename": "depth_anything_v2_vitb.pth",
                },
                "depth_anything_v2_vitl": {
                    "encoder": "vitl", "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                    "repo": "depth-anything/Depth-Anything-V2-Large",
                    "filename": "depth_anything_v2_vitl.pth",
                },
            }

            # Also accept CoreML variant names and map to PyTorch
            coreml_to_pytorch = {
                "DepthAnythingV2SmallF16": "depth_anything_v2_vits",
                "DepthAnythingV2SmallF16INT8": "depth_anything_v2_vits",
                "DepthAnythingV2SmallF32": "depth_anything_v2_vits",
            }

            pytorch_variant = coreml_to_pytorch.get(variant_id, variant_id)
            cfg = pytorch_configs.get(pytorch_variant)
            if not cfg:
                result = {"error": f"Unknown variant: {variant_id}. Available: {list(pytorch_configs.keys())}"}
                print(json.dumps(result))
                return

            _log(f"Loading PyTorch model: {pytorch_variant} on {device}")
            t0 = time.perf_counter()

            weights_path = hf_hub_download(cfg["repo"], cfg["filename"])
            model = DepthAnythingV2(
                encoder=cfg["encoder"],
                features=cfg["features"],
                out_channels=cfg["out_channels"],
            )
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            model_load_ms = (time.perf_counter() - t0) * 1000
            backend = "pytorch"
            _log(f"PyTorch model loaded in {model_load_ms:.0f}ms on {device}")

        except Exception as e:
            result = {"error": f"Failed to load model: {e}"}
            print(json.dumps(result))
            return

    # ── Run benchmark ──────────────────────────────────────────────

    timings = []
    errors = 0
    extraction_data = None

    for i in range(num_runs):
        try:
            t0 = time.perf_counter()

            if backend == "coreml":
                from PIL import Image as PILImage
                # CoreML inference
                input_w, input_h = 518, 392
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (input_w, input_h))
                pil_img = PILImage.fromarray(resized, mode="RGB")
                prediction = model.predict({"image": pil_img})
                output_key = list(prediction.keys())[0]
                depth_map = np.array(prediction[output_key])
                if depth_map.ndim > 2:
                    depth_map = np.squeeze(depth_map)
            else:
                # PyTorch inference
                import torch
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    depth_map = model.infer_image(rgb)

            # Normalize and colorize
            d_min, d_max = depth_map.min(), depth_map.max()
            depth_norm = ((depth_map - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, colormap_id)
            depth_colored = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(elapsed_ms)

            # Capture extraction result from last run
            if i == num_runs - 1:
                import base64
                _, buf = cv2.imencode(".jpg", depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extraction_data = base64.b64encode(buf).decode("ascii")

            _log(f"  Run {i+1}/{num_runs}: {elapsed_ms:.1f}ms")

        except Exception as e:
            _log(f"  Run {i+1}/{num_runs}: ERROR — {e}")
            errors += 1

    # ── Build results ──────────────────────────────────────────────

    if not timings:
        result = {"error": f"All {num_runs} runs failed"}
        print(json.dumps(result))
        return

    avg = statistics.mean(timings)
    fps = 1000.0 / avg if avg > 0 else 0

    result = {
        "model_id": variant_id,
        "variant_id": variant_id,
        "backend": backend,
        "device": device,
        "num_runs": num_runs,
        "successful_runs": len(timings),
        "errors": errors,
        "avg_time_ms": round(avg, 2),
        "min_time_ms": round(min(timings), 2),
        "max_time_ms": round(max(timings), 2),
        "std_time_ms": round(statistics.stdev(timings), 2) if len(timings) > 1 else 0.0,
        "fps": round(fps, 2),
        "model_load_ms": round(model_load_ms, 1),
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "colormap": colormap_name,
    }

    # Include depth map preview from last run
    if extraction_data:
        result["extraction_result"] = {
            "success": True,
            "feature_type": "depth_estimation",
            "feature_data": extraction_data,
            "processing_time": round(timings[-1], 2),
            "metadata": {
                "backend": backend,
                "device": device,
                "colormap": colormap_name,
            },
        }

    _log(f"Results: {avg:.1f}ms avg, {fps:.1f} FPS, {len(timings)}/{num_runs} successful")

    # Output JSON on stdout (Aegis parses the last line)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Benchmark")
    parser.add_argument("--variant", type=str, required=True,
                        help="Model variant ID (e.g. depth_anything_v2_vits, DepthAnythingV2SmallF16)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of benchmark runs (default: 10)")
    parser.add_argument("--colormap", type=str, default="viridis",
                        choices=["inferno", "viridis", "plasma", "magma", "jet", "turbo", "hot", "cool"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--compute-units", type=str, default="all",
                        help="Compute units for CoreML (all, cpu, cpu_and_ne, gpu, npu, cpu_npu)")
    parser.add_argument("--test-image-url", type=str,
                        default="https://ultralytics.com/images/bus.jpg",
                        help="URL of test image to download")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
