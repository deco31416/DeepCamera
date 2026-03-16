#!/usr/bin/env python3
"""
Native TensorRT Benchmark for Depth Anything V2.

Builds a TensorRT engine from ONNX, benchmarks FP32 and FP16, 
and compares against vanilla PyTorch CUDA and ONNX CUDA.
"""
import sys, os, time, json, statistics
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

def _log(msg):
    print(f"[TRT] {msg}", file=sys.stderr, flush=True)

def get_test_image():
    cache = Path.home() / ".aegis-ai" / "tmp" / "benchmark" / "test_image.jpg"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists(): return str(cache)
    from urllib.request import urlretrieve
    _log("Downloading test image...")
    urlretrieve("https://ultralytics.com/images/bus.jpg", str(cache))
    return str(cache)

def stats(times, label):
    avg = statistics.mean(times)
    return {"label": label, "runs": len(times),
            "avg_ms": round(avg,2), "min_ms": round(min(times),2),
            "max_ms": round(max(times),2),
            "std_ms": round(statistics.stdev(times),2) if len(times)>1 else 0,
            "fps": round(1000/avg,2) if avg>0 else 0}

def build_trt_engine(onnx_path, fp16=False):
    """Build a TensorRT engine from an ONNX model."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    _log(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                _log(f"  ONNX Parse Error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Handle dynamic shapes — set optimization profile for batch dimension
    inp = network.get_input(0)
    if any(d == -1 for d in inp.shape):
        profile = builder.create_optimization_profile()
        # Use fixed shape (batch=1) for the actual input dimensions
        shape_list = list(inp.shape)
        fixed_shape = tuple(1 if d == -1 else d for d in shape_list)
        profile.set_shape(inp.name, fixed_shape, fixed_shape, fixed_shape)
        config.add_optimization_profile(profile)
        _log(f"  Set optimization profile: {fixed_shape}")
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        _log("  Building TRT engine (FP16)...")
    else:
        _log("  Building TRT engine (FP32)...")

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_time = (time.perf_counter() - t0) * 1000
    
    if serialized is None:
        _log("  Engine build failed!")
        return None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    _log(f"  Engine built in {build_time:.0f}ms")
    return engine, build_time

def bench_trt_engine(engine, input_data, num_runs=15, warmup=5, label="TRT"):
    """Benchmark a TensorRT engine."""
    import tensorrt as trt
    import numpy as np
    
    try:
        # Try newer API first (TRT 10+)
        context = engine.create_execution_context()
        
        # Get binding info
        num_io = engine.num_io_tensors
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        input_shape = engine.get_tensor_shape(input_name)
        output_shape = engine.get_tensor_shape(output_name)
        
        _log(f"  Input: {input_name} {list(input_shape)}")
        _log(f"  Output: {output_name} {list(output_shape)}")
        
        # Allocate CUDA memory
        import ctypes
        
        # Use pycuda or cuda-python for memory management
        try:
            import cuda  # Try nvidia cuda-python
            has_cuda_python = True
        except ImportError:
            has_cuda_python = False
        
        # Fallback: use torch for GPU memory management (simplest)
        import torch
        
        input_tensor = torch.from_numpy(input_data).cuda()
        
        # Determine output shape - handle dynamic dims
        out_shape = list(output_shape)
        for i, s in enumerate(out_shape):
            if s == -1:
                if i == 0: out_shape[i] = input_data.shape[0]  # batch
                else: out_shape[i] = 1  # placeholder
        
        # Set input shape for dynamic dims
        context.set_input_shape(input_name, input_data.shape)
        
        # Get actual output shape after setting input
        actual_out_shape = context.get_tensor_shape(output_name)
        output_tensor = torch.empty(list(actual_out_shape), dtype=torch.float32, device='cuda')
        
        # Set tensor addresses
        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())
        
        # Get CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        
        # Warmup
        for _ in range(warmup):
            context.execute_async_v3(stream)
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for i in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            context.execute_async_v3(stream)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            _log(f"  [{label}] Run {i+1}/{num_runs}: {elapsed:.1f}ms")
        
        return times
        
    except Exception as e:
        _log(f"  Engine execution error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None

def main():
    import torch, cv2, numpy as np

    device = "cuda"
    N, W = 15, 5

    _log(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    _log(f"GPU: {torch.cuda.get_device_name(0)}")

    import tensorrt as trt
    _log(f"TensorRT: {trt.__version__}")

    # Load image and model
    image = cv2.imread(get_test_image())
    _log(f"Image: {image.shape[1]}x{image.shape[0]}")

    from depth_anything_v2.dpt import DepthAnythingV2
    from huggingface_hub import hf_hub_download
    cfg = {"encoder":"vits","features":64,"out_channels":[48,96,192,384],
           "repo":"depth-anything/Depth-Anything-V2-Small","filename":"depth_anything_v2_vits.pth"}
    weights = hf_hub_download(cfg["repo"], cfg["filename"])
    model = DepthAnythingV2(encoder=cfg["encoder"], features=cfg["features"], out_channels=cfg["out_channels"])
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor, (h, w) = model.image2tensor(rgb, input_size=518)
    img_tensor = img_tensor.to(device)
    _log(f"Tensor: {img_tensor.shape}")

    results = []

    # 1. PyTorch CUDA baseline
    _log("\n== 1. PyTorch CUDA ==")
    for _ in range(W):
        with torch.no_grad(): model.forward(img_tensor); torch.cuda.synchronize()
    times = []
    for i in range(N):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): model.forward(img_tensor)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t0)*1000)
        _log(f"  [PyTorch] Run {i+1}/{N}: {times[-1]:.1f}ms")
    results.append(stats(times, "PyTorch CUDA"))

    # 2. ONNX CUDA
    _log("\n== 2. ONNX CUDA ==")
    onnx_path = Path.home() / ".aegis-ai" / "tmp" / "benchmark" / "dav2_small.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _log("  Exporting ONNX...")
        torch.onnx.export(model, img_tensor, str(onnx_path),
            input_names=["input"], output_names=["depth"],
            dynamic_axes={"input":{0:"batch"}, "depth":{0:"batch"}}, opset_version=17)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        inp = img_tensor.cpu().numpy()
        for _ in range(W): sess.run(None, {in_name: inp})
        times = []
        for i in range(N):
            t0 = time.perf_counter()
            sess.run(None, {in_name: inp})
            times.append((time.perf_counter()-t0)*1000)
            _log(f"  [ONNX CUDA] Run {i+1}/{N}: {times[-1]:.1f}ms")
        results.append(stats(times, "ONNX CUDA"))
    except Exception as e:
        _log(f"  FAILED: {e}")
        results.append({"label":"ONNX CUDA","error":str(e)[:100]})

    # 3. TensorRT FP32
    _log("\n== 3. TensorRT FP32 ==")
    try:
        engine_result = build_trt_engine(str(onnx_path), fp16=False)
        if engine_result:
            engine, build_ms = engine_result
            inp = img_tensor.cpu().numpy()
            times = bench_trt_engine(engine, inp, N, W, "TRT FP32")
            if times:
                r = stats(times, "TensorRT FP32")
                r["build_ms"] = round(build_ms, 0)
                results.append(r)
            else:
                results.append({"label":"TensorRT FP32","error":"execution failed"})
        else:
            results.append({"label":"TensorRT FP32","error":"engine build failed"})
    except Exception as e:
        _log(f"  FAILED: {e}")
        results.append({"label":"TensorRT FP32","error":str(e)[:100]})

    # 4. TensorRT FP16
    _log("\n== 4. TensorRT FP16 ==")
    try:
        engine_result = build_trt_engine(str(onnx_path), fp16=True)
        if engine_result:
            engine, build_ms = engine_result
            inp = img_tensor.cpu().numpy()
            times = bench_trt_engine(engine, inp, N, W, "TRT FP16")
            if times:
                r = stats(times, "TensorRT FP16")
                r["build_ms"] = round(build_ms, 0)
                results.append(r)
            else:
                results.append({"label":"TensorRT FP16","error":"execution failed"})
        else:
            results.append({"label":"TensorRT FP16","error":"engine build failed"})
    except Exception as e:
        _log(f"  FAILED: {e}")
        results.append({"label":"TensorRT FP16","error":str(e)[:100]})

    # Summary
    _log("\n" + "="*70)
    _log(f"{'Backend':<22} {'Avg(ms)':>8} {'Min(ms)':>8} {'FPS':>7} {'Speedup':>8}")
    _log("-"*70)
    base = results[0].get("avg_ms",1)
    for r in results:
        if "error" in r:
            _log(f"{r['label']:<22} {'FAIL':>8}  {r['error'][:45]}")
        else:
            su = base/r["avg_ms"] if r["avg_ms"]>0 else 0
            _log(f"{r['label']:<22} {r['avg_ms']:>8.1f} {r['min_ms']:>8.1f} {r['fps']:>7.1f} {su:>7.2f}x")

    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "results": results}, indent=2))

if __name__ == "__main__":
    main()
