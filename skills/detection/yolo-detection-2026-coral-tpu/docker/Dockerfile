# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Coral Edge TPU Model Compiler
#
# Converts YOLO 2026 nano (.pt) → TFLite INT8 → Edge TPU (.tflite)
#
# IMPORTANT: Must run on linux/amd64. edgetpu_compiler only exists for
# x86_64 Linux. On Apple Silicon or Windows, use:
#   docker run --platform linux/amd64 ...
# ─────────────────────────────────────────────────────────────────────────────

FROM --platform=linux/amd64 python:3.11-slim

LABEL maintainer="Aegis-AI / DeepCamera"
LABEL description="Compiles YOLO 2026 nano to Google Coral Edge TPU .tflite"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        apt-transport-https \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Google Coral edgetpu_compiler (x86_64 Debian package) ────────────────────
# Official Coral apt repository — edgetpu_compiler is only available here.
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] \
        https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends edgetpu-compiler \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# ultralytics: YOLO export to TFLite INT8
# tensorflow-cpu: required by ultralytics INT8 export (calibration + quantization)
RUN pip install --no-cache-dir \
        "ultralytics>=8.3.0" \
        "tensorflow-cpu>=2.17.0,<3.0" \
        "numpy>=1.24.0,<2.0" \
        "Pillow>=10.0.0" \
        "onnx>=1.16.0"

# ── Copy compile script ───────────────────────────────────────────────────────
WORKDIR /compile
COPY scripts/compile_model.py /compile/compile_model.py

# ── Output volume ─────────────────────────────────────────────────────────────
# Mount the skill's models/ directory here at runtime:
#   docker run -v /path/to/models:/compile/output ...
VOLUME ["/compile/output"]

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "/compile/compile_model.py"]
CMD ["--model", "yolo26n", "--size", "320", "--output", "/compile/output"]
