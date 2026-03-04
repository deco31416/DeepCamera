---
name: Home Security AI Benchmark
description: LLM & VLM evaluation suite for home security AI applications
version: 1.0.0
category: analysis
---

# Home Security AI Benchmark

Comprehensive benchmark suite that evaluates LLM and VLM models on tasks specific to **home security AI assistants** — deduplication, event classification, knowledge extraction, tool use, and scene analysis.

## Quick Start

```bash
# Standalone (provide gateway URL)
node scripts/run-benchmark.cjs --gateway http://localhost:5407

# With VLM tests
node scripts/run-benchmark.cjs --gateway http://localhost:5407 --vlm http://localhost:5405

# Generate HTML report from results
node scripts/generate-report.cjs
```

When spawned by Aegis, configuration is automatic via environment variables.

## Protocol

### Aegis → Skill (env vars)
```
AEGIS_GATEWAY_URL=http://localhost:5407   # LLM gateway
AEGIS_VLM_URL=http://localhost:5405       # VLM server
AEGIS_SKILL_ID=home-security-benchmark    # Skill ID
AEGIS_SKILL_PARAMS={}                     # JSON params from skill config
```

### Skill → Aegis (stdout, JSON lines)
```jsonl
{"event": "ready", "model": "Qwen3.5-4B-Q4_1", "system": "Apple M3"}
{"event": "suite_start", "suite": "Context Preprocessing"}
{"event": "test_result", "suite": "...", "test": "...", "status": "pass", "timeMs": 123}
{"event": "suite_end", "suite": "...", "passed": 4, "failed": 0}
{"event": "complete", "passed": 23, "total": 26, "timeMs": 95000, "resultFile": "..."}
```

Human-readable output goes to **stderr** (visible in Aegis console tab).

## Test Suites

| Suite | Tests | Domain |
|-------|-------|--------|
| Context Preprocessing | 4 | Conversation dedup accuracy |
| Topic Classification | 4 | Topic extraction & change detection |
| Knowledge Distillation | 3 | Fact extraction, slug matching |
| Event Deduplication | 3 | Security event classification |
| Tool Use | 4 | Tool selection & parameter extraction |
| Chat & JSON Compliance | 7 | Persona, memory, structured output |
| VLM Scene Analysis | 4 | Frame description & object detection |

## Metrics Collected

- **Per-test**: latency (ms), prompt/completion tokens, pass/fail
- **Per-run**: total time, tokens/sec, memory usage
- **System**: OS, CPU, RAM, GPU, model name, quantization

## Results

Results are saved to `~/.aegis-ai/benchmarks/` as JSON. The HTML report generator reads all historical results for cross-model comparison.

## Requirements

- Node.js ≥ 18
- Running LLM server (llama-cpp, vLLM, or any OpenAI-compatible API)
- Optional: Running VLM server for scene analysis tests
