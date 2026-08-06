# LLM Decode Dispatch Analysis

This repository investigates dispatch overhead in iterative LLM decode.

It demonstrates that small-batch decode is often dispatch-bound rather than compute-bound.

---

## Key Results

| Regime         | Tokens/sec |
|---------------|------------|
| Baseline     | ~4.3k      |
| Graph Static | ~68.8k     |
| Graph Pool   | ~66k       |
| Real Loop    | ~22k       |

CUDA Graph Replay eliminates kernel launch overhead and dramatically reduces CPU dispatch time.

---

## Core Findings

1. Iterative decode is dispatch-bound under small batch sizes.
2. CUDA Graph removes dispatch overhead.
3. Mask-based dynamic batching reduces gains.
4. Graph pooling preserves performance.
5. Host orchestration becomes the next bottleneck.

---

## Repository Structure

baseline/  
graph_static/  
graph_pool/  
realistic_loop/  
profiling/  
docs/  
results/  

---

## Reproducing Experiments

Baseline profiling:

PYTHONPATH=. python benchmarks/baseline_profile.py

Graph profiling:

PYTHONPATH=. python benchmarks/graph_profile.py

Graph Pool profiling:

PYTHONPATH=. python benchmarks/graph_pool_profile.py

Full benchmark comparison:

PYTHONPATH=. python benchmarks/compare_serving_modes.py

Historical aggregation:

PYTHONPATH=. python benchmarks/aggregate_history.py

---

## Full Technical Report

See:

report.md

---

## Why This Matters

Kernel optimization alone is insufficient.

Serving runtime architecture determines real-world decode latency.

This repository provides a minimal, controlled study isolating dispatch and orchestration overhead.

---

## License

MIT

