# LLM Decode Dispatch Analysis

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Profiling](https://img.shields.io/badge/Profiling-Nsight_Systems-blue)
![Focus](https://img.shields.io/badge/Focus-LLM_Runtime_Engineering-purple)

This repository investigates dispatch overhead in iterative LLM decode.

It demonstrates that small-batch decode is often dispatch-bound rather than compute-bound.

---

## Key Results

| Regime        | Tokens/sec |
|--------------|------------|
| Baseline     | ~4.3k      |
| Graph Static | ~68.8k     |
| Graph Pool   | ~66k       |
| Real Loop    | ~22k       |

CUDA Graph Replay eliminates kernel launch overhead and dramatically reduces CPU dispatch time.

---

## Formal Overhead Model

Decode latency can be decomposed into:

T_total = T_compute + T_dispatch + T_orchestration

Where:

- T_compute = actual GPU kernel execution
- T_dispatch = kernel launch and driver overhead
- T_orchestration = host-side request lifecycle logic

CUDA Graph eliminates T_dispatch.

Once dispatch is removed, T_orchestration becomes the dominant factor.

---

## Core Findings

- Iterative decode is dispatch-bound under small batch sizes.
- CUDA Graph removes kernel launch overhead.
- Mask-based dynamic batching reduces gains.
- Graph pooling preserves performance.
- Host orchestration becomes the next bottleneck.

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

## Profiling Evidence

Nsight Systems traces confirm:

- Significant reduction in CUDA API launch calls
- Elimination of CPU dispatch gaps
- GPU compute time remains largely unchanged

(Profiling screenshots available in the profiling/ directory.)

---

## Limitations

- Single-GPU study
- No swap or preemption
- Fixed maximum batch size
- Focused on small-batch regime
- Sampling logic simplified for isolation

---

## Why This Matters

Kernel optimization alone is insufficient.

Serving runtime architecture determines real-world decode latency.

This repository provides a minimal, controlled study isolating dispatch and orchestration overhead.

---

## License

MIT

