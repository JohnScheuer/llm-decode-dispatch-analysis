# LLM Decode Dispatch Analysis

## Executive Summary

Iterative LLM decode is frequently assumed to be compute-bound.  
In practice, small-batch serving regimes are dispatch-bound.

Through controlled experiments, we show:

- Baseline iterative decode: ~4.3k tokens/sec
- CUDA Graph replay (static batch): ~68.8k tokens/sec
- CUDA Graph with graph pooling: ~66k tokens/sec
- Realistic serving loop with host orchestration: ~22k tokens/sec

CUDA Graph eliminates kernel launch and driver dispatch overhead.  
However, host-side orchestration becomes the next dominant bottleneck.

Key insight:

T_total = T_compute + T_dispatch + T_orchestration

CUDA Graph removes T_dispatch.  
It does not remove T_orchestration.

---

## 1. Motivation

Modern LLM serving systems generate tokens iteratively.  
Each decode step typically includes:

- Attention projection
- MLP projection
- KV append
- Sampling
- Python-side scheduling
- CUDA kernel launches

Even when kernels are optimized, repeated dispatch per token can dominate latency.

The goal of this study:

Measure and isolate dispatch overhead in iterative decode.

---

## 2. Experimental Setup

Hardware:
- CPU: Ryzen 5 5600X
- GPU: RTX 2070 (SM75)
- RAM: 32GB

Software:
- CUDA 11.x
- PyTorch 2.x
- Nsight Systems 2023.x

Configuration:
- Hidden size: 1024
- Batch size: 4
- Steps: 200
- Max sequence length: 1024
- Explicit torch.cuda.synchronize() used for accurate timing

All measurements were taken after warmup and synchronization.

---

## 3. Regimes Compared

| Regime         | Tokens/sec | CPU/step | GPU/step |
|---------------|------------|----------|----------|
| Baseline     | ~4.3k      | ~0.91ms  | ~0.78ms  |
| Graph Static | ~68.8k     | ~0.058ms | ~0.78ms  |
| Graph Pool   | ~66k       | ~0.06ms  | ~0.78ms  |
| Real Loop    | ~22k       | ~0.2ms   | ~0.78ms  |

---

## 4. Baseline Decode

CPU dominated the runtime.

Self CPU time total: ~280ms  
Self CUDA time total: ~10ms  

Dispatch and Python orchestration were the primary bottlenecks.

---

## 5. CUDA Graph Replay (Static)

Capturing a single decode step with CUDA Graph:

- Eliminated repeated kernel launches
- Removed Python loop overhead
- Reduced CPU per step by ~15x
- Increased throughput by ~16x

GPU compute time remained nearly constant.

---

## 6. Mask-Based Dynamic Batching

Attempting to support dynamic active sequences using masking:

- Preserved graph validity
- Introduced additional elementwise overhead
- Reduced throughput significantly (~7.4k tok/s)

Conclusion: mask-based batching reintroduces overhead.

---

## 7. Graph Pool Strategy

Maintaining one graph per batch size:

- Preserves static shapes
- Avoids masking
- Avoids recapture
- Maintains ~66k tok/s

Graph pooling is superior to mask-based dynamic batching.

---

## 8. Realistic Serving Loop

Introducing host-side request lifecycle management:

- Active request list
- Completion checks
- Python-level orchestration

Throughput drops to ~22k tok/s.

Dispatch is eliminated, but orchestration remains.

---

## 9. Three-Layer Decode Overhead Model

T_total = T_compute
        + T_dispatch
        + T_orchestration

Where:

T_compute → GPU math
T_dispatch → kernel launch + driver overhead
T_orchestration → host-side scheduling logic

CUDA Graph removes T_dispatch only.

Future optimization must address T_orchestration.

---

## 10. Implications

- Kernel fusion alone does not solve dispatch-bound regimes.
- Python-heavy serving engines leave performance untapped.
- Graph + C++ runtime core is a logical next step.
- Persistent kernels could eliminate dispatch entirely.

---

## 11. Limitations

- Single GPU
- No distributed execution
- No swap/preemption in graph runtime
- Small-batch focused
- Synthetic workload

Results are conceptual and architectural, not absolute production metrics.

---

## Conclusion

Dispatch overhead is a dominant factor in iterative decode.

CUDA Graph Replay significantly reduces per-token latency.

However, runtime architecture becomes the next optimization frontier.

Engineering effort must shift from kernel tuning to runtime orchestration design.

