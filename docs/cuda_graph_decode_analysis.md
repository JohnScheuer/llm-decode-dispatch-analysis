# CUDA Graph Decode Runtime Analysis

## Objective

Reduce per-token decode latency by eliminating:

- Kernel launch overhead
- Python orchestration overhead
- CUDA driver dispatch cost

Focus: iterative decode (1 token per step per active request).

---

## Baseline Decode Engine

Configuration:

- Batch size: 4
- Hidden size: 1024
- Steps: 200

Measured:

CPU per step: ~0.913 ms  
GPU per step: ~0.778 ms  

Total CPU time: ~280 ms  
Total CUDA time: ~10 ms  

### Interpretation

The dominant cost in iterative decode was CPU-side dispatch and kernel launch overhead.

GPU compute time was not the bottleneck.

---

## CUDA Graph Replay (Static Batch)

After capturing the decode step using CUDA Graph:

CPU per step: ~0.058 ms  
Tokens/sec: ~68k  

This represents ~15x improvement in dispatch overhead elimination.

Key insight:

> CUDA Graph removes Python loop + kernel launch overhead almost entirely.

---

## Active Mask Strategy

To support dynamic active sequences, a mask-based approach was implemented.

Result:

CPU per step increased to ~0.536 ms  
Tokens/sec dropped significantly (~7.4k)

Conclusion:

Mask-based dynamic batching reintroduces significant overhead.

---

## Graph Pool Strategy

Instead of masking, a graph pool was created:

- One graph per batch_size (1..max_batch)
- No shape mutation
- No mask multiplication
- No recapture in hot path

Result:

CPU per step: ~0.060 ms  
Tokens/sec: ~66k  

Performance preserved while supporting dynamic batch sizes.

---

## Nsight Profiling Summary

Baseline:

Self CPU time: ~280 ms  
Self CUDA time: ~10 ms  

Graph Pool:

Self CPU time: ~11 ms  
Self CUDA time: ~8 ms  

Conclusion:

Dispatch overhead dominated baseline runtime.  
CUDA Graph eliminated ~96% of CPU dispatch time.

---

## Engineering Conclusions

1. CUDA Graph Replay is highly effective for iterative decode.
2. CPU dispatch overhead can dominate small-batch serving.
3. Mask-based dynamic batching reduces graph efficiency.
4. Graph pooling by batch size preserves maximum performance.
5. Preallocation and pointer stability are mandatory for graph capture.
6. Dynamic memory allocation during capture is invalid.

---

## Limitations

- Fixed max sequence length required
- No dynamic memory allocation allowed during capture
- Swap and preemption disabled in this prototype
- RNG-based sampling must be handled carefully

---

## Final Result

CUDA Graph + Graph Pool achieved:

~15x reduction in CPU overhead  
~16x increase in tokens/sec  
while preserving dynamic batch capability.

This demonstrates that per-token dispatch overhead is a critical bottleneck in LLM serving.

