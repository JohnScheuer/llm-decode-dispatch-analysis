import torch
import time
from src.graph_pool_runtime import GraphPoolRuntime

MAX_BATCH = 8
HIDDEN = 1024
STEPS = 200

runtime = GraphPoolRuntime(MAX_BATCH, HIDDEN)

torch.cuda.synchronize()

# Test varying batch sizes
start = time.perf_counter()

for i in range(STEPS):
    batch = 4 if i < 100 else 2
    x = torch.randn(batch, HIDDEN, device="cuda")
    runtime.replay(x)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print("Graph Pool Runtime:")
print("CPU per step:", elapsed / STEPS)
print("Approx Tokens/sec:", (4 * STEPS) / elapsed)
