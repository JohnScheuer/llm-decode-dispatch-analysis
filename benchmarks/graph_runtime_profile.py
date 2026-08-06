import torch
import time
from src.graph_runtime import GraphRuntime

BATCH = 4
HIDDEN = 1024
STEPS = 500

engine = GraphRuntime(BATCH, HIDDEN)

x = torch.randn(BATCH, HIDDEN, device="cuda")

engine.capture()

torch.cuda.synchronize()
cpu_start = time.perf_counter()

for _ in range(STEPS):
    engine.replay(x)

torch.cuda.synchronize()
cpu_elapsed = time.perf_counter() - cpu_start

print("GraphRuntime:")
print("CPU per step:", cpu_elapsed / STEPS)
print("Tokens/sec:", (BATCH * STEPS) / cpu_elapsed)
