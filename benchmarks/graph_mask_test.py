import torch
import time
from src.graph_runtime import GraphRuntime

BATCH = 4
HIDDEN = 1024
STEPS = 200

engine = GraphRuntime(BATCH, HIDDEN)
x = torch.randn(BATCH, HIDDEN, device="cuda")

engine.capture()

torch.cuda.synchronize()
start = time.perf_counter()

for i in range(STEPS):
    # simulate dynamic active shrink
    active = [0, 1] if i > 50 else [0, 1, 2, 3]
    engine.replay(x, active_indices=active)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print("Graph with Mask:")
print("CPU per step:", elapsed / STEPS)
print("Tokens/sec:", (BATCH * STEPS) / elapsed)
