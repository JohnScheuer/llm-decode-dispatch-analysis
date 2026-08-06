import torch
import time
from src.graph_engine import GraphDecodeEngine

BATCH = 4
HIDDEN = 1024
STEPS = 200

engine = GraphDecodeEngine(BATCH, HIDDEN)

x = torch.randn(BATCH, HIDDEN, device="cuda")

# Capture graph
engine.capture()

# CPU timing
torch.cuda.synchronize()
cpu_start = time.perf_counter()

for _ in range(STEPS):
    engine.replay(x)

torch.cuda.synchronize()
cpu_elapsed = time.perf_counter() - cpu_start

# GPU timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()

for _ in range(STEPS):
    engine.replay(x)

end_event.record()
torch.cuda.synchronize()

gpu_elapsed_ms = start_event.elapsed_time(end_event)

print("===== Graph Profile =====")
print("CPU total time:", cpu_elapsed)
print("CPU per step:", cpu_elapsed / STEPS)
print("GPU total time (ms):", gpu_elapsed_ms)
print("GPU per step (ms):", gpu_elapsed_ms / STEPS)
print("Tokens/sec:", (BATCH * STEPS) / cpu_elapsed)
print("=========================")
