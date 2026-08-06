import time
import torch
from src.graph_decode_runtime import GraphDecodeRuntime

BATCH = 4
HIDDEN = 1024
REQUESTS = 4
TOKENS = 200

runtime = GraphDecodeRuntime(BATCH, HIDDEN)

for i in range(REQUESTS):
    runtime.add_request(i, TOKENS)

start = time.perf_counter()
completed = runtime.run_until_complete(lambda x: None)
elapsed = time.perf_counter() - start

total_tokens = REQUESTS * TOKENS

print("Graph Decode Runtime:")
print("Elapsed:", elapsed)
print("Tokens/sec:", total_tokens / elapsed)
