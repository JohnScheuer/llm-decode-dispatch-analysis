import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.baseline_engine import BaselineDecodeEngine
from src.graph_pool_runtime import GraphPoolRuntime

DEVICE = "cuda"
BATCH = 4
HIDDEN = 1024
STEPS = 200


def run_baseline():
    engine = BaselineDecodeEngine(BATCH, HIDDEN)
    x = torch.randn(BATCH, HIDDEN, device=DEVICE)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False
    ) as prof:
        with record_function("baseline_decode"):
            for _ in range(STEPS):
                engine.step(x)

    return prof


def run_graph_pool():
    engine = GraphPoolRuntime(max_batch_size=8, hidden_size=HIDDEN)
    x = torch.randn(BATCH, HIDDEN, device=DEVICE)

    engine.graphs[BATCH].current_step = 0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False
    ) as prof:
        with record_function("graph_pool_decode"):
            for _ in range(STEPS):
                engine.replay(x)

    return prof


def extract_cuda_time(event):
    if hasattr(event, "self_cuda_time_total"):
        return event.self_cuda_time_total
    if hasattr(event, "cuda_time_total"):
        return event.cuda_time_total
    return 0.0


def print_summary(prof, title):
    print("\n==============================")
    print(title)
    print("==============================")

    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    total_cuda = sum(extract_cuda_time(e) for e in prof.key_averages())
    total_cpu = sum(e.self_cpu_time_total for e in prof.key_averages())

    kernel_calls = sum(e.count for e in prof.key_averages()
                       if extract_cuda_time(e) > 0)

    print(f"\nTotal CUDA time (us): {total_cuda:.2f}")
    print(f"Total CPU time (us): {total_cpu:.2f}")
    print(f"Total CUDA kernel calls: {kernel_calls}")


def main():
    print("Profiling baseline...")
    baseline_prof = run_baseline()
    print_summary(baseline_prof, "BASELINE")

    torch.cuda.synchronize()

    print("\nProfiling graph pool...")
    graph_prof = run_graph_pool()
    print_summary(graph_prof, "GRAPH POOL")


if __name__ == "__main__":
    main()
