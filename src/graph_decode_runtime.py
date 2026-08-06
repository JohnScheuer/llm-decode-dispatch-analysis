import torch
from typing import List, Callable
from src.graph_pool_runtime import GraphPoolRuntime


class DecodeRequest:
    def __init__(self, request_id, max_new_tokens):
        self.request_id = request_id
        self.max_new_tokens = max_new_tokens
        self.generated = 0
        self.finished = False


class GraphDecodeRuntime:
    """
    Minimal decode runtime using CUDA Graph Pool.
    No scheduler, no preemption, no swap.
    Pure iterative decode.
    """

    def __init__(self, max_batch_size, hidden_size, device="cuda"):
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size
        self.device = device

        self.graph_pool = GraphPoolRuntime(max_batch_size, hidden_size)
        self.active_requests: List[DecodeRequest] = []

    # ------------------------------------------------------
    # API
    # ------------------------------------------------------

    def add_request(self, request_id: int, max_new_tokens: int):
        if len(self.active_requests) >= self.max_batch_size:
            raise RuntimeError("Max batch size exceeded")

        self.active_requests.append(
            DecodeRequest(request_id, max_new_tokens)
        )

    def run_until_complete(self, generator: Callable):
        completed = []

        while self.active_requests:
            batch_size = len(self.active_requests)

            x = torch.randn(batch_size, self.hidden_size, device=self.device)

            self.graph_pool.replay(x)

            new_active = []

            for req in self.active_requests:
                req.generated += 1
                if req.generated >= req.max_new_tokens:
                    req.finished = True
                    completed.append(req)
                else:
                    new_active.append(req)

            self.active_requests = new_active

        return completed
