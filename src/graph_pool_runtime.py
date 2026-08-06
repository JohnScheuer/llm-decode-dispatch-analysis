import torch


class GraphPoolRuntime:
    def __init__(
        self,
        max_batch_size: int,
        hidden_size: int,
        max_seq_len: int = 1024,
        device: str = "cuda",
    ):
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.graphs = {}
        self.engines = {}

        # Pre-build graphs for each batch size
        for b in range(1, max_batch_size + 1):
            engine = _GraphUnit(b, hidden_size, max_seq_len, device)
            engine.capture()
            self.graphs[b] = engine
            self.engines[b] = engine

    def replay(self, x):
        batch_size = x.shape[0]

        if batch_size not in self.graphs:
            raise RuntimeError(f"No graph for batch_size={batch_size}")

        engine = self.graphs[batch_size]
        return engine.replay(x)


class _GraphUnit:
    def __init__(self, batch_size, hidden_size, max_seq_len, device):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.kv = torch.zeros(batch_size, max_seq_len, hidden_size, device=device)
        self.current_step = 0

        self.attn_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
        self.mlp = torch.nn.Linear(hidden_size, hidden_size).to(device)

        self.input_buffer = torch.zeros(batch_size, hidden_size, device=device)
        self.output_buffer = torch.zeros(batch_size, hidden_size, device=device)

        self.graph = None

    def _decode_step(self):
        attn_out = self.attn_proj(self.input_buffer)
        mlp_out = self.mlp(attn_out)

        self.kv[:, self.current_step].copy_(mlp_out)
        self.output_buffer.copy_(mlp_out)

    def capture(self):
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()

        self._decode_step()
        torch.cuda.synchronize()

        with torch.cuda.graph(g):
            self._decode_step()

        self.graph = g

    def replay(self, x):
        self.input_buffer.copy_(x)
        self.graph.replay()
        self.current_step += 1
        return self.output_buffer
