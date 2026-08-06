import torch


class GraphDecodeEngine:
    def __init__(self, batch_size, hidden_size, max_seq_len=1024, device="cuda"):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.max_seq_len = max_seq_len

        # Preallocate KV max
        self.kv = torch.zeros(batch_size, max_seq_len, hidden_size, device=device)

        # Fixed modules
        self.attn_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
        self.mlp = torch.nn.Linear(hidden_size, hidden_size).to(device)

        # Static buffers
        self.input_buffer = torch.zeros(batch_size, hidden_size, device=device)
        self.output_buffer = torch.zeros(batch_size, hidden_size, device=device)

        self.current_step = 0
        self.graph = None

    def _decode_step_static(self):
        attn_out = self.attn_proj(self.input_buffer)
        mlp_out = self.mlp(attn_out)

        # Write into fixed slot (no dynamic index inside capture)
        self.kv[:, self.current_step].copy_(mlp_out)

        self.output_buffer.copy_(mlp_out)

    def capture(self):
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()

        # Warmup one static step
        self._decode_step_static()

        torch.cuda.synchronize()

        with torch.cuda.graph(g):
            self._decode_step_static()

        self.graph = g

    def replay(self, x):
        self.input_buffer.copy_(x)
        self.graph.replay()
        self.current_step += 1
        return self.output_buffer
