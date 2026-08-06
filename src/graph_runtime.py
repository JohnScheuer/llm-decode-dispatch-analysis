import torch


class GraphRuntime:
    def __init__(
        self,
        batch_size: int,
        hidden_size: int,
        max_seq_len: int = 1024,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.max_seq_len = max_seq_len

        # Preallocate KV
        self.kv = torch.zeros(
            batch_size, max_seq_len, hidden_size, device=device
        )

        self.current_step = 0

        # Active mask (1 = active, 0 = inactive)
        self.active_mask = torch.ones(batch_size, device=device)

        # Model layers
        self.attn_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
        self.mlp = torch.nn.Linear(hidden_size, hidden_size).to(device)

        # Static buffers
        self.input_buffer = torch.zeros(batch_size, hidden_size, device=device)
        self.output_buffer = torch.zeros(batch_size, hidden_size, device=device)

        self.graph = None

    # --------------------------------------------------
    # STATIC DECODE STEP (MASKED)
    # --------------------------------------------------

    def _decode_step(self):
        attn_out = self.attn_proj(self.input_buffer)
        mlp_out = self.mlp(attn_out)

        # Apply active mask (broadcast)
        mlp_out = mlp_out * self.active_mask.unsqueeze(-1)

        # Write into fixed slot
        self.kv[:, self.current_step].copy_(mlp_out)

        self.output_buffer.copy_(mlp_out)

    # --------------------------------------------------
    # CAPTURE
    # --------------------------------------------------

    def capture(self):
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()

        # Warmup
        self._decode_step()
        torch.cuda.synchronize()

        with torch.cuda.graph(g):
            self._decode_step()

        self.graph = g

    # --------------------------------------------------
    # REPLAY
    # --------------------------------------------------

    def replay(self, x, active_indices=None):
        """
        active_indices: list of indices that are active
        """

        # Reset mask
        self.active_mask.zero_()

        if active_indices is None:
            self.active_mask.fill_(1.0)
        else:
            self.active_mask[active_indices] = 1.0

        self.input_buffer.copy_(x)
        self.graph.replay()

        self.current_step += 1
        return self.output_buffer
