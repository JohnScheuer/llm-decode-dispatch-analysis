import torch


class BaselineDecodeEngine:
    def __init__(self, batch_size, hidden_size, device="cuda"):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device

        self.kv = torch.zeros(batch_size, 1024, hidden_size, device=device)
        self.pos = torch.zeros(batch_size, dtype=torch.int32, device=device)

        self.attn_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
        self.mlp = torch.nn.Linear(hidden_size, hidden_size).to(device)

    def step(self, x):
        # Simulate attention
        attn_out = self.attn_proj(x)

        # Simulate MLP
        mlp_out = self.mlp(attn_out)

        # Append to KV
        for b in range(self.batch_size):
            p = self.pos[b]
            self.kv[b, p] = mlp_out[b]
            self.pos[b] += 1

        return mlp_out
