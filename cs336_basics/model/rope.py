import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange




def get_sin_cos_rope(theta_base: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> (
        tuple)[Float: [Tensor, "max_seq_len d_k // 2"], Float: [Tensor, "max_seq_len d_k // 2"]]:

    j = torch.arange(0, d_k // 2, device=device)
    inv_freq: Float[Tensor, d_k // 2] = theta_base ** (-2 * j / d_k)
    thetas: Float[Tensor, max_seq_len, d_k // 2] = torch.outer(torch.arange(max_seq_len, device=device), inv_freq)
    return thetas.cos(), thetas.sin()


class RopeEmbeddings(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        cos, sin = get_sin_cos_rope(theta, d_k, max_seq_len, device=device)
        self.register_buffer("sin", sin, persistent=False) # persistent=False means they’re not saved in checkpoints, which is reasonable since they’re deterministic.
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: (Float[Tensor, "... sequence_length d_k"]), token_positions: Int[Tensor, "... sequence_length"]) -> Float[Tensor, "... sequence_length d_k"]:
        in_type = x.dtype

        if token_positions is not None:
            token_positions = token_positions.long()
            sin = self.sin[token_positions].unsqueeze(1)
            cos = self.cos[token_positions].unsqueeze(1)
        else:
            sin = self.sin[:x.size(-2)] #Advanced: arbitrary positions (useful for KV cache, packed sequences, sliding windows)
            cos = self.cos[:x.size(-2)]
        # print(x.shape, sin.shape, cos.shape)
        x_pairs = rearrange(x.to(torch.float32), "... seq_len (d_k2 t) -> ... seq_len d_k2 t", t=2)
        # print(x_pairs.shape)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        # print(x1.shape, x2.shape)

        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos

        x_rotated = torch.stack((x_rotated_1, x_rotated_2), dim=-1)
        x_rotated = rearrange(x_rotated, "... seq_len d_k2 t -> ... seq_len (d_k2 t)", t=2)
        return x_rotated.to(in_type)


if __name__ == "__main__":
    rot = RopeEmbeddings(10, 32, 16)
