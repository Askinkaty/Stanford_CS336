import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.base_functions import RMSNorm, SwiGLU

_MAX_SEQ_LEN = 4096

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_seq_len: int = _MAX_SEQ_LEN, theta: float = 10000.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads,
                                                max_seq_len=max_seq_len, theta=theta)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "b seq d_model"],
        token_positions: Int[Tensor, "b seq"] | None = None,
    ) -> Float[Tensor, "b seq d_model"]:

        x = x + self.attn(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x