import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.base_functions import RMSNorm, SwiGLU, Embedding, Linear, softmax

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


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int,
                 d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                             max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_projection = Linear(d_model, vocab_size)

    def forward(
        self,
            x: Int[Tensor, "b seq"],
    ) -> Float[Tensor, "b seq vocab_size"]:

        emb = self.token_embedding(x)
        seq_len = x.size(1)
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        for layer in self.layers:
            emb = layer(emb, token_positions=token_positions)
        normed = self.norm(emb)
        logits = self.output_projection(normed)
        return logits
