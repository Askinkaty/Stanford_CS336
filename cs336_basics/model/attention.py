import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
from cs336_basics.model.base_functions import softmax
from cs336_basics.model.rope import RopeEmbeddings

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Float[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:

        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        d_k = Q.size(-1)
        scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype))
        scores *= scale
        # print('scores', scores.shape)
        if mask is not None:
            scores.masked_fill_(~mask, float("-inf"))
        attn_weights = softmax(scores, dim=-1)
        # print(attn_weights.shape)
        # print(V.shape)
        # a = attn_weights @ V
        # print(a.shape)

        output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
        # return attn_weights @ V
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Projections match test fixtures that provide weight-only state_dicts.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention()
        self.rope = (
            RopeEmbeddings(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            if theta is not None
            else None
        )

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Int[Tensor, "b seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        batch, seq, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # print(Q.shape, K.shape, V.shape)

        Q = rearrange(Q, "b seq (head d_k) -> b head seq d_k", head=self.num_heads)
        K = rearrange(K, "b seq (head d_k) -> b head seq d_k", head=self.num_heads)
        V = rearrange(V, "b seq (head d_v) -> b head seq d_v", head=self.num_heads)
 
        # print(Q.shape, K.shape, V.shape)
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones((seq, seq), dtype=torch.bool, device=Q.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        # print(mask.shape)

        attn_output = self.attention(Q, K, V, mask=mask)
        # print('att output', attn_output.shape)

        attn_output = rearrange(attn_output, "b head seq d_v -> b seq (head d_v)")
 
        # print(attn_output.shape)

        output = self.o_proj(attn_output)
 
        # print('final output', output.shape)
        return output
