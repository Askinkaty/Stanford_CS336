import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight: Float[Tensor, "o i"] = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sigma: float = 2 / (in_features + out_features)
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: Float[Tensor, "b ... i"]) -> Float[Tensor, "b ... o"]:
        out = einsum(x, self.weight, "b ... i, o i -> b ... o")
        return out


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        sigma: float = 2 / (num_embeddings + embedding_dim)
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: Int[Tensor, "batch ..."]) -> Float[Tensor, "batch ... d_model"]:
        out = torch.index_select(self.weight, dim=0, index=x.reshape(-1))
        return out.reshape(*x.shape, self.weight.size(1))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.gain: Float[Tensor, "d_model"] = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / norm_x
        result = x_normalized * self.gain
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Float[Tensor, "... d_model"]:
        super().__init__()
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        w1_out = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        w3_out = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        gated = (w1_out * torch.sigmoid(w1_out)) * w3_out
        output = einsum(gated, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return output


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x
