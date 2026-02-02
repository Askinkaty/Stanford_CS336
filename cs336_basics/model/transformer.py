import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.base_functions import RMSNorm, SwiGLU, Embedding, Linear, softmax

_MAX_SEQ_LEN = 4096

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_seq_len: int = _MAX_SEQ_LEN, theta: float = 10000.0,
                 device: str = "cpu", dtype: torch.dtype | None = None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

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
                 d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: str = "cpu",
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_projection = Linear(d_model, vocab_size, device=device, dtype=dtype)

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


    def generate(
        self,
        x: Int[Tensor, "b seq"],
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_p: int | None = None,
    ) -> Int[Tensor, "b seq+new_seq"]:

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = self.forward(x)
                if temperature == 0:
                    logits = logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = softmax(logits[:, -1, :] / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        mask = cumulative_probs <= top_p
                        mask[:, 0] = True # ensure at least one token
                        original_mask = mask.gather(-1, sorted_indices.argsort(dim=-1))
                        for i in range(probs.size(0)):
                            probs[i][~original_mask[i]] = 0.0
                            probs[i] /= probs[i].sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=-1)
                if (next_token[-1:] == eos_token_id).all(dim=-1).item():
                    break
        return x







if __name__ == "__main__":
    d_model = 1024
    d_ff = 2048
    num_heads = 8
    theta = 10_000
    block = TransformerBlock(d_model, num_heads, theta, d_ff / d_model)
    x = torch.randn(4, 64, 1024)

    print(block(x).shape)
