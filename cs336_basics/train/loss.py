import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange



def cross_entropy_loss(
    logits: Float[Tensor, "batch seq vocab_size"],
    targets: Int[Tensor, "batch seq"],
) -> Float[Tensor, ""]:

    """
     loss = -sum(log(p(y_i|x_i))) / n
     p(y_i|x_i) = exp(logits[y_i]) / sum(exp(logits[j])) # softmax
    """

    if targets.dtype != torch.long:
        targets = targets.long()
    # logits: [B, S, V], targets: [B, S]
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1)) + max_logits.squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()
