import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange



def cross_entropy_loss(
    logits: Float[Tensor, "batch vocab_size"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:

    """
     loss = -sum(log(p(y_i|x_i))) / n
     p(y_i|x_i) = exp(logits[y_i]) / sum(exp(logits[j])) # softmax
    """

    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    max = torch.max(logits, dim=-1, keepdim=True).values
    log_sum = torch.stack([torch.logsumexp(lg - max[i].unsqueeze(-1), dim=-1) for i, lg in enumerate(logits)])
    loss = max - target_logits + log_sum
    return loss.mean()