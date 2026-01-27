import math
from typing import Any
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        super().__init__(params=params, defaults=dict(lr=lr))

    def step(self, closure: Any | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 1e-6):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Any | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "t" not in state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                grad = p.grad.data
                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]


                m.mul_(beta1).add_(grad, alpha=1 - beta1) # m * beta1 + (1 - beta1) * grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v * beta2 + (1 - beta2) * (grad ** 2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr / bias_correction1 # lr * m_hat = lr * m / (1 - beta1**t)

                denom = v.sqrt().div_(math.sqrt(bias_correction2)).add_(eps) # sqrt(v_hat) + eps = sqrt(v / (1 - beta2**t)) + eps
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay) # data <- data - lr * weight_decay * data
                p.data.addcdiv_(m, denom, value=-step_size) # data <- data - step_size * m / denom
        return loss


def get_cosine_lr(it: int, max_learning_rate: float, min_learning_rate: float,
                  warmup_iters: int, cosine_cycle_iters: int, ) -> float:
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    elif it > cosine_cycle_iters:
        lr = min_learning_rate
    else:
        phase = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * phase)) * (max_learning_rate - min_learning_rate)
    return lr


def gradient_clipping(parameters: Iterable[Tensor], max_norm: float):
    total_squared_norm = torch.zeros((1,), device=parameters.__iter__().__next__().device)
    for p in parameters:
        if p.grad is not None:
            # total_squared_norm += p.grad.pow(2).sum()
            total_squared_norm += torch.linalg.norm(p.grad) ** 2
    total_norm = torch.sqrt(total_squared_norm)

    if total_norm > max_norm:
        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_( max_norm / (total_norm + 1e-6) )



if __name__ == "__main__":
    weights = nn.Parameter(torch.randn(10, 10))
    opt = SGD([weights], lr=1e0)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).norm()
        print(loss.item())
        loss.backward()
        opt.step()
