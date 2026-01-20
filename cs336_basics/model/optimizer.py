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



if __name__ == "__main__":
    weights = nn.Parameter(torch.randn(10, 10))
    opt = SGD([weights], lr=1e0)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).norm()
        print(loss.item())
        loss.backward()
        opt.step()