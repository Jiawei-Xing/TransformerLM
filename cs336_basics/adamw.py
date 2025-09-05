from collections.abc import Callable, Iterable
from typing import Optional
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lamb = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # update first moment
                m = state.get("m", 0)
                grad1 = p.grad.data
                m_new = beta1 * m + (1 - beta1) * grad1
                state["m"] = m_new

                # update second moment
                v = state.get("v", 0)
                grad2 = grad1 * grad1
                v_new = beta2 * v + (1 - beta2) * grad2
                state["v"] = v_new

                # update loss
                t = state.get("t", 1) # prevent division by zero
                alpha_t = alpha * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.data -= alpha_t * m_new / (v_new ** 0.5 + eps)
                p.data *= (1 - alpha * lamb)
                state["t"] = t + 1

        return loss

