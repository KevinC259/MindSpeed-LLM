import math
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class AdaMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        betas=(0.9, 0.95),
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            betas=betas,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            beta1, beta2 = group.get("betas", (0.9, 0.95))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if p.ndim >= 2:
                    if p.ndim > 2:
                        g_in = g.view(g.size(0), -1)
                    else:
                        g_in = g

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g_in)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g_in)
                    update_in = g_in.add(buf, alpha=momentum) if nesterov else buf

                    sign_update = torch.sign(update_in)
                    ortho = zeropower_via_newtonschulz5(sign_update, steps=ns_steps)
                    flat = ortho.flatten()

                    if "v_buffer" not in state:
                        state["v_buffer"] = torch.zeros_like(flat)
                    v = state["v_buffer"]
                    v.mul_(momentum).addcmul_(flat, flat, value=(1 - momentum))
                    normed = flat.div(v.sqrt().add(eps))

                    a, b = p.shape[:2]
                    scale = 0.2 * math.sqrt(a * b) / (normed.norm() + eps)
                    update = (normed * scale).view_as(p)

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                else:
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.lerp_(g.pow(2), 1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    step_size = lr * math.sqrt(bc2) / bc1
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

