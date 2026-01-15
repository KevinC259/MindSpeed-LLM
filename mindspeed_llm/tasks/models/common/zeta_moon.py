import math
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    norm = X.norm() + eps
    X = X / norm
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class ZetaMoon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        trust_region_threshold: float = 0.1,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            trust_region_threshold=trust_region_threshold,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    def _get_moonshot_scale(self, param_shape):
        if len(param_shape) < 2:
            return 1.0
        a, b = param_shape[:2]
        return 0.2 * math.sqrt(max(a, b))

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
            trust_thresh = group["trust_region_threshold"]
            eps = group.get("eps", 1e-8)
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

                    update_norm = update_in.norm()
                    ortho_update = zeropower_via_newtonschulz5(update_in, steps=ns_steps)
                    scale = self._get_moonshot_scale(p.shape)
                    scaled_lr = lr * scale
                    gate = torch.tanh(update_norm / (trust_thresh + 1e-6)).pow(2)
                    final_lr = scaled_lr * gate

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)

                    update = ortho_update.view_as(p)
                    p.data.add_(update, alpha=-final_lr)
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

