import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

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

class ZetaSophia(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        trust_region_threshold: float = 0.1,
        betas=(0.965, 0.99),
        rho: float = 0.04,
        mode: str = "post_scale",
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
            rho=rho,
            mode=mode,
        )
        super().__init__(params, defaults)

    def get_moonshot_scale(self, param_shape):
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
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            rho = group["rho"]
            trust_thresh = group["trust_region_threshold"]
            beta1, beta2 = group.get("betas", (0.965, 0.99))
            mode = group["mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1

                hess = state["exp_avg_sq"]
                hess.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                mom = state["exp_avg"]
                mom.mul_(beta1).add_(grad, alpha=1 - beta1)
                update_in = grad.add(mom, alpha=beta1) if nesterov else mom

                consistency = 0.0
                if p.ndim >= 2:
                    if p.ndim > 2:
                        m_in = update_in.view(mom.size(0), -1)
                        h_in = hess.view(hess.size(0), -1)
                    else:
                        m_in = update_in
                        h_in = hess

                    g_flat = grad.view(-1)
                    m_flat = mom.view(-1)
                    consistency = F.cosine_similarity(g_flat, m_flat, dim=0, eps=1e-6)

                    if mode == "pre_condition":
                        denom = h_in.sqrt().add_(1e-6)
                        pre_m = m_in / denom
                        update = zeropower_via_newtonschulz5(pre_m, steps=ns_steps)
                        scale = self.get_moonshot_scale(p.shape)
                        final_update = update * scale
                    else:
                        update_norm = update_in.norm()
                        ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
                        hess_clamped = torch.clamp(h_in, min=rho)
                        h_mean = hess_clamped.mean()
                        sophia_scale = h_mean / hess_clamped
                        boost_factor = 1.0
                        boost_thresh = 0.6
                        if consistency > boost_thresh:
                            ratio = (consistency - boost_thresh) / (1.0 - boost_thresh)
                            boost_factor = 1.0 + ratio * (trust_thresh - 1.0)
                        scale = self.get_moonshot_scale(p.shape)
                        gate = torch.tanh(update_norm / (trust_thresh + 1e-6)).pow(2)
                        final_update = ortho_direction * sophia_scale * boost_factor * scale * gate

                    update = final_update.view_as(p)
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1_f, beta2_f = 0.9, 0.95
                    exp_avg.lerp_(grad, 1 - beta1_f)
                    exp_avg_sq.lerp_(grad.pow(2), 1 - beta2_f)
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    bc1 = 1 - beta1_f ** state["step"]
                    bc2 = 1 - beta2_f ** state["step"]
                    step_size = lr * math.sqrt(bc2) / bc1
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
