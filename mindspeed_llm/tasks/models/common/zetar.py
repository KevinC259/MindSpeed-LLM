# copyright  AIGCode 2025
# @author: chenqiuwu@aigcode.net
# @file: zetar_tmp.py
# @brief: ZetaR optimizer temporary extraction
# @details: ZetaR optimizer implementation extracted for modification
# @version: 1.0.0
# @date: 2025-12-28
# @copyright: Copyright (c) 2025 AIGCode
# @license: MIT
# @contact: contact@aigcode.net
import math
import torch
import torch.optim as optim
import torch.nn.functional as F

#@torch.compile
def polar_decomposition_ns(G, steps=5, eps=1e-7):
    """Stable polar decomposition in float32."""
    assert len(G.shape) == 2
    X = G.to(torch.float32)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    scale = X.norm(p='fro') + eps
    X = X / scale
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype), scale

@torch.no_grad()
def get_moonshot_scale(param_shape):
    if len(param_shape) < 2:
        return 1.0
    A, B = param_shape[:2]
    return 0.2 * math.sqrt(max(A, B))

class ZetaR(optim.Optimizer):
    """
    ZetaR: Fast & Stable Adaptive Optimizer for LLM Pretraining.
    
    Key innovations:
      - Curvature-preconditioned Muon: NS applied to m / sqrt(H)
      - Prodigy global LR adaptation
      - Landscape-aware flatness control
      - Cold-start acceleration + dynamic regularization
    """

    def __init__(
        self,
        params,
        lr=1.0,
        lr_base=0.08,          # ↑ Increased from 0.02 for faster start
        weight_decay=0.01,
        betas=(0.9, 0.95),
        rho=0.1,               # ↓ Reduced (pre-conditioning handles curvature)
        ns_steps=5,
        d0=1e-2,               # ↑ Larger initial D
        d_coef=1.5,
        max_lr=0.15,           # ↑ Slightly higher cap
        flat_thresh=1.0,
        ravine_damp=0.5,
        flat_boost=2.0,
        trust_region_factor=2.5,
        initial_boost_steps=100,
        cold_start_decay_steps=100,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            lr_base=lr_base,
            weight_decay=weight_decay,
            betas=betas,
            rho=rho,
            ns_steps=ns_steps,
            d0=d0,
            d_coef=d_coef,
            max_lr=max_lr,
            flat_thresh=flat_thresh,
            ravine_damp=ravine_damp,
            flat_boost=flat_boost,
            trust_region_factor=trust_region_factor,
            initial_boost_steps=initial_boost_steps,
            cold_start_decay_steps=cold_start_decay_steps,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        """Update Hessian using current gradients (from sampled loss)."""
        for group in self.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.full_like(p, 1e-3)
                # SophiaMuon style update
                state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_base = group['lr_base']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            rho_nominal = group['rho']
            ns_steps = group['ns_steps']
            d0 = group['d0']
            d_coef = group['d_coef']
            max_lr = group['max_lr']
            flat_thresh = group['flat_thresh']
            ravine_damp = group['ravine_damp']
            flat_boost = group['flat_boost']
            trust_region_factor = group['trust_region_factor']
            initial_boost_steps = group['initial_boost_steps']
            cold_start_decay_steps = group['cold_start_decay_steps']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad.zero_()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.full_like(p, 1e-3)
                    state['s'] = torch.zeros_like(p)
                    state['d'] = torch.tensor(d0, device=p.device, dtype=torch.float32)
                    state['d_denom'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                d = state['d']
                d_denom = state['d_denom']

                # --- Update momentum and Hessian ---
                exp_avg.lerp_(grad, 1 - beta1)
                # SophiaMuon style update
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # --- Adaptive Hessian Damping (Cold Start) ---
                rho_adaptive = rho_nominal * (1.0 + 9.0 * math.exp(-step / cold_start_decay_steps))
                if p.ndim >= 2:
                    if p.ndim > 2:
                        orig_shape = p.shape
                        m_view = exp_avg.view(p.size(0), -1)
                        h_view = exp_avg_sq.view(p.size(0), -1)
                        s_view = s.view(p.size(0), -1)
                    else:
                        orig_shape = None
                        m_view, h_view, s_view = exp_avg, exp_avg_sq, s



                    # --- Landscape Flatness Indicator ---
                    m_norm = m_view.norm()
                    h_trace = h_view.mean()
                    flatness = m_norm / (torch.sqrt(h_trace) + rho_adaptive + eps)

                    if flatness > flat_thresh:
                        alpha = 1.0 + (flat_boost - 1.0) * (1.0 - flat_thresh / (flatness + eps))
                    else:
                        alpha = ravine_damp + (1.0 - ravine_damp) * (flatness / (flat_thresh + eps))
                    alpha = torch.clamp(alpha, ravine_damp, flat_boost)

                    # ---------------------------------------------------------
                    # ✅ CHANGED: Post-scale mode (SophiaMuon style)
                    # 1. Orthogonalize momentum directly
                    # 2. Scale by 1/Hessian
                    # ---------------------------------------------------------
                    
                    # 1. Muon: Orthogonalize momentum
                    U, g_scale = polar_decomposition_ns(m_view, steps=ns_steps)
                    
                    # 2. Sophia: Scaling
                    hess_clamped = h_view + rho_adaptive
                    sophia_scale = 1.0 / hess_clamped
                    # Optional: clamp scale to prevent explosion (common in Sophia)
                    sophia_scale = sophia_scale.clamp(min=0.05, max=20.0)
                    
                    raw_update = U * sophia_scale

                    # --- Global Scaling & Landscape ---
                    moon_scale = get_moonshot_scale(p.shape)
                    raw_update = raw_update * moon_scale * alpha

                    # --- Initial Exploration Boost ---
                    if step <= initial_boost_steps:
                        boost = 1.0 + (flat_boost - 1.0) * (1.0 - (step - 1) / initial_boost_steps)
                        raw_update = raw_update * boost

                    # --- Adaptive Trust Region ---
                    update_rms = raw_update.norm() / (raw_update.numel() ** 0.5 + eps)
                    ref_rms = moon_scale / (raw_update.size(-1) ** 0.5 + eps)
                    scale_norm = g_scale / math.sqrt(m_view.numel())
                    adaptive_tr_factor = 1.0 + 1.5 * torch.exp(-scale_norm * 5.0)
                    clip_threshold = adaptive_tr_factor * ref_rms
                    if update_rms > clip_threshold:
                        raw_update.mul_(clip_threshold / (update_rms + eps))

                    final_update = raw_update.view_as(p) if orig_shape else raw_update

                    # --- Prodigy Warm-Start ---
                    if step == 1:
                        grad_norm = grad.norm()
                        if grad_norm > 1e-6:
                            initial_d = lr_base / (grad_norm + 1e-6)
                            d.fill_(max(d0, initial_d))

                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr

                    # --- Prodigy State Update ---
                    s.add_(final_update, alpha=lr_eff)
                    g_norm_sq = grad.norm().pow(2)
                    d_denom_new = d_denom + lr_eff * g_norm_sq
                    if d_denom_new > eps:
                        d_num = s.norm()
                        d_hat = d_num / torch.sqrt(d_denom_new)
                        d_new = torch.clamp(d_hat, d.item(), d.item() * d_coef)
                        if d_new * lr_base > max_lr:
                            d_new = max_lr / lr_base
                        d.copy_(d_new)
                        d_denom.copy_(d_denom_new)

                    # --- Dynamic Weight Decay ---
                    if wd > 0:
                        dynamic_wd_factor = 1.0 + 0.3 * torch.tanh(scale_norm * 10.0)
                        effective_wd = wd * dynamic_wd_factor
                        p.data.mul_(1 - lr_eff * effective_wd)

                    p.data.add_(final_update, alpha=-lr_eff)

                    if orig_shape is not None:
                        s.copy_(s_view.view_as(s))

                else:
                    bias_correction2 = 1 - beta2 ** step
                    h_corrected = exp_avg_sq / bias_correction2
                    denom = torch.sqrt(h_corrected) + rho_adaptive
                    scale_1d = 1.0 / denom
                    scale_1d = torch.clamp(scale_1d, 0.1, 10.0)
                    update_1d = exp_avg * scale_1d

                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr

                    if wd > 0:
                        p.data.mul_(1 - lr_eff * wd)
                    p.data.add_(update_1d, alpha=-lr_eff)
                    # Fallback to standard Sophia for 1D params (bias, layernorm)
                    '''ratio = (exp_avg.abs() / (rho_adaptive * 5120 * exp_avg_sq + 1e-15)).clamp(None, 1)
                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr
                    if wd > 0:
                        p.data.mul_(1 - lr_eff * wd)
                    p.addcmul_(exp_avg.sign(), ratio, value=-lr_eff)
                    # Fallback to AdamW for 1D
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr
                    step_size = lr_eff * math.sqrt(bias_correction2) / bias_correction1
                    if wd > 0:
                        p.data.mul_(1 - lr_eff * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)'''

        return loss
