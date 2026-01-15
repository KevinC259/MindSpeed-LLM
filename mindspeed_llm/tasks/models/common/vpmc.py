# Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# @author: chenqiuwu@aigcode.net

import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

def get_moonshot_scale(param_shape):
    if len(param_shape) < 2:
        return 1.0
    A, B = param_shape[:2]
    return 0.2 * math.sqrt(max(A, B))
import torch_npu

def safe_linalg_solve(A, B):
    """NPU-safe linalg.solve with fallback."""
    if hasattr(torch_npu, 'npu_linear_solve'):
        try:
            return torch_npu.npu_linear_solve(A, B)
        except:
            pass
    # Fallback (should not happen on CANN>=7.0)
    return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)

class VPMC(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.03,
        betas=(0.9, 0.995),
        rho=0.04,
        ns_steps=5,
        eps=1e-12,
        use_cayley=True, 
        weight_decay=0.01, momentum=0.95, nesterov=True,
        d0=1e-3,
        d_coef=1.5,
        max_lr=0.15,
        landscape_warmup_steps=50,
    ):
        defaults = dict(
            lr=lr, betas=betas, rho=rho, ns_steps=ns_steps,
            eps=eps, use_cayley=use_cayley, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            d0=d0, d_coef=d_coef, max_lr=max_lr,
            landscape_warmup_steps=landscape_warmup_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def newton_schulz(self, G, steps=3, eps=1e-7):
        if G.numel() == 0:
            return G
        frob_norm = G.norm()
        if frob_norm < eps:
            return G
        X = G / frob_norm

        orig_shape = X.shape
        # Work with tall matrix for stability
        if X.shape[0] < X.shape[1]:
            X = X.T
            transposed = True
        else:
            transposed = False

        for _ in range(steps):
            XTX = X.T @ X
            eye = torch.eye(XTX.size(0), device=X.device, dtype=X.dtype)
            X = 0.5 * X @ (3 * eye - XTX)

        if transposed:
            X = X.T
        
        # Safety: ensure output shape matches input
        if X.shape != orig_shape:
            return G
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_base = group['lr']
            beta1, beta2 = group['betas']
            ns_steps = group['ns_steps']
            use_cayley = group['use_cayley']
            eps = group['eps']
            wd = group['weight_decay']
            rho = group['rho']
            d0 = group['d0']
            d_coef = group['d_coef']
            max_lr = group['max_lr']
            landscape_warmup = group['landscape_warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    param_id += 1
                    continue

                # Get original shape from DistributedOptimizer
                if hasattr(self, '_param_shapes'):
                    orig_shape = self._param_shapes[param_id]
                    orig_dim = self._param_dims[param_id]
                else:
                    orig_shape = p.shape
                    orig_dim = p.dim()

                # Restore 2D+ view for structured update
                if orig_dim >= 2 and p.numel() == orig_shape.numel():
                    p_view = p.view(orig_shape)
                    grad_view = p.grad.view(orig_shape)
                else:
                    p_view = p
                    grad_view = p.grad

                grad = grad_view
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad.zero_()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
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

                # ONLY apply Cayley to square matrices
                is_square = (orig_dim == 2 and orig_shape[0] == orig_shape[1])
                apply_cayley = use_cayley and is_square

                if apply_cayley:
                    grad_proj = grad - p_view @ (grad.T @ p_view)
                else:
                    grad_proj = grad

                exp_avg.lerp_(grad_proj, 1 - beta1)
                exp_avg_sq.lerp_(grad_proj * grad_proj, 1 - beta2)

                # --- Prodigy LR with safe warm-start ---
                lr_eff_raw = (d * lr_base).item()
                if lr_eff_raw > max_lr:
                    lr_eff_raw = max_lr

                if step == 1:
                    grad_norm = grad.norm()
                    if grad_norm > 1e-6:
                        initial_d = lr_base / (grad_norm + 1e-6)
                        initial_d = min(initial_d, 1.0)
                        d.fill_(max(d0, initial_d))
                        lr_eff_raw = (d * lr_base).item()
                        if lr_eff_raw > max_lr:
                            lr_eff_raw = max_lr

                # --- Landscape-aware scaling ---
                blend_steps = 10
                if step <= landscape_warmup:
                    landscape_factor = 1.0
                else:
                    g_norm = grad.norm()
                    m_norm = exp_avg.norm()
                    flatness1 = 0.5
                    if g_norm > 1e-8 and m_norm > 1e-8:
                        cos_sim = torch.dot(grad.view(-1), exp_avg.view(-1)) / (g_norm * m_norm)
                        cos_sim = cos_sim.clamp(-1.0, 1.0)
                        flatness1 = (cos_sim + 1.0) / 2.0

                    h_mean = exp_avg_sq.mean()
                    h_max = exp_avg_sq.max()
                    curv_cond = (h_max + 1e-8) / (h_mean + 1e-8)
                    flatness2 = 1.0 / (1.0 + torch.log(curv_cond + 1e-8))
                    flatness2 = flatness2.clamp(0.1, 0.99)

                    flatness = 0.6 * flatness1 + 0.4 * flatness2
                    dynamic_factor = 0.5 + 1.5 * flatness
                    dynamic_factor = max(dynamic_factor, 0.8)

                    if step <= landscape_warmup + blend_steps:
                        t = (step - landscape_warmup) / blend_steps
                        landscape_factor = (1 - t) * 1.0 + t * dynamic_factor
                    else:
                        landscape_factor = dynamic_factor

                lr_eff = lr_eff_raw * landscape_factor
                lr_eff = min(lr_eff, max_lr * 1.5)
                lr_eff = max(lr_eff, 1e-8)

                # --- Handle non-square parameters ---
                if not apply_cayley:
                    bias_corr1 = 1 - beta1 ** step
                    bias_corr2 = 1 - beta2 ** step
                    exp_avg_corr = exp_avg / bias_corr1
                    exp_avg_sq_corr = exp_avg_sq / bias_corr2
                    denom = exp_avg_sq_corr.sqrt().add_(eps)
                    step_size = lr_eff * math.sqrt(bias_corr2) / bias_corr1

                    if wd > 0:
                        p.data.mul_(1 - lr_eff * wd)
                    p.data.addcdiv_(exp_avg_corr, denom, value=-step_size)

                    s.add_(exp_avg_corr, alpha=lr_eff)
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
                    continue

                # --- Square parameter update (Cayley path) ---
                if orig_dim > 2:
                    m_view = exp_avg.view(orig_shape[0], -1)
                    h_view = exp_avg_sq.view(orig_shape[0], -1)
                else:
                    m_view = exp_avg
                    h_view = exp_avg_sq

                sqrt_h = h_view.sqrt() + rho
                precond_m = m_view / sqrt_h
                precond_m = precond_m.clamp(min=-10.0, max=10.0)

                U = self.newton_schulz(precond_m, steps=ns_steps)
                moon_scale = get_moonshot_scale(p_view.shape)
                direction = U * moon_scale

                # Trust region
                update_rms = direction.norm() / (direction.numel() ** 0.5 + eps)
                ref_rms = moon_scale / (direction.size(-1) ** 0.5 + eps)
                if update_rms > 2.0 * ref_rms:
                    direction.mul_(2.0 * ref_rms / (update_rms + eps))

                # === 显存优化版 Cayley + Vector Transport ===
                W = p_view
                G = direction
                
                # Reuse exp_avg as temporary buffer for A
                temp_A = exp_avg  # Shape: [n, n]
                temp_A.copy_(G @ W.T)
                temp_A.sub_(W @ G.T)  # A = G@W.T - W@G.T

                alpha_val = 0.5 * lr_eff
                temp_A.mul_(alpha_val)  # alpha * A

                # Q_minus = I - alpha * A (in-place)
                Q_minus = temp_A
                Q_minus.neg_()  # -alpha * A
                Q_minus.diagonal(dim1=-2, dim2=-1).add_(1.0)  # + I

                # Pre-allocate target buffer if needed
                if not hasattr(self, '_cayley_temp') or self._cayley_temp.shape != W.shape:
                    self._cayley_temp = torch.empty_like(W)
                temp_target = self._cayley_temp
                torch.matmul(temp_A, W, out=temp_target)  # (alpha * A) @ W
                temp_target.add_(W)  # + W

                W_new = safe_linalg_solve(Q_minus, temp_target)
                p.copy_(W_new)

                # Vector transport: exp_avg ← W_new @ (W.T @ exp_avg)
                if not hasattr(self, '_vt_temp') or self._vt_temp.shape != exp_avg.shape:
                    self._vt_temp = torch.empty_like(exp_avg)
                vt_temp = self._vt_temp
                torch.matmul(W.T, exp_avg, out=vt_temp)      # W.T @ exp_avg
                torch.matmul(W_new, vt_temp, out=exp_avg)    # W_new @ (W.T @ exp_avg)

                # Prodigy state update
                s.add_(direction, alpha=lr_eff)
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

                if wd > 0:
                    exp_avg.data.mul_(1 - lr_eff * wd)

        return loss
    
class VPMC_b2(optim.Optimizer):
    """
    VPMC + Prodigy: Vector-Transported Muon-Cayley with Prodigy Adaptive LR

    Design:
    - Prodigy only controls a global scalar learning rate multiplier `d`.
    - Cayley update uses effective LR: lr_eff = d * lr_base.
    - Prodigy states (s, d_denom, d) are maintained per parameter.
    - 黎曼结构 (Cayley + vector transport) is preserved.
    """
    def __init__(
        self,
        params,
        lr=0.08,                  # This is lr_base for Prodigy
        betas=(0.9, 0.995),
        rho=0.04,
        ns_steps=5,
        eps=1e-12,
        use_cayley=True,
        weight_decay=0.01,
        d0=1e-2,                  # Prodigy initial d
        d_coef=1.5,               # Prodigy growth rate
        max_lr=0.15,              # Prodigy max lr
    ):
        defaults = dict(
            lr=lr, betas=betas, rho=rho, ns_steps=ns_steps,
            eps=eps, use_cayley=use_cayley, weight_decay=weight_decay,
            d0=d0, d_coef=d_coef, max_lr=max_lr
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def newton_schulz(self, G, steps=5, eps=1e-7):
        if G.numel() == 0:
            return G
        frob_norm = G.norm()
        if frob_norm < eps:
            return G
        X = G / frob_norm

        a, b = X.shape
        transposed = False
        if a < b:
            X = X.T
            transposed = True

        for _ in range(steps):
            XTX = X.T @ X
            eye = torch.eye(XTX.size(0), device=X.device, dtype=X.dtype)
            X = 0.5 * X @ (3 * eye - XTX)

        if transposed:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_base = group['lr']          # Prodigy base LR
            beta1, beta2 = group['betas']
            ns_steps = group['ns_steps']
            use_cayley = group['use_cayley']
            eps = group['eps']
            wd = group['weight_decay']
            rho = group['rho']
            d0 = group['d0']
            d_coef = group['d_coef']
            max_lr = group['max_lr']

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
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)               # Prodigy: running sum of updates
                    state['d'] = torch.tensor(d0, device=p.device, dtype=torch.float32)
                    state['d_denom'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)

                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                d = state['d']
                d_denom = state['d_denom']

                # --- 1. Determine if parameter is on Stiefel manifold ---
                is_stiefel = (p.ndim == 2 and p.shape[0] >= p.shape[1])
                apply_cayley = use_cayley and is_stiefel

                # --- 2. Riemannian Projection (if on Stiefel) ---
                if apply_cayley:
                    grad_proj = grad - p @ (grad.T @ p)
                else:
                    grad_proj = grad

                # --- 3. Update EMA states (for Sophia-style curvature, optional) ---
                exp_avg.lerp_(grad_proj, 1 - beta1)
                exp_avg_sq.lerp_(grad_proj * grad_proj, 1 - beta2)

                # --- 4. Prodigy: Adaptive Learning Rate (Scalar d) ---
                # Compute effective learning rate
                lr_eff = (d * lr_base).item()
                if lr_eff > max_lr:
                    lr_eff = max_lr

                # Warm-start d on first step
                if step == 1:
                    grad_norm = grad.norm()
                    if grad_norm > 1e-6:
                        initial_d = lr_base / (grad_norm + 1e-6)
                        d.fill_(max(d0, initial_d.item()))
                        lr_eff = (d * lr_base).item()
                        if lr_eff > max_lr:
                            lr_eff = max_lr
                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                exp_avg_sq_corr = exp_avg_sq / bias_corr2
                hess_mean = exp_avg_sq_corr.mean().sqrt() + eps
                adaptive_scale = (1.0 / torch.clamp(hess_mean, min=1e-3, max=10.0))
                # --- 5. Handle non-Stiefel parameters (AdamW + Prodigy LR) ---
                if not apply_cayley:

                    exp_avg_corr = exp_avg / bias_corr1
                    
                    denom = exp_avg_sq_corr.sqrt().add_(eps)
                    step_size = lr_eff * math.sqrt(bias_corr2) / bias_corr1

                    denom = exp_avg_sq_corr.sqrt().add_(eps)
                    step_size = lr_eff * adaptive_scale * math.sqrt(bias_corr2) / bias_corr1
                    if wd > 0:
                        p.data.mul_(1 - lr_eff * wd)
                    p.data.addcdiv_(exp_avg_corr, denom, value=-step_size)
                    
                    # Prodigy state update for 1D params
                    s.add_(exp_avg_corr, alpha=lr_eff)
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
                    continue

                # --- 6. Stiefel Manifold Update (Cayley + Muon) ---
                orig_shape = None
                if p.ndim > 2:
                    orig_shape = p.shape
                    p_view = p.view(p.size(0), -1)
                    m_view = exp_avg.view(p.size(0), -1)
                    h_view = exp_avg_sq.view(p.size(0), -1)
                else:
                    p_view = p
                    m_view = exp_avg
                    h_view = exp_avg_sq

                # Precondition momentum by curvature (optional Sophia-style)
                sqrt_h = h_view.sqrt() + rho
                precond_m = m_view / sqrt_h
                precond_m = precond_m.clamp(min=-10.0, max=10.0)

                # Muon: get orthogonal direction
                U = self.newton_schulz(precond_m, steps=ns_steps)

                # Moonshot scaling
                moon_scale = get_moonshot_scale(p_view.shape)
                direction = U * moon_scale

                # Trust region clipping
                update_rms = direction.norm() / (direction.numel() ** 0.5 + eps)
                ref_rms = moon_scale / (direction.size(-1) ** 0.5 + eps)
                if update_rms > 2.0 * ref_rms:
                    direction.mul_(2.0 * ref_rms / (update_rms + eps))

                if orig_shape is not None:
                    direction = direction.view_as(p)

                # --- Cayley Transform with Prodigy LR ---
                W = p
                G = direction
                A = G @ W.T - W @ G.T  # skew-symmetric

                alpha = 0.5 * lr_eff
                I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                Q_minus = I - alpha * A
                target = W + alpha * (A @ W)
                #W_new = W + alpha * (A @ W) + 0.5 * (alpha ** 2) * (A @ (A @ W))
                W_new = safe_linalg_solve(Q_minus, target)

                # Update parameter
                p.copy_(W_new)

                # Vector transport: rotate momentum
                exp_avg_rotated = W_new @ (W.T @ exp_avg)
                exp_avg.copy_(exp_avg_rotated)

                # --- Prodigy State Update (using the direction as "effective gradient") ---
                # Note: We use `direction` (not raw grad) because it's the actual update direction
                s.add_(direction, alpha=lr_eff)
                g_norm_sq = grad.norm().pow(2)  # use original grad norm for stability
                d_denom_new = d_denom + lr_eff * g_norm_sq

                if d_denom_new > eps:
                    d_num = s.norm()
                    d_hat = d_num / torch.sqrt(d_denom_new)
                    d_new = torch.clamp(d_hat, d.item(), d.item() * d_coef)
                    if d_new * lr_base > max_lr:
                        d_new = max_lr / lr_base
                    d.copy_(d_new)
                    d_denom.copy_(d_denom_new)

                # Weight decay (applied to momentum in tangent space)
                if wd > 0:
                    exp_avg.data.mul_(1 - lr_eff * wd)

        return loss
