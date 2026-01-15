# Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# @author: chenqiuwu@aigcode.net

import math
from re import S
import torch
import torch.optim as optim
import torch.nn.functional as F


#@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Optimized Newton-Schulz iteration (No changes).
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    norm = X.norm() + eps
    X = X / norm
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


#@torch.compile
def cayley_update(W, G, lr, steps=2):
    """
    Zeta-Cayley 核心算子：
    使用 Neumann 级数近似 Cayley 变换，实现流形上的更新。
    不需要矩阵求逆，只用矩阵乘法。
    
    W: 当前权重 (Ortho)
    G: 梯度
    lr: 学习率
    """
    # 1. 构造黎曼梯度 (Riemannian Gradient) 对应的反对称矩阵 A
    # A = G @ W^T - W @ G^T
    # 这代表了切空间中的 "旋转力矩"
    WT = W.T
    A = G @ WT - W @ G.T
    
    # 2. Cayley 变换近似: W_new = (I + lr/2 * A)(I - lr/2 * A)^(-1) W
    # 令 X = lr/2 * A
    # 近似 (I - X)^(-1) ≈ I + X + X^2
    # Update ≈ (I + X)(I + X + X^2) W ≈ (I + 2X + 2X^2) W
    
    X = (lr * 0.5) * A
    
    # Neumann Series Iteration (Fixed Point Iteration)
    # 这种写法比纯级数展开数值更稳
    # Q = I
    # for _ in range(steps):
    #    Q = I + X @ Q
    # W_new = (I + X) @ Q @ W
    
    # 极简二阶近似 (速度最快，对于小 LR 足够)
    # W_new = (I + 2X + 2X^2) W
    #       = W + 2XW + 2X(XW)
    #       = W + lr*A @ W + 0.5 * lr^2 * A @ (A @ W)
    
    AW = A @ W
    W_new = W + lr * AW + 0.5 * (lr**2) * (A @ AW)
    
    return W_new

# --- 统一入口类 ---
class ZetaS_factory:
    def __new__(cls, params, version="Pro", **kwargs):
        # 1. 动态拼接类名
        target_class_name = f"ZetaS_{version}"
        
        # 2. 从当前全局作用域中查找该类
        target_cls = globals().get(target_class_name)
        
        if target_cls is None:
            raise AttributeError(f"未找到类实现: {target_class_name}。请检查配置值 '{version}' 是否正确。")
        
        # 3. 实例化并返回具体实现
        return target_cls(params, **kwargs)
 




#@torch.compile
def polar_decomposition_ns(G, steps=5, eps=1e-7):
    """
    Compute the orthogonal polar factor of G via Newton-Schulz iteration.
    Always compute in float32 for numerical stability.
    Returns:
        U: orthogonal matrix (same dtype as G)
        _norm: spectral norm of input (for diagnostics only)
    """
    assert len(G.shape) == 2
    X = G.to(torch.float32)  # Critical: avoid bfloat16 precision loss

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    _norm = X.norm() + eps
    X = X / _norm

    # 5th-order Newton-Schulz coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X.to(G.dtype), _norm


@torch.no_grad()
def get_moonshot_scale(param_shape):
    """Moonshot-style adaptive scaling based on leading dimensions."""
    if len(param_shape) < 2:
        return 1.0
    A, B = param_shape[:2]
    return 0.2 * math.sqrt(max(A, B))




class ZetaS_Prodigy(optim.Optimizer):
    """
    ZetaS with Prodigy-style global LR + Landscape-Aware Flatness Control.
    
    Combines:
      - Muon: orthogonal direction (U)
      - Sophia: per-element curvature scale (1/(h+rho))
      - Prodigy: global adaptive LR (d)
      - Landscape Awareness: flatness indicator F = ||m|| / sqrt(Tr(H))
    """

    def __init__(
        self,
        params,
        lr=1.0,                # Prodigy: initial d=1, actual LR = d * lr_base
        lr_base=0.02,          # Reference base LR
        weight_decay=0.01,
        betas=(0.9, 0.95),
        rho=0.04,
        ns_steps=5,
        d0=1e-6,               # Initial D estimate
        d_coef=1.0,            # Prodigy growth coefficient
        max_lr=0.1,            # Hard cap on d * lr_base
        flat_thresh=1.0,       # F < flat_thresh → ravine; F > flat_thresh → flat
        ravine_damp=0.5,       # Min alpha in ravine
        flat_boost=2.0,        # Max alpha in flat region
        trust_region_factor=2.5,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,              # This is 'd' in Prodigy
            lr_base=lr_base,
            weight_decay=weight_decay,
            betas=betas,
            rho=rho,
            ns_build=ns_steps,
            d0=d0,
            d_coef=d_coef,
            max_lr=max_lr,
            flat_thresh=flat_thresh,
            ravine_damp=ravine_damp,
            flat_boost=flat_boost,
            trust_region_factor=trust_region_factor,
            eps=eps,
        )
        super().__init__(params, defaults)

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
            rho = group['rho']
            ns_steps = group['ns_build']
            d0 = group['d0']
            d_coef = group['d_coef']
            max_lr = group['max_lr']
            flat_thresh = group['flat_thresh']
            ravine_damp = group['ravine_damp']
            flat_boost = group['flat_boost']
            trust_region_factor = group['trust_region_factor']
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
                    state['s'] = torch.zeros_like(p)          # Prodigy: running sum of updates
                    state['d'] = torch.tensor(d0, device=p.device, dtype=torch.float32)  # scalar
                    state['d_denom'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                d = state['d']
                d_denom = state['d_denom']

                # --- 1. Update momentum and Hessian ---
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.float().pow(2), 1 - beta2)

                # --- 2. Core update for 2D+ params ---
                if p.ndim >= 2:
                    if p.ndim > 2:
                        orig_shape = p.shape
                        p_view = p.view(p.size(0), -1)
                        m_view = exp_avg.view(p.size(0), -1)
                        h_view = exp_avg_sq.view(p.size(0), -1)
                        s_view = s.view(p.size(0), -1)
                    else:
                        orig_shape = None
                        m_view, h_view, s_view = exp_avg, exp_avg_sq, s

                    # --- 2.1 Compute flatness indicator F ---
                    m_norm = m_view.norm()  # ||m||
                    h_trace = h_view.mean()  # Tr(H)/d ≈ avg curvature
                    flatness = m_norm / (torch.sqrt(h_trace) + rho + eps)  # F = ||m|| / sqrt(Tr(H))

                    # --- 2.2 Landscape-aware alpha ---
                    if flatness > flat_thresh:
                        # Flat region: boost exploration
                        alpha = 1.0 + (flat_boost - 1.0) * (1.0 - flat_thresh / (flatness + eps))
                    else:
                        # Ravine or transition: dampen
                        alpha = ravine_damp + (1.0 - ravine_damp) * (flatness / (flat_thresh + eps))
                    alpha = torch.clamp(alpha, ravine_damp, flat_boost)

                    # --- 2.3 Muon direction + Sophia scale ---
                    U, g_scale = polar_decomposition_ns(m_view, steps=ns_steps)
                    scale_norm = g_scale / math.sqrt(m_view.numel())
                    sophia_scale = 1.0 / (h_view + rho)
                    sophia_scale = sophia_scale.clamp(min=0.05, max=20.0)
                    raw_update = U * sophia_scale

                    # --- 2.4 Moonshot global scale ---
                    moon_scale = get_moonshot_scale(p.shape)
                    raw_update = raw_update * moon_scale * alpha

                    # --- 2.5 Trust-region clipping ---
                    update_rms = raw_update.norm() / (raw_update.numel() ** 0.5 + eps)
                    ref_rms = moon_scale / (raw_update.size(-1) ** 0.5 + eps)
                    #clip_threshold = trust_region_factor * ref_rms
                    adaptive_tr_factor = 1.0 + 1.5 * torch.exp(-scale_norm * 5.0)
                    clip_threshold = adaptive_tr_factor * ref_rms
                    if update_rms > clip_threshold:
                        raw_update.mul_(clip_threshold / (update_rms + eps))

                    final_update = raw_update.view_as(p) if orig_shape else raw_update

                    # --- 2.6 Prodigy: update global LR d ---
                    # Effective LR for this param: d * lr_base
                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr

                    # Update Prodigy state
                    s.add_(final_update, alpha=lr_eff)
                    g_norm_sq = grad.norm().pow(2)
                    d_denom_new = d_denom + lr_eff * g_norm_sq

                    # Estimate new d
                    if d_denom_new > eps:
                        d_num = s.norm()
                        d_hat = d_num / torch.sqrt(d_denom_new)
                        # Apply growth constraint
                        d_new = torch.clamp(d_hat, d.item(), d.item() * d_coef)
                        # Hard cap by max_lr
                        if d_new * lr_base > max_lr:
                            d_new = max_lr / lr_base
                        d.copy_(d_new)
                        d_denom.copy_(d_denom_new)

                    # --- 2.7 Apply update ---
                    if wd > 0:
                        dynamic_wd_factor = 1.0 + 0.3 * torch.tanh(scale_norm * 10.0)
                        effective_wd = wd * dynamic_wd_factor
                        #p.data.mul_(1 - lr_eff * wd)
                        p.data.mul_(1 - lr_eff * effective_wd)
                    p.data.add_(final_update, alpha=-lr_eff)

                    # Save s back
                    if orig_shape:
                        s.copy_(s_view.view_as(s))

                else:
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
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class ZetaS(optim.Optimizer):
    """
    ZetaS Optimizer: Direction-Scale Decoupled Adaptive Optimization.
    
    - Direction: Muon (Newton-Schulz polar decomposition of momentum)
    - Scale: Sophia (1 / (Hessian + rho)) per-element curvature adaptation
    - Global normalization: Moonshot (dimension-aware scaling)
    - Stability: Trust-region RMS clipping + Hessian initialization

    Designed for high-dimensional, long-sequence LLM pretraining.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        rho=0.04,
        ns_steps=5,
        h_accum_steps=1,        # Gradient square accumulation for Hessian
        trust_region_factor=2.5,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            rho=rho,
            ns_steps=ns_steps,
            h_accum_steps=h_accum_steps,
            trust_region_factor=trust_region_factor,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            rho = group['rho']
            ns_steps = group['ns_steps']
            h_accum_steps = group['h_accum_steps']
            trust_region_factor = group['trust_region_factor']
            eps = group['eps']
            h_accum_steps = 8

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # NaN/Inf guard
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad.zero_()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Initialize Hessian with small positive value for stability
                    state['exp_avg_sq'] = torch.full_like(p, 1e-3)
                    if h_accum_steps > 1:
                        state['hess_buffer'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # --- 1. Hessian Estimation (with optional accumulation) ---
                grad_sq = grad.float().pow(2)

                if h_accum_steps > 1:
                    state['hess_buffer'].add_(grad_sq)
                    if step % h_accum_steps == 0:
                        avg_grad_sq = state['hess_buffer'] / h_accum_steps
                        exp_avg_sq.mul_(beta2).add_(avg_grad_sq, alpha=1 - beta2)
                        state['hess_buffer'].zero_()
                    elif step == 1:
                        # Ensure Hessian is initialized on first step
                        exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                # --- 2. Momentum Update ---
                exp_avg.lerp_(grad, 1 - beta1)  # exp_avg = beta1 * exp_avg + (1-beta1) * grad

                # --- 3. Core Update for 2D+ Parameters ---
                if p.ndim >= 2:
                    # Reshape to 2D if needed
                    if p.ndim > 2:
                        orig_shape = p.shape
                        p_view = p.view(p.size(0), -1)
                        m_view = exp_avg.view(p.size(0), -1)
                        h_view = exp_avg_sq.view(p.size(0), -1)
                    else:
                        orig_shape = None
                        p_view, m_view, h_view = p, exp_avg, exp_avg_sq

                    # --- 3.1 Muon: Get orthogonal direction from momentum ---
                    U, g_scale = polar_decomposition_ns(m_view, steps=ns_steps)  # (A, B)
                    scale_norm = g_scale / math.sqrt(m_view.numel())

                    # --- 3.2 Sophia: Per-element curvature-aware scaling ---
                    sophia_scale = 1.0 / (h_view + rho)
                    sophia_scale = sophia_scale.clamp(min=0.05, max=20.0)

                    # --- 3.3 Combine: Direction * Scale ---
                    update_view = U * sophia_scale

                    # --- 3.4 Moonshot: Global dimension-aware scaling ---
                    moon_scale = get_moonshot_scale(p.shape)
                    update_view = update_view * moon_scale

                    # --- 3.5 Trust-Region: RMS Clipping for stability ---
                    numel = update_view.numel()
                    update_rms = update_view.norm() / (numel ** 0.5 + eps)
                    ref_rms = moon_scale / (update_view.size(-1) ** 0.5 + eps)
                    #clip_threshold = trust_region_factor * ref_rms
                    # Make trust region tighter when momentum is large
                    adaptive_tr_factor = 1.0 + 1.5 * torch.exp(-scale_norm * 5.0)
                    clip_threshold = adaptive_tr_factor * ref_rms
                    if update_rms > clip_threshold:
                        update_view.mul_(clip_threshold / (update_rms + eps))

                    # Reshape back
                    if orig_shape is not None:
                        final_update = update_view.view_as(p)
                    else:
                        final_update = update_view

                    # Apply decoupled weight decay
                    if wd > 0:
                        dynamic_wd_factor = 1.0 + 0.3 * torch.tanh(scale_norm * 10.0)
                        effective_wd = wd * dynamic_wd_factor
                        p.data.mul_(1 - lr * effective_wd)

                    p.data.add_(final_update, alpha=-lr)

                # --- 4. Fallback for 1D Parameters (AdamW-style) ---
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



class ZetaS_poly(optim.Optimizer):
    """
    ZetaPro Optimizer: Muon + Moonshot + Sophia + Trust-Region Gating.

    Fixes numerical instability issues in high-dimensional, long-sequence training.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.01,
        trust_region_threshold=0.1,
        betas=(0.9, 0.95),
        rho=0.04,
        mode='post_scale', 
        boost_disable_step=150,  # disable consistency boost after this step
        eps=1e-7,
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
            boost_disable_step=boost_disable_step,  
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            rho = group['rho']
            trust_thresh = group['trust_region_threshold']
            beta1, beta2 = group['betas']
            #boost_disable_step = group['boost_disable_step']
            h_accum_steps = 8
            transport = True
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                # -----------------------------------------------------------
                # 1. 状态初始化
                # -----------------------------------------------------------
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.full_like(p, 1e-3)
                    # [New] Initialize Hessian Buffer if accumulation is enabled
                    if h_accum_steps > 1:
                        state['hess_buffer'] = torch.full_like(p, 1e-3)

                state['step'] += 1
                curr_step = state['step']
                
                # --- 1. Sophia Hessian Estimation (Accumulated) ---
                
                # Current Gradient Squared (float32)
                grad_sq = g.float().pow(2)
                
                if h_accum_steps > 1:
                    # A. Accumulation Mode
                    # Add to buffer
                    state['hess_buffer'].add_(grad_sq)
                    
                    # Only update actual Hessian state every N steps
                    if curr_step % h_accum_steps == 0:
                        # Average the accumulated squared gradients
                        avg_grad_sq = state['hess_buffer'] / h_accum_steps
                        
                        # Update EMA: H_t = beta * H_{t-1} + (1-beta) * avg_g^2
                        state['exp_avg_sq'].mul_(beta2).add_(avg_grad_sq, alpha=1 - beta2)
                        
                        # Reset buffer
                        state['hess_buffer'].zero_()
                        
                    # Corner case: First few steps before first update.
                    # We rely on 'rho' to handle the zero-Hessian case safely.
                    # Or we can force an update on step 1 to initialize.
                    elif curr_step == 1:
                        state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                else:
                    # B. Standard Per-Step Mode
                    state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                # Fetch current Hessian estimate (might be stale by < N steps, which is fine)
                hess = state['exp_avg_sq']

                # --- 2. Momentum Update ---
                buf = state['exp_avg']
                buf.mul_(beta1).add_(g, alpha=1 - beta1)
                
                if nesterov:
                    update_in = g.add(buf, alpha=beta1)
                else:
                    update_in = buf
                
                # -----------------------------------------------------------
                # 4. 核心融合逻辑 (针对 2D 矩阵)
                # -----------------------------------------------------------
                if p.ndim >= 2:
                    if p.ndim > 2:
                        #g_in = g.view(g.size(0), -1)
                        g_in = update_in.view(g.size(0), -1)
                        #m_in = update_in.view(momentum.size(0), -1)
                        #h_in = hess.view(hess.size(0), -1)
                    else:
                        g_in = g
                        #m_in = update_in
                        #h_in = hess

                    #g_flat = g.view(-1)
                    #m_flat = momentum.view(-1)
                    # 避免完全 norm 计算的开销，仅作近似或采样，这里写全量为了准确性
                    #consistency = F.cosine_similarity(g_flat, m_flat, dim=0, eps=1e-6)
                    # === 1. 计算当前梯度的几何基 (U_t) ===
                    # U_t 是当前点切空间的 "正交基底"
                    U_t, g_scale = polar_decomposition_ns(g_in, steps=ns_steps)
                    
                    # === 2. 黎曼动量传输 (Parallel Transport) ===
                    # 核心思想：不要直接加 m_{t-1}。
                    # 我们假设 m_{t-1} 是在上一个基底 U_{t-1} 下定义的。
                    # 如果流形弯曲了，U_t 和 U_{t-1} 会有一个旋转 R。
                    # m_new = m_old * R (近似)
                    
                    m_prev = state['exp_avg']
                    
                    if transport and state['step'] > 1:
                        # 这是一个简化的 Transport：投影修正
                        # 我们把旧动量 m_prev 投影到新基底 U_t 上，只保留 "顺路" 的分量
                        # 并丢弃那些因为流形旋转而变成 "阻力" 的垂直分量
                        
                        # Projection: P = U_t @ U_t.T
                        # m_aligned = P @ m_prev
                        # 实际上，我们可以做得更激进：直接用 U_t 替换方向，保留 m_prev 的模长
                        
                        m_norm = m_prev.norm()
                        g_dir = U_t # 这是当前梯度的纯方向
                        
                        # 混合策略：
                        # 在欧氏空间，m = beta * m + g
                        # 在黎曼空间，我们混合的是 "系数" 而不是 "向量"
                        # m_t = (beta * m_norm + g_scale) * U_t ???
                        # 不，这样丢失了历史方向信息。
                        
                        # 高级策略：Lie Transport
                        # m_transported = U_t @ (U_t.T @ m_prev) 
                        # 这把旧动量拉到了当前的正交标架下
                        m_transported = U_t @ (U_t.T @ m_prev)
                        
                        # 混合
                        m_new = buf * m_transported + g_in
                    else:
                        # 标准欧氏动量
                        m_new = buf * m_prev + g_in

                    # 更新动量状态
                    state['exp_avg'].copy_(m_new)
                    
                    # === 3. 测地线更新 (Geodesic Update) ===
                    # 我们再次对混合后的动量做正交化，确保走在流形上
                    # 这就是 Muon 的 update，但输入是经过 Transport 修正的 m_new
                    U_update, m_scale = polar_decomposition_ns(m_new, steps=ns_steps)
                    
                    # === 4. 谱还原 (Spectral Restoration) ===
                    # Muon 丢弃了 scale。Zeta-Riemann 试图还原它。
                    # 我们使用 m_scale (动量的模长) 作为置信度指标
                    # 但为了防止爆炸，必须结合 Moonshot scaling
                    
                    moon_scale = self.get_moonshot_scale(p.shape)
                    
                    # 创新点：动态曲率感知
                    # 如果 g_scale (当前梯度模长) 和 m_scale (动量模长) 差异巨大，说明曲率剧变
                    # 我们引入一个 "Curvature Friction"
                    curvature_consistency = 2 * (g_scale * m_scale) / (g_scale**2 + m_scale**2)
                    final_lr = lr * moon_scale * curvature_consistency
                    
                    # 简化版：直接使用 Moonshot
                    #final_lr = lr * moon_scale
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    update = U_update.view_as(p)
                    p.data.add_(update, alpha=-final_lr)

                # ---- Fallback: 1D params (embeddings, biases) -> AdamW-style ----
                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1_fallback, beta2_fallback = 0.9, 0.95

                    exp_avg.lerp_(g, 1 - beta1_fallback)
                    exp_avg_sq.lerp_(g.pow(2), 1 - beta2_fallback)

                    denom = exp_avg_sq.sqrt().add_(1e-8)

                    bias_correction1 = 1 - beta1_fallback ** state['step']
                    bias_correction2 = 1 - beta2_fallback ** state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
   
   
class ZetaS_Pro(optim.Optimizer):
    """
    ZetaPro Optimizer: Muon + Moonshot + Sophia + Trust-Region Gating.

    Fixes numerical instability issues in high-dimensional, long-sequence training.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.01,
        trust_region_threshold=0.1,
        betas=(0.9, 0.95),
        rho=1.0,
        mode='post_scale', 
        boost_disable_step=150,  # disable consistency boost after this step
        eps=1e-7,
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
            boost_disable_step=boost_disable_step,  
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            rho = group['rho']
            trust_thresh = group['trust_region_threshold']
            beta1, beta2 = group['betas']
            boost_disable_step = group['boost_disable_step']
            h_accum_steps = 8

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # ---- NaN / Inf Guard ----
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad.zero_()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if h_accum_steps > 1:
                        state['hess_buffer'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                curr_step = state['step']
                
                # --- 1. Sophia Hessian Estimation ---
                grad_sq = grad.float().pow(2)
                
                if h_accum_steps > 1:
                    if 'hess_buffer' not in state: # Safety check
                         state['hess_buffer'] = torch.zeros_like(p, dtype=torch.float32)
                    state['hess_buffer'].add_(grad_sq)
                    if curr_step % h_accum_steps == 0:
                        avg_grad_sq = state['hess_buffer'] / h_accum_steps
                        state['exp_avg_sq'].mul_(beta2).add_(avg_grad_sq, alpha=1 - beta2)
                        state['hess_buffer'].zero_()
                    elif curr_step == 1:
                        state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                else:
                    state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # ---- Sophia Hessian Diagonal Estimate ----
                #exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # ---- Momentum (EMA) ----
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if nesterov:
                    update_in = grad.add(exp_avg, alpha=beta1)
                else:
                    update_in = exp_avg

                # ---- Handle 2D+ Parameters with Muon + Sophia ----
                if p.ndim >= 2:
                    # Flatten to 2D if needed
                    if p.ndim > 2:
                        m_in = update_in.view(update_in.size(0), -1)
                        h_in = exp_avg_sq.view(exp_avg_sq.size(0), -1)
                        orig_shape = p.shape
                    else:
                        m_in = update_in
                        h_in = exp_avg_sq
                        orig_shape = None

                    # ---- Consistency Boost (only early training) ----
                    boost_factor = 1.0
                    if step <= boost_disable_step:
                        g_flat = grad.view(-1)
                        m_flat = exp_avg.view(-1)
                        g_norm = g_flat.norm()
                        m_norm = m_flat.norm()
                        if g_norm > 1e-8 and m_norm > 1e-8:
                            consistency = F.cosine_similarity(g_flat, m_flat, dim=0).clamp(-1.0, 1.0)
                            boost_thresh = 0.6
                            if consistency > boost_thresh:
                                ratio = (consistency - boost_thresh) / (1.0 - boost_thresh)
                                boost_factor = 1.0 + ratio * (trust_thresh - 1.0)  # note: trust_thresh < 1

                    # Muon: get orthogonal direction (rotation only)
                    ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)

                    # Sophia: per-element scaling with damping
                    hess_clamped = h_in + rho  # use rho as damping (critical!)
                    sophia_scale = 1.0 / hess_clamped

                    # Prevent extreme per-element scaling
                    sophia_scale = sophia_scale.clamp(min=0.05, max=20.0)

                    # Apply Moonshot scale
                    moonshot_scale = get_moonshot_scale(p.shape)

                    # Combine
                    raw_update = ortho_direction * sophia_scale * boost_factor * moonshot_scale

                    # ---- Trust Region: RMS Clipping (per-parameter update stability) ----
                    numel = raw_update.numel()
                    update_rms = raw_update.norm() / (numel ** 0.5 + 1e-8)
                    # Theoretical RMS of Muon output: ~ moonshot_scale / sqrt(last_dim)
                    ref_rms = moonshot_scale / (ortho_direction.size(-1) ** 0.5 + 1e-8)
                    clip_threshold = 2.5 * ref_rms  # allow some flexibility

                    if update_rms > clip_threshold:
                        raw_update.mul_(clip_threshold / (update_rms + 1e-8))

                    final_update = raw_update


                    # Reshape if needed
                    if orig_shape is not None:
                        final_update = final_update.view_as(p)

                    # Decoupled Weight Decay
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)

                    p.data.add_(final_update, alpha=-lr)

                # ---- Fallback: 1D params (embeddings, biases) -> AdamW-style ----
                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1_fallback, beta2_fallback = 0.9, 0.95

                    exp_avg.lerp_(grad, 1 - beta1_fallback)
                    exp_avg_sq.lerp_(grad.pow(2), 1 - beta2_fallback)

                    denom = exp_avg_sq.sqrt().add_(1e-8)

                    bias_correction1 = 1 - beta1_fallback ** state['step']
                    bias_correction2 = 1 - beta2_fallback ** state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
   
class ZetaS_passive(optim.Optimizer):
    """
    ZetaS-v4: With Enhanced Dynamic RMS Blending.
    
    New Features:
        - rms_blend_rate (float): Base blending ratio. 0.5 means 50/50 mix.
        - dynamic_blending (bool): If True, adjusts blend ratio based on signal strength.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.01,
                 betas=(0.9, 0.95), rho=0.03, 
                 hessian_accum_steps=1,
                 rms_blend_rate=0.5, # Base blend rate
                 dynamic_blending=True, eps=1e-7): # Enable adaptive blending
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        betas=betas, rho=rho, 
                        hessian_accum_steps=hessian_accum_steps,
                        rms_blend_rate=rms_blend_rate,
                        dynamic_blending=dynamic_blending, eps=eps)

        super().__init__(params, defaults)

    def get_moonshot_scale(self, param_shape):
        if len(param_shape) < 2:
            return 1.0
        return 0.2 * math.sqrt(max(param_shape[0], param_shape[1]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            rho = group['rho']
            beta1, beta2 = group['betas']
            h_accum_steps = group['hessian_accum_steps']
            
            # Blend Params
            base_blend = group['rms_blend_rate']
            use_dynamic = group['dynamic_blending']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # --- State Init ---
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                    if h_accum_steps > 1:
                        state['hess_buffer'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                curr_step = state['step']
                
                # --- 1. Sophia Hessian Estimation ---
                grad_sq = grad.float().pow(2)
                
                if h_accum_steps > 1:
                    if 'hess_buffer' not in state: # Safety check
                         state['hess_buffer'] = torch.zeros_like(p, dtype=torch.float32)
                    state['hess_buffer'].add_(grad_sq)
                    if curr_step % h_accum_steps == 0:
                        avg_grad_sq = state['hess_buffer'] / h_accum_steps
                        state['exp_avg_sq'].mul_(beta2).add_(avg_grad_sq, alpha=1 - beta2)
                        state['hess_buffer'].zero_()
                    elif curr_step == 1:
                        state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                else:
                    state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                hess = state['exp_avg_sq']

                # --- 2. Momentum Update ---
                buf = state['exp_avg']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                if nesterov:
                    update_in = grad.add(buf, alpha=momentum)
                else:
                    update_in = buf

                # --- 3. 2D Matrix Logic ---
                if p.ndim >= 2:
                    if p.ndim > 2:
                        m_in = update_in.view(buf.size(0), -1)
                        h_in = hess.view(hess.size(0), -1)
                    else:
                        m_in = update_in
                        h_in = hess

                    # A. Calculate RMS of current momentum (Signal Strength)
                    # m_rms represents the "natural" scale of the update
                    m_rms = m_in.norm() / math.sqrt(m_in.numel())
                    
                    # B. Get Moonshot Scale (Structural Constraints)
                    moonshot_scale = self.get_moonshot_scale(p.shape)
                    
                    # C. Enhanced Dynamic Blending Logic
                    if use_dynamic:
                        # Logic: 
                        # If RMS is very small (convergence), trust RMS more (decay naturally).
                        # If RMS is huge (explosion/early training), trust Moonshot more (clamp it).
                        
                        # Compute ratio: How strong is the signal vs the heuristic?
                        # Add eps to avoid div zero
                        ratio = m_rms / (moonshot_scale + 1e-8)
                        
                        # Sigmoid-like transition centered at ratio=1.0
                        # If ratio > 1 (High Energy), blend -> 1.0 (Moonshot dominated)
                        # If ratio < 1 (Low Energy), blend -> 0.0 (RMS dominated)
                        # We map ratio to a weight [0, 1]
                        # Using a soft switch: tanh((ratio - 1) * steepness)
                        # Let's map it simply:
                        
                        # Determine alpha (Weight for Moonshot)
                        # We want alpha high when ratio is high.
                        # Clamp ratio to reasonable bounds [0, 2] for calculation
                        clamped_ratio = torch.clamp(ratio, 0.0, 2.0)
                        
                        # Linear interpolation around 1.0
                        # If ratio=1, alpha = base_blend (e.g., 0.5)
                        # If ratio=2, alpha -> 0.9 (Trust Moonshot to constrain)
                        # If ratio=0, alpha -> 0.1 (Trust RMS to decay)
                        alpha = base_blend + (clamped_ratio - 1.0) * 0.4
                        alpha = torch.clamp(alpha, 0.1, 0.9)
                        
                    else:
                        alpha = base_blend

                    # Final Scale Composition
                    # Scale = alpha * Moonshot + (1-alpha) * RMS
                    final_scale = alpha * moonshot_scale + (1.0 - alpha) * m_rms

                    # D. Newton-Schulz Orthogonalization
                    ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
                    
                    # E. Relative Sophia Scaling
                    h_clamped = torch.clamp(h_in, min=rho)
                    h_mean = h_clamped.mean()
                    sophia_shape = h_mean / h_clamped
                    sophia_shape.clamp_(min=0.5, max=2.0)
                    sophia_shape = sophia_shape.to(p.dtype)

                    # Apply Updates
                    update = ortho_direction * sophia_shape * final_scale
                    
                    if p.ndim > 2:
                        update = update.view_as(p)

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(update, alpha=-lr)

                # --- 4. 1D Vector Logic ---
                else:
                    denom = hess.sqrt().add_(1e-8)
                    bc1 = 1 - momentum ** curr_step
                    bc2 = 1 - beta2 ** curr_step
                    step_size = lr * math.sqrt(bc2) / bc1
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(buf, denom, value=-step_size)

        return loss


class ZetaS_vfix(optim.Optimizer):
    """
    ZetaS-v3: With Fixed-Step Hessian Accumulation.
    
    New Args:
        hessian_accum_steps (int): Accumulate gradients for N steps before updating Hessian estimates.
                                   Recommended: 4, 8, or 16. Helps denoise curvature.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.01,
                 betas=(0.9, 0.95), rho=0.03, 
                 hessian_accum_steps=1, eps=1e-7): # 默认为1，即每步更新
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        betas=betas, rho=rho, 
                        hessian_accum_steps=hessian_accum_steps, eps=eps)

        super().__init__(params, defaults)

    def get_moonshot_scale(self, param_shape):
        if len(param_shape) < 2:
            return 1.0
        return 0.2 * math.sqrt(max(param_shape[0], param_shape[1]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            rho = group['rho']
            beta1, beta2 = group['betas']
            h_accum_steps = group['hessian_accum_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # --- State Init ---
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    # Use float32 for Hessian to prevent underflow
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                    
                    # [New] Initialize Hessian Buffer if accumulation is enabled
                    if h_accum_steps > 1:
                        state['hess_buffer'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                curr_step = state['step']
                
                # --- 1. Sophia Hessian Estimation (Accumulated) ---
                
                # Current Gradient Squared (float32)
                grad_sq = grad.float().pow(2)
                
                if h_accum_steps > 1:
                    # A. Accumulation Mode
                    # Add to buffer
                    state['hess_buffer'].add_(grad_sq)
                    
                    # Only update actual Hessian state every N steps
                    if curr_step % h_accum_steps == 0:
                        # Average the accumulated squared gradients
                        avg_grad_sq = state['hess_buffer'] / h_accum_steps
                        
                        # Update EMA: H_t = beta * H_{t-1} + (1-beta) * avg_g^2
                        state['exp_avg_sq'].mul_(beta2).add_(avg_grad_sq, alpha=1 - beta2)
                        
                        # Reset buffer
                        state['hess_buffer'].zero_()
                        
                    # Corner case: First few steps before first update.
                    # We rely on 'rho' to handle the zero-Hessian case safely.
                    # Or we can force an update on step 1 to initialize.
                    elif curr_step == 1:
                        state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                else:
                    # B. Standard Per-Step Mode
                    state['exp_avg_sq'].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                # Fetch current Hessian estimate (might be stale by < N steps, which is fine)
                hess = state['exp_avg_sq']

                # --- 2. Momentum Update ---
                buf = state['exp_avg']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                if nesterov:
                    update_in = grad.add(buf, alpha=momentum)
                else:
                    update_in = buf

                # --- 3. 2D Matrix Logic (Muon + Zeta) ---
                if p.ndim >= 2:
                    if p.ndim > 2:
                        m_in = update_in.view(buf.size(0), -1)
                        h_in = hess.view(hess.size(0), -1)
                    else:
                        m_in = update_in
                        h_in = hess

                    # RMS Recovery (Previous Fix)
                    m_rms = m_in.norm() / math.sqrt(m_in.numel())
                    
                    # Newton-Schulz Orthogonalization
                    ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
                    
                    # Relative Sophia Scaling
                    # Clamp Hessian to avoid division by zero (Rho)
                    # If Hessian is 0 (first few steps of accumulation), it becomes Rho.
                    h_clamped = torch.clamp(h_in, min=rho)
                    h_mean = h_clamped.mean()
                    
                    sophia_shape = h_mean / h_clamped
                    sophia_shape.clamp_(min=0.01, max=2.0)
                    sophia_shape = sophia_shape.to(p.dtype)

                    # Final Composition
                    moonshot_scale = self.get_moonshot_scale(p.shape)
                    
                    # Optional: Blend RMS (Currently using strict Moonshot scale for stability)
                    m_rms = m_rms.to(p.dtype)
                    moonshot_scale = moonshot_scale * 0.5 + m_rms * 0.5
                    update = ortho_direction * sophia_shape * moonshot_scale
                    
                    if p.ndim > 2:
                        update = update.view_as(p)

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(update, alpha=-lr)

                # --- 4. 1D Vector Logic (AdamW Fallback) ---
                else:
                    # For vectors, we also use the accumulated Hessian if available
                    denom = hess.sqrt().add_(1e-8)
                    
                    # Bias Correction logic needs care with accumulation
                    # But for simplicity, we treat step count normally
                    bc1 = 1 - momentum ** curr_step
                    bc2 = 1 - beta2 ** curr_step
                    
                    step_size = lr * math.sqrt(bc2) / bc1
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.addcdiv_(buf, denom, value=-step_size)

        return loss
    
class ZetaS2(optim.Optimizer):
    """
    Zeta-S Optimizer:
    Combines Muon's second-order preconditioning + Moonshot's dimension scaling + Zeta's trust-region damping.

    Advantages:
    1. Inherits Muon's ability for fast escape from ravines.
    2. Inherits Moonshot's adaptive scaling for different layer widths.
    3. [Zeta unique] Introduces Trust-Region Gating to solve Muon's oscillation problem
       when gradients are very small (near optimum).

    Args:
        params: Model parameters
        lr: Learning rate
        momentum: Momentum factor
        nesterov: Whether to use Nesterov momentum
        ns_steps: Newton-Schulz iteration steps
        weight_decay: Weight decay coefficient
        trust_region_threshold: Threshold for trust region gating
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.01,
                 trust_region_threshold=0.1,
                 betas=(0.9, 0.95), rho=0.04, mode='post_scale', eps: float = 1e-7):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        trust_region_threshold=trust_region_threshold,
                        betas=betas, rho=rho, mode=mode, eps=eps
                        )

        super().__init__(params, defaults)

    def get_moonshot_scale(self, param_shape):
        """Moonshot paper core: Scaling based on matrix dimension"""
        if len(param_shape) < 2:
            return 1.0
        A, B = param_shape[:2]
        # 0.2 * sqrt(max_dim) is the heuristic recommended by Moonshot
        return 0.2 * math.sqrt(max(A, B))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            rho = group['rho']
            trust_thresh = group['trust_region_threshold']
            beta1, beta2 = group['betas']
            mode = group['mode']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # -----------------------------------------------------------
                # 1. 状态初始化
                # -----------------------------------------------------------
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                # -----------------------------------------------------------
                # 2. Sophia 核心：Hessian 对角线估计
                #    使用 Gauss-Newton-Bartlett 近似: E[g * g]
                # -----------------------------------------------------------
                hess = state['exp_avg_sq']
                hess.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)

                # -----------------------------------------------------------
                # 3. 动量更新 (EMA)
                # -----------------------------------------------------------
                buf = state['exp_avg']
                buf.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Nesterov 修正: 使用预见动量进行更新计算
                if nesterov:
                    update_in = grad.add(buf, alpha=beta1)
                else:
                    update_in = buf

                #consistency = 0.0
                # -----------------------------------------------------------
                # 4. 核心融合逻辑 (针对 2D 矩阵)
                # -----------------------------------------------------------
                if p.ndim >= 2:
                    # 将多维张量展平为 2D 矩阵以适应 Muon
                    if p.ndim > 2:
                        m_in = update_in.view(momentum.size(0), -1)
                        h_in = hess.view(hess.size(0), -1)
                    else:
                        m_in = update_in
                        h_in = hess

                    m_rms = m_in.norm() / math.sqrt(m_in.numel())
                    
                    # === [Correction 2] Newton-Schulz Orthogonalization ===
                    # This gives us the "Pure Direction" (Spectrally Normalized)
                    ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
                    
                    # === [Correction 3] Relative Sophia Scaling ===
                    # Instead of 1/h, we use mean(h)/h.
                    # This applies curvature shaping WITHOUT changing the global learning rate scale.
                    # This is critical for compatibility with Muon's spectral LR.
                    
                    # Clamp Hessian to avoid division by zero (Rho)
                    h_clamped = torch.clamp(h_in, min=rho)
                    h_mean = h_clamped.mean()
                    
                    # "Shape Factor": >1 means flat direction (accelerate), <1 means steep (decelerate)
                    sophia_shape = h_mean / h_clamped
                    
                    # Soft Clamp to prevent explosions (Trust Region)
                    sophia_shape.clamp_(min=0.5, max=2.0)
                    
                    # Cast back to param dtype
                    sophia_shape = sophia_shape.to(p.dtype)

                    # === Final Composition ===
                    # Update = Ortho_Dir * Sophia_Shape * Moonshot_Scale
                    
                    moonshot_scale = self.get_moonshot_scale(p.shape)
                    
                    # Re-inject a fraction of RMS to handle warmup/decay transitions smoothly
                    # We blend the fixed Moonshot scale with the dynamic RMS scale.
                    # As training progresses, m_rms drops, acting as an implicit scheduler.
                    # Heuristic: 0.5 * Moonshot + 0.5 * RMS (Scalable)
                    # For strict Muon replication: just use moonshot_scale. 
                    # But for ZetaS win: use the blend.
                    #final_scale = moonshot_scale # Keeping it strict for now to isolate Sophia fix
                    final_scale = moonshot_scale * 0.5 + m_rms * 0.5
                    update = ortho_direction * sophia_shape * final_scale

                    # 恢复形状
                    if p.ndim > 2:
                        update = update.view_as(p)

                    # Weight Decay (Decoupled)
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(update, alpha=-lr)

                # --- 2. Handle 1D vectors/embeddings (Fallback to AdamW logic) ---
                else:
                    # Standard AdamW is fine for vectors
                    denom = hess.sqrt().add_(1e-8)
                    
                    # Bias Correction
                    step = state['step']
                    bc1 = 1 - momentum ** step
                    bc2 = 1 - beta2 ** step
                    
                    # Adaptive Step Size
                    step_size = lr * math.sqrt(bc2) / bc1
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.addcdiv_(buf, denom, value=-step_size)

        return loss
