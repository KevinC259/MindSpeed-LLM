# copyright  AIGCode 2025
# @author: chenqiuwu@aigcode.net
# @file: zetar.py
# @brief: Zeta optimizer implementation
# @details: Zeta optimizer implementation
# @version: 1.0.0
# @date: 2025-12-27
# @copyright: Copyright (c) 2025 AIGCode
# @license: MIT
# @contact: contact@aigcode.net

import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

@torch.compile
def polar_decomposition_ns(G, steps=5, eps=1e-7):
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

class VPMC(optim.Optimizer):
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
            #landscape_warmup = group['landscape_warmup_steps']
            landscape_warmup = 50

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
                    param_id += 1
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
    
class Zeta_backup(optim.Optimizer):
    """
    Zeta: Plateau-Aware Optimizer with Mathematically-Safe Feature Activation.
    
    Fixes:
      - Plateau confidence now includes gradient direction consistency
      - CGNI noise scales with grad norm, not Hessian
      - IBE injects noise in low-curvature directions
      - AdEMAMix disables long-term momentum in plateau
    """

    def __init__(
        self,
        params,
        lr=1.0,
        lr_base=0.08,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        rho=0.1,
        ns_steps=5,
        d0=1e-2,
        d_coef=1.5,
        max_lr=0.15,
        flat_thresh=1.0,
        ravine_damp=0.5,
        flat_boost=2.0,
        trust_region_factor=2.5,
        initial_boost_steps=120,
        cold_start_decay_steps=100,
        eps=1e-8,
        # --- Plateau Detection ---
        plateau_window_size=200,
        plateau_loss_std_thresh=1e-5,
        plateau_grad_norm_thresh=1e-5,
        # --- Feature Switches ---
        enable_cgni=True,
        enable_dmr=True,
        enable_mce=True,
        enable_ibe=True,
        enable_ademamix=True,
        # --- Feature Hyperparams ---
        cgni_sigma=1e-4,
        dmr_cos_thresh=0.0,
        mce_kappa_thresh=1e-3,
        mce_gamma=0.5,
        ibe_topk_ratio=0.01,
        ibe_sigma=1e-4,
        # --- AdEMAMix Params ---
        beta3=0.999,
        alpha0=0.9,
        alphainf=0.1,
        tau=1000,
        momentum=0.95, nesterov=True,
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
            plateau_window_size=plateau_window_size,
            plateau_loss_std_thresh=plateau_loss_std_thresh,
            plateau_grad_norm_thresh=plateau_grad_norm_thresh,
            enable_cgni=enable_cgni,
            enable_dmr=enable_dmr,
            enable_mce=enable_mce,
            enable_ibe=enable_ibe,
            enable_ademamix=enable_ademamix,
            cgni_sigma=cgni_sigma,
            dmr_cos_thresh=dmr_cos_thresh,
            mce_kappa_thresh=mce_kappa_thresh,
            mce_gamma=mce_gamma,
            ibe_topk_ratio=ibe_topk_ratio,
            ibe_sigma=ibe_sigma,
            beta3=beta3,
            alpha0=alpha0,
            alphainf=alphainf,
            tau=tau,
            momentum=momentum,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        
        # Global plateau state
        self._plateau_loss_window = deque(maxlen=plateau_window_size)
        self._plateau_grad_norm_window = deque(maxlen=plateau_window_size)
        self._plateau_confidence = 0.0
        self._in_plateau = False
        self._prev_grads = []  # For direction consistency

    def update_plateau_state(self, loss, total_grad_norm, step, current_grads):
        """Smooth plateau detection with gradient direction consistency."""
        if loss is not None:
            self._plateau_loss_window.append(loss)
        self._plateau_grad_norm_window.append(total_grad_norm)
        
        # Update gradient history for direction check
        self._prev_grads.append(current_grads)
        if len(self._prev_grads) > 2:
            self._prev_grads.pop(0)
        
        if len(self._plateau_loss_window) == self._plateau_loss_window.maxlen:
            loss_std = torch.tensor(list(self._plateau_loss_window)).std().item()
            avg_grad_norm = sum(self._plateau_grad_norm_window) / len(self._plateau_grad_norm_window)
            
            # Direction consistency: cosine similarity between recent grads
            dir_conf = 1.0
            if len(self._prev_grads) == 2:
                g1, g2 = self._prev_grads[0], self._prev_grads[1]
                if g1.numel() == g2.numel():
                    cos_sim = F.cosine_similarity(g1.view(-1), g2.view(-1), dim=0).item()
                    dir_conf = max(0.0, 1.0 - abs(cos_sim))  # 0: consistent, 1: inconsistent
            
            loss_conf = max(0.0, 1.0 - loss_std / self.defaults['plateau_loss_std_thresh'])
            grad_conf = max(0.0, 1.0 - avg_grad_norm / self.defaults['plateau_grad_norm_thresh'])
            
            # Conservative plateau confidence
            plateau_conf = min(loss_conf, grad_conf, dir_conf)
            early_penalty = min(1.0, step / 1000.0)
            self._plateau_confidence = plateau_conf * early_penalty
            self._in_plateau = self._plateau_confidence > 0.1
        else:
            self._plateau_confidence = 0.0
            self._in_plateau = False

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.full_like(p, 1e-3)
                state['exp_avg_sq'].lerp_(p.grad.float().pow(2), 1 - beta2)
                
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
                loss = loss_val.item()
        else:
            loss_val = None
            loss = None

        # Compute total grad norm and collect current grads
        total_grad_norm = 0.0
        current_grads_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
                    current_grads_list.append(p.grad.clone().view(-1))
        total_grad_norm = math.sqrt(total_grad_norm)
        current_grads = torch.cat(current_grads_list) if current_grads_list else torch.tensor([])

        global_step = 0
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'step' in self.state[p]:
                    global_step = max(global_step, self.state[p]['step'])
        if global_step == 0:
            global_step = 1

        # Update plateau state
        self.update_plateau_state(loss, total_grad_norm, global_step, current_grads)

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
            
            enable_cgni = group['enable_cgni']
            enable_dmr = group['enable_dmr']
            enable_mce = group['enable_mce']
            enable_ibe = group['enable_ibe']
            enable_ademamix = group['enable_ademamix']
            
            cgni_sigma = group['cgni_sigma']
            dmr_cos_thresh = group['dmr_cos_thresh']
            mce_kappa_thresh = group['mce_kappa_thresh']
            mce_gamma = group['mce_gamma']
            ibe_topk_ratio = group['ibe_topk_ratio']
            ibe_sigma = group['ibe_sigma']

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
                    if enable_ademamix:
                        state['exp_avg_long'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # --- AdEMAMix with plateau-aware alpha ---
                if enable_ademamix:
                    exp_avg_short = state.get('exp_avg', torch.zeros_like(p))
                    exp_avg_short.lerp_(grad, 1 - beta1)
                    exp_avg_long = state.get('exp_avg_long', torch.zeros_like(p))
                    exp_avg_long.lerp_(grad, 1 - group['beta3'])
                    
                    # In plateau, use only short-term momentum
                    if self._in_plateau:
                        alpha_t = group['alphainf']
                    else:
                        alpha_t = group['alphainf'] + (group['alpha0'] - group['alphainf']) * math.exp(-step / group['tau'])
                    
                    exp_avg = (1 - alpha_t) * exp_avg_short + alpha_t * exp_avg_long
                    state['exp_avg'] = exp_avg_short
                    state['exp_avg_long'] = exp_avg_long
                else:
                    exp_avg = state['exp_avg']
                    exp_avg.lerp_(grad, 1 - beta1)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.lerp_(grad.float().pow(2), 1 - beta2)

                s = state['s']
                d = state['d']
                d_denom = state['d_denom']

                # --- Adaptive rho ---
                rho_adaptive = rho_nominal * (1.0 + 2.0 * math.exp(-step / cold_start_decay_steps))

                if p.ndim >= 2:
                    if p.ndim > 2:
                        orig_shape = p.shape
                        m_view = exp_avg.view(p.size(0), -1)
                        h_view = exp_avg_sq.view(p.size(0), -1)
                        s_view = s.view(p.size(0), -1)
                    else:
                        orig_shape = None
                        m_view, h_view, s_view = exp_avg, exp_avg_sq, s

                    # Bias-corrected Hessian
                    bias_correction2 = 1 - beta2 ** step
                    h_view_corrected = h_view / bias_correction2

                    m_norm = m_view.norm()
                    h_trace = h_view_corrected.mean()
                    flatness = m_norm / (torch.sqrt(h_trace) + rho_adaptive + eps)

                    if flatness > flat_thresh:
                        alpha = 1.0 + (flat_boost - 1.0) * (1.0 - flat_thresh / (flatness + eps))
                    else:
                        alpha = ravine_damp + (1.0 - ravine_damp) * (flatness / (flat_thresh + eps))
                    alpha = torch.clamp(alpha, ravine_damp, flat_boost)

                    sqrt_h = torch.sqrt(h_view_corrected) + rho_adaptive
                    precond_m = m_view / sqrt_h
                    precond_m = precond_m.clamp(min=-1e3, max=1e3)

                    U, g_scale = polar_decomposition_ns(precond_m, steps=ns_steps)
                    raw_update = U
                    moon_scale = get_moonshot_scale(p.shape)
                    raw_update = raw_update * moon_scale * alpha

                    if step <= initial_boost_steps:
                        boost = 1.0 + (flat_boost - 1.0) * (1.0 - (step - 1) / initial_boost_steps)
                        raw_update = raw_update * boost

                    # --- MCE: Only if confident AND in true plateau ---
                    if enable_mce and self._in_plateau:
                        kappa = torch.sqrt(torch.mean(h_view_corrected**2))
                        if kappa < mce_kappa_thresh:
                            mce_factor = 1.0 + mce_gamma * (1.0 - kappa / (mce_kappa_thresh + eps))
                            # Apply but capped
                            raw_update = raw_update * min(mce_factor, 1.5)

                    update_rms = raw_update.norm() / (raw_update.numel() ** 0.5 + eps)
                    ref_rms = moon_scale / (raw_update.size(-1) ** 0.5 + eps)
                    scale_norm = g_scale / math.sqrt(m_view.numel())
                    adaptive_tr_factor = 1.0 + 1.5 * torch.exp(-scale_norm * 5.0)
                    clip_threshold = adaptive_tr_factor * ref_rms
                    if update_rms > clip_threshold:
                        raw_update.mul_(clip_threshold / (update_rms + eps))

                    final_update = raw_update.view_as(p) if orig_shape else raw_update

                    # --- CGNI: Safe noise scaling ---
                    if enable_cgni and self._in_plateau:
                        noise_amp = cgni_sigma * (grad.norm().item() + eps)
                        noise = torch.randn_like(final_update) * noise_amp
                        final_update.add_(noise)

                    # --- IBE: Inject in LOW-curvature directions ---
                    if enable_ibe and self._in_plateau:
                        num_params = h_view_corrected.numel()
                        topk = max(1, int(num_params * ibe_topk_ratio))
                        # Note: largest=False → smallest Hessian (low curvature)
                        _, topk_indices = torch.topk(h_view_corrected.view(-1), topk, largest=False)
                        ibe_noise = torch.zeros_like(final_update)
                        ibe_noise.view(-1)[topk_indices] = torch.randn(topk, device=p.device) * ibe_sigma
                        final_update.add_(ibe_noise)

                    # --- Prodigy ---
                    if step == 1:
                        grad_norm = grad.norm()
                        if grad_norm > 1e-6:
                            initial_d = lr_base / (grad_norm + 1e-6)
                            d.fill_(max(d0, initial_d))

                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr

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

                    if wd > 0:
                        dynamic_wd_factor = 1.0 + 0.3 * torch.tanh(scale_norm * 10.0)
                        effective_wd = wd * dynamic_wd_factor
                        p.data.mul_(1 - lr_eff * effective_wd)

                    p.data.add_(final_update, alpha=-lr_eff)

                    if orig_shape is not None:
                        s.copy_(s_view.view_as(s))

                else:
                    # ---- 1D Parameters: Bias-corrected curvature scaling ----
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1_fallback, beta2_fallback = 0.9, 0.95
                    
                    exp_avg.lerp_(grad, 1 - beta1_fallback)
                    exp_avg_sq.lerp_(grad.pow(2), 1 - beta2_fallback)

                    ratio = (exp_avg.abs() / (rho_adaptive * 5120 * exp_avg_sq + 1e-15)).clamp(None, 1)
                    bias_correction1 = 1 - beta1_fallback ** state['step']
                    bias_correction2 = 1 - beta2_fallback ** state['step']
                    step_size = lr_base * math.sqrt(bias_correction2) / bias_correction1

                    if wd > 0:
                        p.data.mul_(1 - lr_base * wd)
                    p.addcmul_(exp_avg.sign(), ratio, value=-step_size)

        return loss_val

class Zeta_27_29am(optim.Optimizer):
    """
    ZetaR_Plus: Plateau-Aware Optimizer with Smooth Feature Activation.
    
    Fixes:
      - Removed hard switch at step=1000 for 1D parameters
      - Smooth activation of plateau features via confidence weighting
      - Bias-corrected Hessian for consistent dynamics
    """

    def __init__(
        self,
        params,
        lr=1.0,
        lr_base=0.08,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        rho=0.1,
        ns_steps=5,
        d0=1e-2,
        d_coef=1.5,
        max_lr=0.15,
        flat_thresh=1.0,
        ravine_damp=0.5,
        flat_boost=2.0,
        trust_region_factor=2.5,
        initial_boost_steps=120,
        cold_start_decay_steps=100,
        eps=1e-8,
        # --- Plateau Detection ---
        plateau_window_size=200,
        plateau_loss_std_thresh=1e-5,
        plateau_grad_norm_thresh=1e-5,
        # --- Feature Switches ---
        enable_cgni=True,
        enable_dmr=True,
        enable_mce=True,
        enable_ibe=True,
        enable_ademamix=True,
        # --- Feature Hyperparams ---
        cgni_sigma=1e-4,
        dmr_cos_thresh=0.0,
        mce_kappa_thresh=1e-3,
        mce_gamma=0.5,
        ibe_topk_ratio=0.01,
        ibe_sigma=1e-4,
        # --- AdEMAMix Params ---
        beta3=0.999,
        alpha0=0.9,
        alphainf=0.1,
        tau=1000,
        momentum=0.95, nesterov=True,
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
            # Plateau detection
            plateau_window_size=plateau_window_size,
            plateau_loss_std_thresh=plateau_loss_std_thresh,
            plateau_grad_norm_thresh=plateau_grad_norm_thresh,
            # Feature switches
            enable_cgni=enable_cgni,
            enable_dmr=enable_dmr,
            enable_mce=enable_mce,
            enable_ibe=enable_ibe,
            enable_ademamix=enable_ademamix,
            # Feature params
            cgni_sigma=cgni_sigma,
            dmr_cos_thresh=dmr_cos_thresh,
            mce_kappa_thresh=mce_kappa_thresh,
            mce_gamma=mce_gamma,
            ibe_topk_ratio=ibe_topk_ratio,
            ibe_sigma=ibe_sigma,
            # AdEMAMix
            beta3=beta3,
            alpha0=alpha0,
            alphainf=alphainf,
            tau=tau,
            momentum=momentum,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        
        # Global plateau state
        self._plateau_loss_window = deque(maxlen=plateau_window_size)
        self._plateau_grad_norm_window = deque(maxlen=plateau_window_size)
        self._plateau_confidence = 0.0
        self._in_plateau = False

    def update_plateau_state(self, loss, total_grad_norm, step):
        """Smooth plateau detection with confidence weighting."""
        if loss is not None:
            self._plateau_loss_window.append(loss)
        self._plateau_grad_norm_window.append(total_grad_norm)
        
        if len(self._plateau_loss_window) == self._plateau_loss_window.maxlen:
            loss_std = torch.tensor(list(self._plateau_loss_window)).std().item()
            avg_grad_norm = sum(self._plateau_grad_norm_window) / len(self._plateau_grad_norm_window)
            
            # Confidence: 0.0 (not plateau) to 1.0 (deep plateau)
            loss_conf = max(0.0, 1.0 - loss_std / self.defaults['plateau_loss_std_thresh'])
            grad_conf = max(0.0, 1.0 - avg_grad_norm / self.defaults['plateau_grad_norm_thresh'])
            plateau_conf = min(loss_conf, grad_conf)
            
            # Early training penalty
            early_penalty = min(1.0, step / 1000.0)
            self._plateau_confidence = plateau_conf * early_penalty
            self._in_plateau = self._plateau_confidence > 0.1
        else:
            self._plateau_confidence = 0.0
            self._in_plateau = False

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
                state['exp_avg_sq'].lerp_(p.grad.float().pow(2), 1 - beta2)
                
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
                loss = loss_val.item()
        else:
            loss_val = None
            loss = None

        # Compute total grad norm
        total_grad_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = math.sqrt(total_grad_norm)

        # Get global step
        global_step = 0
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'step' in self.state[p]:
                    global_step = max(global_step, self.state[p]['step'])
        if global_step == 0:
            global_step = 1

        # Update plateau state
        self.update_plateau_state(loss, total_grad_norm, global_step)

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
            
            # Feature switches
            enable_cgni = group['enable_cgni']
            enable_dmr = group['enable_dmr']
            enable_mce = group['enable_mce']
            enable_ibe = group['enable_ibe']
            enable_ademamix = group['enable_ademamix']
            
            # Feature params
            cgni_sigma = group['cgni_sigma']
            dmr_cos_thresh = group['dmr_cos_thresh']
            mce_kappa_thresh = group['mce_kappa_thresh']
            mce_gamma = group['mce_gamma']
            ibe_topk_ratio = group['ibe_topk_ratio']
            ibe_sigma = group['ibe_sigma']

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
                    if enable_ademamix:
                        state['exp_avg_long'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # --- AdEMAMix Momentum ---
                if enable_ademamix:
                    exp_avg_short = state.get('exp_avg', torch.zeros_like(p))
                    exp_avg_short.lerp_(grad, 1 - beta1)
                    
                    exp_avg_long = state.get('exp_avg_long', torch.zeros_like(p))
                    exp_avg_long.lerp_(grad, 1 - group['beta3'])
                    
                    alpha_t = group['alphainf'] + (group['alpha0'] - group['alphainf']) * math.exp(-step / group['tau'])
                    exp_avg = (1 - alpha_t) * exp_avg_short + alpha_t * exp_avg_long
                    
                    state['exp_avg'] = exp_avg_short
                    state['exp_avg_long'] = exp_avg_long
                else:
                    exp_avg = state['exp_avg']
                    exp_avg.lerp_(grad, 1 - beta1)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.lerp_(grad.float().pow(2), 1 - beta2)

                s = state['s']
                d = state['d']
                d_denom = state['d_denom']

                # --- Adaptive rho (cold-start damping) ---
                rho_adaptive = rho_nominal * (1.0 + 2.0 * math.exp(-step / cold_start_decay_steps))

                if p.ndim >= 2:
                    if p.ndim > 2:
                        orig_shape = p.shape
                        m_view = exp_avg.view(p.size(0), -1)
                        h_view = exp_avg_sq.view(p.size(0), -1)
                        s_view = s.view(p.size(0), -1)
                    else:
                        orig_shape = None
                        m_view, h_view, s_view = exp_avg, exp_avg_sq, s

                    # Bias-corrected Hessian
                    bias_correction2 = 1 - beta2 ** step
                    h_view_corrected = h_view / bias_correction2

                    m_norm = m_view.norm()
                    h_trace = h_view_corrected.mean()
                    flatness = m_norm / (torch.sqrt(h_trace) + rho_adaptive + eps)

                    if flatness > flat_thresh:
                        alpha = 1.0 + (flat_boost - 1.0) * (1.0 - flat_thresh / (flatness + eps))
                    else:
                        alpha = ravine_damp + (1.0 - ravine_damp) * (flatness / (flat_thresh + eps))
                    alpha = torch.clamp(alpha, ravine_damp, flat_boost)

                    # Pre-condition momentum
                    sqrt_h = torch.sqrt(h_view_corrected) + rho_adaptive
                    precond_m = m_view / sqrt_h
                    precond_m = precond_m.clamp(min=-1e3, max=1e3)

                    U, g_scale = polar_decomposition_ns(precond_m, steps=ns_steps)
                    raw_update = U
                    moon_scale = get_moonshot_scale(p.shape)
                    raw_update = raw_update * moon_scale * alpha

                    if step <= initial_boost_steps:
                        boost = 1.0 + (flat_boost - 1.0) * (1.0 - (step - 1) / initial_boost_steps)
                        raw_update = raw_update * boost

                    # --- MCE: Smooth activation ---
                    if enable_mce:
                        kappa = torch.sqrt(torch.mean(h_view_corrected**2))
                        if kappa < mce_kappa_thresh:
                            mce_factor = 1.0 + mce_gamma * (1.0 - kappa / (mce_kappa_thresh + eps))
                            raw_update = raw_update * (1.0 + (mce_factor - 1.0) * self._plateau_confidence)

                    update_rms = raw_update.norm() / (raw_update.numel() ** 0.5 + eps)
                    ref_rms = moon_scale / (raw_update.size(-1) ** 0.5 + eps)
                    scale_norm = g_scale / math.sqrt(m_view.numel())
                    adaptive_tr_factor = 1.0 + 1.5 * torch.exp(-scale_norm * 5.0)
                    clip_threshold = adaptive_tr_factor * ref_rms
                    if update_rms > clip_threshold:
                        raw_update.mul_(clip_threshold / (update_rms + eps))

                    final_update = raw_update.view_as(p) if orig_shape else raw_update

                    # --- CGNI: Smooth activation ---
                    if enable_cgni:
                        tau = h_view_corrected.mean()
                        w = torch.exp(-h_view_corrected / (tau + eps))
                        noise = torch.randn_like(final_update) * w * cgni_sigma
                        final_update.add_(noise * self._plateau_confidence)

                    # --- IBE: Smooth activation (only if confident) ---
                    if enable_ibe and self._in_plateau:
                        num_params = h_view_corrected.numel()
                        topk = max(1, int(num_params * ibe_topk_ratio))
                        _, topk_indices = torch.topk(h_view_corrected.view(-1), topk, largest=True)
                        ibe_noise = torch.zeros_like(final_update)
                        ibe_noise.view(-1)[topk_indices] = torch.randn(topk, device=p.device) * ibe_sigma
                        final_update.add_(ibe_noise * self._plateau_confidence)

                    # --- Prodigy ---
                    if step == 1:
                        grad_norm = grad.norm()
                        if grad_norm > 1e-6:
                            initial_d = lr_base / (grad_norm + 1e-6)
                            d.fill_(max(d0, initial_d))

                    lr_eff = (d * lr_base).item()
                    if lr_eff > max_lr:
                        lr_eff = max_lr

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

                    if wd > 0:
                        dynamic_wd_factor = 1.0 + 0.3 * torch.tanh(scale_norm * 10.0)
                        effective_wd = wd * dynamic_wd_factor
                        p.data.mul_(1 - lr_eff * effective_wd)

                    p.data.add_(final_update, alpha=-lr_eff)

                    if orig_shape is not None:
                        s.copy_(s_view.view_as(s))

                else:
                    # ---- 1D Parameters: Unified bias-corrected curvature scaling ----

                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1_fallback, beta2_fallback = 0.9, 0.95
                    
                    exp_avg.lerp_(grad, 1 - beta1_fallback)
                    exp_avg_sq.lerp_(grad.pow(2), 1 - beta2_fallback)

                    #denom = exp_avg_sq.sqrt().add_(1e-8)
                    ratio = (exp_avg.abs() / (rho_adaptive * 5120 * exp_avg_sq + 1e-15)).clamp(None, 1)

                    bias_correction1 = 1 - beta1_fallback ** state['step']
                    bias_correction2 = 1 - beta2_fallback ** state['step']
                    step_size = lr_base * math.sqrt(bias_correction2) / bias_correction1

                    if wd > 0:
                        p.data.mul_(1 - lr_base * wd)

                    #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    p.addcmul_(exp_avg.sign(), ratio, value=-step_size)

                    '''bias_correction2 = 1 - beta2 ** step
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
                    p.data.add_(update_1d, alpha=-lr_eff)'''

        return loss_val
    
class Zeta_v2(optim.Optimizer):
    """
    Zeta-S Turbo: A stable fusion of Muon (orthogonal momentum) and Sophia (curvature-aware step size).

    Key Features:
    - For 2D+ parameters: uses orthogonalized momentum direction + directional Newton step.
    - For 1D parameters: falls back to bias-corrected AdamW.
    - Fully decoupled weight decay.
    - No arbitrary scaling factors (Moonshot removed by default).

    Args:
        lr (float): Learning rate (recommended: 0.01~0.05 for large models).
        betas (tuple): (beta1, beta2) for momentum and Hessian estimation.
        ns_steps (int): Newton-Schulz iteration steps (3~5).
        weight_decay (float): Decoupled weight decay.
        use_moonshot (bool): Enable optional Moonshot scaling (default: False).
    """

    def __init__(self, params, lr=0.02, betas=(0.95, 0.99), ns_steps=4,
                 weight_decay=0.01, rho=0.04, use_moonshot=False):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            rho=rho,
            use_moonshot=use_moonshot
        )
        super().__init__(params, defaults)

    def _moonshot_scale(self, shape):
        """Optional: scale based on parameter dimension."""
        if len(shape) < 2:
            return 1.0
        return 0.2 * math.sqrt(max(shape[0], shape[1]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']
            rho = group['rho']
            use_moonshot = group['use_moonshot']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # === Initialize state ===
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']

                # === Update moments ===
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Use momentum as search direction (simpler than Nesterov)
                direction_in = exp_avg  # or: beta1 * exp_avg + grad  # Nesterov variant

                # === High-dimensional parameters (2D+) ===
                if p.ndim >= 2 and p.numel() > 4:
                    # Reshape to (out, in_prod)
                    original_shape = p.shape
                    if p.ndim > 2:
                        dir_mat = direction_in.view(direction_in.size(0), -1)
                        h_mat = exp_avg_sq.view(exp_avg_sq.size(0), -1)
                    else:
                        dir_mat = direction_in
                        h_mat = exp_avg_sq

                    # 1. Orthogonalize direction (Muon core)
                    try:
                        U_mat = newton_schulz_orthogonal(dir_mat, steps=ns_steps)
                        U = U_mat.view(original_shape)
                    except Exception:
                        # Fallback if NS fails
                        U = direction_in

                    # 2. Normalize direction to unit Frobenius norm
                    U_norm = U / (U.norm() + 1e-12)

                    # 3. Compute directional curvature: <U, H U> ≈ sum(U^2 * h)
                    h_clamped = torch.clamp(h_mat, min=rho)
                    if p.ndim > 2:
                        h_clamped = h_clamped.view(original_shape)
                    curvature = (U_norm.pow(2) * h_clamped).sum()
                    newton_step = 1.0 / (curvature + 1e-12)

                    # 4. Clamp step size for stability
                    local_scale = torch.clamp(newton_step, 0.5, 2.0)

                    # 5. Optional Moonshot scaling (disabled by default)
                    if use_moonshot:
                        moon_scale = self._moonshot_scale(p.shape)
                        local_scale = local_scale * moon_scale

                    update = U_norm * local_scale

                    # Apply weight decay (decoupled)
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)

                    p.add_(update, alpha=-lr)

                # === 1D parameters (bias, LayerNorm, Embedding) ===
                else:
                    # Bias-corrected AdamW
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                    update = (exp_avg / bias_correction1).div(denom)

                    if wd > 0:
                        p.mul_(1.0 - lr * wd)

                    p.add_(update, alpha=-lr)

        return loss
    
class Zetahipro(optim.Optimizer):
    """
    Zeta-S Turbo: High-throughput version of Zeta-S.
    
    Key Optimizations:
    1. Removed expensive cosine similarity checks.
    2. Clamped Sophia-Scaling to preserve Muon's orthogonality.
    3. Reduced overhead in Newton-Schulz.
    4. Simplified trust-region logic.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=4, weight_decay=0.01,
                 betas=(0.95, 0.99), rho=0.04):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        betas=betas, rho=rho)

        super().__init__(params, defaults)

    def get_moonshot_scale(self, param_shape):
        """Moonshot scaling factor."""
        if len(param_shape) < 2:
            return 1.0
        # Fast path for standard shapes
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
            momentum = group['exp_avg']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            rho = group['rho']
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # --- Initialization ---
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                
                # --- 1. Sophia Hessian Estimation (Diagonal) ---
                # H_t+1 = beta * H_t + (1-beta) * (g * g)
                hess = state['exp_avg_sq']
                hess.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # --- 2. Momentum Update ---
                # m_t+1 = beta * m_t + (1-beta) * g
                mom = state['exp_avg']
                mom.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Nesterov lookahead
                if nesterov:
                    update_in = grad.add(mom, alpha=beta1)
                else:
                    update_in = mom

                # --- 3. High-Dim Processing (Muon + Sophia Logic) ---
                if p.ndim >= 2:
                    # Flatten to 2D
                    if p.ndim > 2:
                        m_in = update_in.view(mom.size(0), -1)
                        h_in = hess.view(hess.size(0), -1)
                    else:
                        m_in = update_in
                        h_in = hess

                    # A. Newton-Schulz Orthogonalization (The Muon Core)
                    # This projects the momentum onto the Stiefel manifold (approx)
                    ortho_direction = newton_schulz_optimized(m_in, steps=ns_steps)
                    
                    # B. Robust Sophia Scaling
                    # Instead of pure 1/h, we use a relative scaling centered around 1.0
                    # This preserves the orthogonality from step A while adapting to local curvature.
                    
                    # h_clamped = max(h, rho)
                    h_clamped = torch.clamp(h_in, min=rho)
                    
                    # Calculate mean curvature of this layer
                    h_mean = h_clamped.mean()
                    
                    # Relative Scale: h_mean / h_local
                    # If local curvature is high (steep), we slow down (scale < 1)
                    # If local curvature is low (flat), we speed up (scale > 1)
                    sophia_scale = h_mean.div(h_clamped)
                    
                    # CLAMPING: Crucial for stability! 
                    # Don't let Sophia distort the Muon direction by more than 2x or less than 0.5x
                    # This prevents outliers in Hessian estimate from exploding the gradient
                    sophia_scale.clamp_(min=0.5, max=2.0)
                    
                    # C. Moonshot Scaling
                    moonshot_scale = self.get_moonshot_scale(p.shape)
                    
                    # D. Final Composition
                    # update = Ortho * Sophia * Moonshot
                    final_update = ortho_direction.mul_(sophia_scale).mul_(moonshot_scale)
                    
                    # Reshape and Apply
                    if p.ndim > 2:
                        final_update = final_update.view_as(p)
                        
                    # Decoupled Weight Decay (Applied to params, not mixed with update direction)
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.add_(final_update, alpha=-lr)

                # --- 4. Low-Dim / Vector Processing (AdamW Fallback) ---
                else:
                    # Optimized AdamW logic for 1D params (bias, layernorm)
                    # Re-using the calculated momentum and hessian
                    
                    # In AdamW context, hessian state is actually v_t (second moment)
                    # So sqrt(hessian) is the denominator
                    denom = hess.sqrt().add_(1e-8)
                    
                    # Bias correction
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    
                    # Adam step
                    step_size = lr * math.sqrt(bc2) / bc1
                    
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                        
                    p.data.addcdiv_(update_in, denom, value=-step_size)

        return loss