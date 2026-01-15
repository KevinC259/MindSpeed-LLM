# Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# @author: chenqiuwu@aigcode.net

import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List
from collections import deque

@torch.compile
def polar_decomposition_ns(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for polar decomposition
    """
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

class SignMuon(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99),
                 weight_decay=1e-2, *, maximize: bool = False,
                 capturable: bool = False, ns_steps=5, eps: float = 1e-8,
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
                 # --- Prodigy Params ---
                 d0=1e-2,
                 d_coef=1.5,
                 max_lr=0.15,
                 ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable,
                        ns_steps=ns_steps, eps=eps,
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
                        d0=d0,
                        d_coef=d_coef,
                        max_lr=max_lr)
        super(SignMuon, self).__init__(params, defaults)
        
        # Global plateau state
        self._plateau_loss_window = deque(maxlen=plateau_window_size)
        self._plateau_grad_norm_window = deque(maxlen=plateau_window_size)
        self._plateau_confidence = 0.0
        self._in_plateau = False
        self._prev_grads = []

    def update_plateau_state(self, loss, total_grad_norm, step, current_grads):
        """Smooth plateau detection with gradient direction consistency."""
        if loss is not None:
            self._plateau_loss_window.append(loss)
        self._plateau_grad_norm_window.append(total_grad_norm)
        
        # Update gradient history for direction check
        self._prev_grads.append(current_grads)
        if len(self._prev_grads) > 2:
            self._prev_grads.pop(0)
        
        # Use grad norm window size as the primary trigger since loss might be missing
        if len(self._plateau_grad_norm_window) == self._plateau_grad_norm_window.maxlen:
            loss_conf = 1.0
            if len(self._plateau_loss_window) >= 10:
                loss_std = torch.tensor(list(self._plateau_loss_window)).std().item()
                loss_conf = max(0.0, 1.0 - loss_std / self.defaults['plateau_loss_std_thresh'])

            avg_grad_norm = sum(self._plateau_grad_norm_window) / len(self._plateau_grad_norm_window)
            
            # Direction consistency: cosine similarity between recent grads
            dir_conf = 1.0
            if len(self._prev_grads) == 2:
                g1, g2 = self._prev_grads[0], self._prev_grads[1]
                if g1.numel() == g2.numel() and g1.numel() > 0:
                    cos_sim = F.cosine_similarity(g1.view(-1), g2.view(-1), dim=0).item()
                    dir_conf = max(0.0, 1.0 - abs(cos_sim))  # 0: consistent, 1: inconsistent
            
            grad_conf = max(0.0, 1.0 - avg_grad_norm / self.defaults['plateau_grad_norm_thresh'])
            
            # Conservative plateau confidence
            plateau_conf = min(loss_conf, grad_conf, dir_conf)
            early_penalty = min(1.0, step / 1000.0)
            self._plateau_confidence = plateau_conf * early_penalty
            self._in_plateau = self._plateau_confidence > 0.1
        else:
            self._plateau_confidence = 0.0
            self._in_plateau = False

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # Compute total grad norm and collect current grads for plateau detection
        total_grad_norm = 0.0
        current_grads_list = []
        for group in self.param_groups:
            for p in group['params']:
                grad = getattr(p, 'decoupled_grad', None)
                if grad is None:
                    grad = p.grad
                if grad is not None:
                    total_grad_norm += grad.norm().item() ** 2
                    current_grads_list.append(grad.clone().view(-1))
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
        self.update_plateau_state(loss.item() if loss is not None else None, total_grad_norm, global_step, current_grads)

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            # Feature flags and params
            enable_cgni = group['enable_cgni']
            enable_dmr = group['enable_dmr']
            enable_mce = group['enable_mce']
            enable_ibe = group['enable_ibe']
            enable_ademamix = group['enable_ademamix']
            
            cgni_sigma = group['cgni_sigma']
            mce_kappa_thresh = group['mce_kappa_thresh']
            mce_gamma = group['mce_gamma']
            ibe_topk_ratio = group['ibe_topk_ratio']
            ibe_sigma = group['ibe_sigma']
            
            # AdEMAMix params
            beta3 = group['beta3']
            alpha0 = group['alpha0']
            alphainf = group['alphainf']
            tau = group['tau']
            
            # Prodigy params
            d0 = group['d0']
            d_coef = group['d_coef']
            max_lr = group['max_lr']
            lr_base = group['lr'] # Using lr as lr_base for Prodigy logic compatibility

            for p in group['params']:
                grad = getattr(p, 'decoupled_grad', None)
                if grad is None:
                    grad = p.grad
                if grad is None:
                    continue
                
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad.zero_()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.full_like(p, 1e-3) # For MCE/IBE curvature estimation
                    state['s'] = torch.zeros_like(p) # For Prodigy
                    state['d'] = torch.tensor(d0, device=p.device, dtype=torch.float32) # For Prodigy
                    state['d_denom'] = torch.tensor(0.0, device=p.device, dtype=torch.float32) # For Prodigy
                    if enable_ademamix:
                        state['exp_avg_long'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step'].item()
                
                # --- AdEMAMix with plateau-aware alpha ---
                if enable_ademamix:
                    exp_avg_short = state.get('exp_avg', torch.zeros_like(p))
                    exp_avg_short.lerp_(grad, 1 - beta1)
                    exp_avg_long = state.get('exp_avg_long', torch.zeros_like(p))
                    exp_avg_long.lerp_(grad, 1 - beta3)
                    
                    # In plateau, use only short-term momentum
                    if self._in_plateau:
                        alpha_t = alphainf
                    else:
                        alpha_t = alphainf + (alpha0 - alphainf) * math.exp(-step / tau)
                    
                    exp_avg = (1 - alpha_t) * exp_avg_short + alpha_t * exp_avg_long
                    state['exp_avg'] = exp_avg_short
                    state['exp_avg_long'] = exp_avg_long
                else:
                    exp_avg = state['exp_avg']
                    exp_avg.lerp_(grad, 1 - beta1)
                
                # Update Hessian approx for MCE/IBE
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.lerp_(grad.float().pow(2), 1 - beta2)

                # --- Prodigy Step Size Adaptation ---
                s = state['s']
                d = state['d']
                d_denom = state['d_denom']
                
                if step == 1:
                    grad_norm_val = grad.norm()
                    if grad_norm_val > 1e-6:
                        initial_d = lr_base / (grad_norm_val + 1e-6)
                        d.fill_(max(d0, initial_d))

                lr_eff = (d * lr_base).item()
                if lr_eff > max_lr:
                    lr_eff = max_lr

                # Apply Weight Decay
                if group['weight_decay'] > 0:
                     p.data.mul_(1 - lr_eff * group['weight_decay'])

                # --- Muon / SignMuon Logic ---
                final_update = None
                
                if p.ndim >= 2:
                    m_in = exp_avg
                    if p.ndim > 2:
                        m_in = m_in.view(m_in.size(0), -1)
                    
                    # Apply Newton-Schulz
                    ortho_direction, g_scale = polar_decomposition_ns(m_in, steps=group['ns_steps'])
                    
                    # Restore shape
                    if p.ndim > 2:
                        ortho_direction = ortho_direction.view_as(exp_avg)
                    
                    # SignMuon base update
                    raw_update = ortho_direction.sign()
                    
                    # --- MCE: Only if confident AND in true plateau ---
                    if enable_mce and self._in_plateau:
                        # Estimate curvature kappa
                        h_view_corrected = exp_avg_sq.view(-1) / (1 - beta2 ** step)
                        kappa = torch.sqrt(torch.mean(h_view_corrected**2))
                        if kappa < mce_kappa_thresh:
                            mce_factor = 1.0 + mce_gamma * (1.0 - kappa / (mce_kappa_thresh + group['eps']))
                            raw_update = raw_update * min(mce_factor, 1.5)
                            
                    final_update = raw_update

                    # --- CGNI: Safe noise scaling ---
                    if enable_cgni and self._in_plateau:
                        noise_amp = cgni_sigma * (grad.norm().item() + group['eps'])
                        noise = torch.randn_like(final_update) * noise_amp
                        final_update.add_(noise)

                    # --- IBE: Inject in LOW-curvature directions ---
                    if enable_ibe and self._in_plateau:
                        h_view_corrected = exp_avg_sq.view(-1) / (1 - beta2 ** step)
                        num_params = h_view_corrected.numel()
                        topk = max(1, int(num_params * ibe_topk_ratio))
                        # Note: largest=False → smallest Hessian (low curvature)
                        _, topk_indices = torch.topk(h_view_corrected, topk, largest=False)
                        ibe_noise = torch.zeros_like(final_update)
                        ibe_noise.view(-1)[topk_indices] = torch.randn(topk, device=p.device) * ibe_sigma
                        final_update.add_(ibe_noise)

                else:
                    # 1D Parameters: SignSGD
                    final_update = exp_avg.sign()

                # Prodigy state update
                s.add_(final_update, alpha=lr_eff)
                g_norm_sq = grad.norm().pow(2)
                d_denom_new = d_denom + lr_eff * g_norm_sq
                if d_denom_new > group['eps']:
                    d_num = s.norm()
                    d_hat = d_num / torch.sqrt(d_denom_new)
                    d_new = torch.clamp(d_hat, d.item(), d.item() * d_coef)
                    if d_new * lr_base > max_lr:
                        d_new = max_lr / lr_base
                    d.copy_(d_new)
                    d_denom.copy_(d_denom_new)

                # Final Parameter Update
                p.add_(final_update, alpha=-lr_eff)

        return loss

