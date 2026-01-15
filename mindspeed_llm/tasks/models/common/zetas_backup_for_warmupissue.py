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
    Zeta-MM improved Newton-Schulz iteration with eps added to prevent division by zero,
    maintaining numerical stability in bfloat16.

    Args:
        G: Input matrix
        steps: Number of Newton-Schulz iterations
        eps: Small epsilon for numerical stability
    """
    assert len(G.shape) == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(0) > G.size(1):
        X = X.T

    # Spectral normalization (the essence of Muon): force eigenvalues to be close to 1
    norm = X.norm() + eps
    X = X / norm

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X

class ZetaS(optim.Optimizer):
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
                 betas=(0.965, 0.99), rho=0.04, mode='post_scale', eps: float = 1e-7):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        trust_region_threshold=trust_region_threshold,
                        betas=betas, rho=rho, mode=mode
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
                hess.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # -----------------------------------------------------------
                # 3. 动量更新 (EMA)
                # -----------------------------------------------------------
                momentum = state['exp_avg']
                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Nesterov 修正: 使用预见动量进行更新计算
                if nesterov:
                    update_in = grad.add(momentum, alpha=beta1)
                else:
                    update_in = momentum

                consistency = 0.0
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

                    g_flat = grad.view(-1)
                    m_flat = momentum.view(-1)
                    # 避免完全 norm 计算的开销，仅作近似或采样，这里写全量为了准确性
                    consistency = F.cosine_similarity(g_flat, m_flat, dim=0, eps=1e-6)
                    # === 方案 A: Sophia-Preconditioned Muon ===
                    if mode == 'pre_condition':
                        # Step 1: 利用 Hessian 对动量进行预处理 (Whitening the scale)
                        # 我们希望把那些曲率大的方向（h_in 大）缩放得小一点
                        # 加上 eps 防止除零
                        denom = (h_in.sqrt() + 1e-6)
                        pre_conditioned_m = m_in / denom
                        
                        # Step 2: Newton-Schulz 正交化
                        # 这一步是在 "Hessian-Aware" 的空间里找正交基
                        update = zeropower_via_newtonschulz5(pre_conditioned_m, steps=ns_steps)
                        
                        # Step 3: 应用更新 (Scale 主要由 LR 控制)
                        # 注意：这里我们不需要再乘回 Hessian，因为 NS 已经归一化了谱
                        scale = self.get_moonshot_scale(p.shape)
                        final_update = update * scale

                    # === 方案 B: Sophia-Scaled Muon (推荐) ===
                    elif mode == 'post_scale':
                        # Step 1: Muon 负责找“最佳方向” (Rotation)
                        # 这一步消除了病态条件数的影响，找到了指向谷底的方向
                        update_norm = update_in.norm()
                        ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
                        
                        # Step 2: Sophia 负责找“最佳步长” (Scale)
                        # 这一步根据每个参数的局部陡峭程度，决定在这个方向上走多远
                        # Sophia 的 clipping 机制： max(h, rho)
                        # 我们计算一个逐元素的缩放因子
                        
                        # 为了数值稳定性，Muon 的输出谱范数为1，元素值较小 (~1/sqrt(d))
                        # Sophia 的 Scaling 是 1/h。
                        # 为了让两者结合时不至于数值爆炸，我们需要精细控制
                        
                        # Sophia 的标准更新是 m / max(h, rho)
                        # 我们这里用 ortho_direction 替代 m
                        # 并且我们需要补偿 Muon 归一化带来的幅度损失
                        
                        # 获取 Hessian 阻尼后的倒数
                        hess_clamped = torch.clamp(h_in, min=rho)
                        #sophia_scale = 1.0 / hess_clamped
                        hess_clamped = torch.clamp(h_in, min=rho)
                        h_mean = hess_clamped.mean()
                        # relative_sophia_scale = h_mean / h_in
                        # 这样即使后期 h 变大，这个 scale 依然在 1.0 附近浮动
                        sophia_scale = h_mean / hess_clamped
                        boost_factor = 1.0
                        boost_thresh = 0.6
                        if consistency > boost_thresh:
                            # 线性映射: [thresh, 1.0] -> [1.0, limit]
                            ratio = (consistency - boost_thresh) / (1.0 - boost_thresh)
                            boost_factor = 1.0 + ratio * (trust_thresh - 1.0)
                        # 关键点：Muon 丢失了幅度信息 (RMS约为 1/sqrt(D))
                        # Sophia 试图恢复幅度。
                        # 我们直接相乘： Direction * Scale
                        scale = self.get_moonshot_scale(p.shape)
                        #gate = torch.clamp(update_norm / (trust_thresh + 1e-6), max=10.0)
                        gate = torch.tanh(update_norm / (trust_thresh + 1e-6)).pow(2)
                        final_update = ortho_direction * sophia_scale * boost_factor * scale * gate
                        
                        # 修正系数：由于 Muon 输出了归一化矩阵，而 Sophia Scale 假设输入是原始梯度的量级
                        # 这里可能需要一个与维度相关的系数来对齐量级，但在纯 Sophia 中通常不需要
                        # 为了保险，我们引入一个 0.2 的全局缩放 (类 Moonshot) 来稳定初期
                        #final_update.mul_(0.2) 

                    # 恢复形状
                    update = final_update.view_as(p)

                    # Decoupled Weight Decay
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    
                    p.data.add_(update, alpha=-lr)

                # --- 2. Handle 1D vectors/embeddings (Fallback to AdamW logic) ---
                else:
                    state = self.state[p]

                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = 0.9, 0.95  # AdamW defaults

                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.pow(2), 1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(1e-8)

                    # Bias correction
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bc2) / bc1

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
