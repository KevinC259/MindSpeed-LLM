# Copyright (c) 2025, AIGCode CORPORATION. All rights reserved. 
# @author: chenqiuwu@aigcode.net

import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List

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

class SignMuon(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99),
                 weight_decay=1e-2, *, maximize: bool = False,
                 capturable: bool = False, ns_steps=5, eps: float = 1e-8):
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
                        ns_steps=ns_steps)
        super(SignMuon, self).__init__(params, defaults)

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

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []

            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('SignMuon does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])

            signmuon(params_with_grad,
                    grads,
                    exp_avgs,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    maximize=group['maximize'],
                    capturable=group['capturable'],
                    ns_steps=group['ns_steps'])

        return loss


def signmuon(params: List[torch.Tensor],
            grads: List[torch.Tensor],
            exp_avgs: List[torch.Tensor],
            state_steps: List[torch.Tensor],
            capturable: bool = False,
            *,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            maximize: bool,
            ns_steps: int):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    _single_tensor_signmuon(params,
                           grads,
                           exp_avgs,
                           state_steps,
                           beta1=beta1,
                           beta2=beta2,
                           lr=lr,
                           weight_decay=weight_decay,
                           maximize=maximize,
                           capturable=capturable,
                           ns_steps=ns_steps)


def _single_tensor_signmuon(params: List[torch.Tensor],
                           grads: List[torch.Tensor],
                           exp_avgs: List[torch.Tensor],
                           state_steps: List[torch.Tensor],
                           *,
                           beta1: float,
                           beta2: float,
                           lr: float,
                           weight_decay: float,
                           maximize: bool,
                           capturable: bool,
                           ns_steps: int):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            param = torch.view_as_real(param)

        step_t += 1
        param.mul_(1 - lr * weight_decay)
        
        # Momentum update
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # --- Muon integration start ---
        # Apply Newton-Schulz to exp_avg (momentum) if it's a matrix (>=2D)
        # Muon usually works on >1D parameters.
        if param.ndim >= 2:
            m_in = exp_avg
            if param.ndim > 2:
                m_in = m_in.view(m_in.size(0), -1)
            
            # Apply Newton-Schulz
            ortho_direction = zeropower_via_newtonschulz5(m_in, steps=ns_steps)
            
            # Restore shape
            if param.ndim > 2:
                ortho_direction = ortho_direction.view_as(exp_avg)
                
            # SignMuon: effectively hessian=0 behavior in SophiaMuon
            # which results in update_val = sign(ortho_direction)
            
            update_val = ortho_direction.sign()
            param.add_(update_val, alpha=-lr)

        else:
            # Fallback to standard SignSGD for 1D params (bias, layernorm)
            # SophiaMuon with hessian=0 does: param - lr * sign(exp_avg)
            update_val = exp_avg.sign()
            param.add_(update_val, alpha=-lr)
