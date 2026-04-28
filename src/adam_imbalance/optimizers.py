"""Paper-facing optimizer implementations.

The legacy experiment scripts under ``py/`` contain the full research history.
This module keeps the optimizer variants from the ICLR 2026 submission in a
small, reusable form:

* AdamLDN: Adam with layer-wise dynamics normalization.
* AdamS: Adam with a layer-specific initial-parameter scale.
* AdamSLDN: AdamS with layer-wise dynamics normalization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


def _rms(tensor: Tensor) -> Tensor:
    return tensor.pow(2).mean().sqrt()


class _LayerScaleMixin:
    """Initial-parameter RMS scaling used by Adam-S style optimizers."""

    def _init_layer_scale(
        self,
        param: Tensor,
        state: dict,
        eps: float,
        fallback_scale: Optional[Tensor],
        bias_scale: float,
    ) -> Tensor:
        scale = _rms(param.detach())
        if scale <= eps and fallback_scale is not None:
            scale = fallback_scale.detach().clone() * bias_scale
        elif scale <= eps:
            scale = torch.ones((), dtype=param.dtype, device=param.device)

        state["layer_scale"] = scale.detach().clone()
        return state["layer_scale"]


class AdamLDN(Optimizer):
    """Adam-LDN from Algorithm 3.

    Adam-LDN keeps Adam's first-moment dynamics but replaces element-wise second
    moment normalization with one RMS denominator per parameter tensor.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(
            params,
            dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamLDN does not support sparse gradients")

                grad = param.grad
                if weight_decay:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta).add_(grad, alpha=1.0 - beta)
                dynamics = exp_avg / (1.0 - beta ** state["step"])
                denom = _rms(dynamics).add(eps)
                param.add_(dynamics / denom, alpha=-lr)

        return loss


class AdamS(_LayerScaleMixin, Optimizer):
    """Adam-S from Algorithm 4.

    This is Adam with a layer-specific multiplier initialized from the RMS of
    the initial parameter tensor. For near-zero tensors such as many biases, the
    previous non-zero layer scale is reused with ``bias_scale``.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_scale: float = 0.1,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                bias_scale=bias_scale,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            bias_scale = group["bias_scale"]
            fallback_scale = None

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamS does not support sparse gradients")

                grad = param.grad
                if weight_decay:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    self._init_layer_scale(param, state, eps, fallback_scale, bias_scale)

                fallback_scale = state["layer_scale"]
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = lr * state["layer_scale"] / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamSLDN(_LayerScaleMixin, AdamLDN):
    """Adam-S-LDN from Algorithm 5."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_scale: float = 0.1,
    ) -> None:
        super().__init__(params, lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            group["bias_scale"] = bias_scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            bias_scale = group["bias_scale"]
            fallback_scale = None

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamSLDN does not support sparse gradients")

                grad = param.grad
                if weight_decay:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    self._init_layer_scale(param, state, eps, fallback_scale, bias_scale)

                fallback_scale = state["layer_scale"]
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta).add_(grad, alpha=1.0 - beta)
                dynamics = exp_avg / (1.0 - beta ** state["step"])
                denom = _rms(dynamics).add(eps)
                param.add_(dynamics / denom, alpha=-(lr * state["layer_scale"]))

        return loss
