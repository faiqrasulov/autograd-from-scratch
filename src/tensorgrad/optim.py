"""Optimizers. Update rules follow ``torch.optim`` exactly so their trajectories can be compared."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .nn import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with optional (Nesterov) momentum and L2 weight decay.

    ``g = grad + wd * p``; ``buf = momentum * buf + g``; ``p -= lr * buf``
    (Nesterov uses ``g + momentum * buf`` as the step direction).
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr)
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        self.momentum, self.weight_decay, self.nesterov = momentum, weight_decay, nesterov
        self.velocity: list[np.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            if self.momentum:
                buf = self.velocity[i]
                # torch initialises the buffer with the first gradient (no dampening)
                buf = g.copy() if buf is None else self.momentum * buf + g
                self.velocity[i] = buf
                g = g + self.momentum * buf if self.nesterov else buf
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction and L2 weight decay (not AdamW).

    m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g^2
    p -= lr / (1-b1^t) * m / (sqrt(v) / sqrt(1-b2^t) + eps)
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = [0] * len(self.params)

    def step(self) -> None:
        b1, b2 = self.beta1, self.beta2
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            self.t[i] += 1
            t = self.t[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            bias1, bias2 = 1 - b1**t, 1 - b2**t
            denom = np.sqrt(self.v[i]) / np.sqrt(bias2) + self.eps
            p.data -= (self.lr / bias1) * self.m[i] / denom
