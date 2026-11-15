"""tensorgrad: a small reverse-mode autodiff engine and neural-network library in pure NumPy."""

from . import functional, nn, optim
from .gradcheck import gradcheck
from .tensor import Tensor, is_grad_enabled, no_grad

__all__ = ["Tensor", "no_grad", "is_grad_enabled", "gradcheck", "functional", "nn", "optim"]
__version__ = "0.1.0"
