"""
Anti-aliasing modules for WaveNet layers.

Provides two strategies:
1. AliasFreeActivation: wraps an activation with a frozen lowpass filter
2. AntiAliasFilter: standalone frozen lowpass filter (e.g. for residual outputs)

Both use a windowed sinc lowpass filter with configurable cutoff and filter size.
"""

import math as _math

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
from pydantic import BaseModel as _BaseModel


def sinc_lowpass_kernel(filter_size: int, cutoff: float = 0.45) -> _torch.Tensor:
    """
    Generate a windowed sinc lowpass filter kernel.

    :param filter_size: Number of taps (should be even for symmetric padding).
    :param cutoff: Normalized cutoff frequency as a fraction of the sampling rate
        (0, 0.5]. 0.45 means 90% of Nyquist.
    :returns: 1D tensor of shape (filter_size,), normalized to sum to 1.
    """
    assert filter_size >= 1, f"filter_size must be >= 1, got {filter_size}"
    assert 0.0 < cutoff <= 0.5, f"cutoff must be in (0, 0.5], got {cutoff}"

    if filter_size == 1:
        return _torch.ones(1)

    # Centered indices
    n = _torch.arange(filter_size, dtype=_torch.float64) - (filter_size - 1) / 2.0
    # Sinc function (normalized: sinc(x) = sin(pi*x) / (pi*x))
    # cutoff is fraction of fs; Nyquist = fs/2 = pi in normalized freq
    omega_c = 2 * cutoff * _math.pi
    sinc = _torch.where(
        n == 0,
        _torch.tensor(omega_c / _math.pi, dtype=_torch.float64),
        _torch.sin(omega_c * n) / (_math.pi * n),
    )
    # Blackman window
    window = (
        0.42
        - 0.5 * _torch.cos(2 * _math.pi * _torch.arange(filter_size, dtype=_torch.float64) / (filter_size - 1))
        + 0.08 * _torch.cos(4 * _math.pi * _torch.arange(filter_size, dtype=_torch.float64) / (filter_size - 1))
    )
    kernel = sinc * window
    # Normalize to unit DC gain
    kernel = kernel / kernel.sum()
    return kernel.float()


class AntiAliasConfig(_BaseModel):
    active: bool = False
    filter_size: int = 4
    cutoff: float = 0.45


class AntiAliasFilter(_nn.Module):
    """
    Frozen (non-trainable) depthwise lowpass filter.
    Applies a windowed sinc lowpass to each channel independently.
    Uses same-padding so output length == input length.
    """

    def __init__(self, channels: int, filter_size: int = 4, cutoff: float = 0.45):
        super().__init__()
        self.channels = channels
        self.filter_size = filter_size
        kernel = sinc_lowpass_kernel(filter_size, cutoff)
        # Shape: (channels, 1, filter_size) for depthwise conv
        self.register_buffer("kernel", kernel.view(1, 1, -1).expand(channels, 1, -1).clone())

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        # Same-padding: pad left more if filter_size is even
        pad_total = self.filter_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = _F.pad(x, (pad_left, pad_right))
        return _F.conv1d(x, self.kernel, groups=self.channels)


class AliasFreeActivation(_nn.Module):
    """
    Wraps an activation function with a frozen anti-aliasing lowpass filter.
    The filter is applied after the activation to suppress aliasing introduced
    by the nonlinearity.
    """

    def __init__(
        self,
        activation: _nn.Module,
        channels: int,
        filter_size: int = 8,
        cutoff: float = 0.45,
    ):
        super().__init__()
        self.activation = activation
        self.filter = AntiAliasFilter(channels, filter_size, cutoff)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self.filter(self.activation(x))
