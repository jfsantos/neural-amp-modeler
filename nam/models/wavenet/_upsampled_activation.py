"""
Upsampled activation for WaveNet layers.

Implements upsample -> activate -> downsample using a Kaiser-windowed sinc
half-band filter, matching NeuralAmpModelerCore's Resampler2x exactly.

The key idea: nonlinear activations introduce harmonics that alias back into
the audio band. By temporarily upsampling before the activation and
downsampling after, the aliased harmonics are pushed above the new Nyquist
and removed by the anti-aliasing filter.

Uses polyphase decomposition of the half-band filter for efficiency:
only the 32 non-zero odd-indexed taps need to be applied, and all
filtering happens at the base rate (not the upsampled rate).
"""

import math

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
from pydantic import BaseModel as _BaseModel


# Half-band filter parameters matching C++ Resampler2x
_M = 16
_FILTER_LENGTH = 4 * _M + 1  # 65 taps
_PHASE1_NUM_TAPS = 2 * _M  # 32 non-zero odd-indexed taps
_KAISER_BETA = 10.0


class UpsampledActivationConfig(_BaseModel):
    active: bool = False
    factor: int = 2  # 2, 4, or 8


def _bessel_i0(x: float) -> float:
    """Modified Bessel function of the first kind, order 0 (series expansion).

    Matches C++ Resampler2x::_bessel_i0 exactly.
    """
    sum_val = 1.0
    term = 1.0
    x_half = x * 0.5
    for k in range(1, 31):
        term *= (x_half / k) * (x_half / k)
        sum_val += term
        if term < 1e-12 * sum_val:
            break
    return sum_val


def _compute_phase1_coefficients() -> _torch.Tensor:
    """Compute the 32 non-zero phase1 coefficients of the half-band filter.

    These are the odd-indexed taps h[1], h[3], ..., h[63] of the 65-tap
    Kaiser-windowed sinc half-band filter. They are symmetric:
    phase1[k] = phase1[2M-1-k].

    Matches C++ Resampler2x::_compute_filter() exactly.
    """
    inv_i0_beta = 1.0 / _bessel_i0(_KAISER_BETA)
    center = 2 * _M  # 32

    phase1 = _torch.zeros(_PHASE1_NUM_TAPS, dtype=_torch.float64)
    for k in range(_PHASE1_NUM_TAPS):
        pos = 2 * k + 1  # Position in full filter [0, 4M]
        n_offset = float(pos - center)

        # Ideal lowpass sinc at omega_c = pi/2
        sinc_val = math.sin(math.pi * n_offset * 0.5) / (math.pi * n_offset)

        # Kaiser window
        normalized = n_offset / float(center)
        arg = 1.0 - normalized * normalized
        window = _bessel_i0(_KAISER_BETA * math.sqrt(max(0.0, arg))) * inv_i0_beta

        phase1[k] = sinc_val * window

    return phase1.float()


# Pre-compute coefficients (same for all instances)
_PHASE1_COEFFS = _compute_phase1_coefficients()


def _compute_half_band_filter() -> _torch.Tensor:
    """Reconstruct the full 65-tap half-band filter from phase1 coefficients."""
    h = _torch.zeros(_FILTER_LENGTH)
    h[2 * _M] = 0.5
    for k in range(_PHASE1_NUM_TAPS):
        h[2 * k + 1] = _PHASE1_COEFFS[k]
    return h


class _Resampler2x(_nn.Module):
    """Polyphase half-band FIR resampler for 2x up/downsampling.

    Uses polyphase decomposition of a Kaiser-windowed sinc half-band filter
    for efficient resampling. All filtering is done at the base rate using
    only the 32 non-zero phase1 coefficients, avoiding the naive approach
    of filtering at the upsampled rate with the full 65-tap kernel.

    Matches C++ Resampler2x numerically. The phase1 coefficients are
    symmetric (phase1[k] = phase1[2M-1-k]), so F.conv1d (cross-correlation)
    and convolution produce identical results.
    """

    def __init__(self):
        super().__init__()
        # Phase1 coefficients shaped for depthwise conv: (1, 1, 32)
        self.register_buffer(
            "_phase1", _PHASE1_COEFFS.clone().view(1, 1, -1)
        )

    def upsample(self, x: _torch.Tensor) -> _torch.Tensor:
        """Upsample by factor 2 using polyphase decomposition.

        Even outputs: pass-through with M-sample delay (group delay alignment).
        Odd outputs: FIR interpolation with phase1 coefficients, scaled by 2.

        :param x: (B, C, L) at base rate
        :return: (B, C, 2*L) at 2x rate
        """
        B, C, L = x.shape
        phase1 = self._phase1.expand(C, 1, -1)

        # Even outputs: x delayed by M samples
        even = _F.pad(x, (_M, 0))[:, :, :L]

        # Odd outputs: 2x scaled FIR with 32 phase1 taps at base rate
        odd = 2.0 * _F.conv1d(
            _F.pad(x, (_PHASE1_NUM_TAPS - 1, 0)), phase1, groups=C
        )

        # Interleave into 2x-rate output
        out = _torch.empty(B, C, 2 * L, device=x.device, dtype=x.dtype)
        out[:, :, ::2] = even
        out[:, :, 1::2] = odd
        return out

    def downsample(self, x: _torch.Tensor) -> _torch.Tensor:
        """Downsample by factor 2 using polyphase decomposition.

        Splits the 2x-rate input into even and odd polyphase components,
        then combines using the half-band structure:
          y[n] = 0.5 * x_even[n-M] + sum_j phase1[j] * x_odd[n-j-1]

        :param x: (B, C, 2*L) at 2x rate
        :return: (B, C, L) at base rate
        """
        B, C, L2 = x.shape
        L = L2 // 2
        phase1 = self._phase1.expand(C, 1, -1)

        # Split into even/odd polyphase components
        x_even = x[:, :, ::2]  # (B, C, L)
        x_odd = x[:, :, 1::2]  # (B, C, L)

        # Even contribution: center tap (0.5) with M-sample delay
        even_contrib = 0.5 * _F.pad(x_even, (_M, 0))[:, :, :L]

        # Odd contribution: phase1 FIR on odd samples, delayed by 1
        # conv gives sum_j phase1[j] * x_odd[n-j]; we need x_odd[n-j-1]
        odd_conv = _F.conv1d(
            _F.pad(x_odd, (_PHASE1_NUM_TAPS - 1, 0)), phase1, groups=C
        )
        odd_contrib = _F.pad(odd_conv, (1, 0))[:, :, :L]

        return even_contrib + odd_contrib


class _CascadedResampler(_nn.Module):
    """Cascaded 2x resamplers for 2x, 4x, or 8x oversampling.

    Matches C++ CascadedResampler: chains multiple Resampler2x stages.
    """

    def __init__(self, factor: int):
        super().__init__()
        assert factor in (1, 2, 4, 8), f"factor must be 1, 2, 4, or 8, got {factor}"
        self._factor = factor
        self._num_stages = 0
        f = factor
        while f > 1:
            self._num_stages += 1
            f >>= 1
        self._stages = _nn.ModuleList(
            [_Resampler2x() for _ in range(self._num_stages)]
        )

    def upsample(self, x: _torch.Tensor) -> _torch.Tensor:
        for stage in self._stages:
            x = stage.upsample(x)
        return x

    def downsample(self, x: _torch.Tensor) -> _torch.Tensor:
        for stage in reversed(self._stages):
            x = stage.downsample(x)
        return x

    def get_round_trip_delay(self) -> int:
        """Round-trip latency in base-rate samples.

        Matches C++ CascadedResampler::GetRoundTripLatency().
        Each 2x stage adds M samples delay at its operating rate.
        """
        return 2 * self._get_latency()

    def _get_latency(self) -> int:
        if self._num_stages == 0:
            return 0
        return 2 * _M - 2 * _M // (1 << self._num_stages)


class UpsampledActivation(_nn.Module):
    """Wraps an activation with upsample -> activate -> downsample.

    Uses the same Kaiser-windowed sinc half-band filter as C++
    NeuralAmpModelerCore's Resampler2x, ensuring identical computation
    for the same weights.
    """

    def __init__(self, activation: _nn.Module, factor: int = 2):
        super().__init__()
        assert factor in (2, 4, 8), f"factor must be 2, 4, or 8, got {factor}"
        self.activation = activation
        self._factor = factor
        self._resampler = _CascadedResampler(factor)

    @property
    def round_trip_delay(self) -> int:
        return self._resampler.get_round_trip_delay()

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """Upsample, activate, downsample.

        :param x: (B, C, L) - input (may have 2*C channels for paired activations)
        :return: (B, C', L) - same length, content delayed by round_trip_delay.
            C' = C for simple activations, C/2 for paired activations.
        """
        up = self._resampler.upsample(x)
        activated = self.activation(up)
        return self._resampler.downsample(activated)
