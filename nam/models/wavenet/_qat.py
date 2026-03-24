"""
Quantization-Aware Training (QAT) for Q15 fixed-point WaveNet inference.

Inserts fake quantization operations at the boundaries that match the
nam2c Q15 fused pipeline:

1. Rechannel output -> Q15 (layer_buf)
2. Conv1D weights (per-layer symmetric)
3. Post conv+mixin accumulator -> Q15 + activation
4. Layer1x1 weights (per-layer symmetric)
5. Layer1x1 output + residual with saturation -> Q15
6. Head accumulation rescaling

Usage:
    from nam.models.wavenet._qat import enable_qat, disable_qat

    model = WaveNet(...)  # or load from .nam
    enable_qat(model, act_scales=[...])  # optional per-layer act_scales
    # ... train ...
    disable_qat(model)
    # ... export ...
"""

import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Q15_MAX = 32767


# ---------------------------------------------------------------------------
# Fake quantization primitives
# ---------------------------------------------------------------------------


class FakeQuantizeQ15(nn.Module):
    """Differentiable fake quantization simulating Q15 fixed-point.

    Uses straight-through estimator (STE) for gradients.
    Scale is learnable via log parameterization to stay positive.
    """

    def __init__(
        self,
        initial_scale: float = 1.0,
        q_max: int = Q15_MAX,
        learnable: bool = True,
    ):
        super().__init__()
        self.q_max = q_max
        if learnable:
            self.log_scale = nn.Parameter(
                torch.tensor(float(initial_scale)).clamp(min=1e-8).log()
            )
        else:
            self.register_buffer(
                "log_scale",
                torch.tensor(float(initial_scale)).clamp(min=1e-8).log(),
            )

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fake_quantize_activation(x, self.scale, self.q_max)

    def extra_repr(self) -> str:
        return f"scale={self.scale.item():.6f}, q_max={self.q_max}"


def _fake_quantize_activation(
    x: torch.Tensor, scale: torch.Tensor, q_max: int = Q15_MAX
) -> torch.Tensor:
    """Fake quantize activations: quantize then dequantize with STE.

    Simulates: q = clamp(round(x / scale), -q_max, q_max)
               x_hat = q * scale / q_max

    STE is applied only to round() (zero gradient replaced by identity).
    clamp() keeps its real gradient (1 inside, 0 outside) so the scale
    parameter receives a meaningful gradient: it learns to widen the range
    where activations would clip, or tighten it to improve resolution.
    """
    x_scaled = x / scale
    # STE: straight-through only for round()
    x_rounded = x_scaled + (torch.round(x_scaled) - x_scaled).detach()
    # clamp keeps its gradient (0 where clipped, 1 otherwise)
    x_q = torch.clamp(x_rounded, -q_max, q_max)
    return x_q * scale / float(q_max)


def _fake_quantize_weight(
    w: torch.Tensor, q_max: int = Q15_MAX
) -> torch.Tensor:
    """Fake quantize weights with per-tensor symmetric scale.

    Scale = max(|w|), matching nam2c's quantize_q15().
    """
    max_abs = w.detach().abs().max()
    if max_abs == 0:
        return w
    inv_scale = float(q_max) / max_abs
    w_q = torch.clamp(torch.round(w * inv_scale), -q_max, q_max)
    w_hat = w_q / inv_scale
    return w + (w_hat - w).detach()


def _conv_with_fake_weight(
    conv: nn.Conv1d, x: torch.Tensor, q_max: int = Q15_MAX
) -> torch.Tensor:
    """Run conv1d with fake-quantized weights."""
    w = _fake_quantize_weight(conv.weight, q_max)
    return F.conv1d(
        x,
        w,
        conv.bias,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )


# ---------------------------------------------------------------------------
# Patched forward methods
# ---------------------------------------------------------------------------


def _qat_layer_forward(self, x, h, out_length):
    """Replacement forward for _Layer with QAT fake quantization.

    Matches the original _Layer.forward() but inserts:
    - Fake weight quantization on conv and layer1x1
    - Fake activation quantization after activation (accumulator -> Q15)
    - Fake saturation on residual (SSAT16 on Q15 layer_buf)
    """

    def _c(t_len, tensor=h):
        return tensor[:, :, -t_len:]

    # Step 1: Conv with fake-quantized weights
    conv_input = x
    if self._conv_pre_film is not None:
        pre_conv_length = min(conv_input.shape[2], h.shape[2])
        conv_input = self._conv_pre_film(
            _c(pre_conv_length, tensor=conv_input), _c(pre_conv_length)
        )
    zconv = _conv_with_fake_weight(self._conv, conv_input, self._qat_q_max)
    if self._conv_post_film is not None:
        post_conv_length = min(zconv.shape[2], h.shape[2])
        zconv = self._conv_post_film(
            _c(post_conv_length, tensor=zconv), _c(post_conv_length)
        )

    # Step 2: Input mixin
    mixin_input = h
    if self._input_mixin_pre_film is not None:
        mixin_input = self._input_mixin_pre_film(mixin_input, h)
    mix_out = self._input_mixer(mixin_input)[:, :, -zconv.shape[2] :]
    if self._input_mixin_post_film is not None:
        mix_out = self._input_mixin_post_film(mix_out, _c(mix_out.shape[2]))

    # Step 3: Add + activation
    z1len = min(zconv.shape[2], mix_out.shape[2])
    z1 = zconv[:, :, -z1len:] + mix_out[:, :, -z1len:]
    if self._activation_pre_film is not None:
        z1 = self._activation_pre_film(z1, _c(z1.shape[2]))

    post_activation = self._activation(z1)
    if self._activation_post_film is not None:
        post_activation = self._activation_post_film(
            post_activation, _c(post_activation.shape[2])
        )

    # ** QAT: fake quantize post-activation (accumulator -> Q15) **
    post_activation = self._qat_post_act(post_activation)

    # Step 4: layer1x1 + residual
    layer_output = post_activation
    if self._layer1x1 is not None:
        # ** QAT: fake quantize layer1x1 weights **
        layer_output = _conv_with_fake_weight(
            self._layer1x1, layer_output, self._qat_q_max
        )
        if self._layer1x1_post_film is not None:
            layer_output = self._layer1x1_post_film(
                layer_output, _c(layer_output.shape[2])
            )

    # Head output (no QAT needed - head path stays float)
    head_output = post_activation
    if self._head1x1 is not None:
        head_output = self._head1x1(head_output)[:, :, -out_length:]
        if self._head1x1_post_film is not None:
            head_output = self._head1x1_post_film(
                head_output, _c(head_output.shape[2])
            )
    else:
        head_output = head_output[:, :, -out_length:]

    residual = x[:, :, -layer_output.shape[2] :] + layer_output

    # ** QAT: fake saturate residual (SSAT16 on Q15 layer_buf) **
    residual = self._qat_residual(residual)

    return (residual, head_output)


def _qat_layer_array_forward(self, x, c, head_input=None):
    """Replacement forward for _LayerArray with QAT fake quantization.

    Inserts fake quantization after rechannel (float -> Q15 layer_buf).
    """
    out_length = min(x.shape[2], c.shape[2]) - (self.receptive_field - 1)
    x = self._rechannel(x)

    # ** QAT: fake quantize rechannel output (float -> Q15 layer_buf) **
    x = self._qat_rechannel_q(x)

    for layer in self._layers:
        x, head_term = layer(x, c, out_length)
        head_input = (
            head_term
            if head_input is None
            else head_input[:, :, -out_length:] + head_term
        )
    return self._head_rechannel(head_input), x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enable_qat(
    model,
    act_scales: Optional[List[float]] = None,
    default_act_scale: float = 1.0,
    q_max: int = Q15_MAX,
    learnable_scales: bool = True,
):
    """Enable quantization-aware training on a WaveNet model.

    Instruments the model's forward pass with fake quantization ops that
    simulate the nam2c Q15 fused inference pipeline. The fake quantization
    modules are added as proper sub-modules so their learnable scale
    parameters participate in optimization.

    :param model: A _WaveNet instance (the inner model, not the WaveNet wrapper).
        If a WaveNet wrapper is passed, the inner _WaveNet is instrumented.
    :param act_scales: Optional per-layer activation scales from profiling.
        If provided, must have one entry per layer across all layer arrays.
        These initialize the learnable FakeQuantizeQ15 scales.
    :param default_act_scale: Default activation scale when act_scales is not
        provided. 1.0 works well for audio in [-1, 1].
    :param q_max: Maximum quantized value (32767 for full Q15 range).
    :param learnable_scales: If True, activation scales are learnable parameters.
    """
    # Unwrap LightningModule -> WaveNet -> _WaveNet if needed
    inner = model
    while hasattr(inner, "_net") and not hasattr(inner, "_layer_arrays"):
        inner = inner._net

    if not hasattr(inner, "_layer_arrays"):
        raise TypeError(
            f"Expected a WaveNet model with _layer_arrays, got {type(inner)}"
        )

    if hasattr(inner, "_qat_enabled") and inner._qat_enabled:
        raise RuntimeError("QAT is already enabled on this model")

    scale_idx = 0

    for la_idx, la in enumerate(inner._layer_arrays):
        # Rechannel output fake quantize
        rechannel_scale = default_act_scale
        if act_scales is not None and len(act_scales) > 0:
            # Use max of the layer scales in this array as the buf_scale
            n_layers = len(la._layers)
            la_scales = act_scales[scale_idx : scale_idx + n_layers]
            rechannel_scale = max(la_scales) if la_scales else default_act_scale

        fq_rechannel = FakeQuantizeQ15(
            initial_scale=rechannel_scale,
            q_max=q_max,
            learnable=learnable_scales,
        )
        la.add_module("_qat_rechannel_q", fq_rechannel)

        # Save original forward and patch
        la._qat_orig_forward = la.forward
        la.forward = types.MethodType(_qat_layer_array_forward, la)

        # Instrument each layer
        for l_idx, layer in enumerate(la._layers):
            # Per-layer activation scale
            if act_scales is not None and scale_idx < len(act_scales):
                layer_scale = act_scales[scale_idx]
            else:
                layer_scale = default_act_scale
            scale_idx += 1

            # Post-activation fake quantize (accumulator -> Q15)
            fq_post_act = FakeQuantizeQ15(
                initial_scale=layer_scale,
                q_max=q_max,
                learnable=learnable_scales,
            )
            layer.add_module("_qat_post_act", fq_post_act)

            # Residual fake quantize (SSAT16 saturation on Q15 layer_buf)
            # Uses the buf_scale (max across layer array) for the residual path
            fq_residual = FakeQuantizeQ15(
                initial_scale=rechannel_scale,
                q_max=q_max,
                learnable=learnable_scales,
            )
            layer.add_module("_qat_residual", fq_residual)

            layer._qat_q_max = q_max

            # Save original forward and patch
            layer._qat_orig_forward = layer.forward
            layer.forward = types.MethodType(_qat_layer_forward, layer)

    inner._qat_enabled = True


def disable_qat(model):
    """Disable QAT and restore original forward methods.

    The learned activation scales are preserved in the modules and can be
    extracted with :func:`get_learned_act_scales` before disabling.
    """
    inner = model
    while hasattr(inner, "_net") and not hasattr(inner, "_layer_arrays"):
        inner = inner._net

    if not getattr(inner, "_qat_enabled", False):
        return

    for la in inner._layer_arrays:
        if hasattr(la, "_qat_orig_forward"):
            la.forward = la._qat_orig_forward
            del la._qat_orig_forward
        # Remove the QAT module but keep it accessible for scale extraction
        if hasattr(la, "_qat_rechannel_q"):
            delattr(la, "_qat_rechannel_q")

        for layer in la._layers:
            if hasattr(layer, "_qat_orig_forward"):
                layer.forward = layer._qat_orig_forward
                del layer._qat_orig_forward
            if hasattr(layer, "_qat_q_max"):
                del layer._qat_q_max
            for attr in ("_qat_post_act", "_qat_residual"):
                if hasattr(layer, attr):
                    delattr(layer, attr)

    inner._qat_enabled = False


def get_learned_act_scales(model) -> List[float]:
    """Extract learned per-layer activation scales from a QAT-enabled model.

    Returns a flat list of scales, one per layer across all layer arrays,
    suitable for passing to nam2c --act-profile.
    """
    inner = model
    while hasattr(inner, "_net") and not hasattr(inner, "_layer_arrays"):
        inner = inner._net

    scales = []
    for la in inner._layer_arrays:
        for layer in la._layers:
            fq = getattr(layer, "_qat_post_act", None)
            if fq is not None:
                scales.append(fq.scale.item())
            else:
                scales.append(1.0)
    return scales


def get_qat_scales_from_state_dict(
    state_dict: dict,
) -> Optional[dict]:
    """Extract learned QAT scales from a checkpoint state dict.

    Returns a dict with "per_layer_act_scale" and "buf_scale" lists,
    or None if no QAT parameters are found.
    The keys in the state dict are expected to match the pattern
    ``*._qat_post_act.log_scale`` and ``*._qat_rechannel_q.log_scale``.
    """
    import math

    act_scales = {}  # (la_idx, l_idx) -> scale
    buf_scales = {}  # la_idx -> scale

    for key, val in state_dict.items():
        if "_qat_post_act.log_scale" in key:
            # e.g. _net._net._layer_arrays.0._layers.3._qat_post_act.log_scale
            parts = key.split(".")
            la_idx = int(parts[parts.index("_layer_arrays") + 1])
            l_idx = int(parts[parts.index("_layers") + 1])
            act_scales[(la_idx, l_idx)] = math.exp(val.item())
        elif "_qat_rechannel_q.log_scale" in key:
            parts = key.split(".")
            la_idx = int(parts[parts.index("_layer_arrays") + 1])
            buf_scales[la_idx] = math.exp(val.item())

    if not act_scales:
        return None

    # Flatten to ordered lists
    max_la = max(la for la, _ in act_scales) + 1
    per_layer = []
    for la_idx in range(max_la):
        layer_keys = sorted(
            [(la, l) for la, l in act_scales if la == la_idx], key=lambda x: x[1]
        )
        for k in layer_keys:
            per_layer.append(act_scales[k])

    buf_list = [buf_scales.get(i, 1.0) for i in range(max_la)]

    return {"per_layer_act_scale": per_layer, "buf_scale": buf_list}


def get_learned_buf_scales(model) -> List[float]:
    """Extract learned per-layer-array buffer scales (rechannel/residual).

    Returns one scale per layer array.
    """
    inner = model
    while hasattr(inner, "_net") and not hasattr(inner, "_layer_arrays"):
        inner = inner._net

    scales = []
    for la in inner._layer_arrays:
        fq = getattr(la, "_qat_rechannel_q", None)
        if fq is not None:
            scales.append(fq.scale.item())
        else:
            scales.append(1.0)
    return scales
