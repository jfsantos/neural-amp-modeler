# File: wiener_hammerstein.py
# Created Date: Sunday March 23rd 2026
# Author: Joao Felipe Santos

"""
Wiener-Hammerstein model: cascaded Linear -> Nonlinear stages, followed by a
final linear (FIR) post-filter.

Single stage (LNL):   Filter -> NL -> PostFilter
Two stages (LNLNL):   Filter -> NL -> Filter -> NL -> PostFilter
N stages:              (Filter -> NL) x N -> PostFilter

This maps naturally to guitar amp signal chains where multiple gain stages
are interleaved with interstage filtering.
"""

from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Sequence as _Sequence
from typing import Union as _Union

import numpy as _np
import torch as _torch
import torch.nn as _nn

from .._core import InitializableFromConfig as _InitializableFromConfig
from ._abc import ImportsWeights as _ImportsWeights
from ._activations import get_activation as _get_activation
from .base import BaseNet as _BaseNet


def _make_fir(length: int) -> _nn.Conv1d:
    conv = _nn.Conv1d(1, 1, length, bias=False, padding=0)
    _nn.init.zeros_(conv.weight)
    conv.weight.data[0, 0, 0] = 1.0
    return conv


def _make_mlp(
    hidden_sizes: _List[int],
    activation: _Union[str, _Dict[str, _Any]],
    context: int,
) -> _nn.Sequential:
    layers: _List[_nn.Module] = []
    in_size = context
    for h in hidden_sizes:
        layers.append(_nn.Linear(in_size, h))
        layers.append(_get_activation(activation))
        in_size = h
    layers.append(_nn.Linear(in_size, 1))
    return _nn.Sequential(*layers)


def _apply_mlp(
    h: _torch.Tensor, mlp: _nn.Sequential, context: int
) -> _torch.Tensor:
    """
    Apply a pointwise (or quasi-memoryless) MLP to a signal.

    :param h: (N, 1, L)
    :param mlp: the MLP module
    :param context: number of input samples per MLP call
    :return: (N, 1, L) if context==1, else (N, 1, L-context+1)
    """
    if context == 1:
        N, C, L = h.shape
        h_flat = h.permute(0, 2, 1).reshape(N * L, 1)
        h_flat = mlp(h_flat)
        return h_flat.reshape(N, L, 1).permute(0, 2, 1)
    else:
        N, C, L = h.shape
        h_seq = h.squeeze(1)  # (N, L)
        h_unf = h_seq.unfold(1, context, 1)  # (N, L-ctx+1, ctx)
        L2 = h_unf.shape[1]
        h_flat = h_unf.reshape(N * L2, context)
        h_flat = mlp(h_flat)
        return h_flat.reshape(N, L2, 1).permute(0, 2, 1)


class WienerHammerstein(_BaseNet, _ImportsWeights):
    """
    Cascaded Wiener-Hammerstein model.

    Signal flow: (Filter -> NL) x num_stages -> PostFilter

    :param stages: List of stage configs, each a dict with keys:
        - filter_length: int — FIR taps for this stage's linear filter
        - hidden_sizes: list[int] — MLP hidden layer sizes
        - activation: str or dict — activation for the MLP
        - context: int — samples per MLP call (1 = pointwise)
    :param post_filter_length: FIR taps for the final output filter.
    """

    def __init__(
        self,
        stages: _List[_Dict[str, _Any]],
        post_filter_length: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._stages_config = _deepcopy(stages)
        self._post_filter_length = post_filter_length

        self._filters = _nn.ModuleList()
        self._mlps = _nn.ModuleList()
        self._contexts: _List[int] = []

        for stage in stages:
            self._filters.append(_make_fir(stage["filter_length"]))
            self._mlps.append(
                _make_mlp(
                    stage["hidden_sizes"],
                    stage.get("activation", "Tanh"),
                    stage.get("context", 1),
                )
            )
            self._contexts.append(stage.get("context", 1))

        self._post_filter = _make_fir(post_filter_length)

    @classmethod
    def parse_config(cls, config):
        config = _deepcopy(config)
        # Support the old flat single-stage format
        if "stages" not in config:
            config["stages"] = [
                {
                    "filter_length": config.pop("pre_filter_length"),
                    "hidden_sizes": config.pop("hidden_sizes"),
                    "activation": config.pop("activation", "Tanh"),
                    "context": config.pop("context", 1),
                }
            ]
        return config

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        # Each filter and context window consumes (length - 1) samples
        rf = 1
        for stage, ctx in zip(self._stages_config, self._contexts):
            rf += stage["filter_length"] - 1
            rf += ctx - 1
        rf += self._post_filter_length - 1
        return rf

    def _forward(self, x: _torch.Tensor, **kwargs) -> _torch.Tensor:
        """
        :param x: (N, L) where L >= receptive_field
        :return: (N, L - receptive_field + 1)
        """
        h = x[:, None]  # (N, 1, L)

        for fir, mlp, ctx in zip(self._filters, self._mlps, self._contexts):
            h = fir(h)
            h = _apply_mlp(h, mlp, ctx)

        y = self._post_filter(h)
        return y[:, 0]

    def _export_config(self) -> _Dict[str, _Any]:
        return {
            "stages": _deepcopy(self._stages_config),
            "post_filter_length": self._post_filter_length,
        }

    def _export_weights(self) -> _np.ndarray:
        params = []
        for fir, mlp in zip(self._filters, self._mlps):
            params.append(fir.weight.flatten())
            for module in mlp:
                if hasattr(module, "weight"):
                    params.append(module.weight.flatten())
                if hasattr(module, "bias") and module.bias is not None:
                    params.append(module.bias.flatten())
        params.append(self._post_filter.weight.flatten())
        return _torch.cat(params).detach().cpu().numpy()

    def import_weights(self, weights: _Sequence[float], i: int = 0) -> int:
        weights_tensor = (
            weights if isinstance(weights, _torch.Tensor) else _torch.tensor(weights)
        )

        for fir, mlp in zip(self._filters, self._mlps):
            # FIR
            n = fir.weight.numel()
            fir.weight.data = (
                weights_tensor[i : i + n]
                .reshape(fir.weight.shape)
                .to(fir.weight.device)
            )
            i += n
            # MLP
            for module in mlp:
                if hasattr(module, "weight"):
                    n = module.weight.numel()
                    module.weight.data = (
                        weights_tensor[i : i + n]
                        .reshape(module.weight.shape)
                        .to(module.weight.device)
                    )
                    i += n
                if hasattr(module, "bias") and module.bias is not None:
                    n = module.bias.numel()
                    module.bias.data = (
                        weights_tensor[i : i + n]
                        .reshape(module.bias.shape)
                        .to(module.bias.device)
                    )
                    i += n

        # Post-filter
        n = self._post_filter.weight.numel()
        self._post_filter.weight.data = (
            weights_tensor[i : i + n]
            .reshape(self._post_filter.weight.shape)
            .to(self._post_filter.weight.device)
        )
        i += n

        return i
