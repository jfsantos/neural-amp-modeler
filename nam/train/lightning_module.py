# File: base.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Implements the base PyTorch Lightning module.
This is meant to combine an actual model (subclassed from `..models.base.BaseNet`)
along with loss function boilerplate.

For the base *PyTorch* model containing the actual architecture, see `..models.base`.
"""

import logging as _logging
from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import NamedTuple as _NamedTuple
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import pytorch_lightning as _pl
import torch as _torch
import torch.nn as _nn

from .._core import InitializableFromConfig as _InitializableFromConfig
from .._dependencies import auraloss as _auraloss
from ..models.base import BaseNet as _BaseNet
from ..models.conv_net import ConvNet as _ConvNet
from ..models.factory import init as _init_model
from ..models.factory import register as _register_model
from ..models.linear import Linear as _Linear
from ..models.losses import apply_pre_emphasis_filter as _apply_pre_emphasis_filter
from ..models.losses import esr as _esr
from ..models.losses import mse as _mse
from ..models.losses import mse_fft as _mse_fft
from ..models.losses import multi_resolution_stft_loss as _multi_resolution_stft_loss
from ..models.recurrent import LSTM as _LSTM
from ..models.wavenet import WaveNet as _WaveNet
from ..util import init as _init

logger = _logging.getLogger(__name__)


class ValidationLoss(_Enum):
    """
    mse: mean squared error
    esr: error signal ratio (Eq. (10) from
        https://www.mdpi.com/2076-3417/10/3/766/htm
        NOTE: Be careful when computing ESR on minibatches! The average ESR over
        a minibatch of data not the same as the ESR of all of the same data in
        the minibatch calculated over at once (because of the denominator).
        (Hint: think about what happens if one item in the minibatch is all
        zeroes...)
    """

    MSE = "mse"
    ESR = "esr"


class _CustomLoss(_NamedTuple):
    weight: float
    func: _Callable[[_torch.Tensor, _torch.Tensor], _torch.Tensor]


@_dataclass
class LossConfig(_InitializableFromConfig):
    """
    :param mse_weight: Weight for the MSE loss term. If None, MSE is not computed.
    :param mrstft_weight: Multi-resolution short-time Fourier transform loss
        coefficient. None means to skip; 2e-4 works pretty well if one wants to use it.
    :param mask_first: How many of the first samples to ignore when computing the loss.
    :param dc_weight: Weight for the DC loss term. If 0, ignored.
    :params val_loss: Which loss to track for the best model checkpoint. If a string is
        provided, then it must match the name of a custom loss.
    :param pre_emph_coef: Coefficient of 1st-order pre-emphasis filter from
        https://www.mdpi.com/2076-3417/10/3/766. Paper value: 0.95.
    :param pre_
    """

    class ValLossNameError(ValueError):
        """
        Error thrown when a validation loss name is invalid.
        """

        pass

    mse_weight: _Optional[float] = 1.0
    mrstft_weight: _Optional[float] = None
    fourier: bool = False
    mask_first: int = 0
    dc_weight: float = None
    val_loss: _Union[ValidationLoss, str] = ValidationLoss.MSE
    pre_emph_weight: _Optional[float] = None
    pre_emph_coef: _Optional[float] = None
    pre_emph_mrstft_weight: _Optional[float] = None
    pre_emph_mrstft_coef: _Optional[float] = None
    custom_losses: _Optional[_Dict[str, _CustomLoss]] = None

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)

        def parse_custom_losses(config):
            def init_custom_loss(
                name: str, kwargs: _Dict[str, _Any], weight: float
            ) -> _CustomLoss:
                func = _init(name, **kwargs)
                return _CustomLoss(weight, func)

            if "custom_losses" in config:
                return {
                    k: init_custom_loss(v["name"], v["kwargs"], v["weight"])
                    for k, v in config["custom_losses"].items()
                }
            else:
                return None

        def parse_val_loss():
            # TODO: Pydantic
            value = config.get("val_loss", "mse")
            try:
                return ValidationLoss(value)
            except ValueError:
                # Now we need to check if it's a name of a custom loss
                if value not in custom_losses:
                    raise cls.ValLossNameError(f"Invalid validation loss: {value}")
                return value

        def get_mrstft_weight() -> _Optional[float]:
            key = "mrstft_weight"
            wrong_key = "mstft_key"  # Backward compatibility
            if key in config:
                if "mstft_weight" in config:
                    raise ValueError(
                        f"Received loss configuration with both '{key}' and "
                        f"'{wrong_key}'. Provide only '{key}'."
                    )
                return config[key]
            elif wrong_key in config:
                logger.warning(
                    f"Use of '{wrong_key}' is deprecated and will be removed in a future "
                    f"version. Use '{key}' instead."
                )
                return config[wrong_key]
            else:
                return None

        custom_losses = parse_custom_losses(config)
        val_loss = parse_val_loss()
        mrstft_weight = get_mrstft_weight()

        return {
            "fourier": config.get("fourier", False),
            "mask_first": config.get("mask_first", 0),
            "dc_weight": config.get("dc_weight"),
            "val_loss": val_loss,
            "pre_emph_coef": config.get("pre_emph_coef"),
            "pre_emph_weight": config.get("pre_emph_weight"),
            "mrstft_weight": mrstft_weight,
            "pre_emph_mrstft_weight": config.get("pre_emph_mrstft_weight"),
            "pre_emph_mrstft_coef": config.get("pre_emph_mrstft_coef"),
            "custom_losses": custom_losses,
        }

    def apply_mask(self, *args):
        """
        :param args: (L,) or (B,)
        :return: (L-M,) or (B, L-M)
        """
        return tuple(a[..., self.mask_first :] for a in args)


class _LossItem(_NamedTuple):
    weight: _Optional[float]
    value: _Optional[_torch.Tensor]


class LightningModule(_pl.LightningModule, _InitializableFromConfig):
    """
    The PyTorch Lightning Module that unites the model with its loss and
    optimization recipe.
    """

    def __init__(
        self,
        net: _BaseNet,
        optimizer_config: _Optional[dict] = None,
        scheduler_config: _Optional[dict] = None,
        loss_config: _Optional[LossConfig] = None,
        optimizer_small_config: _Optional[dict] = None,
        freeze_small: _Optional[dict] = None,
    ):
        """
        :param scheduler_config: contains
            Required:
            * "class"
            * "kwargs"
            Optional (defaults to Lightning defaults):
            * "interval" ("epoch" of "step")
            * "frequency" (int)
            * "monitor" (str)
        :param optimizer_small_config: Optimizer kwargs for the small model step
            in dual-optimizer slimmable training. If None, uses the main optimizer
            config for both. Only used when the net is a slimmable WaveNet.
        :param freeze_small: Controls when to freeze the small model and switch
            to training the large model. Supported keys:
            - "after_epoch" (int): Hard cutoff — freeze after this epoch.
            - "patience" (int): Freeze after this many epochs without improvement.
            - "min_delta" (float): Minimum improvement to reset patience counter
              (default: 0.0).
            When both "after_epoch" and "patience" are set, whichever triggers
            first wins. If None, both models are trained every step (no phasing).
        """
        super().__init__()
        self._net = net
        self._optimizer_config = {} if optimizer_config is None else optimizer_config
        self._optimizer_small_config = optimizer_small_config
        self._scheduler_config = scheduler_config
        self._loss_config = LossConfig() if loss_config is None else loss_config
        self._mrstft = None  # Multi-resolution short-time Fourier transform loss
        # Where to compute the MRSTFT.
        # Keeping it on-device is preferable, but if that fails, then remember to drop
        # it to cpu from then on.
        self._mrstft_device: _Optional[_torch.device] = None

        # Freeze-small config
        self._freeze_small_config = freeze_small
        self._freeze_small_after_epoch: _Optional[int] = None
        self._freeze_small_patience: _Optional[int] = None
        self._freeze_small_min_delta: float = 0.0
        if freeze_small is not None:
            self._freeze_small_after_epoch = freeze_small.get("after_epoch")
            self._freeze_small_patience = freeze_small.get("patience")
            self._freeze_small_min_delta = freeze_small.get("min_delta", 0.0)

        # Patience tracking state (reset in on_train_start if needed)
        self._small_best_loss: _Optional[float] = None
        self._small_patience_counter: int = 0
        self._small_frozen: bool = False
        self._small_frozen_at_epoch: _Optional[int] = None

        # Dual-optimizer mode for slimmable WaveNet
        self._dual_optimizer = self._is_dual_optimizer_applicable()
        if self._dual_optimizer:
            self.automatic_optimization = False
            # Tell the WaveNet wrapper that slimming is controlled externally
            self._net._external_slimming_control = True

    def _is_dual_optimizer_applicable(self) -> bool:
        """Check if the net is a slimmable WaveNet that supports dual optimizers."""
        if not isinstance(self._net, _WaveNet):
            return False
        return self._net._net.is_slimmable() and (
            self._optimizer_small_config is not None
            or self._freeze_small_config is not None
        )

    @classmethod
    def init_from_config(cls, config):
        checkpoint_path = config.get("checkpoint_path")
        config = cls.parse_config(config)
        return (
            cls(**config)
            if checkpoint_path is None
            else cls.load_from_checkpoint(checkpoint_path, **config)
        )

    @classmethod
    def parse_config(cls, config):
        """
        e.g.

        {
            "net": {
                "name": "ConvNet",
                "config": {...}
            },
            "loss": {
                "dc_weight": 0.1
            },
            "optimizer": {
                "lr": 0.0003
            },
            "lr_scheduler": {
                "class": "ReduceLROnPlateau",
                "kwargs": {
                    "factor": 0.8,
                    "patience": 10,
                    "cooldown": 15,
                    "min_lr": 1e-06,
                    "verbose": true
                },
                "monitor": "val_loss"
            }
        }
        """
        config = super().parse_config(config)
        net_config = config["net"]
        # A little hacky--assumes "init_from_config"-style factory.
        net = _init_model(
            name=net_config["name"], kwargs={"config": net_config["config"]}
        )
        loss_config = LossConfig.init_from_config(config.get("loss", {}))
        result = {
            "net": net,
            "optimizer_config": config["optimizer"],
            "scheduler_config": config["lr_scheduler"],
            "loss_config": loss_config,
        }
        if "optimizer_small" in config:
            result["optimizer_small_config"] = config["optimizer_small"]
        if "freeze_small" in config:
            val = config["freeze_small"]
            # Accept a plain int as shorthand for {"after_epoch": N}
            if isinstance(val, int):
                result["freeze_small"] = {"after_epoch": val}
            else:
                result["freeze_small"] = val
        return result

    @classmethod
    def register_net_initializer(cls, name, constructor, overwrite: bool = False):
        logger.warning(f"Deprecated: use models.factory.register instead")
        _register_model(name=name, constructor=constructor, overwrite=overwrite)

    @property
    def net(self) -> _nn.Module:
        return self._net

    def configure_optimizers(self):
        if self._dual_optimizer:
            return self._configure_dual_optimizers()

        optimizer = _torch.optim.Adam(self.parameters(), **self._optimizer_config)
        if self._scheduler_config is None:
            return optimizer
        else:
            lr_scheduler = getattr(
                _torch.optim.lr_scheduler, self._scheduler_config["class"]
            )(optimizer, **self._scheduler_config["kwargs"])
            lr_scheduler_config = {"scheduler": lr_scheduler}
            for key in ("interval", "frequency", "monitor"):
                if key in self._scheduler_config:
                    lr_scheduler_config[key] = self._scheduler_config[key]
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _configure_dual_optimizers(self):
        small_config = (
            self._optimizer_small_config
            if self._optimizer_small_config is not None
            else self._optimizer_config
        )
        opt_small = _torch.optim.Adam(self.parameters(), **small_config)
        opt_large = _torch.optim.Adam(self.parameters(), **self._optimizer_config)

        optimizers = [opt_small, opt_large]
        schedulers = []

        if self._scheduler_config is not None:
            for opt in optimizers:
                lr_scheduler = getattr(
                    _torch.optim.lr_scheduler, self._scheduler_config["class"]
                )(opt, **self._scheduler_config["kwargs"])
                schedulers.append(lr_scheduler)

        if schedulers:
            return optimizers, schedulers
        return optimizers

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)  # TODO deprecate--use self.net() instead.

    def on_load_checkpoint(self, checkpoint: _Dict[str, _Any]) -> None:
        # Resolves https://github.com/sdatkinson/neural-amp-modeler/issues/351
        self.net.sample_rate = checkpoint["sample_rate"]

    def on_save_checkpoint(self, checkpoint: _Dict[str, _Any]) -> None:
        # Resolves https://github.com/sdatkinson/neural-amp-modeler/issues/351
        checkpoint["sample_rate"] = self.net.sample_rate

    def _shared_step(
        self, batch
    ) -> _Tuple[_torch.Tensor, _torch.Tensor, _Dict[str, _LossItem]]:
        """
        B: Batch size
        L: Sequence length

        :return: (B,L), (B,L)
        """
        args, targets = batch[:-1], batch[-1]
        preds = self(*args, pad_start=False)

        return preds, targets, self._get_loss_dict(preds, targets)

    def training_step(self, batch, batch_idx):
        if self._dual_optimizer:
            return self._dual_optimizer_training_step(batch)

        _, _, loss_dict = self._shared_step(batch)

        loss = 0.0
        for v in loss_dict.values():
            if v.weight is not None and v.weight > 0.0:
                loss = loss + v.weight * v.value
        return loss

    def _dual_optimizer_training_step(self, batch):
        opt_small, opt_large = self.optimizers()

        if not self._small_frozen:
            # Phase 1: Train only the small model
            self._net._net.set_slimming(0.0)
            _, _, loss_dict_small = self._shared_step(batch)
            loss_small = sum(
                v.weight * v.value
                for v in loss_dict_small.values()
                if v.weight is not None and v.weight > 0.0
            )
            opt_small.zero_grad()
            self.manual_backward(loss_small)
            self._mask_gradients_to_small()
            opt_small.step()
            self._net._net.set_slimming(1.0)
            self.log("train_loss_small", loss_small, prog_bar=True)
            return loss_small
        else:
            # Phase 2: Small is frozen, train only the large model (boosting
            # detaches the small region's gradients automatically)
            self._net._net.set_slimming(1.0)
            _, _, loss_dict_large = self._shared_step(batch)
            loss_large = sum(
                v.weight * v.value
                for v in loss_dict_large.values()
                if v.weight is not None and v.weight > 0.0
            )
            opt_large.zero_grad()
            self.manual_backward(loss_large)
            opt_large.step()
            self.log("train_loss_large", loss_large, prog_bar=True)
            return loss_large

    def _mask_gradients_to_small(self):
        """Zero gradients outside the smallest channel slice.

        After the small-model backward pass, only the small region should
        receive gradient updates. This masks out everything else.
        """
        from ..models.wavenet._slimmable_conv import SlimmableConv1dBase

        for module in self._net.modules():
            if not isinstance(module, SlimmableConv1dBase):
                continue
            small_in = module._allowed_in_channels[0]
            small_out = module._allowed_out_channels[0]

            w = module.weight
            if w.grad is not None:
                # Zero grad outside [:small_out, :small_in, :]
                if small_out < w.shape[0]:
                    w.grad[small_out:, :, :] = 0.0
                if small_in < w.shape[1]:
                    w.grad[:, small_in:, :] = 0.0

            b = module.bias
            if b is not None and b.grad is not None:
                if small_out < b.shape[0]:
                    b.grad[small_out:] = 0.0

    def on_train_epoch_end(self):
        if not self._dual_optimizer:
            return

        # Check freeze triggers at end of epoch (before stepping schedulers)
        if not self._small_frozen:
            self._check_freeze_small()

        # Only step the active phase's scheduler
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            if not self._small_frozen:
                # Phase 1: step small scheduler
                if len(schedulers) > 0 and schedulers[0] is not None:
                    schedulers[0].step()
            else:
                # Phase 2: step large scheduler
                if len(schedulers) > 1 and schedulers[1] is not None:
                    schedulers[1].step()

    def _check_freeze_small(self):
        """Check if the small model should be frozen this epoch."""
        # Hard cutoff
        if (
            self._freeze_small_after_epoch is not None
            and self.current_epoch >= self._freeze_small_after_epoch
        ):
            self._freeze_small_now()
            return

        # Patience-based: use the epoch-averaged train_loss_small
        if self._freeze_small_patience is None:
            return

        callback_metrics = self.trainer.callback_metrics
        if "train_loss_small" not in callback_metrics:
            return
        current_loss = callback_metrics["train_loss_small"].item()

        if (
            self._small_best_loss is None
            or current_loss < self._small_best_loss - self._freeze_small_min_delta
        ):
            self._small_best_loss = current_loss
            self._small_patience_counter = 0
        else:
            self._small_patience_counter += 1

        if self._small_patience_counter >= self._freeze_small_patience:
            self._freeze_small_now()

    def _freeze_small_now(self):
        self._small_frozen = True
        self._small_frozen_at_epoch = self.current_epoch
        logger.info(
            f"Small model frozen at epoch {self.current_epoch}. "
            f"Switching to large model training."
        )

    def validation_step(self, batch, batch_idx):
        preds, targets, loss_dict = self._shared_step(batch)

        def get_val_loss():
            # "esr" -> "ESR"
            # "mse" -> "MSE"
            # Others unsupported...
            # TODO better mapping from Enum to dict keys
            val_loss_type = self._loss_config.val_loss
            val_loss_key_for_loss_dict = (
                val_loss_type
                if isinstance(val_loss_type, str)
                else val_loss_type.value.upper()
            )
            if val_loss_key_for_loss_dict in loss_dict:
                return loss_dict[val_loss_key_for_loss_dict].value
            else:
                raise RuntimeError(
                    f"Undefined validation loss routine for {val_loss_type}"
                )

        loss_dict["ESR"] = _LossItem(None, self._esr_loss(preds, targets))
        val_loss = get_val_loss()
        self.log_dict(
            {
                "val_loss": val_loss,
                **{key: value.value for key, value in loss_dict.items()},
            }
        )
        return val_loss

    def _esr_loss(self, preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
        """
        Error signal ratio aka ESR loss.

        Eq. (10), from
        https://www.mdpi.com/2076-3417/10/3/766/htm

        B: Batch size
        L: Sequence length

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        return _esr(preds, targets)

    def _get_loss_dict(self, preds, targets) -> _Dict[str, _LossItem]:
        """
        Compute all of the losses.
        """
        # Compute all relevant losses.
        loss_dict = {}  # Mind keys versus validation loss requested...

        def get_mse_loss():
            if self._loss_config.mse_weight is None:
                return

            if self._loss_config.fourier:
                loss_dict["MSE_FFT"] = _LossItem(1.0, _mse_fft(preds, targets))
            else:
                loss_dict["MSE"] = _LossItem(1.0, self._mse_loss(preds, targets))

        get_mse_loss()

        # Pre-emphasized MSE
        if self._loss_config.pre_emph_weight is not None:
            if (self._loss_config.pre_emph_coef is None) != (
                self._loss_config.pre_emph_weight is None
            ):
                raise ValueError("Invalid pre-emph")
            loss_dict["Pre-emphasized MSE"] = _LossItem(
                self._loss_config.pre_emph_weight,
                self._mse_loss(
                    preds, targets, pre_emph_coef=self._loss_config.pre_emph_coef
                ),
            )
        # Multi-resolution short-time Fourier transform loss
        if self._loss_config.mrstft_weight is not None:
            loss_dict["MRSTFT"] = _LossItem(
                self._loss_config.mrstft_weight, self._mrstft_loss(preds, targets)
            )
        # Pre-emphasized MRSTFT
        if self._loss_config.pre_emph_mrstft_weight is not None:
            loss_dict["Pre-emphasized MRSTFT"] = _LossItem(
                self._loss_config.pre_emph_mrstft_weight,
                self._mrstft_loss(
                    preds, targets, pre_emph_coef=self._loss_config.pre_emph_mrstft_coef
                ),
            )
        # DC loss
        dc_weight = self._loss_config.dc_weight
        if dc_weight is not None and dc_weight > 0.0:
            # Denominator could be a bad idea. I'm going to omit it esp since I'm
            # using mini batches
            mean_dims = _torch.arange(1, preds.ndim).tolist()
            dc_loss = _nn.MSELoss()(
                preds.mean(dim=mean_dims), targets.mean(dim=mean_dims)
            )
            loss_dict["DC MSE"] = _LossItem(dc_weight, dc_loss)

        def get_custom_losses():
            if self._loss_config.custom_losses is None:
                return
            for name, loss in self._loss_config.custom_losses.items():
                loss_dict[name] = _LossItem(loss.weight, loss.func(preds, targets))

        get_custom_losses()
        return loss_dict

    def _mse_loss(self, preds, targets, pre_emph_coef: _Optional[float] = None):
        if pre_emph_coef is not None:
            preds, targets = [
                _apply_pre_emphasis_filter(z, pre_emph_coef) for z in (preds, targets)
            ]
        return _mse(preds, targets)

    def _mrstft_loss(
        self,
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        pre_emph_coef: _Optional[float] = None,
    ) -> _torch.Tensor:
        """
        Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
        B: Batch size
        L: Sequence length

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        if self._mrstft is None:
            self._mrstft = _auraloss.freq.MultiResolutionSTFTLoss()
        backup_device = "cpu"

        if pre_emph_coef is not None:
            preds, targets = [
                _apply_pre_emphasis_filter(z, pre_emph_coef) for z in (preds, targets)
            ]

        try:
            return _multi_resolution_stft_loss(
                preds, targets, self._mrstft, device=self._mrstft_device
            )
        except Exception as e:
            if self._mrstft_device == backup_device:
                raise e
            logger.warning("MRSTFT failed on device; falling back to CPU")
            self._mrstft_device = backup_device
            return _multi_resolution_stft_loss(
                preds, targets, self._mrstft, device=self._mrstft_device
            )
