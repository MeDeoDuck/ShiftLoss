"""Global shift loss for trend alignment.

Usage example:
    loss_fn = DBLossWithShift(db_loss_module, lambda_shift=0.1, k=5, mode="mse")
    total_loss, details = loss_fn(y_hat, y, return_details=True)
    print(details["shift_loss"].item(), details["delta_star"])
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def _ensure_three_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(-1)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(
        f"Expected tensor with shape [B, T] or [B, T, C], got {tensor.shape}."
    )


def _normalize_time_first(tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return [B, T, C] tensor and whether a transpose was applied."""
    tensor = _ensure_three_dim(tensor)
    time_first = True
    if tensor.size(1) < tensor.size(2):
        tensor = tensor.transpose(1, 2)
        time_first = False
    return tensor, time_first


def _ema_trend(tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute EMA trend along the time dimension."""
    ema = torch.zeros_like(tensor)
    ema[:, 0, :] = tensor[:, 0, :]
    for t in range(1, tensor.size(1)):
        ema[:, t, :] = alpha * tensor[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
    return ema


def _shift_error(
    y_t_hat: torch.Tensor,
    y_t: torch.Tensor,
    delta: int,
    mode: str,
) -> torch.Tensor:
    time_steps = y_t.size(1)
    if delta >= 0:
        pred = y_t_hat[:, delta:time_steps, :]
        target = y_t[:, : time_steps - delta, :]
    else:
        shift = -delta
        pred = y_t_hat[:, : time_steps - shift, :]
        target = y_t[:, shift:time_steps, :]

    if pred.numel() == 0:
        raise ValueError("Overlap length is zero; check input length and K.")

    if mode == "mse":
        error = (pred - target).pow(2)
    elif mode == "mae":
        error = (pred - target).abs()
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'mse' or 'mae'.")

    return error.mean()


def compute_global_shift_loss(
    y_t_hat: torch.Tensor,
    y_t: torch.Tensor,
    k: int,
    mode: str = "mse",
) -> torch.Tensor:
    """Compute hard-min global shift loss for trend components."""
    loss, _ = compute_global_shift_loss_with_delta(y_t_hat, y_t, k, mode=mode)
    return loss


def compute_global_shift_loss_with_delta(
    y_t_hat: torch.Tensor,
    y_t: torch.Tensor,
    k: int,
    mode: str = "mse",
) -> Tuple[torch.Tensor, int]:
    """Compute hard-min global shift loss and return best delta for logging."""
    y_t_hat = _ensure_three_dim(y_t_hat)
    y_t = _ensure_three_dim(y_t)

    if y_t_hat.size(1) != y_t.size(1):
        raise ValueError("Trend tensors must have the same time dimension.")

    time_steps = y_t.size(1)
    if time_steps <= 0:
        raise ValueError("Time dimension must be >= 1.")

    k_eff = min(max(k, 0), max(time_steps - 1, 0))
    deltas = torch.arange(-k_eff, k_eff + 1, device=y_t.device)

    errors = []
    for delta in deltas.tolist():
        errors.append(_shift_error(y_t_hat, y_t, int(delta), mode))

    error_tensor = torch.stack(errors, dim=0)
    min_error, min_index = torch.min(error_tensor, dim=0)
    delta_star = int(deltas[min_index].item())
    return min_error, delta_star


class DBLossWithShift(nn.Module):
    """Wrap DBLoss with a global shift loss on the trend component."""

    def __init__(
        self,
        db_loss_module: nn.Module,
        lambda_shift: float = 0.1,
        k: int = 5,
        mode: str = "mse",
        ema_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.db_loss_module = db_loss_module
        self.lambda_shift = lambda_shift
        self.k = k
        self.mode = mode
        self.ema_alpha = ema_alpha

    def _extract_trend(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        y_t_hat: Optional[torch.Tensor] = None,
        y_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if y_t_hat is not None and y_t is not None:
            return y_t_hat, y_t

        if hasattr(self.db_loss_module, "decompose") and callable(
            getattr(self.db_loss_module, "decompose")
        ):
            outputs = self.db_loss_module.decompose(y_hat, y)
            if isinstance(outputs, (tuple, list)) and len(outputs) == 4:
                _, y_t_hat, _, y_t = outputs
                return y_t_hat, y_t
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                return outputs[0], outputs[1]
            raise ValueError("db_loss_module.decompose must return 2 or 4 tensors.")

        y_hat_norm, hat_time_first = _normalize_time_first(y_hat)
        y_norm, y_time_first = _normalize_time_first(y)
        if hat_time_first != y_time_first:
            raise ValueError("Input tensors must share the same time dimension layout.")

        y_t_hat = _ema_trend(y_hat_norm, self.ema_alpha)
        y_t = _ema_trend(y_norm, self.ema_alpha)

        if not hat_time_first:
            y_t_hat = y_t_hat.transpose(1, 2)
            y_t = y_t.transpose(1, 2)

        return y_t_hat, y_t

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        y_t_hat: Optional[torch.Tensor] = None,
        y_t: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, int]]]]:
        db_loss_output = self.db_loss_module(y_hat, y)
        if isinstance(db_loss_output, (tuple, list)):
            db_loss = db_loss_output[0]
        else:
            db_loss = db_loss_output

        if y_t_hat is None or y_t is None:
            y_t_hat, y_t = self._extract_trend(y_hat, y, y_t_hat=y_t_hat, y_t=y_t)

        shift_loss, delta_star = compute_global_shift_loss_with_delta(
            y_t_hat, y_t, self.k, mode=self.mode
        )
        total_loss = db_loss + self.lambda_shift * shift_loss

        if return_details:
            details = {
                "db_loss": db_loss,
                "shift_loss": shift_loss,
                "delta_star": delta_star,
            }
            return total_loss, details

        return total_loss
