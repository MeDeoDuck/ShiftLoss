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
    raise ValueError(f"Expected tensor with shape [B, T] or [B, T, C], got {tensor.shape}.")


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
    ) -> None:
        super().__init__()
        self.db_loss_module = db_loss_module
        self.lambda_shift = lambda_shift
        self.k = k
        self.mode = mode

    def _extract_trend(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.db_loss_module, "decompose"):
            outputs = self.db_loss_module.decompose(y_hat, y)
            if isinstance(outputs, (tuple, list)) and len(outputs) == 4:
                _, y_t_hat, _, y_t = outputs
                return y_t_hat, y_t
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                return outputs[0], outputs[1]
            raise ValueError("db_loss_module.decompose must return 2 or 4 tensors.")
        raise ValueError(
            "Trend tensors were not provided and db_loss_module has no decompose method."
        )

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
            y_t_hat, y_t = self._extract_trend(y_hat, y)

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
