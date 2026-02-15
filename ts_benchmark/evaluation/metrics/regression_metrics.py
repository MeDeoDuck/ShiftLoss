# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "wape",
    "msmape",
    "mae_norm",
    "mse_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
    "DTW_norm",
    "TDI",
]


def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Simple error"""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Percentage error"""
    return (actual - predicted) / actual


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error(actual, predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 2,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """

    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Simple error"""
    return scaler.transform(actual) - scaler.transform(predicted)


def _percentage_error_norm(
    actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs
):
    """Percentage error"""
    return (scaler.transform(actual) - scaler.transform(predicted)) / scaler.transform(
        actual
    )


def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse_norm(actual, predicted, scaler))


def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def mase_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    hist_data: np.ndarray,
    seasonality: int = 2,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    hist_data = scaler.transform(hist_data)
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error_norm(actual, predicted, scaler))) * 100


def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    epsilon: float = 0.1,
    **kwargs
):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series

def dtw_tdi_1d(
    x: np.ndarray,
    y: np.ndarray,
    z_norm: bool = True,
    dist: str = "sq",
    window: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute DTW normalized distance and TDI for two 1D series.

    DTW_norm = (sum of local costs along optimal path) / K
    TDI = mean(|i - j| / T) along the optimal path, where T = max(N, M)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size < 1 or y.size < 1:
        raise ValueError("Input series must have at least one element.")

    if z_norm:
        x_std = x.std()
        y_std = y.std()
        x = (x - x.mean()) / (x_std if x_std > 0 else 1.0)
        y = (y - y.mean()) / (y_std if y_std > 0 else 1.0)

    if dist not in ("sq", "abs"):
        raise ValueError(f"Unsupported dist: {dist}")

    n, m = x.size, y.size
    if window is not None:
        window = int(window)
        if window < 0:
            raise ValueError("window must be >= 0.")
        window = max(window, abs(n - m))

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        if window is None:
            j_start, j_end = 1, m
        else:
            j_start = max(1, i - window)
            j_end = min(m, i + window)
            if j_start > j_end:
                continue
        xi = x[i - 1]
        for j in range(j_start, j_end + 1):
            diff = xi - y[j - 1]
            cost = diff * diff if dist == "sq" else abs(diff)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    i, j = n, m
    path_len = 0
    tdi_sum = 0.0
    t_scale = max(n, m)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            tdi_sum += abs((i - 1) - (j - 1)) / t_scale
        path_len += 1
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        diag = dp[i - 1, j - 1]
        up = dp[i - 1, j]
        left = dp[i, j - 1]
        if diag <= up and diag <= left:
            i -= 1
            j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1

    dtw_norm = float(dp[n, m] / path_len) if path_len > 0 else np.nan
    tdi = float(tdi_sum / path_len) if path_len > 0 else np.nan
    return dtw_norm, tdi


def compute_batch_dtw_tdi(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    z_norm: bool = True,
    dist: str = "sq",
    window: Optional[int] = None,
    downsample: int = 1,
    channel_reduce: str = "mean",
) -> dict:
    """
    Compute average DTW_norm and TDI over a batch.

    Supported shapes: [B, T], [B, T, C], and single-series [T, C]/[T].
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1.")
    if channel_reduce not in ("mean", "per_channel"):
        raise ValueError("channel_reduce must be 'mean' or 'per_channel'.")

    def _to_numpy(arr: np.ndarray) -> np.ndarray:
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr)

    y_hat = _to_numpy(y_hat)
    y = _to_numpy(y)

    def _standardize(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[None, :, None]
        if arr.ndim == 2:
            if arr.shape[1] == 1 and arr.shape[0] > 1:
                return arr[None, :, :]
            if arr.shape[0] >= arr.shape[1] and arr.shape[1] > 1:
                return arr[None, :, :]
            return arr[:, :, None]
        if arr.ndim == 3:
            return arr
        raise ValueError("Input must have shape [B,T], [B,T,C], [T,C], or [T].")

    y_hat = _standardize(y_hat)
    y = _standardize(y)

    if y_hat.shape[0] != y.shape[0]:
        raise ValueError("Batch sizes of y_hat and y must match.")
    if y_hat.shape[2] != y.shape[2]:
        raise ValueError("Channel sizes of y_hat and y must match.")

    if downsample > 1:
        y_hat = y_hat[:, ::downsample, :]
        y = y[:, ::downsample, :]

    dtw_vals = []
    tdi_vals = []
    batch_size, _, channels = y.shape
    for b in range(batch_size):
        if channel_reduce == "mean":
            x_series = y_hat[b].mean(axis=-1)
            y_series = y[b].mean(axis=-1)
            dtw_norm, tdi = dtw_tdi_1d(
                x_series, y_series, z_norm=z_norm, dist=dist, window=window
            )
        else:
            dtw_list = []
            tdi_list = []
            for c in range(channels):
                dtw_norm, tdi = dtw_tdi_1d(
                    y_hat[b, :, c],
                    y[b, :, c],
                    z_norm=z_norm,
                    dist=dist,
                    window=window,
                )
                dtw_list.append(dtw_norm)
                tdi_list.append(tdi)
            dtw_norm = float(np.mean(dtw_list))
            tdi = float(np.mean(tdi_list))
        dtw_vals.append(dtw_norm)
        tdi_vals.append(tdi)

    return {
        "DTW_norm": float(np.mean(dtw_vals)),
        "TDI": float(np.mean(tdi_vals)),
    }


def DTW_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    z_norm: bool = True,
    dist: str = "sq",
    window: Optional[int] = None,
    downsample: int = 1,
    channel_reduce: str = "mean",
    **kwargs,
):
    """DTW normalized distance metric (average cost along optimal path)."""
    metrics = compute_batch_dtw_tdi(
        predicted,
        actual,
        z_norm=z_norm,
        dist=dist,
        window=window,
        downsample=downsample,
        channel_reduce=channel_reduce,
    )
    return metrics["DTW_norm"]


def TDI(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    z_norm: bool = True,
    dist: str = "sq",
    window: Optional[int] = None,
    downsample: int = 1,
    channel_reduce: str = "mean",
    **kwargs,
):
    """Time Distortion Index metric derived from DTW optimal path."""
    metrics = compute_batch_dtw_tdi(
        predicted,
        actual,
        z_norm=z_norm,
        dist=dist,
        window=window,
        downsample=downsample,
        channel_reduce=channel_reduce,
    )
    return metrics["TDI"]
