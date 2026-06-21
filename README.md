# Shift-Aware DBLoss — A Differentiable Shift-Alignment Extension of DBLoss

> **⚠️ Attribution first.** This repository is **an extension built on top of the official DBLoss codebase** (NeurIPS'25, *DBLoss: Decomposition-based Loss Function for Time Series Forecasting*, **Qiu et al., ECNU**). The original **DBLoss** method, the `DECOMP`/`EMA` decomposition blocks, and the underlying **TFB** benchmark framework are **not my work** — full credit to the original authors (see [Acknowledgements & Citation](#acknowledgements--citation)).
>
> **My contribution** is the `DBLossWithShift` extension and its supporting shift-loss functions, described in [What I Added](#what-i-added). The original project README is preserved at [`ts_benchmark/readme`](ts_benchmark/readme).

📎 **Repo:** _link to be added_

---

## TL;DR

Point-wise losses like MSE penalize a forecast value-by-value at each timestep, so a prediction with the **right shape but the wrong phase** (shifted slightly earlier/later in time) is punished as harshly as a structurally wrong one. This extension adds a **differentiable shift-alignment loss** that explicitly tolerates small time-axis offsets, on top of DBLoss's seasonal/trend decomposition loss.

```
total_loss = DBLoss(pred, target)  +  λ_shift · L_shift(pred, target)
```

`L_shift` is a **soft-min over discrete time shifts**, so the "best shift" is selected in a fully differentiable way and the model can be trained end-to-end.

---

## What I Added

All additions live in [`ts_benchmark/baselines/utils.py`](ts_benchmark/baselines/utils.py) and are wired into the trainer in [`ts_benchmark/baselines/deep_forecasting_model_base.py`](ts_benchmark/baselines/deep_forecasting_model_base.py).

| Component | Location | Description |
|---|---|---|
| `DBLossWithShift` | `utils.py` | Wrapper combining the original `DBLoss` with a shift-alignment term: `total = DBLoss + λ_shift · L_shift`. Optional `return_logs=True` exposes `L_DB`, `L_shift`, and the estimated shift per window. |
| `compute_windowed_shift_loss_softmin` | `utils.py` | The shift loss actually used in training. Splits the horizon into windows and estimates a **local** shift per window via soft-min, then averages — capturing piecewise phase drift. |
| `compute_global_shift_loss_softmin` | `utils.py` | A single-shift (global) variant. **Implemented but not used by `DBLossWithShift`** (windowed is the default); kept for ablation/comparison. |
| Loss selection + hyperparameters | `deep_forecasting_model_base.py` | `loss="DBLossWithShift"` branch + hyperparameters plumbed through the trainer. |

### How the shift loss works

For shift candidates `δ ∈ [−k, k]`, the overlap error `e(δ)` (MSE or MAE) between `pred` and the `δ`-shifted `target` is computed. Instead of a non-differentiable `min`, a temperature-scaled **soft-min** via `logsumexp` is used:

```
L_shift = − (1/τ) · logsumexp( −τ · e(δ) )   over δ ∈ [−k, k]
```

- As `τ → ∞`, `L_shift` approaches the hard minimum (the single best shift).
- The soft weights `softmax(−τ·e(δ))` also yield a differentiable **estimated shift** (`delta_soft`) for inspection.
- The **windowed** version applies this per window of length `window_size` and averages, so different parts of the horizon can align at different offsets.

---

## Usage

This follows the original DBLoss / TFB workflow — you just select the new loss via hyperparameters.

```shell
# (recommended) Python 3.8
pip install -r requirements.txt

# Place the preprocessed datasets under ./dataset  (see original instructions in ts_benchmark/readme)
```

Select `DBLossWithShift` in any experiment by passing it in `--model-hyper-params`:

```shell
python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "ETTh1.csv" \
  --strategy-args '{"horizon": 96}' \
  --adapter "transformer_adapter" \
  --model-name "time_series_library.PatchTST" \
  --model-hyper-params '{"loss": "DBLossWithShift", "lambda_shift": 0.03, "shift_k": 5, "shift_mode": "mse", "shift_tau": 1.0, "shift_window_size": 96, "seq_len": 96, "horizon": 96}' \
  --gpus 0 --num-workers 1 --save-path "ETTh1/PatchTST"
```

### Shift-loss hyperparameters

| Param | Meaning |
|---|---|
| `lambda_shift` | Weight of the shift term in the total loss. |
| `shift_k` | Max absolute shift candidate (δ ∈ [−k, k]); clamped to horizon−1. |
| `shift_window_size` | Window length for piecewise (windowed) shift estimation. |
| `shift_tau` | Soft-min temperature (larger ⇒ closer to a hard min). |
| `shift_mode` | Per-shift error metric: `"mse"` or `"mae"`. |

Ready-to-run scripts are under [`scripts/multivariate_forecast/`](scripts/multivariate_forecast/), covering backbones **PatchTST, DLinear, iTransformer, Amplifier** on **ETT, Electricity, Solar, Weather** (with per-horizon `lambda_shift`, mainly for horizons 96/192/336).

---

## Status

- ✅ Loss implemented and integrated into the training pipeline (end-to-end differentiable, verified by code).
- ✅ Experiment scripts prepared across multiple models and datasets.
- ⏳ **Benchmark results are not included in this snapshot** — the `result/` directory is empty, so **no MSE/MAE numbers are reported here.** Any performance claims should come only after running the scripts above.

> The figure `docs/figures/exp.png` is the **original DBLoss paper's** results table (not measurements from this extension).

---

## Acknowledgements & Citation

This work would not exist without the original DBLoss authors and the TFB benchmark.

- **DBLoss** — *DBLoss: Decomposition-based Loss Function for Time Series Forecasting*, NeurIPS 2025. ([arXiv:2510.14510](https://arxiv.org/pdf/2510.14510))
- **TFB** — Time Series Forecasting Benchmark, decisionintelligence/TFB.

```bibtex
@inproceedings{qiu2025DBLoss,
  title     = {DBLoss: Decomposition-based Loss Function for Time Series Forecasting},
  author    = {Xiangfei Qiu and Xingjian Wu and Hanyin Cheng and Xvyuan Liu and Chenjuan Guo and Jilin Hu and Bin Yang},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

If you use the **shift-alignment extension** specifically, please also link back to this repository.
