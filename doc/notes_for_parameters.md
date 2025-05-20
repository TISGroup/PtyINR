# Hyperparameter Tuning Guide for PtyINR

This guide outlines key hyperparameters used in `parameters.py` and provides recommended adjustment ranges and tuning strategies.

---

## 1. Object Neural Networks

| Parameter | Suggested Range | Notes |
|----------|------------------|-------|
| **first_omega** | `30 – 300` | Controls the extent of recovered object detail. A larger value enables finer details, especially with more diffraction patterns. However, when data is insufficient, setting larger values will result in artifacts. |
| **LR** | `1e-5 – 1e-4` | Learning rate for the object's amplitude network. |
| **LR2** | `1e-5 – 1e-4` | Learning rate for the object's phase network. Should be ≤ `LR`. |

---

## 2. Probe Neural Networks

| Parameter | Suggested Range | Notes |
|----------|------------------|-------|
| **LR3** | `1e-5 – 1e-4` | Learning rate for the probe's amplitude network. Should be ≤ learning rate of the object networks. |
| **LR4** | `1e-5 – 1e-4` | Learning rate for the probe's phase network. Should be ≤ `LR3`. |

---

## 3. Training Loss Terms

| Parameter | Suggested Range | Notes |
|----------|------------------|-------|
| **diffraction_scale** | `400 – 2000` | Scales the raw diffraction patterns, which in turn normalizes the recovered object and probe amplitudes to a range between 0 and 1, thereby improving numerical stability and facilitating more consistent network updates. |
| **beta_for_smoothl1** | `0 – 1` | Balances between L1 and MSE loss:<br>– Larger values → MSE-like (higher quality but may diverge)<br>– Smaller values → L1-like (more stable and faster convergence but possible sub-optimal quality)<br>Start with `1e-3` and adjust based on convergence behavior. |
| **regularized_loss_weight** | `0 – 1` | Regularizes probe shape to be focused at early training stages.<br>– Use non-zero value for circular probes<br>– Set to `0` for rectangular probes (e.g. in MLLS systems) |
| **regularized_steps** | `0 – 200` | Number of steps applying probe amplitude regularization.<br>– The total regularization effect is:<br>**`regularized_loss_weight × regularized_steps`**<br>– Increase either if the probe diverges<br>– Consider decreasing this value if the probe retains a concentrated shape—such as a single bright pixel—even after the number of training steps exceeds the regularized step count, indicating that the amplitude may have been over-regularized.|

> **Tip:** When unsure, start with default values in `parameters.py` and adjust one parameter at a time. Use visual feedback from the probe/object reconstructions to guide tuning.

---

For more information, refer to the code in `parameters.py`.
