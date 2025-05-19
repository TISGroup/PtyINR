# 🧪 Hyperparameter Tuning Guide for PtyINR

This guide outlines key hyperparameters used in `Parameters.py` and provides recommended adjustment ranges and tuning strategies.

---

## 1. 🧠 Object Neural Networks

| Parameter | Suggested Range | Notes |
|----------|------------------|-------|
| **first_omega** | `30 – 300` | Controls the extent of recovered object detail. A larger value enables finer details, especially with more diffraction patterns. |
| **lr_obj_amp** | `1e-5 – 1e-4` | Learning rate for the object's amplitude network. |
| **lr_obj_phase** | `1e-5 – 1e-4` | Learning rate for the object's phase network. Should be ≤ `lr_obj_amp`. |

---

## 2. 🔦 Probe Neural Networks

| Parameter | Suggested Range | Notes |
|----------|------------------|-------|
| **lr_probe_amp** | `1e-5 – 1e-4` | Learning rate for the probe's amplitude network. Should be ≤ learning rate of the object networks. |
| **lr_probe_phase** | `1e-5 – 1e-4` | Learning rate for the probe's phase network. Should be ≤ `lr_probe_amp`. |

---

## 3. 🎯 Training Loss Terms

| Parameter | Suggested Range | Notes |
|----------|------------------|-------|
| **diffraction_scale** | `400 – 2000` | Scales raw diffraction patterns. Use existing recovered probes as reference (e.g. divide by max amplitude). Helps stabilize network updates. |
| **beta_for_smoothl1** | `0 – 1` | Balances between L1 and MSE loss:<br>– Larger values → MSE-like (higher quality but may diverge)<br>– Smaller values → L1-like (more stable and faster convergence)<br>Start with `0.5` and adjust based on convergence behavior. |
| **regularized_loss_weight** | `0 – 1` | Regularizes probe shape at early training stages.<br>– Use non-zero value for circular probes<br>– Set to `0` for rectangular probes (e.g. in MLLS systems) |
| **regularized_steps** | `0 – 200` | Number of steps applying probe amplitude regularization.<br>– The total regularization effect is:<br>**`regularized_loss_weight × regularized_steps`**<br>– Reduce either if the probe diverges<br>– Increase if the probe fails to maintain focus |

> 💡 **Tip:** When unsure, start with default values in `Parameters.py` and adjust one parameter at a time. Use visual feedback from the probe/object reconstructions to guide tuning.

---

## 📌 Summary

- Use **lower learning rates** for phase networks compared to amplitude networks.
- Use **smoothly scaled diffraction** to avoid instability.
- Tune **beta_for_smoothl1** to balance quality and stability.
- Use **regularization** to control early-stage probe formation, especially for circular probes.

---

For more information, refer to the code in `Parameters.py` and example results in the notebooks.