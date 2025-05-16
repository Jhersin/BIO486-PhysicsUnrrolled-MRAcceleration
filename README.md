# Project BIO580 - Binary Detectability in MR Reconstruction

Magnetic Resonance Imaging (MRI) provides detailed anatomical images, but acquiring high-quality images can be time-consuming. Acceleration techniques that skip k-space lines are commonly used to reduce acquisition time, but this leads to image artifacts and loss of diagnostic information.

This project explores binary detectability in undersampled MRI reconstructions using Hotelling Observer (HO) and Channelized Hotelling Observer (CHO) models. The focus is to evaluate how acceleration, noise, and lesion characteristics affect the ability to detect small structures in reconstructed images—going beyond traditional metrics such as SSIM and MSE.

---

## 🔬 Problem Setup

We simulate elliptical lesions embedded in background patches to model signal-present and signal-absent conditions under varying intensity of the lesion, noise and acceleration levels.

### Hypotheses

- **Signal Present (H₁):**
g₁ = H(f_s + f_b) + n
- **Signal Absent (H₂):**
g₂ = H(f_b) + n

Where:
- `f_s`: signal (elliptical lesion)  
- `f_b`: background  (images)
- `n`: additive noise  (gaussina correlate noise)
- `H`: reconstruction operator (reconstruction)  

---

## 🧠 Hotelling Observer

The Hotelling observer template is defined as: W_hot = [0.5 * (K₁ + K₂)]⁻¹ · (E[g₁] - E[g₂])

- `K₁`, `K₂`: Covariance matrices under hypotheses H₁ and H₂  
- `E[g₁]`, `E[g₂]`: Mean vectors of signal-present and signal-absent images

The test statistic for a given patch `g` is: t_hot = W_hotᵀ · g

This allows us to compute the **AUC** and evaluate detectability performance across varying conditions.

---

## 📦 Project Structure

| Module                      | Description                                                                          |
|----------------------------|---------------------------------------------------------------------------------------|
| `HO_save_preprocesing'     | Simulates signal-present/absent/reference patches with varying noise and acceleration |
| `CHO_HO_Calculation`       | Calculate AUC for hotelling and chanelizing hoteling observer                         |
| `CHO_HO_Channels_effect'   | Visualize the effect of moving parameters in the gambor chaneel for CHO               |
| `HO_exploration`           | Explore the lesion formation and different condition in the parameters                |
| `Utils`                    | Provid useful function for the process                                                |

---

## 📊 Metrics

While traditional metrics like:

- **MSE** – Average pixel-wise error  
- **SSIM** – Structural Similarity Index
- **AUC_HO** - AUC for hotelling observer
- **AUC_CHO** - AUC for hotelling observer
  
are included, this project emphasizes **task-based** metrics like AUC derived from observer models, which are more aligned with clinical goals.

---

