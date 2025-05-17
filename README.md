# 🧠 Project BIO486 – Physics-Based Model in MR Reconstruction

Magnetic Resonance Imaging (MRI) is a vital diagnostic tool used to capture detailed internal images of the human body. However, acquiring high-quality images can be time-consuming. To accelerate the process, certain lines in the k-space data are skipped beyond the Nyquist limit, reducing scan time but introducing artifacts that degrade image quality.

This project explores a **physics-based unrolled approach** to reconstruct missing k-space data and improve image quality. It focuses on the case where only **25% of k-space lines** are collected to accelerate MRI scans. The method combines **undersampled k-space data** with **fully reconstructed images** in a training loop that integrates **MRI physics** with **deep learning**. The goal is to learn a model capable of transforming incomplete k-space data into a consistent, high-quality image.

---

## 🔬 Goal of the Project

This project combines **classical** and **deep learning** techniques to leverage the strengths of each:

- A **classical iterative algorithm** enforces consistency and reduces noise.
- A **CNN-based model** enhances the visual quality of the reconstruction.

### Key Components:
- Simulate undersampled MRI data (4× acceleration)
- Implement an iterative reconstruction algorithm
- Develop a CNN model
- Build a unified unrolling loop that integrates physics and learning
- Evaluate reconstruction performance

---

## 🧮 Mathematical Formulation

This project models MRI reconstruction as an inverse problem with L1 regularization:

$$
f(x) = \min \| s - E x \|_2^2 + \lambda \| x \|_1
$$

An iterative approximation of the solution is:

$$
c_{k+1} = T_\lambda \left( (I - \alpha E^T E) c_k + \alpha E^T s \right)
$$

If we replace \( T_\lambda \) with a U-Net, we get the **unrolled physics-based model**:

$$
c_{k+1} = \text{UNet}_\theta \left( (I - \alpha E^T E) c_k + \alpha E^T s \right)
$$

The training loss function is defined as:

$$
\min_{\theta} \sum_{i = 1}^{n_{\text{data}}} \left\| U_{\theta}(\hat{m}_{\text{zf,us,cs}}^{(i)}) - m_{\text{gt}}^{(i)} \right\|_2^2
$$

Where:

- `zf` – Zero-filled reconstruction  
- `us` – Undersampled k-space data  
- `cs` – Coil sensitivity map  
- `gt` – Ground truth image  

---

## 📦 Project Structure

| Folder / Module                | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `Exploring_data`              | Demonstrates operations implemented in the physics module                   |
| `Preprocesing_Unrolled`       | Preprocessing steps to generate ZF, US, CS, and GT subfolders               |
| `Unrolled_model_final_Local`  | Step-by-step physics-based model on a small dataset                         |
| `Unrolled_model_final_Server` | In-progress: Full-scale implementation for larger datasets                  |
| `Utils2`                      | Helper functions: dataset loader, sampling, physics/deep learning modules   |

---

## 📊 Evaluation Metrics

While traditional metrics are included:

- **MSE** – Mean Squared Error  
- **SSIM** – Structural Similarity Index  

---

## 📁 Dataset Information

- Data simulated using 4× undersampling with Cartesian masks from FastMRI dataset
- Coil sensitivity estimated were simulate with spigy.
- Ground truth images included for supervised training

---

## 🚀 Future Work

- Improve generalization to other anatomies
- Add support for non-Cartesian sampling
- Incorporate real clinical datasets
