# Project BIO486 - Physics Based model in MR Reconstruction

Magnetic Resonance Imaging (MRI) is a vital diagnostic tool used to capture detailed internal images of the human body. However, acquiring high-quality images can be time-consuming. To accelerate the process, certain lines in the k-space data are skipped beyond the Nyquist limit, which reduces scan time but introduces artifacts that degrade image quality.

This project explores a physics-based unrolled approach to reconstruct missing k-space data and improve image quality. It focuses on the case where only 25\% of the k-space lines are collected to speed up MRI scans. The method uses both the undersampled k-space data and the corresponding fully reconstructed images in a training loop that combines the physical principles of MRI with deep learning. The goal is to train a model that can convert the incomplete k-space data into a consistent, fully reconstructed image.

---

## 🔬 Goal of the project

This project combines both classical and deep learning approaches to leverage the strengths of each. A classical iterative algorithm is used to enforce consistency in the final reconstruction and reduce inconsistent noise, while a deep learning model enhances the visual quality of the reconstructed image. The project covers several key components: simulating undersampled data with a 4 times acceleration factor, implementing an iterative reconstruction algorithm, developing a convolutional neural network (CNN), and building a unified unrolling loop that integrates both approaches. Finally, the report presents the results of the reconstruction process.

### Mathematical formulation.

This is the way that a inverse problem is solve with L1 regularization.

f(x) = \min \| s - E x \|_2^2 + \lambda \| x \|_1

If we solve for x the equation above using a itertive solution, we got the following aproximation.

c_{k+1} = T_\lambda \left( (I - \alpha E^T E) c_k + \alpha E^T s \right)

If we replace T_\lambda in the equation above for a UNET. You got the unrolled physics base model.

c_{k+1} = T_\lambda \left( (I - \alpha E^T E) c_k + \alpha E^T s \right)

In addition, for training purpose we need to consider that the loss function is given by the following formula.

\min_{\theta} \sum_{i = 1}^{n_{\text{data}}} \left| U_{\theta}(\hat{m}_{\text{zf,us,cs}}^{(i)}) - m_{\text{gt}}^{(i)} \right|^2

Where:
- `zf`: ZeroFill (Zerofill reconstructed images)  
- `us`: UnderSampling (k-space data)
- `cs`: Coil Sensitivity image
- `gt`: ground true  
---


## 📦 Project Structure

| Module                      | Description                                                                          
|------------------------------|----------------------------------------------------------------------------------------|
| `Exploring_data'             | Show the exploration of the operations implementet in the physics module               |
| `Preprocesing_Unrolled`      | Show preprocesing steps to obtein 4 subfolders for zf, us, cs and gt                   |
| `Unrolled_model_final_Local' | Show the implementation of the physics base model in a small dataset step by step      |
| `Unrolled_model_final_Server`| Show a implementation the future implementation with larger data, but it is still under|       
                               | construction                                                                           | 
| `Utils2`                     | Show the final helpers of the netwok. You can find: Custom_dataset, Sampling functions,|
                               | Physics_module, deep_learning_module, and utils functions.                             |

---

## 📊 Metrics

While traditional metrics like:

- **MSE** – Average pixel-wise error  
- **SSIM** – Structural Similarity Index
  
are included, this project emphasizes **task-based** metrics like AUC derived from observer models, which are more aligned with clinical goals.

---

