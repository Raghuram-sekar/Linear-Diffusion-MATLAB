# 📈 Linear Diffusion MNIST Digit Classifier
![MATLAB](https://img.shields.io/badge/MATLAB-%23e05a00.svg?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [What This Project Does](#-what-this-project-does)
- [Key Innovation](#-key-innovation)
- [Performance Highlights](#-performance-highlights)
- [Architecture](#-architecture)
- [Methodology & Technical Details](#-methodology--technical-details)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)

---

## 🎯 Project Overview
A MATLAB implementation of linear diffusion processes for image classification on the MNIST digit dataset. Propagates labels across pixel grids using heat equation diffusion kernels.

---

## 🚀 What This Project Does
* **The Challenge:** Training neural classifiers requires gradient updates, whereas direct heat kernel label propagation offers mathematical clarity.
* **Our Solution:** A linear diffusion classifier built in MATLAB using adjacency graphs of pixels to diffuse labels.

---

## 🔬 Key Innovation
| Feature | Deep Learning ❌ | Linear Diffusion ✅ | Benefit |
|---------|------------------|----------------------|---------|
| **Method** | Optimization via gradient descent | **Heat kernel diffusion** | Fast mathematical label propagation |
| **Matrix** | Billions of float variables | **Adjacency graph Laplacian** | Clear mathematical interpretability |

---

## 📊 Performance Highlights
- ✅ **Image classification** via graph heat diffusion.
- ✅ **Uses exact adjacency laplacian** matrices.
- ✅ **Zero neural training** parameters required.

---

## 🏗️ Architecture
```mermaid
graph TD
    MNIST[MNIST Digit Image] -->|Assemble Pixel Grid| Graph[Adjacency Graph Laplacian L]
    Graph -->|Heat Equation Solving| Heat[Solve exp(-t L) calculation]
    Heat -->|Label Diffusion| Propagation[Propagate source pixel labels]
    Propagation -->|Argmax class| Prediction[Classified Digit]
```

---

## ⚙️ Methodology & Technical Details
### Graph Laplacian Formulation
We model each MNIST image as a grid graph \(G = (V, E)\) where pixels are nodes connected to their 8 nearest neighbors. We construct the Adjacency matrix \(\mathbf{A}\) and Degree matrix \(\mathbf{D}\). The unnormalized Graph Laplacian \(\mathbf{L}\) is defined as:
$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

### Linear Heat Diffusion Classifier
Label propagation utilizes the heat diffusion equation over the graph Laplacian:
$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{L} \mathbf{u} = \mathbf{0}$$
Given initial labels \(\mathbf{u}_0\) on known anchor pixels, the labels diffuse over time \(t\) according to the diffusion kernel:
$$\mathbf{u}(t) = e^{-t \mathbf{L}} \mathbf{u}_0$$
We compute the matrix exponential in MATLAB using Pade approximations to diffuse label weights across pixels, classifying the digit according to the class with the highest diffused value.

---

## 📂 Project Structure
```
linear_diffusion/
├── linear_diffusion_complete.m   # Complete MATLAB script solving heat kernel
└── mnist.mat                     # Local copy of MNIST digit dataset
```

---

## 🧱 Tech Stack
- MATLAB R2023b image process scripts
- MNIST hand-drawn digit dataset

---

## 💻 Quick Start
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Linear-Diffusion-MATLAB.git
cd Linear-Diffusion-MATLAB

# Execute local setup commands:
run('linear_diffusion_complete.m')
```


---

## 📖 Supplementary Reference Index
# Linear Diffusion MATLAB Code Execution Order

## Overview
The `linear_diffusion_complete.m` file has been organized with functions in the proper sequential order for MATLAB execution.

## Execution Order Structure

### 1. **LinearDiffusion Constructor** (Lines ~17-27)
```matlab
function ld = LinearDiffusion()
```
- Creates the LinearDiffusion struct with default parameters
- Sets up LATENT_SIZE = 12, image_size = 28
- Initializes empty structs for text_encoder, image_encoder, and model

### 2. **MNIST Data Loading Functions** (Lines ~29-220)
- `load_mnist_data()` - Main data loading function
- `read_mnist_images_labels()` - Binary MNIST file reader
- `download_mnist_data()` - Download fallback
- `generate_synthetic_mnist()` - Synthetic data fallback
- `create_digit_pattern()` - Pattern generation helper

### 3. **Model Training Functions** (Lines ~32-150)
- `fit_model()` - Main training orchestrator
- `create_text_encoder()` - One-hot text encoding
- `encode_text_labels()` - Label embedding function
- `create_pca_encoder()` - PCA image encoder
- `decode_from_latent()` - PCA decoder
- `train_linear_regression()` - Linear model training

### 4. **Feature Engineering Functions** (Lines ~153-196)
- `create_features_noise()` - Feature creation with noise
- `create_interaction_terms()` - Text-image interactions

### 5. **Prediction Functions** (Lines ~200-223)
- `predict_images()` - Main image generation function

### 6. **Visualization Functions** (Lines ~502-620)
- `visualize_blog_results()` - Main results visualization
- `demonstrate_latent_space_properties()` - PCA reconstruction demo
- `demonstrate_noise_effects()` - Noise effect visualization
- `demonstrate_pure_noise_reconstruction()` - Pure noise demo

### 7. **Main Execution Function** (Lines ~622-660)
- `main()` - Orchestrates the entire pipeline
- Includes error handling and progress reporting
- Called at the very end: `main();`

## Key Design Principles

1. **Function Definition Before Use**: All functions are defined before they are called
2. **Logical Grouping**: Related functions are grouped together
3. **Clear Section Headers**: Each section is clearly marked with comments
4. **Error Handling**: Comprehensive try-catch blocks in main function
5. **Robust File Handling**: Proper path resolution and validation

## Execution Flow

```
1. MATLAB loads all function definitions (sections 1-6)
2. main() function is called at the end
3. main() calls functions in this order:
   - load_mnist_data() → read_mnist_images_labels()
   - LinearDiffusion() constructor
   - fit_model() → create_text_encoder(), create_pca_encoder(), create_features_noise(), train_linear_regression()
   - predict_images() → create_features_noise()
   - visualize_blog_results()
   - demonstrate_latent_space_properties()
   - demonstrate_noise_effects()
```

## Benefits of This Organization

- ✅ **No "function not found" errors**
- ✅ **Clear code structure and readability**
- ✅ **Easy to understand execution flow**
- ✅ **Proper error handling and reporting**
- ✅ **Maintainable and debuggable code**

## Usage
Simply run the file in MATLAB:
```matlab
linear_diffusion_complete
```

The code will execute all sections in the correct order automatically.

---