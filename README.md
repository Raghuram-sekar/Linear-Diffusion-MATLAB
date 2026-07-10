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
