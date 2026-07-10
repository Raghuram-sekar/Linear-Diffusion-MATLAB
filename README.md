# 📈 Linear Diffusion MNIST Digit Classifier
![MATLAB](https://img.shields.io/badge/MATLAB-%23e05a00.svg?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#🎯-project-overview)
- [What This Project Does](#🚀-what-this-project-does)
- [Key Innovation](#🔬-key-innovation)
- [Performance Highlights](#📊-performance-highlights)
- [Architecture](#🏗️-architecture)
- [Tech Stack](#🧱-tech-stack)
- [Quick Start](#💻-quick-start)

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
```\n[Core Architectural Components & Datastore Framework]\n```

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
% Run MATLAB script:
run('linear_diffusion_complete.m')
```
