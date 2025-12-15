# PED: Paper Explain With Documentation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat&logo=Jupyter)
![License](https://img.shields.io/badge/license-MIT-green)

Welcome to the official repository for **Paper Explain With Documentation (PED)**. This project hosts clean, educational implementations of cutting-edge AI papers, accompanied by deep-dive explanations on my Telegram channel.

📢 **Telegram Channel:** [Paper Explain With Documentation](https://t.me/paperExplainWithDocumentation)

> **Note:** Comprehensive video walkthroughs are primarily in **Farsi (Persian)**.

---

## 📚 Implemented Papers & Visuals

### 🧠 Reasoning
#### [Hierarchical Reasoning Model (HRM)](./HRM)
A novel architecture for complex reasoning and planning tasks, capable of solving Sudoku and ARC challenges.
<p align="center">
  <img src="./HRM/assets/hrm.png" width="800" alt="HRM Architecture">
</p>

#### [Grokking](./grokking)
Exploring generalization beyond overfitting on small algorithmic datasets.

---

### 🌟 Computer Vision & Graphics

#### [3D Gaussian Splatting (3DGS)](./3DGS)
Real-time radiance field rendering of 3D scenes using 3D Gaussians.
<p align="center">
  <img src="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/teaser.jpg" width="800" alt="3DGS Teaser">
</p>

#### [Neural Radiance Fields (NeRF)](./NeRF)
Representing scenes as neural radiance fields for novel view synthesis.
<p align="center">
  <img src="https://raw.githubusercontent.com/bmild/nerf/master/imgs/legogif.gif" width="600" alt="NeRF Lego GIF">
</p>

#### [ResNet (Residual Networks)](./ResNet)
Deep Residual Learning for Image Recognition - solving the vanishing gradient problem with skip connections.
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/ResNet_Block.svg/640px-ResNet_Block.svg.png" width="400" alt="ResNet Block">
</p>

#### [Inception (GoogLeNet)](./Inception)
"Going Deeper with Convolutions" - efficient deep learning architectures.
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Inception_module_Inc-v1.png/640px-Inception_module_Inc-v1.png" width="500" alt="Inception Module">
</p>

---

## 🛠️ Installation

```bash
git clone https://github.com/ali-hamedi/PED.git
cd PED
pip install torch torchvision numpy matplotlib jupyter
# For HRM specifics:
pip install -r HRM/requirements.txt
```

## 🚀 Usage

Navigate to any paper's directory and launch the Jupyter Notebook:

```bash
cd NeRF
jupyter lab nerf.ipynb
```

---

<div align="center">
  <sub>Created by Ali Hamedi</sub>
</div>
