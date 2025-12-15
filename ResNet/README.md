# ResNet Implementation & Analysis (PED)

This project is part of the **Paper Explanation with Documentation (PED)** series. It focuses on understanding and implementing Deep Residual Learning (ResNet), specifically analyzing its performance on the CIFAR-10 dataset.

## Project Overview

The core goal of this project is to explore the "degradation problem" in deep neural networks and how residual connections solve it. We experiment with ResNet architectures to verify the hypothesis that it is easier to optimize a residual mapping than an original, unreferenced mapping.

### Key Concepts Covered
- **Datasets:** Analysis of ImageNet, PASCAL VOC, COCO, and CIFAR-10.
- **The Degradation Problem:** Why deeper networks doesn't always equate to better accuracy (accuracy saturation and degradation).
- **Residual Block:** $H(x) = F(x) + x$. Learning the residual $F(x)$ instead of the underlying mapping $H(x)$.
- **Identity Mapping:** The concept that if identity were optimal, pushing residual to zero is easier for the network.

## File Structure

- **`resnet_cifar10.ipynb` / `resnet_cifar10-2.ipynb`**: Jupyter notebooks containing the implementation of ResNet on the CIFAR-10 dataset. Includes model definition, training loops, and evaluation.
- **`2016a.txt`**: Rough notes and theoretical background on datasets (ImageNet, VOC, COCO) and ResNet mechanics (degradation, residual connections).
- **`degradation_plot.png`**: Visualization illustrating the degradation problem (training error vs. number of layers).
- **`resnet_results.png`**: Results obtained from the experiments.

## Dataset Details (CIFAR-10)

The primary dataset used for experimentation is **CIFAR-10**:
- **Images:** 60,000 (32x32 color images)
- **Classes:** 10
- **Split:** 50,000 training (45k train, 5k val) / 10,000 test

## Getting Started

1.  Clone this repository.
2.  Ensure you have the necessary dependencies installed (PyTorch/TensorFlow, Matplotlib, NumPy, Jupyter).
3.  Run the notebooks to reproduce the training and evaluation results.

---
*This repository is maintained as part of the PED Telegram channel resources.*
