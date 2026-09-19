# 3D Gaussian Splatting implementation

This directory contains my implementation of the 3D Gaussian Splatting (3DGS) training pipeline for my bachelor's thesis. The Python training code and notebook implement COLMAP loading, Gaussian parameterization, camera setup, training, adaptive density control, optimizer state updates, evaluation, and output generation. Rendering calls the CUDA rasterizer from the [original 3DGS project](https://github.com/graphdeco-inria/gaussian-splatting); the CUDA kernel itself is not my implementation.

The code here is for the 3DGS stage and validation on public scenes. It does not contain the entire toy car capture, segmentation, and structure-from-motion pipeline described in the thesis.

## Start here

| File | Purpose |
| --- | --- |
| `3dgs.py` | Main training and evaluation script. |
| `3dgs.ipynb` | Notebook version with editable configuration, plots, and visual comparisons. |
| `3DGS.pdf` | Paper for the 3DGS method. |
| `results/` | Thesis metrics, selected toy car renders, a seed 42 Gaussian model, and available videos from an earlier toy car run. |

## Setup and data

Use a CUDA-capable PyTorch environment compatible with the original rasterizer. The script imports `torch`, `numpy`, `tqdm`, `scikit-learn`, `imageio`, `Pillow`, `pytorch-msssim`, and `lpips`; the notebook also uses `matplotlib`. Install the [upstream CUDA rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization) with a working CUDA build toolchain. Dependency versions are not pinned here, so a fresh environment may need versions matched to its installed PyTorch and CUDA releases.

Run commands from this `3DGS` directory. The script expects a COLMAP reconstruction and images under:

```text
data/<scene>/
  images/                 # or images_2/, images_4/, images_8/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

The public [mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) provides the bonsai and counter validation scenes. Input data and training checkpoints are not included in this directory.

## Usage

The following are example commands to run after setting up the environment and data; they have not been executed as part of this documentation update.

```bash
python 3dgs.py bonsai --downsample 2
python 3dgs.py bicycle --downsample 4 --max-points 4000000
python 3dgs.py bonsai --eval-only results/gaussian_model_bonsai_30000.pth
```

The default run uses 30,000 iterations, holds out every eighth view for evaluation, and writes outputs under `results/`. Indoor examples use `images_2`; outdoor examples use `images_4`. The point cap is an optional memory limit and changes the unconstrained paper protocol. Use `python 3dgs.py --help` to inspect the script's other options when you are ready to run it.

For the notebook, select a CUDA-enabled kernel, edit the configuration cell (`SCENE`, `DOWNSAMPLE`, paths and output settings), and run the cells in order. The notebook is an explanatory companion; `3dgs.py` is the main entry point.

## Results

[`results/metrics.csv`](results/metrics.csv) records the PSNR values reported in Chapter 4 for bonsai and counter, plus the three-seed toy car test results. The public-scene values are 32.18 dB for bonsai and 29.06 dB for counter, compared with 31.98 dB and 28.70 dB respectively in the paper. Those two rows are thesis-reported metrics; no bonsai or counter video or checkpoint is available here.

The toy car result uses the thesis's eight-position masked dataset and three 30,000-iteration runs. Mean object-region test scores across seeds 42, 123 and 2026 are **25.829 ± 0.051 dB PSNR**, **0.8701 ± 0.0010 SSIM**, and **0.1763 ± 0.0006 LPIPS** (sample standard deviation). [`results/toycar/model/splat_30000_s42.ply`](results/toycar/model/splat_30000_s42.ply) is the final Gaussian model for seed 42. It is a PLY export, not a training checkpoint for the public-scene script.

The selected [toy car examples](results/toycar/examples/) are the best (`h033`), median (`d025`), and worst (`f001`) seed 42 test views by object-region PSNR. Each has a render, reference image (`gt`), mask, and error map. The selections and scores follow the thesis's Chapter 4 qualitative comparison. The toy car training pipeline and source data are separate from this public-scene script.

Two [toy car videos](results/toycar/previous_ring_run/) are available from an **earlier ring reconstruction**: `render.mp4` and `orbit_ring_superglue_8_ring_30000.mp4`. The accompanying `metrics.txt` belongs to that earlier run (60,650 Gaussians and 27.0927 dB object PSNR). These videos are illustrative and are not outputs of the final three-seed experiment summarized above.

## Scope of the claim

“Independent implementation” here refers to the Python training pipeline and its algorithmic control. It does not mean that the rendering kernel or the underlying 3DGS method was invented in this project. The method is due to Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023).
