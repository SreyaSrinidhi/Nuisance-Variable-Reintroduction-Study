# Dual-SAL Decoder for Nuisance Reintroduction  
**Author:** Sreya Kadagattur Srinidhi  
**Course:** CSDS 440 — Advanced Machine Learning  
**Project:** Minimal Representations and the Dual of SAL

---

# Overview

This repository contains the full implementation of:

1. **SAL Encoders** for viewpoint, brightness, and contrast  
2. **A dual-SAL decoder** that reintroduces viewpoint into a minimal representation  
3. **Experiments** demonstrating the hierarchy of nuisances and the feasibility of nuisance reintroduction  

The project is structured as follows:

- `Project/sks190_Project_Report.md` — Project report containing background theory, methodology and results
- `Project/nerf-pytorch/` — pytorch version of nerf that was used for baseline testing of nuisance variables. This also contains the encodre, decoder, data, datasets and experiments
- `Project/README.md` (this file) — instructions to run code and reproduce results  

---

# Repository Structure

```
Project/
│
├── sks190_Project_Report.md      # Full project paper
├── README.md                     # This README
│
├── nerf-pytorch/
│   ├── decoders/                 # Baseline, 128-channel, and ResNet decoders
│   ├── datasets/                 # Fern + multi-scene datasets
│   |   ├── utils/                    # Pose loading, image utilities
│   ├── encoder.py                # SAL encoder for viewpoint, brightness, contrast, training/eval
│   ├── decoder.py                # Dual-SAL decoder training/eval
│   └── requirements.txt          # Python dependencies
│
└── embedding_files/              # Saved embeddings from SAL encoder
    ├── embeddings_fern_view.pt   # Minimal embeddings used for decoder experiments
    └── ... (other embeddings)
```

---

# 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n nerf python=3.10 -y
conda activate nerf
```

### 1.2 Install Requirements

```bash
pip install -r Project/code/requirements.txt
```

The key dependencies include:

- PyTorch  
- TorchVision  
- NumPy  
- Matplotlib  
- Pillow  
- TQDM  

CUDA is optional but strongly recommended.

---

# 2. Dataset Setup (LLFF / Fern)

The decoders use images and poses from the **LLFF Fern** dataset.

Download LLFF data:

```bash
cd Project
mkdir -p nerf/data
```

Place Fern folder here:

```
Project/nerf/data/nerf_llff_data/fern/
    ├── images/                # image set
    ├── poses_bounds.npy       # camera poses
    └── ...
```

If using multiple LLFF scenes:

```
Project/nerf/data/nerf_llff_data/<scene_name>/
```

---

# 3. Running the SAL Encoder (optional for reproduction)

To generate minimal embeddings:

```bash
python code/encoder.py \
  --mode view \
  --image_path nerf/data/nerf_llff_data/fern/images \
  --nuisance_path nerf/data/nerf_llff_data/fern/poses_bounds.npy \
  --save_path embedding_files/embeddings_fern_view.pt
```

This produces a file:

```
embedding_files/embeddings_fern_view.pt
```

containing:

- `embeddings`: minimal z vectors  
- `paths`: image paths  

This is used directly by the decoder.

---

# 4. Running the Dual-SAL Decoder

The decoder reconstructs:

\[
(z, g) \mapsto \hat{y}_g
\]

where:

- `z` is the scene-level minimal SAL representation  
- `g` is the viewpoint vector (12-D camera extrinsic)  

---

## 4.1 Train the Baseline Decoder

```bash
python code/decoder.py \
  --mode view \
  --decoder_arch baseline \
  --embeddings_path embedding_files/embeddings_fern_view.pt \
  --nuisance_path nerf/data/nerf_llff_data/fern/poses_bounds.npy \
  --image_path nerf/data/nerf_llff_data/fern/images \
  --epochs 20 \
  --save_path decoders/trained_decoders/baseline_decoder_20epochs.pth
```

## 4.2 Train the 128-Channel Decoder

```bash
python code/decoder.py \
  --mode view \
  --decoder_arch 128 \
  --embeddings_path embedding_files/embeddings_fern_view.pt \
  --nuisance_path nerf/data/nerf_llff_data/fern/poses_bounds.npy \
  --image_path nerf/data/nerf_llff_data/fern/images \
  --epochs 20 \
  --save_path decoders/trained_decoders/basech128_decoder_20epochs.pth
```

## 4.3 Train the ResNet Decoder

```bash
python code/decoder.py \
  --mode view \
  --decoder_arch resnet \
  --embeddings_path embedding_files/embeddings_fern_view.pt \
  --nuisance_path nerf/data/nerf_llff_data/fern/poses_bounds.npy \
  --image_path nerf/data/nerf_llff_data/fern/images \
  --epochs 20 \
  --save_path decoders/trained_decoders/resnet64_decoder_20epochs.pth
```

---

# 5. Evaluating a Trained Decoder

Example:

```bash
python code/decoder.py \
  --eval_only \
  --mode view \
  --decoder_arch resnet \
  --embeddings_path embedding_files/embeddings_fern_view.pt \
  --nuisance_path nerf/data/nerf_llff_data/fern/poses_bounds.npy \
  --image_path nerf/data/nerf_llff_data/fern/images \
  --eval_pth_file decoders/trained_decoders/resnet64_decoder_20epochs.pth
```

During evaluation, the script prints:

- L1(recon_i, GT_i)  
- L1(recon_i, GT_j)  
- L1(swap, GT_j)  
- L1(swap, GT_i)

And renders:

- GT(i)  
- GT(j)  
- Reconstruction  
- Swap reconstruction  

These metrics demonstrate whether the decoder correctly responds to nuisance variable g.

---

# 6. Reproducing Paper Results

To reproduce the exact results reported in the project writeup:

1. Use **embeddings_fern_view.pt**  
2. Train all 3 decoders for 20 epochs  
3. Run the evaluation commands above  
4. Observe:

   - `L1(swap,GT_j) < L1(swap,GT_i)` for all models  
   - ResNet has the strongest viewpoint response  

These correspond to the quantitative experiments in Section 3.3 of the report.

---

# 7. Troubleshooting

### Missing CUDA
If PyTorch cannot find CUDA:

```bash
pip install torch==2.1.0+cpu torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### Dataset path errors
Ensure Fern directory structure matches:

```
nerf/data/nerf_llff_data/fern/images
nerf/data/nerf_llff_data/fern/poses_bounds.npy
```

### Embedding mismatch
Embeddings must correspond to:

- SAME scene  
- SAME image ordering  
- SAME SAL encoder mode (view / brightness / contrast)

---

# 8. Credits

This project uses:

- PyTorch  
- torchvision  
- LLFF dataset  
- NeRF-Pytorch utilities (poses)  
- ChatGPT assistance for editing, formatting, and clarity (per course policy)

All custom SAL algorithms, decoders, and nuisance-injection logic were written by me.

---

