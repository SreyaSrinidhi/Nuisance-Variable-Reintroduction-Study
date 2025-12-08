# Dual-SAL Decoder for Nuisance Reintroduction  
**Author:** Sreya Kadagattur Srinidhi  
**Course:** CSDS 440 — Advanced Machine Learning  
**Project:** Minimal Representations and the Dual of SAL

---

# Overview

This repository implements:

1. **SAL Encoders** for viewpoint, brightness, and contrast  
2. **A Dual-SAL Decoder** that reintroduces nuisance variables removed by SAL  
3. **Experiments** demonstrating nuisance hierarchy, invariance structure, and successful viewpoint reintroduction  

Everything needed to reproduce the experiments in the project report is included.

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
│   │   ├── utils/                # Pose loading, image utilities
│   ├── encoder.py                # SAL encoder (view, brightness, contrast)
│   ├── decoder.py                # Dual-SAL decoder training/eval
│   └── requirements.txt          # All Python dependencies
    └── embedding_files/              # Saved SAL embeddings
        ├── embeddings_fern_view.pt   # Minimal embeddings used for decoder training
        ├── embeddings_sal.pt         # All-scene embeddings (older)
        └── additional embeddings...
```

---

# 1. Environment Setup

## 1.1 Create Conda Environment
```bash
conda create -n nerf python=3.10 -y
conda activate nerf
```

## 1.2 Install Dependencies
```bash
pip install -r nerf-pytorch/requirements.txt
```

Key dependencies:

- PyTorch  
- TorchVision  
- NumPy  
- Matplotlib  
- Pillow  
- TQDM  

CUDA strongly recommended.

---

# 2. Dataset Setup (LLFF / Fern)

This project uses the LLFF dataset (Fern scene + others).

Download LLFF from [here](https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full)

Extract it into:

```
Project/nerf-pytorch/data/
    ├── fern/
    │   ├── images/
    │   └── poses_bounds.npy
    ├── flower/
    ├── fortress/
    ├── horns/
    ├── leaves/
    ├── orchids/
    ├── room/
    └── trex/
```

---

# 3. Running the SAL Encoder  
The SAL encoder produces minimal embeddings that remove viewpoint, brightness, or contrast variation.

Run encoder through:

```
nerf-pytorch/encoder.py
```

## 3.1 Encoder Arguments

### **Required**
| Argument | Description |
|----------|-------------|
| `--dataset` | Path to a single image OR a folder containing images. |

### **Optional**
| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--mode` | `view` | `view`, `brightness`, `contrast` | Which nuisance to marginalize out. |
| `--output_path` | `embeddings_sal.pt` | any path | Where to save the `.pt` file of embeddings. |

### Nuisance Modes Explained

| Mode | Description | What SAL Removes |
|------|-------------|------------------|
| `view` | Samples camera viewpoints | Geometric differences between poses |
| `brightness` | Samples brightness scaling | Global illumination changes |
| `contrast` | Samples contrast curves | Photometric tone changes |

The output file (`output_path`) contains:

```
{
  "embeddings": Tensor (N, 128),
  "paths": List[str]
}
```

## 3.2 Example Commands

### Viewpoint-invariant embeddings (used for decoder)
```bash
cd nerf-pytorch
python encoder.py \
    --dataset data/fern/images \
    --mode view \
    --output_path embedding_files/embeddings_fern_view.pt
```

### Brightness-invariant embeddings
```bash
python encoder.py \
    --dataset data/fern/images \
    --mode brightness \
    --output_path embedding_files/embeddings_fern_brightness.pt
```

### Contrast-invariant embeddings
```bash
python encoder.py \
    --dataset data/fern/images \
    --mode contrast \
    --output_path embedding_files/embeddings_fern_contrast.pt
```

---

# 4. Running the Dual-SAL Decoder

The decoder solves:

\[
(z_{\text{scene}}, g) \mapsto \hat{y}_g,
\]

reintroducing a nuisance variable \(g\) that SAL removed.

Use:

```
nerf-pytorch/decoder.py
```

---

# 4.1 Decoder Arguments (Full Documentation)

### **Required**
| Argument | Description |
|----------|-------------|
| `--embeddings_path` | Path to encoder-generated `.pt` embedding file. |

### **Dataset / Nuisance Parameters**
| Argument | Default | Description |
|----------|---------|-------------|
| `--nuisance_path` | `data/fern/poses_bounds.npy` | Nuisance values (camera poses, brightness scalars, etc.). |
| `--image_path` | `data/fern/images` | Ground-truth target images. |
| `--dataset_name` | `fern` | `fern` (1 scene) or `multiscene` (8 LLFF scenes). |
| `--mode` | `view` | Which nuisance to reintroduce (`view`, `brightness`, `contrast`). |

### **Model Architecture**
| Argument | Default | Options | Meaning |
|----------|---------|---------|---------|
| `--decoder_arch` | `128` | `baseline`, `128`, `resnet` | Choose decoder capacity. |
| `--img_size` | `64` | int | Output resolution. |

### **Training Parameters**
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Training epochs. |
| `--batch_size` | 8 | Batch size. |
| `--lr` | `1e-4` | Learning rate (Adam). |
| `--save_path` | `decoder_fern.pth` | Where to save trained model. |

### **Evaluation**
| Argument | Default | Meaning |
|----------|---------|--------|
| `--eval_only` | False | Skip training and only evaluate. |
| `--eval_pth_file` | `decoder_fern.pth` | Trained model to evaluate. |

---

# 4.2 Train the Decoder (Viewpoint Example)

```bash
cd nerf-pytorch
python decoder.py \
    --mode view \
    --decoder_arch baseline \
    --embeddings_path embedding_files/embeddings_fern_view.pt \
    --nuisance_path data/fern/poses_bounds.npy \
    --image_path data/fern/images \
    --epochs 20 \
    --save_path decoders/trained_decoders/baseline_decoder_20epochs.pth
```

This:

- loads minimal SAL embeddings  
- loads ground-truth camera poses (nuisance g)  
- trains the decoder to reconstruct images for each pose  

---

# 5. Evaluating a Decoder

```bash
python decoder.py \
    --eval_only \
    --mode view \
    --decoder_arch baseline \
    --embeddings_path embedding_files/embeddings_fern_view.pt \
    --nuisance_path data/fern/poses_bounds.npy \
    --image_path data/fern/images \
    --eval_pth_file decoders/trained_decoders/baseline_decoder_20epochs.pth
```

### Evaluation outputs:

**Printed metrics:**

- L1($recon_i$, $GT_i$)  
- L1($recon_i$, $GT_j$)  
- L1(swap, $GT_j$)  
- L1(swap, $GT_i$)

Correct behavior requires:

$$
L1(\text{swap}, GT_j) < L1(\text{swap}, GT_i).
$$

**Displayed images:**

- Ground truth at $g_i$  
- Ground truth at $g_j$ 
- Decoder reconstruction  
- Decoder viewpoint-swapped reconstruction  

---

# 6. Reproducing Results in the Report

To reproduce all results in the paper:

1. Compute scene-level viewpoint-invariant embeddings using:  
   ```
   embeddings_fern_view.pt
   ```
2. Train all three decoders for 20 epochs:  
   ```
   --decoder_arch baseline
   --decoder_arch 128
   --decoder_arch resnet
   ```
3. Evaluate using commands in Section 5  
4. Verify viewpoint behavior:  
   ```
   L1(swap, GT_j) < L1(swap, GT_i)
   ```
5. Compare ResNet vs baseline  
6. Observe SAL minimality causing low-frequency reconstructions  

---

# 7. Exploring the System (Modes + Architectures)

### Nuisance types:
```
--mode view
--mode brightness
--mode contrast
```

### Decoder architectures:
```
--decoder_arch baseline
--decoder_arch 128
--decoder_arch resnet
```

### Hyperparameters:
```
--epochs
--batch_size
--lr
```

### Output resolution:
```
--img_size 64
--img_size 128
```

### Multiple scenes:
```
--dataset_name multiscene
```

---

# 8. Credits

- LLFF dataset  
- NeRF-Pytorch utilities (pose loading, image handling)  
- PyTorch  
- ChatGPT assistance for formatting and clarity (all algorithmic work is original)

---

