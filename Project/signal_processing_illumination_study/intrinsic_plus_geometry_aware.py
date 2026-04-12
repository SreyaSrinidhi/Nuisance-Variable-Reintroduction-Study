import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- IMPORT YOUR EXISTING MODULES ----
from intrinsicnet import load_intrinsic_model, IntrinsicNetWrapper
from geometry_aware import GeometryAware


class IntrinsicGeometryFusion:
    def __init__(self):
        # Load models
        self.intrinsic_model = IntrinsicNetWrapper(load_intrinsic_model())
        self.geometry_model = GeometryAware()

    def decompose(self, img, alpha=0.5):
        img_rgb = img.astype(np.float32) / 255.0

        # ---- IntrinsicNet ----
        R_intr, S_intr, _ = self.intrinsic_model.decompose(img)

        # ---- Geometry ----
        _, S_geom, _ = self.geometry_model.geometry_aware_decompose(img)

        # Resize intrinsic outputs to match original image
        S_intr = cv2.resize(S_intr, (img.shape[1], img.shape[0]))
        R_intr = cv2.resize(R_intr, (img.shape[1], img.shape[0]))

        # ---- Convert shading to grayscale ----
        if S_intr.ndim == 3:
            S_intr_gray = np.mean(S_intr, axis=2)
        else:
            S_intr_gray = S_intr
        if S_geom.ndim == 3:
            S_geom_gray = np.mean(S_geom, axis=2)
        else:
            S_geom_gray = S_geom

        # ---- Normalize both ----
        S_intr_gray = (S_intr_gray - S_intr_gray.min()) / (S_intr_gray.max() - S_intr_gray.min() + 1e-6)
        S_geom_gray = (S_geom_gray - S_geom_gray.min()) / (S_geom_gray.max() - S_geom_gray.min() + 1e-6)

        # ---- Fuse ----
        S_final = alpha * S_geom_gray + (1 - alpha) * S_intr_gray
        # stabilize range (VERY IMPORTANT)
        S_final = S_final / (np.mean(S_final) + 1e-6)
        S_final = np.clip(S_final, 0.2, 2.0)

        # Expand to RGB
        S_final_rgb = np.repeat(S_final[:, :, None], 3, axis=2)

        # ---- Recompute reflectance ----
        R_final = img_rgb / (S_final_rgb + 1e-4)
        R_final = np.clip(R_final, 0, 1)

        return R_final, S_final_rgb, S_final


def visualize(img, R, L, L_norm):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Reflectance (Fused)")
    plt.imshow(R)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Illumination (Fused)")
    plt.imshow(L)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Illumination Heatmap")
    im = plt.imshow(L_norm, cmap='inferno')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = cv2.imread(r"D:\Sreya\\Case_Western\AdvanceMachineLearning\\csds440-f25-sks190\\Project\\nerf-pytorch\data\\fern\\images\\IMG_4027.JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fusion = IntrinsicGeometryFusion()

    R, L, L_norm = fusion.decompose(img, alpha=0.5)

    visualize(img, R, L, L_norm)