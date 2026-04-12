import torch
import cv2
import numpy as np


class GeometryAware:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Load MiDaS model
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def decompose(self, img):
        # normalize
        img_rgb = img.astype(np.float32) / 255.0

        # prepare input
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            depth = self.model(input_batch)

        depth = depth.squeeze().cpu().numpy()

        # normalize depth
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        # treat depth as illumination proxy
        L_gray = depth

        # expand to RGB
        L = np.repeat(L_gray[:, :, None], 3, axis=2)

        # avoid division issues
        L_safe = np.clip(L, 1e-4, 1.0)

        # reflectance
        R = img_rgb / L_safe

        # normalized heatmap
        L_norm = L_gray

        return R, L, L_norm