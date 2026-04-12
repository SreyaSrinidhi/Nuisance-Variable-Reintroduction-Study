import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


class GeometryAware:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def geometry_aware_decompose(self, img):
        img_rgb = img.astype(np.float32) / 255.0
        h, w = img_rgb.shape[:2]

        # ---- Depth ----
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            depth = self.model(input_batch)

        depth = depth.squeeze().cpu().numpy()

        # Resize to match original image
        depth = cv2.resize(depth, (w, h))

        # ---- IMPORTANT: do NOT normalize before gradients ----

        # sharpen depth (boost edges)
        depth_blur = cv2.GaussianBlur(depth, (0,0), 3)
        depth = depth + 2.0 * (depth - depth_blur)

        # ---- Compute gradients (amplified) ----
        dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

        dx *= 30.0
        dy *= 30.0

        # ---- Compute normals ----
        norm = np.sqrt(dx**2 + dy**2 + 1.0)

        nx = dx / norm
        ny = dy / norm
        nz = 1.0 / norm

        # ---- Light direction (angled) ----
        light = np.array([0.7, -0.5, 0.2])
        light = light / np.linalg.norm(light)

        # ---- Shading ----
        L_gray = nx * light[0] + ny * light[1] + nz * light[2]

        # Normalize AFTER shading
        L_gray = (L_gray - L_gray.min()) / (L_gray.max() - L_gray.min() + 1e-6)
        # boost contrast
        L_gray = np.power(L_gray, 0.5)

        # ---- Expand to RGB ----
        L = np.repeat(L_gray[:, :, None], 3, axis=2)

        # ---- Reflectance ----
        R = img_rgb / (L + 1e-4)
        R = np.clip(R, 0, 1)

        # ---- Heatmap ----
        L_norm = L_gray

        return R, L, L_norm


def visualize(img, R, L, L_norm):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Reflectance")
    plt.imshow(R)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Illumination (RGB)")
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

    model = GeometryAware()

    R, L, L_norm = model.geometry_aware_decompose(img)

    visualize(img, R, L, L_norm)