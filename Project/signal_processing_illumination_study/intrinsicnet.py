import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# import THEIR model
from unsupervisedLearningIntrinsicImages.models.pix2pix_model import Pix2PixModel


# -------------------------------
# Minimal Options (REQUIRED)
# -------------------------------
class DummyOpt:
    def __init__(self):
        self.isTrain = False
        self.batchSize = 1
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.which_model_netG = 'unet_256'
        self.use_dropout = False
        self.gpu_ids = [0]
        self.lr = 0.0002
        self.fineSize = 256

        # REQUIRED for loading weights
        self.checkpoints_dir = "./pretrained_models"
        self.name = ""


# -------------------------------
# Model Loader
# -------------------------------
def load_intrinsic_model():
    opt = DummyOpt()

    model = Pix2PixModel(opt)
    model.switch_to_eval()

    print("IntrinsicNet model loaded successfully")

    return model


# -------------------------------
# Wrapper Class
# -------------------------------
class IntrinsicNetWrapper:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def decompose(self, img):
        # Resize (REQUIRED)
        img_resized = cv2.resize(img, (256, 256))
        img_rgb = img_resized.astype(np.float32) / 255.0

        # Convert to tensor
        input_tensor = torch.from_numpy(
            img_rgb.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        # Set input
        self.model.input = input_tensor

        # Forward pass
        with torch.no_grad():
            self.model.forward()

            S = self.model.prediction_S
            R_log = self.model.prediction_R

        # Convert reflectance from log-space
        R = torch.exp(R_log)

        # Convert to numpy
        S = S[0].cpu().numpy().transpose(1, 2, 0)
        R = R[0].cpu().numpy().transpose(1, 2, 0)

        # Clip values (important)
        R = np.clip(R, 0, 1)
        S = np.clip(S, 0, 1)

        # Illumination heatmap
        L_gray = np.mean(S, axis=2)
        L_norm = (L_gray - L_gray.min()) / (L_gray.max() - L_gray.min() + 1e-6)

        return R, S, L_norm


# -------------------------------
# Visualization
# -------------------------------
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


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Make sure checkpoint exists
    if not os.path.exists("./pretrained_models/paper_final_net_G.pth"):
        print("ERROR: Missing model weights at ./pretrained_models/paper_final_net_G.pth")
        exit()

    # Load image
    img = cv2.imread(r"D:\Sreya\\Case_Western\AdvanceMachineLearning\\csds440-f25-sks190\\Project\\nerf-pytorch\data\\fern\\images\\IMG_4027.JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load model
    model = load_intrinsic_model()

    # Wrap model
    intrinsic = IntrinsicNetWrapper(model)

    # Run decomposition
    R, L, L_norm = intrinsic.decompose(img)

    # Visualize
    visualize(img, R, L, L_norm)