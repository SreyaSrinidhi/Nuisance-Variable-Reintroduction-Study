import cv2
import numpy as np
import matplotlib.pyplot as plt

def guided_filter(log_img, radius, eps):
    """
    Apply guided filter per channel
    """
    log_L = np.zeros_like(log_img)
    
    for c in range(log_img.shape[2]):
        log_L[:, :, c] = cv2.ximgproc.guidedFilter(
            guide=log_img[:, :, c],
            src=log_img[:, :, c],
            radius=radius,
            eps=eps
        )
    return log_L


def retinexpp_decompose(img, sigma=30, radii=[5,15,30], alpha=0.6):
    img = img.astype(np.float32) / 255.0

    eps = 1e-6

    # ---------- RETINEX (log domain) ----------
    log_img = np.log(img + eps)
    log_L_retinex = cv2.GaussianBlur(log_img, (0,0), sigma)

    # ---------- GUIDED FILTER (linear domain) ----------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    L_guided_scales = []
    for r in radii:
        L_r = cv2.ximgproc.guidedFilter(
            guide=gray,
            src=gray,
            radius=r,
            eps=1e-3
        )
        L_guided_scales.append(L_r)

    L_guided = np.mean(L_guided_scales, axis=0)

    # bring guided into log space
    log_L_guided = np.log(L_guided[..., None] + eps)

    # ---------- COMBINE ----------
    log_L = alpha * log_L_retinex + (1 - alpha) * log_L_guided

    # ---------- FINAL ----------
    log_R = log_img - log_L

    L = np.exp(log_L)
    R = np.exp(log_R)

    # normalize illumination for visualization
    L_gray = np.mean(L, axis=2)
    L_norm = (L_gray - L_gray.min()) / (L_gray.max() - L_gray.min() + 1e-6)

    return R, L, L_norm

if __name__ == "__main__":
    img = cv2.imread("D:\Sreya\Case_Western\AdvanceMachineLearning\csds440-f25-sks190\Project\\nerf-pytorch\data\\fern\images\IMG_4027.JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R, L, L_norm = retinexpp_decompose(img)

    plt.figure(figsize=(16,4))

    # Original
    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    # Reflectance
    plt.subplot(1,4,2)
    plt.title("Reflectance")
    plt.imshow(np.clip(R, 0, 1))
    plt.axis('off')

    # Illumination (RGB)
    plt.subplot(1,4,3)
    plt.title("Illumination (RGB)")
    plt.imshow(np.clip(L, 0, 1))
    plt.axis('off')

    # Illumination Heatmap
    plt.subplot(1,4,4)
    plt.title("Illumination Heatmap")
    im = plt.imshow(L_norm, cmap='inferno')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.show()