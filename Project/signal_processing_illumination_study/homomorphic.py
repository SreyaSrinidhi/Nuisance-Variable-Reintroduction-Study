import cv2
import numpy as np
import matplotlib.pyplot as plt


def homomorphic_decompose(img):
    img = img.astype(np.float32) / 255.0

    # ---- grayscale ----
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---- log transform ----
    log_img = np.log(img_gray + 1e-6)

    # ---- FFT ----
    fft = np.fft.fft2(log_img)
    fft_shift = np.fft.fftshift(fft)

    # ---- build filters ----
    h, w = img_gray.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    sigma = min(h, w) / 8

    D2 = (x - cx)**2 + (y - cy)**2
    gaussian = np.exp(-D2 / (2 * sigma**2))

    # ---- LOW frequency (illumination) ----
    low_freq = fft_shift * gaussian

    # ---- HIGH frequency (reflectance) ----
    high_freq = fft_shift * (1 - gaussian)

    # ---- inverse FFT ----
    L_log = np.fft.ifft2(np.fft.ifftshift(low_freq))
    R_log = np.fft.ifft2(np.fft.ifftshift(high_freq))

    # ---- back to image space ----
    L = np.exp(np.real(L_log))
    R = np.exp(np.real(R_log))

    # ---- normalize ----
    L_norm = (L - L.min()) / (L.max() - L.min() + 1e-6)
    R_norm = (R - R.min()) / (R.max() - R.min() + 1e-6)

    # ---- expand illumination to RGB ----
    L_rgb = np.repeat(L_norm[:, :, None], 3, axis=2)

    print("FFT magnitude mean:", np.mean(np.abs(fft_shift)))
    print("High freq energy:", np.mean(np.abs(high_freq)))
    print("Low freq energy:", np.mean(np.abs(low_freq)))

    plt.imshow(np.log(np.abs(fft_shift) + 1), cmap='inferno')
    plt.title("FFT Spectrum")
    plt.colorbar()
    plt.show()

    return R_norm, L_rgb, L_norm


def visualize(img, R, L, L_norm):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Reflectance (High Freq)")
    plt.imshow(R)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Illumination (Low Freq)")
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

    R, L, L_norm = homomorphic_decompose(img)

    visualize(img, R, L, L_norm)