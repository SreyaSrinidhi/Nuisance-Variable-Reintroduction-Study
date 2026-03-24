import cv2
import numpy as np
import matplotlib.pyplot as plt

def retinex_decompose(img, sigma=30):
    # convert to float
    img = img.astype(np.float32)/255.0

    eps = 1e-6 # to avoid log(0)
    log_img = np.log(img + eps)

    # illumination (low frequency)
    log_L = cv2.GaussianBlur(log_img, (0,0), sigma)

    # reflectance
    log_R = log_img - log_L

    # convert back
    L = np.exp(log_L)
    R = np.exp(log_R)

    return R, L

if __name__ == "__main__":
    img = cv2.imread("D:\Sreya\Case_Western\AdvanceMachineLearning\csds440-f25-sks190\Project\\nerf-pytorch\data\\fortress\images\IMG_1833.JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R, L = retinex_decompose(img, sigma=30)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Reflectance")
    plt.imshow(np.clip(R, 0, 1))

    plt.subplot(1,3,3)
    plt.title("Illumination")
    plt.imshow(np.clip(L, 0, 1))

    plt.show()