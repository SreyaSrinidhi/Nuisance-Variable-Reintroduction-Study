import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import resnet18, ResNet18_Weights

class ResNetBackbone(nn.Module):
    """
    Docstring for ResNetBackbone
    """
    def __init__(self, feature_dim=128, pretrained=True, freeze=True):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        
        self.features = nn.Sequential(*list(base.children())[:-1])
        in_dim = base.fc.in_features # 512 for resnet18

        self.fc=nn.Linear(in_dim, feature_dim)

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.features(x)        # (B, 512, 1, 1) for resnet18
        x = x.view(x.size(0), -1)   # (B, 512)
        x = self.fc(x)              # (B, feature_dim)
        return x

###########################################
# Base Encoder
###########################################

class NuisanceEncoder(nn.Module):
    """
    Docstring for NuisanceEncoder
    """
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 128,
        pooling: str = "max",
        sal_kernel_size: int = 3,
        sal_sigma_levels: float = 1.0
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        assert pooling in ["max", "lse", "mean", "weighted", "SAL"], \
            "pooling must be max, lse, mean, weighted, or SAL"
        self.pooling = pooling

        if pooling == "SAL":
            K = sal_kernel_size
            assert K % 2 == 1, "sal_kernel_size must be odd"
            half = K // 2
            # relative offsets from the center view index
            idx = torch.arange(-half, half+1, dtype=torch.float32) # [-1, 0, 1] if K=3
            k = torch.exp(-0.5 * (idx/sal_sigma_levels)**2) # [-1, 0, 1] -> [0.24, 1.00, 0.24]
            k = k/k.sum()
            self.register_buffer("sal_kernel", k)
        else:
            self.sal_kernel = None
        
    def get_nuisance_values(self) -> torch.Tensor:
        """
        Docstring for get_nuisance_values
        
        :param self: Description
        :return: Description
        :rtype: Tensor
        """
        raise NotImplementedError
    
    def apply_nuisance(self, x:torch.Tensor, v:float) -> torch.Tensor:
        """
        Docstring for apply_nuisance
        
        :param self: Description
        :param x: Description
        :type x: torch.Tensor
        :param v: Description
        :type v: float
        :return: Description
        :rtype: Tensor
        """
        raise NotImplementedError
    
    def _pool_over_nuisances(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Docstring for _pool_over_nuisances
        
        :param self: Description
        :param feats: Description
        :type feats: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        if self.pooling == "max":
            # Profile Likelihood: takes the max over all views
            pooled, _ = feats.max(dim=0) # (B,D)
        elif self.pooling == "lse":
            # Log-sum-exp pooling: smoother version of marginalization along teh view dimension
            # lse(v) = log(sum_i exp(v_i))
            pooled = torch.logsumexp(feats, dim=0) # (B,D)
        elif self.pooling == "mean":
            # Uniform anti-aliasing over viewpoints
            # Uniform kernel across all angles
            pooled =  feats.mean(dim=0)
        elif self.pooling == "weighted":
            if self.nuisance_weights is None:
                raise RuntimeError("nuisance_weights is None but pooling='weighted'")
            # Gaussian kernel over angles : sum_i w_i h(g_iy)
            pooled = (self.nuisance_weights * feats).sum(dim=0)
        elif self.pooling == "SAL":
            pooled = self._sal_pool(feats)
        else:
            raise ValueError("Unknown pooling mode: {self.pooling}")
        return pooled
    
    def _sal_pool(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Docstring for sal_pool
        
        :param self: Description
        :param feats: Description
        :type feats: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        V, B, D = feats.shape
        K = self.sal_kernel.shape[0]
        half = K//2

        smoothed = []

        # for each view index i, smooth over neighbors j with kernel sal_kernel
        for i in range(V):
            # accumulator: holds weighted sum of neighboring view features
            # acc = sum_{j \in neighbors} k*(i-j) \times h(g_jy)
            # anti-aliased representation at position i
            acc = 0.0 
            # normalization: accumulates total weights of neighbors
            # norm = sum_{j \in neighbors} k(i-j)
            norm = 0.0 
            for k in range(K):
                j = i + (k - half) # neighbor index around i
                if 0 <= j < V:
                    w = self.sal_kernel[k]
                    acc = acc + w * feats[j]
                    norm += w
            smoothed.append(acc/max(norm, 1e-8)) # in case norm is 0
        
        smoothed = torch.stack(smoothed, dim=0) # (V,B,D)

        pooled, _ = smoothed.max(dim=0) # (B,D)
        return pooled
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        :type x: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        B,C,H,W = x.shape
        nuisance_values = self.get_nuisance_values()
        level_feats = []
        
        # Loop over sampled viewpoints g_i
        for v in nuisance_values:
            # 1. Apply viewpoint transform g_i
            x_i = self.apply_nuisance(x, float(v.item()))
            # 2. Extract features with shared backbone - gives us representation of g_iyy
            f_i = self.backbone(x_i) # (B,D)
            level_feats.append(f_i)

        # Stack over views -> (num_views, B, D)
        feats = torch.stack(level_feats, dim=0)

        # 3. SAL-like pooling over views
        z = self._pool_over_nuisances(feats)
        return z

###########################################
# Encoder for Viewpoint
###########################################

class SALEncoderViewpoint(NuisanceEncoder):
    """
    Docstring for SALEncoderViewpoint
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_views: int = 8,
        min_angle_deg: float = -45.0,
        max_angle_deg: float = 45.0,
        feature_dim: int = 128,
        pooling: str = "max",
        # angle_sigma_deg: controls how local the anti-aliasing is by controling 
        # gaussian kernel weight
        angle_sigma_deg: float = 15.0, # used only for weighted
        sal_kernel_size: int = 3, # has to be odd to have a symmetric kernel \
                                  # centered around current view index
        sal_sigma_levels: float = 1.0,
    ):
        super().__init__(
            backbone=backbone,
            feature_dim=feature_dim,
            pooling=pooling,
            sal_kernel_size=sal_kernel_size,
            sal_sigma_levels=sal_sigma_levels
        )

        self.num_views = num_views
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg

        # Precompute viewpoint angles g_i (in degrees)
        if num_views == 1:
            angles = [0.0]
        else:
            angles = torch.linspace(min_angle_deg, max_angle_deg, steps=num_views)
        self.register_buffer("view_angles_deg", angles)

        # Precompute gaussian weights over angles (for "weighted pooling")
        if pooling == "weighted":
            sigma = angle_sigma_deg
            weights = torch.exp(-0.5*(angles/sigma)**2) # Gaussian distribution formula
            weights = weights/weights.sum() # Normalizing weights
            weights = weights.view(num_views, 1, 1)
            self.register_buffer("nuisance_weights", weights)
        else:
            self.nuisance_weights = None

    def get_nuisance_values(self) -> torch.Tensor:
        return self.view_angles_deg

    def apply_nuisance(self, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """
        Docstring for _apply_view_transform
        
        :param self: Description
        :param x: Description
        :param angle_deg: Description
        """
        B,C,H,W = x.shape
        transformed = []
        for b in range(B):
            img = x[b]
            # Convert to PIL-like float tensor [0,1] assumed
            img_rot = TF.rotate(img, angle_deg, interpolation=TF.InterpolationMode.BILINEAR) # Rotated image
            transformed.append(img_rot)
        return torch.stack(transformed, dim=0)

###########################################
# Encoder for Brightness
###########################################

class SALEncoderBrightness(NuisanceEncoder):
    def __init__(
        self,
        backbone: nn.Module,
        num_levels: int = 5,
        min_scale: float = 0.6,
        max_scale: float = 1.4,
        feature_dim: int = 128,
        pooling: str = "max",
        sal_kernel_size: int = 3, 
        sal_sigma_levels: float = 1.0
    ):
        super().__init__(
            backbone = backbone,
            feature_dim = feature_dim,
            pooling = pooling, 
            sal_kernel_size = sal_kernel_size, 
            sal_sigma_levels = sal_sigma_levels
        )

        self.num_levels = num_levels
        self.min_scale = min_scale
        self.max_scale = max_scale

        if num_levels == 1:
            scales = torch.tensor([1.0], dtype=torch.float32)
        else:
            scales = torch.linspace(min_scale, max_scale, steps=num_levels)
        self.register_buffer("brightness_scales", scales)

        # ResNet normalization stats as buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)
    
    def get_nuisance_values(self) -> torch.Tensor:
        return self.brightness_scales
    
    def apply_nuisance(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Docstring for apply_nuisance
        
        :param self: Description
        :param x: Description
        :type x: torch.Tensor
        :param scale: Description
        :type scale: float
        :return: Description
        :rtype: Tensor
        """
        x_denorm = x * self.img_std + self.img_mean
        x_bright = x_denorm * scale
        x_bright = torch.clamp(x_bright, 0.0, 1.0)
        x_norm = (x_bright - self.img_mean) / self.img_std
        return x_norm
    
class SALEncoderContrast(NuisanceEncoder):
    def __init__(
        self,
        backbone: nn.Module,
        num_levels: int = 5,
        min_scale: float = 0.6,
        max_scale: float = 1.4,
        feature_dim: int = 128,
        pooling: str = "max",
        sal_kernel_size: int = 3,
        sal_sigma_levels: float = 1.0,
    ):
        super().__init__(
            backbone=backbone,
            feature_dim=feature_dim,
            pooling=pooling,
            sal_kernel_size=sal_kernel_size,
            sal_sigma_levels=sal_sigma_levels
        )

        self.num_levels = num_levels
        self.min_scale = min_scale
        self.max_scale = max_scale

        if num_levels == 1:
            scales = torch.tensor([1.0], dtype=torch.float32)
        else:
            scales = torch.linspace(min_scale, max_scale, steps=num_levels)
        self.register_buffer("contrast_scales", scales)

        # ResNet normalization stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)

    def get_nuisance_values(self) -> torch.Tensor:
        return self.contrast_scales
    
    def apply_nuisance(self, x: torch.Tensor, scale:float) -> torch.Tensor:
        x_denorm = x * self.img_std + self.img_mean
        # per-image mean over C,H,W
        m = x_denorm.mean(dim=[1, 2, 3], keepdim=True)
        x_contrast = m + scale * (x_denorm - m)
        x_contrast = torch.clamp(x_contrast, 0.0, 1.0)
        x_norm = (x_contrast - self.img_mean) / self.img_std
        return x_norm

def build_backbone(feature_dim: int = 128) -> nn.Module:
    base = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Strip the classifier
    backbone = nn.Sequential(*list(base.children())[:-1])
    feature_dim = base.fc.in_features
    backbone = nn.Sequential(
        backbone,
        nn.Flatten(),
        nn.Linear(feature_dim, 128),
    )
    return backbone

def build_encoder(mode: str, backbone: nn.Module):
    if mode == "view":
        return SALEncoderViewpoint(
            backbone=backbone,
            num_views=5,
            min_angle_deg=-30,
            max_angle_deg=30,
            feature_dim=128,
            pooling="weighted",
            angle_sigma_deg=15.0,
            sal_kernel_size=3,
            sal_sigma_levels=1.0,
        )

    elif mode == "brightness":
        return SALEncoderBrightness(
            backbone=backbone,
            num_levels=5,
            min_scale=0.6,
            max_scale=1.4,
            feature_dim=128,
            pooling="mean",
            sal_kernel_size=3,
            sal_sigma_levels=1.0,
        )

    elif mode == "contrast":
        return SALEncoderContrast(
            backbone=backbone,
            num_levels=5,
            min_scale=0.6,
            max_scale=1.4,
            feature_dim=128,
            pooling="SAL",
            sal_kernel_size=3,
            sal_sigma_levels=1.0,
        )

    else:
        raise ValueError(f"Unknown mode={mode}")


def load_images_from_path(path: str) -> tuple[torch.Tensor, list[str]]:
    """
    Docstring for load_images_from_path
    
    :param path: Description
    :type path: str
    :return: Description
    :rtype: tuple[Tensor, list[str]]
    """
    # ResNet normalization
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if os.path.isdir(path):
        exts = ("*.png", "*.jpg", "*.jpeg")
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(path, ext)))
        image_paths = sorted(image_paths)
    else:
        image_paths = [path]
    
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found at {path}")

    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))

    images_tensor = torch.stack(imgs, dim=0)  # (B, 3, 224, 224)
    return images_tensor, image_paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to an image file or a directory containing images (png/jpg/jpeg).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="embeddings_sal.pt",
        help="Path to save the embeddings tensor (default: embeddings_sal.pt).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="view",
        choices=["view", "brightness", "contrast"],
        help="Nuisance variable to marginalize out (default: view) (view/brightness/contrast)"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build backbone + encoder
    backbone = build_backbone(feature_dim=128).to(device)

    encoder = build_encoder(args.mode, backbone).to(device)
    encoder.eval()

    # 2. Load data
    images, paths = load_images_from_path(args.dataset)
    images = images.to(device)
    print(f"Loaded {len(paths)} image(s). Batch shape: {images.shape}")

    # 3. Inference
    with torch.no_grad():
        z = encoder(images)   # (B, 128)
    print("Output shape:", z.shape)

    # 4. Save
    torch.save(
        {"embeddings": z.cpu(), "paths": paths},
        args.output_path,
    )
    print(f"Saved embeddings to {args.output_path}")