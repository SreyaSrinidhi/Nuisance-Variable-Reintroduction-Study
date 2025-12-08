# decoders/base_decoder.py

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module, ABC):
    """
    Base class for decoders that:
      1) Map (z, nuisance params) -> [B, base_ch, 8, 8] via `encode_condition`
      2) Upsample 8x8 -> 64x64 via shared ConvTranspose2d stack
    """

    def __init__(
        self,
        z_dim: int,
        base_ch: int = 64,
        out_ch: int = 3,
        img_size: int = 64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.base_ch = base_ch
        self.out_ch = out_ch
        self.img_size = img_size

        if img_size != 64:
            raise ValueError(f"BaseDecoder currently assumes img_size=64, got {img_size}")

        # Shared upsampling stack: 8 -> 16 -> 32 -> 64
        self.deconv1 = nn.ConvTranspose2d(base_ch, base_ch // 2, 4, 2, 1)      # 8 -> 16
        self.deconv2 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, 4, 2, 1) # 16 -> 32
        self.deconv3 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, 4, 2, 1) # 32 -> 64

        self.final = nn.Conv2d(base_ch // 8, out_ch, 3, 1, 1)

    @abstractmethod
    def encode_condition(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Map (z, nuisance params) to a [B, base_ch, 8, 8] feature map.
        """
        raise NotImplementedError

    def forward(self, z: torch.Tensor, g: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        1) Use `encode_condition` to get [B, base_ch, 8, 8]
        2) Upsample to [B, out_ch, 64, 64]
        """
        x = self.encode_condition(z, g, **kwargs)       # [B, base_ch, 8, 8]

        x = F.relu(self.deconv1(x))                  # [B, base_ch//2, 16, 16]
        x = F.relu(self.deconv2(x))                  # [B, base_ch//4, 32, 32]
        x = F.relu(self.deconv3(x))                  # [B, base_ch//8, 64, 64]

        x = self.final(x)                            # [B, out_ch, 64, 64]
        x = torch.sigmoid(x)                         # constrain to [0, 1]
        return x
