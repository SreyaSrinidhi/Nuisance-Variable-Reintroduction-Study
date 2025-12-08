# decoders/contrast_decoder.py

import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class ContrastDecoder(BaseDecoder):
    """
    Decoder that takes SAL latent z + contrast parameter and outputs an image.

    contrast: scalar per example, e.g. [0.5, 1.5]
    """

    def __init__(
        self,
        z_dim: int,
        base_ch: int = 64,
        out_ch: int = 3,
        img_size: int = 64,
    ):
        super().__init__(
            z_dim=z_dim,
            base_ch=base_ch,
            out_ch=out_ch,
            img_size=img_size,
        )
        # +1 for contrast
        self.in_dim = z_dim + 1

        # Map (z, contrast) to 8x8 feature map
        self.fc = nn.Linear(self.in_dim, base_ch * 8 * 8)

    def encode_condition(
        self,
        z: torch.Tensor,
        contrast: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        z:        [B, z_dim]
        contrast: [B, 1]
        Returns:
            [B, base_ch, 8, 8]
        """
        device = self.fc.weight.device
        z = z.to(device)
        contrast = contrast.to(device)

        x = torch.cat([z, contrast], dim=-1)          # [B, z_dim + 1]

        x = self.fc(x)                                # [B, base_ch*8*8]
        x = x.view(x.size(0), self.base_ch, 8, 8)     # [B, base_ch, 8, 8]
        return x
