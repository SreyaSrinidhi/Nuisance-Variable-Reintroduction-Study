# decoders/brightness_decoder.py

import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class BrightnessDecoder(BaseDecoder):
    """
    Decoder that takes SAL latent z + brightness parameter and outputs an image.

    brightness: scalar per example, typically in something like [-0.5, 0.5]
                or a multiplicative scale like [0.6, 1.4] depending on how you define it.
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
        # +1 for brightness
        self.in_dim = z_dim + 1

        # Map (z, brightness) to 8x8 feature map
        self.fc = nn.Linear(self.in_dim, base_ch * 8 * 8)

    def encode_condition(
        self,
        z: torch.Tensor,
        brightness: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        z:          [B, z_dim]
        brightness: [B, 1]
        Returns:
            [B, base_ch, 8, 8]
        """
        device = self.fc.weight.device
        z = z.to(device)
        brightness = brightness.to(device)

        x = torch.cat([z, brightness], dim=-1)        # [B, z_dim + 1]

        x = self.fc(x)                                # [B, base_ch*8*8]
        x = x.view(x.size(0), self.base_ch, 8, 8)     # [B, base_ch, 8, 8]
        return x
