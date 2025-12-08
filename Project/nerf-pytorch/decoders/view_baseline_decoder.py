# decoders/viewpoint_decoder.py

import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class ViewpointDecoder(BaseDecoder):
    def __init__(
        self,
        z_dim: int,
        pose_dim: int = 12,
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
        self.pose_dim = pose_dim
        self.in_dim = z_dim + pose_dim

        # Map (z, pose) to 8x8 feature map
        self.fc = nn.Linear(self.in_dim, base_ch * 8 * 8)

    def encode_condition(self, z: torch.Tensor, pose_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Map (z, pose_vec) to [B, base_ch, 8, 8]
        """
        device = self.fc.weight.device
        z = z.to(device)
        pose_vec = pose_vec.to(device)

        x = torch.cat([z, pose_vec], dim=-1)          # [B, z_dim + pose_dim]
        x = self.fc(x)                                # [B, base_ch*8*8]
        x = x.view(x.size(0), self.base_ch, 8, 8)     # [B, base_ch, 8, 8]
        return x
