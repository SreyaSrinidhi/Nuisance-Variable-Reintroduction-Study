import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Simple residual block: Conv-BN-ReLU-Conv-BN + skip connection.
    Keeps the number of channels constant.
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ViewpointDecoderResNet(nn.Module):
    """
    ResNet-style decoder:
        (z, pose) -> fc -> [B, base_ch, 8, 8]
        -> upsample to 16,32,64 with ConvTranspose2d
        -> ResBlocks at each scale for refinement
    """
    def __init__(
        self,
        z_dim,
        pose_dim=12,
        base_ch=64,
        out_ch=3,
        img_size=64,
        num_blocks_per_stage=2,
    ):
        super().__init__()
        assert img_size == 64, "ResNet64 decoder is configured for 64x64 outputs."
        assert base_ch % 4 == 0, "base_ch should be divisible by 4."

        self.z_dim = z_dim
        self.pose_dim = pose_dim
        self.in_dim = z_dim + pose_dim
        self.base_ch = base_ch
        self.img_size = img_size

        # Map (z, pose) to 8x8 feature map
        self.fc = nn.Linear(self.in_dim, base_ch * 8 * 8)

        # 8x8 → 16x16
        self.up1 = nn.ConvTranspose2d(
            in_channels=base_ch,
            out_channels=base_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.stage1 = nn.Sequential(
            *[ResBlock(base_ch) for _ in range(num_blocks_per_stage)]
        )

        # 16x16 → 32x32
        self.up2 = nn.ConvTranspose2d(
            in_channels=base_ch,
            out_channels=base_ch // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.stage2 = nn.Sequential(
            *[ResBlock(base_ch // 2) for _ in range(num_blocks_per_stage)]
        )

        # 32x32 → 64x64
        self.up3 = nn.ConvTranspose2d(
            in_channels=base_ch // 2,
            out_channels=base_ch // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.stage3 = nn.Sequential(
            *[ResBlock(base_ch // 4) for _ in range(num_blocks_per_stage)]
        )

        # Final RGB prediction
        self.final = nn.Conv2d(
            in_channels=base_ch // 4,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z, pose_vec):
        device = self.fc.weight.device
        z = z.to(device)
        pose_vec = pose_vec.to(device)

        # Concatenate latent + pose
        x = torch.cat([z, pose_vec], dim=-1)           # [B, z_dim + pose_dim]

        # [B, base_ch*8*8] -> [B, base_ch, 8, 8]
        x = self.fc(x)
        x = x.view(x.size(0), self.base_ch, 8, 8)

        # 8 → 16
        x = self.up1(x)
        x = F.relu(x, inplace=True)
        x = self.stage1(x)

        # 16 → 32
        x = self.up2(x)
        x = F.relu(x, inplace=True)
        x = self.stage2(x)

        # 32 → 64
        x = self.up3(x)
        x = F.relu(x, inplace=True)
        x = self.stage3(x)

        # RGB logits -> sigmoid
        x = self.final(x)
        x = torch.sigmoid(x)                           # [B, out_ch, 64, 64] in [0,1]
        return x
