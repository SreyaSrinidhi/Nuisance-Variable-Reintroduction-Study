import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewpointDecoder128(nn.Module):
    def __init__(
        self,
        z_dim,
        pose_dim=12,
        base_ch=128,
        out_ch=3,
        img_size=64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.pose_dim = pose_dim
        self.in_dim = z_dim + pose_dim
        self.base_ch = base_ch
        self.img_size = img_size

        self.start_spatial = img_size // 8 

        # Map (z, pose) to 8x8 feature map
        self.fc = nn.Linear(self.in_dim, base_ch*self.start_spatial*self.start_spatial)

        # Upsample: 8 -> 16 -> 32 -> 64
        self.deconv1 = nn.ConvTranspose2d(base_ch, base_ch // 2,4,2,1) # 8-> 16
        self.deconv2 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, 4, 2, 1) # 16→32
        self.deconv3 = nn.ConvTranspose2d(base_ch // 4, base_ch // 8, 4, 2, 1) # 32→64

        self.final = nn.Conv2d(base_ch // 8, out_ch, 3, 1, 1)

    def forward(self, z, pose_vec):
        """
        Docstring for forward
        
        :param self: Description
        :param z: Description
        :param pose_vec: Description
        """
        device = self.fc.weight.device
        z = z.to(device)
        pose_vec = pose_vec.to(device)

        x = torch.cat([z, pose_vec], dim=-1)          # [B, z_dim + pose_dim]
        x = self.fc(x)                                # [B, base_ch*8*8]
        x = x.view(x.size(0), self.base_ch, 8, 8)     # [B, base_ch, 8, 8]

        x = F.relu(self.deconv1(x))                   # [B, base_ch//2,16,16]
        x = F.relu(self.deconv2(x))                   # [B, base_ch//4,32,32]
        x = F.relu(self.deconv3(x))                   # [B, base_ch//8,64,64]

        x = self.final(x)                             # [B, out_ch,64,64]
        x = torch.sigmoid(x)                          # constrain to [0,1]
        return x