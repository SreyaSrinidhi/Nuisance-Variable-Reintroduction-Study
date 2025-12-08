import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
from .utils import load_pose_vecs
from PIL import Image

class FernSALDecoderDataset(Dataset):
    def __init__(self, embeddings_path, poses_vec_path, img_dir, img_size=64):
        super().__init__()

        # Load embedding directory
        emb_dict = torch.load(embeddings_path)
        self.z_sal = emb_dict["embeddings"]
        self.paths = emb_dict["paths"]

        self.N, self.dim  = self.z_sal.shape

        # Forcing decoder to use the same scene
        z_all = self.z_sal                                  # [N, z_dim]
        z_scene = z_all.mean(dim=0, keepdim=True)           # [1, z_dim]
        self.z_sal = z_scene.repeat(self.N, 1)              # [N, z_dim]

        # Load pose vectors
        self.pose_vecs = load_pose_vecs(poses_vec_path)
        assert len(self.pose_vecs) == self.N, "Mismatch embeddings vs poses"
        
        # Image handling
        self.img_dir = img_dir
        self.img_size = img_size
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # Latent embedding z
        z = self.z_sal[idx].clone().detach().float() # [128]

        # Viewpoint nuisance variable g_view
        g = torch.tensor(self.pose_vecs[idx]).float() # [12]

        # Load ground-truth image
        img_path = self.paths[idx] # original store path
        full_path = os.path.join(self.img_dir, os.path.basename(img_path))

        img = Image.open(full_path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() # [H,W,3]

        # Normalizing pixel intensities
        if img.max() > 1.0:
            img /= 255.0

        img = img.permute(2,0,1) #C,H,W

        # Resize to decoder size
        img = F.interpolate(
            img.unsqueeze(0),
            size = (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return z, g, img