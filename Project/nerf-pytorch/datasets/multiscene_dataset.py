import torch
import numpy as np
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset 
from .utils import load_pose_vecs
from PIL import Image

class MultiSceneSALDecoderDataset(Dataset):
    def __init__(self, scene_configs, img_size=64):
        all_embeddings = []
        all_pose_vecs = []
        all_paths = []

        for cfg in scene_configs:
            emb_dict = torch.load(cfg["embeddings_path"])
            z_scene = emb_dict["embeddings"]
            paths_scene = emb_dict["paths"]
            pose_vecs = load_pose_vecs(cfg["poses_path"])

            all_embeddings.append(z_scene)
            all_pose_vecs.append(torch.tensor(pose_vecs).float())
            all_paths += paths_scene

        self.z_sal = torch.cat(all_embeddings, dim=0)
        self.pose_vecs = torch.cat(all_pose_vecs, dim=0)
        self.paths = all_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.z_sal)

    def __getitem__(self, idx):
        z = self.z_sal[idx].float()
        g = self.pose_vecs[idx]

        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float()

        if img.max() > 1:
            img /= 255.0

        img = img.permute(2,0,1)
        img = F.interpolate(img.unsqueeze(0),
                            size=(self.img_size,self.img_size),
                            mode="bilinear",
                            align_corners=False).squeeze(0)

        return z, g, img