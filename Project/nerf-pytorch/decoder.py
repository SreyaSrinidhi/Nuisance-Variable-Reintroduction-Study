import os
import argparse 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.fern_dataset import FernSALDecoderDataset
from datasets.multiscene_dataset import MultiSceneSALDecoderDataset

from decoders.view_baseline_decoder import ViewpointDecoder
from decoders.baseline_decoder_basech_128 import ViewpointDecoder128
from decoders.resnet64_decoder import ViewpointDecoderResNet
from decoders.brightness_baseline_decoder import BrightnessDecoder
from decoders.contrast_baseline_decoder import ContrastDecoder


# -----------------------------
# Dataset selection
# -----------------------------
def get_dataset(dataset_name, params):
    """
    Get appropriate dataset object.

    For 'fern', expects:
        params.embeddings_path
        params.nuisance_path
        params.image_path

    For 'multiscene', expects:
        params.scene_configs (list of dicts:
            { "embeddings_path": "...", "poses_path": "..." }
        )
    """
    dataset_name = dataset_name.lower()  # normalize

    if dataset_name == "fern":
        return FernSALDecoderDataset(
            embeddings_path=params.embeddings_path,
            poses_vec_path=params.nuisance_path,
            img_dir=params.image_path,
            img_size=params.img_size,
        )

    elif dataset_name == "multiscene":
        if not hasattr(params, "scene_configs"):
            raise ValueError(
                "Multiscene dataset requires params.scene_configs = "
                "[{ 'embeddings_path': ..., 'poses_path': ...}, ...]"
            )

        return MultiSceneSALDecoderDataset(
            scene_configs=params.scene_configs,
            img_size=params.img_size
        )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# -----------------------------
# Decoder factory
# -----------------------------
def build_decoder(
    mode: str,
    arch: str,
    z_dim: int,
    img_size: int = 64,
    base_ch: int = 64,
    out_ch: int = 3,
    pose_dim: int = 12,
) -> nn.Module:
    """
    Factory for decoders based on nuisance mode and architecture.

    mode: 'view', 'brightness', or 'contrast'
    arch: 'baseline', '128', 'resnet'
    """

    mode = mode.lower()
    arch = arch.lower()

    if mode == "view":
        if arch == "baseline":
            return ViewpointDecoder(
                z_dim=z_dim,
                pose_dim=pose_dim,
                base_ch=base_ch,
                out_ch=out_ch,
                img_size=img_size,
            )
        elif arch == "128":
            return ViewpointDecoder128(
                z_dim=z_dim,
                pose_dim=pose_dim,
                base_ch=base_ch,
                out_ch=out_ch,
                img_size=img_size,
            )
        elif arch == "resnet":
            return ViewpointDecoderResNet(
                z_dim=z_dim,
                pose_dim=pose_dim,
                base_ch=base_ch,
                out_ch=out_ch,
                img_size=img_size,
                num_blocks_per_stage=2,
            )
        else:
            raise ValueError(f"Unknown decoder_arch={arch} for mode='view'")

    elif mode == "brightness":
        # Right now we only have a baseline brightness decoder
        if arch not in ["baseline", "128", "resnet"]:
            raise ValueError(f"Unknown decoder_arch={arch}")
        return BrightnessDecoder(
            z_dim=z_dim,
            base_ch=base_ch,
            out_ch=out_ch,
            img_size=img_size,
        )

    elif mode == "contrast":
        # Right now we only have a baseline contrast decoder
        if arch not in ["baseline", "128", "resnet"]:
            raise ValueError(f"Unknown decoder_arch={arch}")
        return ContrastDecoder(
            z_dim=z_dim,
            base_ch=base_ch,
            out_ch=out_ch,
            img_size=img_size,
        )

    else:
        raise ValueError(f"Unknown mode={mode}")


# -----------------------------
# Training
# -----------------------------
def train(args):
    """
    Train decoder for the chosen nuisance mode.
    """

    dataset = get_dataset(args.dataset_name, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model
    z_dim = dataset.z_sal.shape[1]

    model = build_decoder(
        mode=args.mode,
        arch=args.decoder_arch,
        z_dim=z_dim,
        img_size=args.img_size,
        base_ch=64,
        out_ch=3,
        pose_dim=12,   # used only for view decoders
    ).to(device)

    print(f"Training decoder - mode={args.mode}, arch={args.decoder_arch}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for z_batch, g_batch, img_batch in tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False
        ):
            z_batch = z_batch.to(device, non_blocking=True)
            g_batch = g_batch.to(device, non_blocking=True)
            img_batch = img_batch.to(device, non_blocking=True)

            # For view mode, g_batch is pose vector; for brightness/contrast it's the scalar g.
            pred = model(z_batch, g_batch)

            loss = criterion(pred, img_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print("Saved model to:", args.save_path)
    return model


# -----------------------------
# Evaluation (generic)
# -----------------------------
def evaluate(
    args,
    idx_i: int = 0,
    idx_j: int = 5,
):
    print("Running evaluation only...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading checkpoint from:", args.eval_pth_file)

    # Dataset
    dataset = get_dataset(args.dataset_name, args)
    z_dim = dataset.z_sal.shape[1]

    # Build the same type of decoder used in training
    model = build_decoder(
        mode=args.mode,
        arch=args.decoder_arch,
        z_dim=z_dim,
        img_size=args.img_size,
        base_ch=64,
        out_ch=3,
        pose_dim=12,
    ).to(device)

    state_dict = torch.load(args.eval_pth_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Pick two samples (i and j)
    z_i, g_i, img_i = dataset[idx_i]
    z_j, g_j, img_j = dataset[idx_j]

    z_i   = z_i.unsqueeze(0).to(device)
    g_i   = g_i.unsqueeze(0).to(device)
    g_j   = g_j.unsqueeze(0).to(device)
    img_i = img_i.unsqueeze(0).to(device)
    img_j = img_j.unsqueeze(0).to(device)

    with torch.no_grad():
        # Recon at original viewpoint g_i
        recon_i = model(z_i, g_i)
        # Recon at swapped viewpoint g_j
        recon_i_gj = model(z_i, g_j)

    # L1 distances
    d_recon_i_to_GT_i  = F.l1_loss(recon_i,    img_i).item()
    d_recon_i_to_GT_j  = F.l1_loss(recon_i,    img_j).item()
    d_swap_to_GT_j     = F.l1_loss(recon_i_gj, img_j).item()
    d_swap_to_GT_i     = F.l1_loss(recon_i_gj, img_i).item()

    print("L1(recon_i,   GT_i) =", d_recon_i_to_GT_i)
    print("L1(recon_i,   GT_j) =", d_recon_i_to_GT_j)
    print("L1(swap,      GT_j) =", d_swap_to_GT_j)
    print("L1(swap,      GT_i) =", d_swap_to_GT_i)

    # Visualization

    def show(t, title):
        t = t.squeeze(0).cpu().permute(1, 2, 0).numpy()
        plt.imshow(t)
        plt.title(title)
        plt.axis("off")

    plt.figure(figsize=(12, 3))

    # GT at viewpoint g_i (index i)
    plt.subplot(1, 4, 1)
    show(img_i, f"GT i (z_i, g_i)")

    # GT at viewpoint g_j (index j) – this is your ground-truth for g_j
    plt.subplot(1, 4, 2)
    show(img_j, f"GT j (z_j, g_j)")

    # Recon using (z_i, g_i)
    plt.subplot(1, 4, 3)
    show(recon_i, "Recon (z_i, g_i)")

    # Recon using (z_i, g_j) – should match GT j if mapping is good
    plt.subplot(1, 4, 4)
    show(recon_i_gj, f"Swap (z_i, g_j)")

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to file containing SAL embeddings"
    )
    parser.add_argument(
        "--nuisance_path",
        type=str,
        default="./data/fern/poses_bounds.npy",
        help="Path to file that contains nuisance variable values (poses or g values)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/fern/images",
        help="Path to folder containing the ground truth images"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="fern",
        help="Choose dataset: 'fern' or 'multiscene'"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="view",
        choices=["view", "brightness", "contrast"],
        help="Nuisance variable to handle (view/brightness/contrast)"
    )
    parser.add_argument(
        "--decoder_arch",
        type=str,
        default="128",
        choices=["baseline", "128", "resnet"],
        help="Decoder architecture (baseline, 128, resnet). "
             "For brightness/contrast, currently only baseline behavior is used."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="decoder_fern.pth"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Set this if you only want to perform evaluation"
    )
    parser.add_argument(
        "--eval_pth_file",
        type=str,
        default="./decoder_fern.pth",
        help="Path to the model checkpoint you want to evaluate"
    )

    args = parser.parse_args()

    # Optional multiscene config
    if args.dataset_name == "multiscene":
        args.scene_configs = [
            {
                "embeddings_path": "./embedding_files/embeddings_fern_view.pt",
                "poses_path": "./data/fern/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_flower_view.pt",
                "poses_path": "./data/flower/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_fortress_view.pt",
                "poses_path": "./data/fortress/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_horns_view.pt",
                "poses_path": "./data/horns/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_leaves_view.pt",
                "poses_path": "./data/leaves/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_orchids_view.pt",
                "poses_path": "./data/orchids/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_room_view.pt",
                "poses_path": "./data/room/poses_bounds.npy"
            },
            {
                "embeddings_path": "./embedding_files/embeddings_trex_view.pt",
                "poses_path": "./data/trex/poses_bounds.npy"
            },
        ]

    if not args.eval_only:
        train(args)

    evaluate(
        args=args,
        idx_i=0,
        idx_j=5,
    )
