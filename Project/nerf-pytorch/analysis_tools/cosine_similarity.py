import torch
from torch.nn.functional import cosine_similarity

def single_cosine():
    data = torch.load("embeddings_sal_contrast.pt")
    z = data["embeddings"]   # (V, 128)
    paths = data["paths"]

    V = z.shape[0]
    print("Num views:", V)

    base = z[0:1]  # take first view as reference

    sims = cosine_similarity(z, base, dim=1)  # (V,)

    for i, (p, s) in enumerate(zip(paths, sims)):
        print(f"{i:02d}  {p}  cosine_sim_to_view0 = {s.item():.4f}")

    print("Mean similarity to view 0:", sims.mean().item())
    print("Min similarity to view 0:", sims.min().item())
    print("Max similarity to view 0:", sims.max().item())

def cosine_comparison():
    files = {
        "contrast mean"   : "embeddings_sal_contrast.pt",
        "view mean" : "embeddings_sal.pt",
    }

    # 1. Load all embeddings
    all_embeddings = {
        name: torch.load(path)["embeddings"]
        for name, path in files.items()
    }

    # 2. Sanity check: all models should have embeddings for the same number of images
    lens = {name: z.shape[0] for name, z in all_embeddings.items()}
    print("Num views/images per model:", lens)

    nums = list(lens.values())
    if len(set(nums)) != 1:
        raise ValueError(f"Models have different numbers of embeddings: {lens}")
    num_imgs = nums[0]
    print("Using", num_imgs, "images")

    # 3. Cosine sim to reference image (index 0) for each model
    cos_sims_to_ref = {}
    for name, z in all_embeddings.items():
        base = z[0:1]                          # reference embedding (1, D)
        sims = cosine_similarity(z, base, dim=1)  # (N,)
        cos_sims_to_ref[name] = sims

    # 4. Cross-model cosine similarity per image: view vs brightness
    z_view   = all_embeddings["contrast mean"]
    z_bright = all_embeddings["view mean"]
    cross_sims = cosine_similarity(z_view, z_bright, dim=1)  # (N,)

    # 5. Pretty print table
    header = (
        "Idx | "
        f"{'contrast→contrast0':>12} | "
        f"{'bright→contrast0':>14} | "
        f"{'contrast↔bright':>12}"
    )
    print(header)
    print("-" * len(header))

    for i in range(num_imgs):
        row = (
            f"{i:3d} | "
            f"{cos_sims_to_ref['contrast mean'][i].item():12.4f} | "
            f"{cos_sims_to_ref['view mean'][i].item():14.4f} | "
            f"{cross_sims[i].item():12.4f}"
        )
        print(row)

    # 6. Summary stats
    print("\n=== Summary ===")
    print("View SAL:    mean =", float(cos_sims_to_ref["contrast mean"].mean()),
          "min =", float(cos_sims_to_ref["contrast mean"].min()),
          "max =", float(cos_sims_to_ref["contrast mean"].max()))
    print("Bright SAL:  mean =", float(cos_sims_to_ref["view mean"].mean()),
          "min =", float(cos_sims_to_ref["view mean"].min()),
          "max =", float(cos_sims_to_ref["view mean"].max()))
    print("View↔Bright: mean =", float(cross_sims.mean()),
          "min =", float(cross_sims.min()),
          "max =", float(cross_sims.max()))

if __name__ == "__main__":
    # single_cosine()
    cosine_comparison()