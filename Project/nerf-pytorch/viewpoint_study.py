import os
import numpy as np
import imageio
import torch

from run_nerf import config_parser, create_nerf, render, device
from load_llff import load_llff_data

# Match run_nerf.py behavior: use CUDA default tensor type when available
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def main():
    # ---------- 1. Parse args ----------
    parser = config_parser()
    args = parser.parse_args()

    # Optional: shrink chunk if you hit OOM
    # args.chunk = 1024 * 8

    # ---------- 2. Load LLFF Fern data (same as train() in run_nerf.py) ----------
    images, poses, bds, render_poses, i_test = load_llff_data(
        args.datadir,
        args.factor,
        recenter=True,
        bd_factor=.75,
        spherify=args.spherify
    )

    # Convert poses to torch on the right device
    poses = torch.Tensor(poses).to(device)

    # poses: [N, 3, 5]; last column is [H, W, focal]
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]   # [N, 3, 4]

    H, W, focal = hwf
    H, W = int(H.item()), int(W.item())
    focal = focal.item()

    # Intrinsics matrix K (on the same device)
    K = torch.tensor([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0,     0,     1.0]
    ], device=device)

    # Near / far bounds (same logic as train() for LLFF)
    if args.no_ndc:
        near = torch.min(bds) * .9
        far = torch.max(bds) * 1.0
    else:
        near, far = 0., 1.
    near = float(near)
    far = float(far)

    print(f"H={H}, W={W}, focal={focal}, near={near:.3f}, far={far:.3f}")

    # ---------- 3. Create NeRF & load checkpoints ----------
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    # Add near/far to kwargs the same way run_nerf does
    render_kwargs_train.update(dict(near=near, far=far))
    render_kwargs_test.update(dict(near=near, far=far))

    # ---------- 4. Viewpoint study: render a subset of camera poses ----------
    os.makedirs('view_study', exist_ok=True)

    # Which camera indices to render (change as you like)
    idxs = [0, 5, 10, 15, 19]

    for i in idxs:
        c2w = poses[i]  # [3, 4] on correct device

        print(f"Rendering index {i}...")
        with torch.no_grad():
            rgb, disp, acc, extras = render(
                H, W, K,
                chunk=args.chunk,
                c2w=c2w,
                **render_kwargs_test
            )

        # rgb: [H, W, 3] in [0,1] on device
        rgb_np = rgb.cpu().numpy()
        rgb8 = (255 * np.clip(rgb_np, 0, 1)).astype(np.uint8)

        out_path = os.path.join('view_study', f'view_{i:03d}.png')
        imageio.imwrite(out_path, rgb8)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
