from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn

# ---- LibMOON path ----
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "libmoon-enhanced" / "libmoon"))

from libmoon.problem.mtl.loaders.multimnist_loader import MultiMNISTData
from libmoon_adapter import LibMoonDatasetAdapter


from experiments.dsprites.models import LeNetHyper, LeNetTarget, ImageSize
from experiments.utils import circle_points

KERNEL_SIZE = [9, 5]


@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device, n_tasks):
    hypernet.eval()
    targetnet.eval()

    loss_fns = [nn.CrossEntropyLoss()] * n_tasks
    results = defaultdict(list)

    for ray in rays:
        ray = torch.tensor(ray, dtype=torch.float32, device=device)
        ray = ray / ray.sum()  # safety normalization

        total = 0
        correct = [0] * n_tasks
        losses = [0.0] * n_tasks

        for img, ys in loader:
            img = img.to(device)
            ys = ys.to(device)
            bs = img.size(0)

            weights = hypernet(ray)
            logits = targetnet(img, weights)

            for i in range(n_tasks):
                losses[i] += loss_fns[i](logits[i], ys[:, i]).item() * bs
                preds = logits[i].argmax(dim=1)
                correct[i] += (preds == ys[:, i]).sum().item()

            total += bs

        results["ray"].append(ray.cpu().numpy().tolist())
        for i in range(n_tasks):
            results[f"task{i}_acc"].append(correct[i] / total)
            results[f"task{i}_loss"].append(losses[i] / total)

    return results


def main():
    parser = argparse.ArgumentParser("MO-MNIST inference (3 rays, LibMOON)")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--modelpath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="inference_outputs")
    parser.add_argument("--ray-hidden", type=int, default=100)
    parser.add_argument("--tasks-ids", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--out-dim", nargs="+", type=int, default=[10, 10])
    parser.add_argument("--img-size", type=str, choices=["36", "64"], default="36")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks = len(args.tasks_ids)

    # ---- dataset (LibMOON, test split only) ----
    test_raw = MultiMNISTData(dataset=args.dataset, split="test")
    test_set = LibMoonDatasetAdapter(test_raw)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # ---- model ----
    img_size = ImageSize.IMG36x36 if args.img_size == "36" else ImageSize.IMG64x64

    hnet = LeNetHyper(
        KERNEL_SIZE,
        ray_hidden_dim=args.ray_hidden,
        out_dim=args.out_dim,
        n_tasks=n_tasks,
        img_size=img_size,
    )
    hnet.load_state_dict(torch.load(args.modelpath, map_location=device))
    hnet.to(device).eval()

    net = LeNetTarget(
        KERNEL_SIZE,
        out_dim=args.out_dim,
        n_tasks=n_tasks,
        img_size=img_size,
    )
    net.to(device).eval()

    # ---- fixed inference rays (INTERPRETABLE & REPORTABLE) ----
    # rays = np.array([
    #     [0.999, 0.001],   # task 0 focused
    #     [0.5,   0.5],     # balanced
    #     [0.001, 0.999],   # task 1 focused
    # ], dtype=np.float32)

    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    # K is n_rays, here 10
    
    rays = circle_points(K = 10, n_tasks=2, min_angle=min_angle, max_angle=max_angle) 

    # ---- evaluation ----
    results = evaluate(hnet, net, test_loader, rays, device, n_tasks)

    # ---- save ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_50epochs_results_10rays.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved inference results to {out_path}")


if __name__ == "__main__":
    main()
