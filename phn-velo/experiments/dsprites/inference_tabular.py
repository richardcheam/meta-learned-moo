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

from libmoon.problem.mtl.loaders.adult_loader import Adult
from libmoon.problem.mtl.loaders.compas_loader import Compas
from libmoon.problem.mtl.loaders.credit_loader import Credit
from libmoon_adapter import LibMoonTabularAdapter

from experiments.dsprites.models import TabularHyperNet, TabularTargetNet
from libmoon.problem.mtl.objectives import (
    DDPHyperbolicTangentRelaxation,
    DEOHyperbolicTangentRelaxation,
)


@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    targetnet.eval()

    bce = nn.BCEWithLogitsLoss()
    ddp_loss_fn = DDPHyperbolicTangentRelaxation()
    deo_loss_fn = DEOHyperbolicTangentRelaxation()

    results = defaultdict(list)

    for ray in rays:
        ray = torch.tensor(ray, dtype=torch.float32, device=device)
        ray = ray / ray.sum()

        total = 0
        correct = 0
        pred_losses = []
        ddp_losses = []
        deo_losses = []

        for x, ys in loader:
            x = x.to(device)
            y = ys[:, 0].to(device)  # prediction label
            s = ys[:, 1].to(device)  # sensitive attribute

            weights = hypernet(ray)
            logits = targetnet(x, weights).squeeze()

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.numel()

            # ddp_losses.append(ddp_loss_fn(logits, y, s).item())
            # deo_losses.append(deo_loss_fn(logits, y, s).item())
            L_ddp = ddp_loss_fn(
                logits=logits,
                labels=y,
                sensible_attribute=s,
            )
            L_deo = deo_loss_fn(
                logits=logits,
                labels=y,
                sensible_attribute=s,
            )

            pred_losses.append(bce(logits, y.float()).item())
            ddp_losses.append(L_ddp.item())
            deo_losses.append(L_deo.item())


        results["ray"].append(ray.cpu().numpy().tolist())
        results["acc"].append(correct / total)
        results["pred_loss"].append(np.mean(pred_losses))
        results["fair_loss_ddp"].append(np.mean(ddp_losses))
        results["fair_loss_deo"].append(np.mean(deo_losses))

    return results


def main():
    parser = argparse.ArgumentParser("Tabular inference (Adult / COMPAS / Credit)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["adult", "compas", "credit"])
    parser.add_argument("--modelpath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="inference_outputs")
    parser.add_argument("--ray-hidden", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load dataset (test split only) ----
    if args.dataset == "adult":
        raw = Adult(split="test", sensible_attribute="gender")
    elif args.dataset == "compas":
        raw = Compas(split="test", sensible_attribute="sex")
    elif args.dataset == "credit":
        raw = Credit(split="test", sensible_attribute="SEX")
    else:
        raise ValueError(args.dataset)

    test_set = LibMoonTabularAdapter(raw)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # ---- Infer input dimension ----
    sample_x, _ = test_set[0]
    in_dim = sample_x.shape[0]

    # ---- Models ----
    n_tasks = 2
    hnet = TabularHyperNet(
        in_dim=in_dim,
        hidden_dim=128,
        ray_hidden_dim=args.ray_hidden,
        n_tasks=n_tasks,
    )
    hnet.load_state_dict(torch.load(args.modelpath, map_location=device))
    hnet.to(device).eval()

    net = TabularTargetNet().to(device).eval()

    # # ---- Fixed inference rays ----
    # rays = np.array([
    #     [0.999, 0.001],   # accuracy-focused
    #     [0.5,   0.5],     # balanced
    #     [0.001, 0.999],   # fairness-focused
    # ], dtype=np.float32)

    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    # K is n_rays, here 10
    from experiments.utils import circle_points
    rays = circle_points(K = 10, n_tasks=2, min_angle=min_angle, max_angle=max_angle) 

    # ---- Evaluate ----
    results = evaluate(hnet, net, test_loader, rays, device)

    # ---- Save ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_results.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved inference results to {out_path}")


if __name__ == "__main__":
    main()
