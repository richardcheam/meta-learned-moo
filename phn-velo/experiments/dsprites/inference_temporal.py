from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# LibMOON path
# -----------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "libmoon-enhanced" / "libmoon"))

# -----------------------------------------------------------------------------
# Dataset + adapter
# -----------------------------------------------------------------------------
from libmoon.problem.mtl.loaders.electricity_demand_loader import ElectricityDemandData
from libmoon_adapter import LibMoonTemporalAdapter

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
from experiments.dsprites.models import TemporalHyperNet, TemporalTargetNet

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
from experiments.utils import circle_points, get_device, set_seed


@torch.no_grad()
def evaluate(hnet, net, loader, rays, device):
    hnet.eval()
    net.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    results = defaultdict(list)

    for ray in rays:
        ray = torch.tensor(ray, dtype=torch.float32, device=device)
        ray = ray / ray.sum()

        task0_losses, task1_losses = [], []

        for x, ys in loader:
            x = x.to(device)
            y0 = ys[:, 0].to(device)
            y1 = ys[:, 1].to(device)

            weights = hnet(ray)
            logits = net(x, weights)

            task0_losses.append(loss_fn(logits, y0.float()).item())
            task1_losses.append(loss_fn(logits, y1.float()).item())

        results["ray"].append(ray.cpu().numpy().tolist())
        results["task0_loss"].append(np.mean(task0_losses))
        results["task1_loss"].append(np.mean(task1_losses))

    return results


def main():
    parser = argparse.ArgumentParser("Temporal inference (Electricity Demand)")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--n-rays", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.no_cuda, args.gpus)

    run_dir = Path(args.run_dir)

    # -------------------------------------------------------------------------
    # Load config
    # -------------------------------------------------------------------------
    cfg = {}
    cfg_files = list(run_dir.glob("config_*.json"))
    if cfg_files:
        with open(cfg_files[0]) as f:
            cfg = json.load(f)

    ray_hidden = cfg.get("ray_hidden", 100)

    # -------------------------------------------------------------------------
    # Dataset (TEST only)
    # -------------------------------------------------------------------------
    test_raw = ElectricityDemandData(
        split="test",
        sequence_length=args.seq_len,
    )
    test_set = LibMoonTemporalAdapter(test_raw)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    n_tasks = 2

    hnet = TemporalHyperNet(
        seq_len=args.seq_len,
        hidden_dim=128,
        ray_hidden_dim=ray_hidden,
        n_tasks=n_tasks,
    ).to(device)

    net = TemporalTargetNet(
        seq_len=args.seq_len,
        hidden_dim=128,
    ).to(device)

    ckpt = next(run_dir.glob("hnet_*.pt"))
    hnet.load_state_dict(torch.load(ckpt, map_location=device))
    hnet.eval()
    net.eval()

    # -------------------------------------------------------------------------
    # Rays (same policy as training)
    # -------------------------------------------------------------------------
    # rays = circle_points(
    #     args.n_rays,
    #     n_tasks,
    #     min_angle=0.1,
    #     max_angle=np.pi / 2 - 0.1,
    # )
    # ---- Fixed inference rays ----
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

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    results = evaluate(hnet, net, test_loader, rays, device)

    # -------------------------------------------------------------------------
    # Save (same layout as tabular)
    # -------------------------------------------------------------------------
    infer_dir = run_dir / "inference"
    infer_dir.mkdir(parents=True, exist_ok=True)

    out_path = infer_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved inference results to {out_path}")


if __name__ == "__main__":
    main()