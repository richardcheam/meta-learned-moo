from pathlib import Path
import sys
import argparse
import json
import logging
import datetime
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

# -----------------------------------------------------------------------------
# LibMOON path
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "libmoon-enhanced" / "libmoon"))

# -----------------------------------------------------------------------------
# LibMOON dataset
# -----------------------------------------------------------------------------
from libmoon.problem.mtl.loaders.electricity_demand_loader import (
    ElectricityDemandData
)

# -----------------------------------------------------------------------------
# Adapters
# -----------------------------------------------------------------------------
from libmoon_adapter import LibMoonTemporalAdapter

# -----------------------------------------------------------------------------
# Hypernetwork + utils (same as tabular)
# -----------------------------------------------------------------------------
from experiments.dsprites.models import TemporalHyperNet
from experiments.utils import (
    circle_points,
    count_parameters,
    get_device,
    save_args,
    set_logger,
    set_seed,
)

# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------
from phn import EPOSolver, LinearScalarizationSolver

# -----------------------------------------------------------------------------
# Temporal target network
# -----------------------------------------------------------------------------
from experiments.dsprites.models import TemporalTargetNet

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(hnet, net, loader, rays, device):
    hnet.eval()
    net.eval()

    results = defaultdict(list)
    loss_fn = nn.BCEWithLogitsLoss()

    for ray in rays:
        ray = torch.tensor(ray, dtype=torch.float32, device=device)
        ray /= ray.sum()

        task0_losses = []
        task1_losses = []

        for x, ys in loader:
            x = x.to(device)
            y0 = ys[:, 0].to(device)  # drop anomaly
            y1 = ys[:, 1].to(device)  # spike anomaly

            weights = hnet(ray)
            logits = net(x, weights)

            task0_losses.append(loss_fn(logits, y0.float()).item())
            task1_losses.append(loss_fn(logits, y1.float()).item())

        results["ray"].append(ray.cpu().numpy().tolist())
        results["task0_loss"].append(np.mean(task0_losses))
        results["task1_loss"].append(np.mean(task1_losses))

    return results


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(
    solver_type: str,
    epochs: int,
    ray_hidden: int,
    lr: float,
    wd: float,
    batch_size: int,
    n_rays: int,
    alpha: float,
    eval_every: int,
    out_dir: str,
    device: torch.device,
    optim_type: str,
    seq_len: int,
):

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    train_raw = ElectricityDemandData(split="train", sequence_length=seq_len)
    val_raw   = ElectricityDemandData(split="val",   sequence_length=seq_len)
    test_raw  = ElectricityDemandData(split="test",  sequence_length=seq_len)

    train_set = LibMoonTemporalAdapter(train_raw)
    val_set   = LibMoonTemporalAdapter(val_raw)
    test_set  = LibMoonTemporalAdapter(test_raw)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    n_tasks = 2  # drop + spike

    # -------------------------------------------------------------------------
    # Networks
    # -------------------------------------------------------------------------
    hnet = TemporalHyperNet(
        seq_len=seq_len,
        hidden_dim=128,
        ray_hidden_dim=ray_hidden,
        n_tasks=n_tasks,
    ).to(device)

    net = TemporalTargetNet(
        seq_len=seq_len,
        hidden_dim=128,
    ).to(device)

    logging.info(f"HyperNet params: {count_parameters(hnet)}")

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    if optim_type == "adam":
        optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)
    elif optim_type == "velo":
        from pylo.optim import VeLO
        optimizer = VeLO(hnet.parameters(), lr=1.0)
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")

    # -------------------------------------------------------------------------
    # Solver
    # -------------------------------------------------------------------------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver)
    solver_cls = solvers[solver_type]

    if solver_type == "epo":
        solver = solver_cls(n_tasks=n_tasks, n_params=count_parameters(hnet))
    else:
        solver = solver_cls(n_tasks=n_tasks)

    # -------------------------------------------------------------------------
    # Rays
    # -------------------------------------------------------------------------
    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1

    rays = circle_points(
        n_rays,
        n_tasks,
        min_angle=min_angle,
        max_angle=max_angle,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    loss_fn = nn.BCEWithLogitsLoss()
    val_results, test_results = {}, {}

    start_time = time.time()
    epoch_iter = trange(epochs)

    for epoch in epoch_iter:
        hnet.train()

        for x, ys in train_loader:
            optimizer.zero_grad()

            x = x.to(device)
            y0 = ys[:, 0].to(device)
            y1 = ys[:, 1].to(device)

            ray = torch.from_numpy(
                np.random.dirichlet([alpha] * n_tasks)
            ).float().to(device)

            weights = hnet(ray)
            logits0 = net(x, weights)
            logits1 = net(x, weights)

            L0 = loss_fn(logits0, y0.float())
            L1 = loss_fn(logits1, y1.float())

            losses = torch.stack([L0, L1])
            loss_sol = solver(losses, ray, list(hnet.parameters()))
            loss_sol.backward()

            if optim_type == "adam":
                optimizer.step()
            else:
                optimizer.step(loss_sol) #velo

        epoch_iter.set_description(
            f"Epoch {epoch+1} | L0={L0.item():.3f} L1={L1.item():.3f}"
        )

        if (epoch + 1) % eval_every == 0:
            val_results[f"epoch_{epoch+1}"] = evaluate(
                hnet, net, val_loader, rays, device
            )
            test_results[f"epoch_{epoch+1}"] = evaluate(
                hnet, net, test_loader, rays, device
            )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().isoformat()

    with open(out_dir / f"config_{now}.json", "w") as f:
        json.dump({
            "epochs": epochs,
            "solver": solver_type,
            "optimizer": optim_type,
            "n_rays": n_rays,
            "ray_hidden": ray_hidden,
            "alpha": alpha,
            "training_time": time.time() - start_time,
        }, f)

    with open(out_dir / f"val_results_{now}.json", "w") as f:
        json.dump(val_results, f)

    with open(out_dir / f"test_results_{now}.json", "w") as f:
        json.dump(test_results, f)

    torch.save(hnet.state_dict(), out_dir / f"hnet_{now}.pt")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train HPN / MML on Electricity Demand (Temporal)"
    )

    parser.add_argument("--optim", choices=["adam", "velo"], default="velo")
    parser.add_argument("--solver", choices=["ls", "epo"], default="epo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ray-hidden", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--n-rays", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--out-dir", type=str, default="outputs-electricity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpus", type=str, default="0")

    args = parser.parse_args()

    set_seed(args.seed)
    set_logger()

    train(
        solver_type=args.solver,
        epochs=args.epochs,
        ray_hidden=args.ray_hidden,
        lr=args.lr,
        wd=args.wd,
        batch_size=args.batch_size,
        n_rays=args.n_rays,
        alpha=args.alpha,
        eval_every=args.eval_every,
        out_dir=args.out_dir,
        device=get_device(args.no_cuda, args.gpus),
        optim_type=args.optim,
        seq_len=args.seq_len,
    )

    save_args(args.out_dir, args)