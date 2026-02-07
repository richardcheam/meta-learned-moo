# import sys
# sys.path
# sys.path.append("/Users/macbookpro/Desktop/M2DS/stageMOO/libmoon-enhanced/libmoon")

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "libmoon-enhanced" / "libmoon"))

#### LIBMOON SOLVERS #####
# from libmoon.solver.gradient.methods import EPOSolver, MGDAUBSolver, RandomSolver, PMGDASolver, MOOSVGDSolver, PMTLSolver, GradHVSolver, GradAggSolver, PCGradSolver, NashMTLSolver
# from libmoon.solver.gradient.methods.core.core_mtl import GradBaseMTLSolver
# from libmoon.solver.gradient.methods.epo_solver import EPOCore
########################## 

import logging
import argparse
import json
from collections import defaultdict
from pathlib import Path
import datetime
import time

import numpy as np
import torch
from torch import nn
from tqdm import trange

from experiments.dsprites.data import Dataset
from experiments.dsprites.models import TabularHyperNet, TabularTargetNet

from experiments.utils import (
    circle_points,
    count_parameters,
    get_device,
    save_args,
    set_logger,
    set_seed,
)
from phn import EPOSolver
from phn import LinearScalarizationSolver
# LibMOON Solver
from phn.libmoon_wrapper import *

from libmoon.problem.mtl.objectives import DDPHyperbolicTangentRelaxation, DEOHyperbolicTangentRelaxation

@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    targetnet.eval()

    results = defaultdict(list)

    bce = nn.BCEWithLogitsLoss()
    ddp_loss_fn = DDPHyperbolicTangentRelaxation()
    deo_loss_fn = DEOHyperbolicTangentRelaxation()

    for ray in rays:
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray /= ray.sum()

        total = 0
        correct = 0

        pred_losses = []
        ddp_losses = []
        deo_losses = []

        for x, ys in loader:
            x = x.to(device)
            y = ys[:, 0].to(device)
            s = ys[:, 1].to(device)

            weights = hypernet(ray)
            logits = targetnet(x, weights)

            # accuracy
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.numel()

            # losses
            L_pred = bce(logits, y.float())
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

            pred_losses.append(L_pred.item())
            ddp_losses.append(L_ddp.item())
            deo_losses.append(L_deo.item())

        results["ray"].append(ray.cpu().numpy().tolist())
        results["acc"].append(correct / total)
        results["pred_loss"].append(np.mean(pred_losses))
        results["fair_loss_ddp"].append(np.mean(ddp_losses))
        results["fair_loss_deo"].append(np.mean(deo_losses))

    return results

def train(
    # path, previously argument to dsprites dataset but not used in the function
    dataset: str,
    solver_type: str,
    epochs: int,
    hidden_dim: int,
    lr: float,
    wd: float,
    bs: int,
    val_size: float,
    n_rays: int,
    alpha: float,
    no_val_eval: bool,
    out_dir: str,
    device: torch.device,
    eval_every: int,
    tasks_ids: list[int],
    optim_type: str,
) -> None:
    # ----
    # data
    # ----
    assert val_size > 0, "please use validation by providing val_size > 0"

    # Dsprties version
    # data = Dataset(path, val_size=val_size, tasks_ids=tasks_ids)
    # train_set, val_set, test_set = data.get_datasets()

    # load tabular dataset form LibMOON 
    from libmoon.problem.mtl.loaders.adult_loader import Adult
    from libmoon.problem.mtl.loaders.compas_loader import Compas
    from libmoon.problem.mtl.loaders.credit_loader import Credit

    from libmoon_adapter import LibMoonTabularAdapter
    # load dataset from LibMOON (preprocessed MultiMNIST)
    if dataset == "adult":
        train_raw = Adult(split="train", sensible_attribute="gender")
        val_raw   = Adult(split="val", sensible_attribute="gender")
        test_raw  = Adult(split="test", sensible_attribute="gender")
    elif dataset == "compas":
        train_raw = Compas(split="train", sensible_attribute="sex")
        val_raw   = Compas(split="val", sensible_attribute="sex")
        test_raw  = Compas(split="test", sensible_attribute= "sex")
    elif dataset == "credit":
        train_raw = Credit(split="train", sensible_attribute="SEX")
        val_raw   = Credit(split="val", sensible_attribute="SEX")           
        test_raw  = Credit(split="test", sensible_attribute="SEX")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # adapt to standard torch dataset to fit the existing training code
    train_set = LibMoonTabularAdapter(train_raw)
    val_set   = LibMoonTabularAdapter(val_raw)
    test_set  = LibMoonTabularAdapter(test_raw)

    # infer input dimension
    sample_x, _ = train_set[0]
    in_dim = sample_x.shape[0]

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=4
    )

    # number of tasks
    n_tasks = len(tasks_ids)

    # ----
    # Nets
    # ----
    hidden_dim_mlp = 128 # fixed for M1 architecture
    hnet = TabularHyperNet(
        in_dim=in_dim,
        hidden_dim=hidden_dim_mlp,
        ray_hidden_dim=hidden_dim,
        n_tasks=n_tasks,
    )
    net = TabularTargetNet()

    logging.info(f"HN size: {count_parameters(hnet)}")

    hnet = hnet.to(device)
    net = net.to(device)

    # optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)
    # from pylo.optim import VeLO
    # optimizer = VeLO(hnet.parameters(), lr=1.0)

    if optim_type == "adam":
        optimizer = torch.optim.Adam(
            hnet.parameters(), lr=lr, weight_decay=wd
        )
    elif optim_type == "velo":
        from pylo.optim import VeLO
        optimizer = VeLO(hnet.parameters(), lr=1.0)
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")


    # ------
    # selecting the solver method
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver)

    # LibMOON solvers 
    # comb = make_combiner(solver_type)

    solver_method = solvers[solver_type]
    if solver_type == "epo":
        # pref = torch.from_numpy(
        #         np.random.dirichlet([alpha] * n_tasks, 1).astype(np.float32).flatten()
        #     )
        solver = solver_method(n_tasks=n_tasks, n_params=count_parameters(hnet))
    else:
        # ls
        solver = solver_method(n_tasks=n_tasks)

    # generate the rays that will be used for testing and validation
    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    test_rays = circle_points(n_rays, n_tasks, min_angle=min_angle, max_angle=max_angle)

    # ----------
    # Train loop
    # ----------
    last_eval = -1
    epoch_iter = trange(epochs)

    val_results = dict()
    test_results = dict()
    
    # loss functions
    pred_loss_fn = nn.BCEWithLogitsLoss()
    fair_loss_fn = DDPHyperbolicTangentRelaxation()

    start_time = time.time()

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            x, ys = batch
            x = x.to(device)
            y = ys[:, 0].to(device) # task 0: prediction
            s = ys[:, 1].to(device) # task 1: fairness

            # draw a vector of n_tasks elements from a Dirichlet distribution
            ray = torch.from_numpy(
                np.random.dirichlet([alpha] * n_tasks, 1).astype(np.float32).flatten()
            ).to(device)

            # get the weights from hnet given ray,
            # the use these weights to make a prediction
            # on the current batch by giving the weights
            # to net (the target network)
            ### forward: hypernet -> weights -> target logits
            weights = hnet(ray)
            logits = net(x, weights)

            # SAME logits for both tasks
            L_pred = pred_loss_fn(logits.squeeze(), y.float())
            L_fair = fair_loss_fn(
                logits=logits.squeeze(),
                labels=y,
                sensible_attribute=s
            )

            losses = torch.stack([L_pred, L_fair])

            # removes an extra batch dimension if ray has shape (1, n_tasks) instead of (n_tasks,)
            # we need 1D preference vector
            ray = ray.squeeze(0)

            # use the selected solver to optimize the hnet parameters
            loss_sol = solver(losses, ray, list(hnet.parameters()))
            loss_sol.backward() # compute gradient wrt to hnet params

            # LibMOON Solver called 
            # loss_sol = comb(losses=losses, ray = ray, params=list(hnet.parameters()))
            # loss_sol.backward()

            desc = (
                f"total: {loss_sol.item():.3f} | "
                f"pred: {L_pred.item():.3f} | "
                f"fair: {L_fair.item():.3f}"
            )

            epoch_iter.set_description(
                desc
                # f", ray {ray.cpu().numpy().tolist()}"
            )

            # For classical optimizers
            # optimizer.step()
            # For VeLO optimizer
            # optimizer.step(loss_sol)
            if optim_type == "adam":
                optimizer.step()
            elif optim_type == "velo":
                optimizer.step(loss_sol)


        if (epoch + 1) % eval_every == 0:
            last_eval = epoch
            if not no_val_eval:
                epoch_results = evaluate(
                    hypernet=hnet,
                    targetnet=net,
                    loader=val_loader,
                    rays=test_rays,
                    device=device,
                )
                val_results[f"epoch_{epoch + 1}"] = epoch_results

            test_epoch_results = evaluate(
                hypernet=hnet,
                targetnet=net,
                loader=test_loader,
                rays=test_rays,
                device=device,
            )
            test_results[f"epoch_{epoch + 1}"] = test_epoch_results

    if epoch != last_eval:
        if not no_val_eval:
            epoch_results = evaluate(
                hypernet=hnet,
                targetnet=net,
                loader=val_loader,
                rays=test_rays,
                device=device,
            )
            val_results[f"epoch_{epoch + 1}"] = epoch_results

        test_epoch_results = evaluate(
            hypernet=hnet,
            targetnet=net,
            loader=test_loader,
            rays=test_rays,
            device=device,
        )
        test_results[f"epoch_{epoch + 1}"] = test_epoch_results

    # Saving the training config & results for later use
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().isoformat()
    with open(Path(out_dir) / f"config_{current_time}.json", "w") as file:
        json.dump({
            "n_epochs": epochs,
            "step": eval_every,
            "learning_rate": lr,
            "weight_decay": wd,
            "alpha": alpha,
            "ray_hidden": hidden_dim,
            "n_rays": n_rays,
            "tasks_ids": tasks_ids,
            "solver_type": solver_type,
            "training_time_seconds": time.time() - start_time,
            "optimizer": optim_type,
            }, file)
    with open(Path(out_dir) / f"val_results_{current_time}.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / f"test_results_{current_time}.json", "w") as file:
        json.dump(test_results, file)

    # save the trained model for future inference
    torch.save(hnet.state_dict(), Path(out_dir) / f"hnet_{current_time}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hypernetwork on Tabular Fairness Dataset")
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=["adult", "compas", "credit"])
    parser.add_argument(
        "--optim",
        type=str,
        choices=["adam", "velo"],
        default="velo",
        help="optimizer type",
    )
    parser.add_argument("--n-epochs", type=int, default=50, help="num. epochs")
    parser.add_argument(
        "--ray-hidden", type=int, default=100, help="lower range for ray"
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for dirichlet")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="train on gpu"
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu device")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--val-size", type=float, default=0.1, help="validation size")
    parser.add_argument(
        "--no-val-eval",
        action="store_true",
        default=False,
        help="evaluate on validation",
    )
    parser.add_argument(
        "--solver", type=str, choices=["ls", "epo"], default="epo", help="solver"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="number of epochs between evaluations",
    )
    parser.add_argument("--out-dir", type=str, default="outputs", help="outputs dir")
    parser.add_argument("--n-rays", type=int, default=25, help="num. rays")
    #(can be removed) for tabular dataset with 2 tasks: prediction and fairness
    parser.add_argument("--tasks-ids", nargs='+', type=int, default=[0, 1], help="list of indices of tasks") 
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    args = parser.parse_args()

    assert args.tasks_ids is not None and len(args.tasks_ids) > 0, "length of tasks_ids must be > 0"

    set_seed(args.seed)
    set_logger()

    train(
        dataset=args.dataset,
        solver_type=args.solver,
        epochs=args.n_epochs,
        hidden_dim=args.ray_hidden,
        lr=args.lr,
        wd=args.wd,
        bs=args.batch_size,
        device=get_device(no_cuda=args.no_cuda, gpus=args.gpus),
        eval_every=args.eval_every,
        no_val_eval=args.no_val_eval,
        val_size=args.val_size,
        n_rays=args.n_rays,
        alpha=args.alpha,
        out_dir=args.out_dir,
        tasks_ids=args.tasks_ids,
        optim_type=args.optim, 
    )

    save_args(folder=args.out_dir, args=args)
