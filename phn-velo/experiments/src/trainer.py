import sys
sys.path
sys.path.append("../../")
sys.path.append("/Users/macbookpro/Desktop/M2DS/stageMOO/libmoon-enhanced/libmoon")
# from libmoon.solver.gradient.methods import EPOSolver, MGDAUBSolver, RandomSolver, PMGDASolver, MOOSVGDSolver, PMTLSolver, GradHVSolver, GradAggSolver, PCGradSolver, NashMTLSolver
# from libmoon.solver.gradient.methods.core.core_mtl import GradBaseMTLSolver
# from libmoon.solver.gradient.methods.epo_solver import EPOCore

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
from experiments.dsprites.models import (
    LeNetHyper,
    LeNetTarget,
    ImageSize
)
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

# the sizes of the first and second convolution layers' kernels
# here we use 9 and 5
KERNEL_SIZE = [9, 5]

@torch.no_grad()
def evaluate(hypernet: LeNetHyper, 
             targetnet: LeNetTarget, 
             loader, rays, device,
             n_tasks: int):
    hypernet.eval()
    loss = [nn.CrossEntropyLoss()] * n_tasks

    results = defaultdict(list)

    for ray in rays:
        total = 0
        task_correct = [0.0] * n_tasks
        l = [0.0] * n_tasks
        # normalize the ray
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray /= ray.sum()
        curr_l = [0.0] * n_tasks
        pred = [0.0] * n_tasks

        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            img, ys = batch
            bs = len(ys)

            # get the weights from hnet given ray,
            # the use these weights to make a prediction
            # on the current batch by giving the weights
            # to net (the target network)
            weights = hypernet(ray)
            logits = targetnet(img, weights)

            # calculate the accuracy and loss for each
            # task and save the results
            for i in range(n_tasks):
                curr_l[i] = loss[i](logits[i], ys[:,i])
                l[i] += curr_l[i] * bs
                pred[i] = logits[i].data.max(1)[1]
                task_correct[i] += pred[i].eq(ys[:,i]).sum()

            # keep the total of images across all batches
            total += bs

        # keep all the accuracy and loss results
        results["ray"].append(ray.squeeze(0).cpu().numpy().tolist())
        for i in range(n_tasks):
            results[f"task{i}_acc"].append(task_correct[i].cpu().item() / total)
            results[f"task{i}_loss"].append(l[i].cpu().item() / total)

    return results



def train(
    path,
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
    out_dim: list[int],
    tasks_ids: list[int],
    img_size: ImageSize
) -> None:
    n_tasks = len(tasks_ids)
    # ----
    # Nets
    # ----
    hnet: nn.Module = LeNetHyper(KERNEL_SIZE, ray_hidden_dim=hidden_dim, out_dim=out_dim, n_tasks=n_tasks, img_size=img_size)
    net: nn.Module = LeNetTarget(KERNEL_SIZE, out_dim=out_dim, n_tasks=n_tasks, img_size=img_size)

    logging.info(f"HN size: {count_parameters(hnet)}")
    print(f"TargetNet size: {count_parameters(net)}")

    hnet = hnet.to(device)
    net = net.to(device)

    # ---------
    # Task loss
    # ---------
    # we use CrossEntropyLoss as it's suited for classification problems
    loss = [nn.CrossEntropyLoss()] * n_tasks

    # optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)
    from pylo.optim import VeLO
    NUM_STEPS=int(597196/bs * epochs)
    print(f"number of train steps: {NUM_STEPS}")
    optimizer = VeLO(hnet.parameters(), lr=1.0, num_steps=NUM_STEPS)

    # ------
    # selecting the solver method
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver)
    comb = make_combiner(solver_type)

    solver_method = solvers[solver_type]
    if solver_type == "epo":
        pref = torch.from_numpy(
                np.random.dirichlet([alpha] * n_tasks, 1).astype(np.float32).flatten()
            )
        solver = solver_method(n_tasks=n_tasks, n_params=count_parameters(hnet))
    else:
        # ls
        solver = solver_method(n_tasks=n_tasks)

    # ----
    # data
    # ----
    assert val_size > 0, "please use validation by providing val_size > 0"
    data = Dataset(path, val_size=val_size, tasks_ids=tasks_ids)
    train_set, val_set, test_set = data.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=4
    )

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

    # losses of the tasks
    l = [0.0] * n_tasks

    start_time = time.time()

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            img, ys = batch
            img = img.to(device)
            ys = ys.to(device)

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
            logits = net(img, weights)
            
            # calculate the loss for each task
            for i in range(n_tasks):
                l[i] = loss[i](logits[i], ys[:,i])

            # turn list l into 1D tensor loss 
            losses = torch.stack(l)

            # removes an extra batch dimension if ray has shape (1, n_tasks) instead of (n_tasks,)
            # we need 1D preference vector
            ray = ray.squeeze(0)

            # use the selected solver to optimize the hnet parameters
            loss_sol = solver(losses, ray, list(hnet.parameters()))
            loss_sol.backward() # compute gradient wrt to hnet params

            #loss_sol = comb(losses=losses, ray = pref, params=list(hnet.parameters()))
            #loss_sol.backward()

            desc = f'total weighted loss: {loss_sol.item():.3f}'
            for i in range(n_tasks):
                desc += f', loss{i}: {l[i].item():.3f}'

            epoch_iter.set_description(
                desc
                # f", ray {ray.cpu().numpy().tolist()}"
            )

            # optimizer.step()
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
                    n_tasks=n_tasks,
                )
                val_results[f"epoch_{epoch + 1}"] = epoch_results

            test_epoch_results = evaluate(
                hypernet=hnet,
                targetnet=net,
                loader=test_loader,
                rays=test_rays,
                device=device,
                n_tasks=n_tasks,
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
                n_tasks=n_tasks,
            )
            val_results[f"epoch_{epoch + 1}"] = epoch_results

        test_epoch_results = evaluate(
            hypernet=hnet,
            targetnet=net,
            loader=test_loader,
            rays=test_rays,
            device=device,
            n_tasks=n_tasks,
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
            "out_dim": out_dim,
            "solver_type": solver_type,
            "img_size": '64' if img_size == ImageSize.IMG64x64 else '32',
            "training_time_seconds": time.time() - start_time,
            }, file)
    with open(Path(out_dir) / f"val_results_{current_time}.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / f"test_results_{current_time}.json", "w") as file:
        json.dump(test_results, file)

    # save the trained model for future inference
    torch.save(hnet.state_dict(), Path(out_dir) / f"hnet_{current_time}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dSprites")
    parser.add_argument(
        "--datapath",
        type=str,
        #default="data/pareto-hypernetworks-master/experiments/dsprites/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        default="data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        help="path to data",
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
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    parser.add_argument("--tasks-ids", nargs='+', type=int, default=[2, 3], help="list of indices of tasks")
    parser.add_argument("--out-dim", nargs='+', type=int, default=[6, 40], help="list of the number of classes of each task")
    parser.add_argument("--img-size", type=str, choices=['64', '32'], default='64', help="size of the images: either original 64x64 or downsized 32x32")
    args = parser.parse_args()

    assert args.out_dim is not None and len(args.out_dim) > 0, "length of out_dim must be > 0"
    assert args.tasks_ids is not None and len(args.tasks_ids) > 0, "length of tasks_ids must be > 0"
    assert len(args.out_dim) == len(args.tasks_ids), "length of out_dim and tasks_ids must be the same"

    set_seed(args.seed)
    set_logger()

    train(
        path=args.datapath,
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
        out_dim=args.out_dim,
        tasks_ids=args.tasks_ids,
        img_size=ImageSize.IMG64x64 if args.img_size == '64' else ImageSize.IMG32x32
    )

    save_args(folder=args.out_dir, args=args)
