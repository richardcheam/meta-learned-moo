import sys
sys.path
sys.path.append("../../")

from experiments.dsprites.data import Dataset
from experiments.dsprites.models import (
    LeNetHyper,
    LeNetTarget,
    ImageSize
)

import argparse

import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split

KERNEL_SIZE = [9, 5]

def get_config():
    parser = argparse.ArgumentParser(description="run dSprites model")
    parser.add_argument("--modelpath",
                        type=str,
                        help="path to the model file (.pt)")
    parser.add_argument("--datapath",
                        type=str,
                        default='data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                        help="path to data to be tested")
    parser.add_argument(
        "--ray-hidden", type=int, default=100, help="lower range for ray"
    )
    parser.add_argument("--tasks-ids", nargs='+', type=int, default=[2, 3], help="list of indices of tasks")
    parser.add_argument("--out-dim", nargs='+', type=int, default=[6, 40], help="list of the number of classes of each task")
    parser.add_argument("--img-size", type=str, choices=['64', '32'], default='64', help="size of the images: either original 64x64 or downsized 32x32")
    args = parser.parse_args()
    assert args.out_dim is not None and len(args.out_dim) > 0, "length of out_dim must be > 0"
    assert args.tasks_ids is not None and len(args.tasks_ids) > 0, "length of tasks_ids must be > 0"
    assert len(args.out_dim) == len(args.tasks_ids), "length of out_dim and tasks_ids must be the same"
    return args

def run(config: dict):
    device = torch.device("cuda:0") if torch.cuda.is_available else "cpu"

    data = np.load(config.datapath)
    X = data['imgs']
    Y = data['latents_classes'][:, config.tasks_ids]
    X, _, Y, _ = train_test_split(X, Y, test_size=0.7, random_state=12)
    n_data = len(X)

    X = torch.from_numpy(X.reshape(n_data, 1, 64, 64)).float()
    Y = torch.from_numpy(Y).long()
    test_set = torch.utils.data.TensorDataset(X, Y)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=128, shuffle=True, num_workers=4
    )

    img_size = ImageSize.IMG64x64 if config.img_size == '64' else ImageSize.IMG32x32
    n_tasks = len(config.tasks_ids)
    hnet = LeNetHyper(KERNEL_SIZE, ray_hidden_dim=config.ray_hidden, out_dim=config.out_dim, n_tasks=n_tasks,
                      img_size=img_size)
    hnet.load_state_dict(torch.load(config.modelpath, weights_only=True))
    net = LeNetTarget(KERNEL_SIZE, out_dim=config.out_dim, n_tasks=n_tasks, img_size=img_size)
    
    hnet.to(device)
    net.to(device)

    hnet.eval()
    loss = [nn.CrossEntropyLoss()] * n_tasks

    rays = [[1.0 / n_tasks] * n_tasks]
    for i in range(n_tasks):
        rays.append([1.0 - (n_tasks - 1) * 0.001 if x == i else (n_tasks - 1) * 0.001 for x in range(n_tasks)])

    rays = np.array(rays)

    with torch.no_grad():
        for ray in rays:
            total = 0.0
            task_correct = [0.0] * n_tasks
            l = [0.0] * n_tasks
            ray = torch.from_numpy(ray.astype(np.float32)).to(device)
            ray /= ray.sum()
            curr_l = [0.0] * n_tasks
            pred = [0.0] * n_tasks

            for batch in test_loader:

                batch = (t.to(device) for t in batch)
                img, ys = batch
                bs = len(ys)

                weights = hnet(ray)
                logits = net(img, weights)

                # loss
                for i in range(n_tasks):
                    curr_l[i] = loss[i](logits[i], ys[:,i])
                    l[i] += curr_l[i] * bs
                    pred[i] = logits[i].data.max(1)[1]
                    task_correct[i] += pred[i].eq(ys[:,i]).sum()

                total += bs

            print("ray", ray.squeeze(0).cpu().numpy().tolist())
            for i in range(n_tasks):
                print(f"task{i}_acc:", task_correct[i].cpu().item() / total)
                print(f"task{i}_loss", l[i].cpu().item() / total)


if __name__ == "__main__":
    config = get_config()
    run(config)
