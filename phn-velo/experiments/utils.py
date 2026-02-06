import json
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def save_args(folder, args, name="config.json", check_exists=False):
    set_logger()
    path = Path(folder)
    if check_exists:
        if path.exists():
            logging.warning(f"folder {folder} already exists! old files might be lost.")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(vars(args), open(path / name, "w"))


def circle_points(K, n_tasks: int=2, min_angle=None, max_angle=None):
    if (n_tasks == 2):
        # generate evenly distributed preference vector
        ang0 = 1e-6 if min_angle is None else min_angle
        ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
        angles = np.linspace(ang0, ang1, K, endpoint=True)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.c_[x, y]
    else:
        rays = [[1.0 / n_tasks] * n_tasks]
        for i in range(n_tasks):
            rays.append([1.0 - (n_tasks - 1) * 0.01 if x == i else (n_tasks - 1) * 0.01 for x in range(n_tasks)])
        return np.array(rays)
        
        # Generates points on a sphere surface, but not usable
        # as such now
        # 
        # points = []
        # n_count = 0
        # a = (4 * np.pi) / K
        # d = np.sqrt(a)
        # M_theta = round(np.pi / d)
        # d_theta = np.pi / M_theta
        # d_phi = a / d_theta
        # for m in range(0, M_theta):
        #     theta = np.pi * (m + 0.5) / M_theta
        #     M_phi = round((2 * np.pi * math.sin(theta)) / d_phi)
        #     for n in range(0, M_phi):
        #         phi = 2 * np.pi * n / M_phi
        #         points.append([math.sin(theta) * math.cos(phi),
        #                        math.sin(theta) * math.sin(phi),
        #                        math.cos(theta)])
        #         n_count += 1
        # return points
