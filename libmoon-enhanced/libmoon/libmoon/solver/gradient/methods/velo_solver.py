import torch
from torch.autograd import Variable
from pymoo.indicators.hv import HV
from libmoon.util.constant import get_hv_ref, solution_eps
from libmoon.util.prefs import pref2angle
from pylo import VeLO
import numpy as np

class VeLOSolver:
    def __init__(self, problem, prefs, n_epoch=500, scalarization='linear'):
        self.epoch = n_epoch
        self.problem = problem
        self.prefs = prefs
        self.scalarization = scalarization.lower()
        assert self.scalarization in ['linear', 'tchebycheff'], "Only 'linear' or 'tchebycheff' supported"

    def solve(self, x_init):
        n_prob, n_obj = self.prefs.shape
        xs_var = Variable(x_init, requires_grad=True)
        optimizer = VeLO([xs_var], lr=1.0)

        ind = HV(ref_point=get_hv_ref(self.problem.problem_name))
        hv_arr, y_arr = [], []

        self.prefs_tensor = torch.Tensor(self.prefs).to(x_init.device)
        ideal_point = torch.zeros(n_obj, device=x_init.device)  # Default ideal point

        for epoch in range(self.epoch):
            fs_var = self.problem.evaluate(xs_var)  # shape: [n_prob, n_obj]
            y_np = fs_var.detach().cpu().numpy()
            y_arr.append(y_np)
            hv_arr.append(ind.do(y_np))

            optimizer.zero_grad()

            if self.scalarization == 'linear':
                # Linear scalarization: weighted sum
                scalarized_losses = torch.sum(self.prefs_tensor * fs_var, dim=1)
                loss = torch.mean(scalarized_losses)

            elif self.scalarization == 'tchebycheff':
                # Tchebycheff scalarization (maximize weighted max deviation from ideal point)
                diff = torch.abs(fs_var - ideal_point)
                scalarized_losses = torch.max(self.prefs_tensor * diff, dim=1).values
                loss = torch.mean(scalarized_losses)

            loss.backward()
            optimizer.step(loss)

            # Constraints
            if hasattr(self.problem, 'lbound'):
                xs_var.data = torch.clamp(xs_var.data,
                                          torch.Tensor(self.problem.lbound).to(xs_var.device) + solution_eps,
                                          torch.Tensor(self.problem.ubound).to(xs_var.device) - solution_eps)
            if self.problem.problem_name in ['MOKL']:
                xs_var.data = torch.clamp(xs_var.data, min=0)
                xs_var.data = xs_var.data / torch.sum(xs_var.data, dim=1, keepdim=True)

        return {
            'x': xs_var.detach().cpu().numpy(),
            'y': y_np,
            'hv_history': hv_arr,
            'y_history': y_arr
        }

