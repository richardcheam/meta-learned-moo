#It wasn't present in libmoon, so I added it.

from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.util.constant import root_name
import os
from torch.optim import SGD
from tqdm import tqdm
from pymoo.indicators.hv import HV
import math
import torch
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch import Tensor
from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.util.constant import solution_eps, get_hv_ref

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCGradCore():
    def __init__(self, n_var, prefs):
        '''
            Input:
            n_var: int, number of variables.
            prefs: (n_prob, n_obj).
        '''
        self.core_name = 'PCGradCore'
        self.prefs = prefs
        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        self.n_var = n_var

    def get_shared_gradients(self, Jacobian):
        '''
            Input:
            Jacobian: (n_obj, n_var), torch.Tensor
            losses: (n_obj,), torch.Tensor
        '''
        obj = self.n_obj
        pc_grads = []
        for i in range(obj):
            g_i = Jacobian[i].clone()

            Jacobian_without_i = torch.cat((Jacobian[:i], Jacobian[i+1:]), dim=0)

            indices = torch.randperm(Jacobian_without_i.size(0))
            Jacobian_shuffle = Jacobian_without_i[indices]

            for g_j in Jacobian_shuffle:
                # inner product g_i·g_j
                g_i_g_j = torch.dot(g_i, g_j)

                if g_i_g_j < 0:
                    g_j_norm_square = g_j.norm()**2
                    coeff = g_i_g_j / (g_j_norm_square)
                    g_i = g_i - coeff * g_j

            pc_grads.append(g_i)

        return torch.stack(pc_grads, dim=0).sum(dim=0)


class PCGradSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3,
                 sigma=0.1, h_tol=1e-3, folder_name=None):
        self.folder_name=folder_name
        self.problem = problem
        self.sigma = sigma
        self.h_tol = h_tol
        self.n_epoch = n_epoch
        self.pcgrad_core = PCGradCore(n_var=problem.n_var, prefs=prefs)
        self.prefs = prefs
        self.solver_name = 'PMGDA'
        super().__init__(step_size, n_epoch, tol, self.pcgrad_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)
