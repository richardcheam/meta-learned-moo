import sys
sys.path
sys.path.append("/Users/macbookpro/Desktop/M2DS/stageMOO/libmoon-enhanced/libmoon")

import numpy as np
import torch
from libmoon.util.mtl import get_dataset, model_from_dataset
from libmoon.util.mtl import numel
from libmoon.util.prefs import get_uniform_pref
import os
import pickle
from libmoon.solver.gradient.methods.epo_solver import EPOCore
from libmoon.solver.gradient.methods.mgda_solver import MGDAUBCore
from libmoon.solver.gradient.methods.pmgda_solver import PMGDACore
from libmoon.solver.gradient.methods.moosvgd_solver import MOOSVGDCore
from libmoon.solver.gradient.methods.gradhv_solver import GradHVCore
from libmoon.solver.gradient.methods.pmtl_solver import PMTLCore
from libmoon.solver.gradient.methods.random_solver import RandomCore
from libmoon.solver.gradient.methods.base_solver import AggCore
from libmoon.solver.gradient.methods.pcgrad_solver import PCGradCore
from libmoon.solver.gradient.methods.nashtmtl_solver import NashMTLCore
from libmoon.solver.gradient.methods.core.core_mtl import GradBaseMTLSolver, GradBaseMTLSolverMnist, GradBaseMTLSolverTemporal
from libmoon.metrics.metrics import compute_indicators

from libmoon.solver.gradient.methods.velo_solver import VeLOSolver
from libmoon.solver.gradient.methods.core.core_mtl import VeLOMTLSolver, VeLOMTLSolverMnist

from libmoon.util.mtl import get_mtl_prefs
import os
from matplotlib import pyplot as plt

def plot_fig_loss(loss, prefs):
    prefs_unit = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    rho = np.max([np.linalg.norm(elem) for elem in loss])
    for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, loss)):
        plt.plot([0, rho*p[0]], [0, rho*p[1]], '--', linewidth=1, label=f'p{i}')
        plt.scatter(l1, l2)
    plt.xlabel('$L_1$'); plt.ylabel('$L_2$')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.show()

def plot_fig_history_loss(loss_history, n_prob):
    LH = np.stack(loss_history, axis=0)
    n_epochs, n_prob, n_obj = LH.shape

    plt.figure(figsize=(12, 5))

    for pref_idx in range(n_prob):
        L1 = LH[:, pref_idx, 0]
        L2 = LH[:, pref_idx, 1]
        
        # Subplot for L1
        plt.subplot(1, 2, 1)
        plt.plot(range(n_epochs), L1, label=f'p{pref_idx}')
        plt.title('Evolution of $L_1$ per préférence')
        plt.xlabel('Epoch')
        plt.ylabel('$L_1$')
        plt.legend(fontsize='small', ncol=2)
        plt.grid(True)
        
        # Subplot for L2
        plt.subplot(1, 2, 2)
        plt.plot(range(n_epochs), L2, label=f'p{pref_idx}')
        plt.title('Evolution of $L_2$ per préférence')
        plt.xlabel('Epoch')
        plt.ylabel('$L_2$')
        plt.legend(fontsize='small', ncol=2)
        plt.grid(True)

    plt.tight_layout()

# DATASET = 'adult'
#Agg - [LS, Tche, PBI, COSMOS, STche, AASF, PNorm, mTche, invagg, SmTche]
import sys

DATASET = sys.argv[1] 
SOLVER = sys.argv[2] 

#SOLVER = 'agg_SmTche'
VARIANT = 'velo'

if __name__ == '__main__':

    save_dir = f'./results/{DATASET}/{SOLVER}-{VARIANT}/'
    os.makedirs(save_dir, exist_ok=True)

    for k in range(5):

        # Nom du fichier pickle
        pickle_file = os.path.join(save_dir, f'res_{DATASET}_{SOLVER}-{VARIANT}_{k}.pkl')

        problem_name = DATASET.lower()
        solver_name = SOLVER
        print(f"Trainign with solver: {SOLVER}-{VARIANT}")
        epoch = 100
        step_size = 1e-4
        batch_size = 1024
        n_prob = 10
        device = 'cpu'
        
        args = {"num_classes" : 2, "hidden_dim" : 128 , "seq_length" : 96 } 
        model = model_from_dataset(problem_name, args=args)
        num_param = numel(model)
        print('Number of parameters: {}'.format(num_param))
        prefs = get_mtl_prefs(problem_name=problem_name, n_prob=n_prob)
        print(prefs)
        #prefs = get_uniform_pref(n_prob=n_prob, n_obj=2, clip_eps=1e-2)
        #prefs= torch.Tensor([[1.0, 0.0], [0.0,1.0]])
        #print(prefs)
        device = torch.device(device) if device in ['mps', 'cuda'] else torch.device('cpu')

        if solver_name == 'epo':
            core_solver = EPOCore(n_var=num_param, prefs=prefs)
        elif solver_name == 'mgdaub':
            core_solver = MGDAUBCore()
        elif solver_name == 'random':
            core_solver = RandomCore()
        elif solver_name == 'pmgda':
            core_solver = PMGDACore(n_var=num_param, prefs=prefs)
        elif solver_name.startswith('agg'):
            agg_name = solver_name.split('_')[1]
            core_solver = AggCore(prefs=prefs, agg_name=agg_name)
        elif solver_name == 'moosvgd':
            core_solver = MOOSVGDCore(n_var=num_param, prefs=prefs)
        elif solver_name == 'hvgrad':
            core_solver = GradHVCore(n_obj=2, n_var=num_param, problem_name=problem_name)
        elif solver_name == 'pmtl':
            core_solver = PMTLCore(n_obj=2, n_var=num_param, n_epoch=epoch, prefs=prefs)
        elif solver_name == 'pcgrad':
            core_solver = PCGradCore(n_var=num_param, prefs=prefs)
        elif solver_name == 'nashmtl' : 
            core_solver = NashMTLCore(n_var=num_param, prefs=prefs)
        else:
            assert False, 'Unknown solver'


        if problem_name in ["credit", "adult", "compass"] :
            solver = GradBaseMTLSolver(problem_name=problem_name, step_size=step_size, epoch=epoch, core_solver=core_solver,
                                        batch_size=batch_size, prefs=prefs)
            # solver = VeLOMTLSolver(problem_name=problem_name, prefs=prefs, batch_size=batch_size, epoch=epoch, device=device, scalarization=SCALARIZATION)
        elif problem_name in ["mnist", "fashion", "fmnist", 'dsprites'] :
            solver = GradBaseMTLSolverMnist(problem_name=problem_name, step_size=step_size, epoch=epoch, core_solver=core_solver,
                                       batch_size=batch_size, prefs=prefs, device=device)
            # solver = VeLOMTLSolverMnist(problem_name=problem_name, prefs=prefs, batch_size=batch_size, epoch=epoch, device=device, scalarization=SCALARIZATION)
        elif problem_name in ["electricity_demand"] :
            solver = GradBaseMTLSolverTemporal(problem_name=problem_name, step_size=step_size, epoch=epoch, core_solver=core_solver,
                                        batch_size=batch_size, prefs=prefs, args=args, device=device)
        else :
            assert("Unsupported problem")

        
        res = solver.solve()
        res['prefs'] = prefs
        res['y'] = res['loss']
        loss = res['loss']
        

        #Sauvegarde de l'objet res
        with open(pickle_file, 'wb') as f:
            pickle.dump(res, f)

        """
        with open('./results/res_agg_SmTche.pkl', 'rb') as f:
            res = pickle.load(f)
        """
        
        # loss = res['loss']
        # loss_history = res['loss_history']

        # indicators = compute_indicators(objs=loss,prefs=prefs, problem_name=problem_name)

        # for name, value in indicators.items():
        #     print(f"{name}: {value:.4f}")


        # plot_fig_loss(loss, prefs)
        # plot_fig_history_loss(loss_history, n_prob)

        
