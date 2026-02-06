import torch
import argparse
from matplotlib import pyplot as plt
import pickle
from libmoon.solver.gradient.methods.core.core_mtl import TabPSLSolver, GradBasePSLMTLSolver
import os
from libmoon.util.mtl import get_mtl_prefs
import numpy as np
import matplotlib.pyplot as plt
from libmoon.metrics.metrics import compute_indicators

def save_pickle(folder_name, res):
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))


def plot_fig_last_loss(loss, prefs):
    prefs_unit = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    rho = np.max([np.linalg.norm(elem) for elem in loss])
    for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, loss)):
        plt.plot([0, rho*p[0]], [0, rho*p[1]], '--', linewidth=1, label=f'p{i}')
        plt.scatter(l1, l2)
    plt.xlabel('$L_1$'); plt.ylabel('$L_2$')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.show()

def plot_loss_by_objective(res):
    # Convert to numpy arrays
    agg = res['loss_history']
    vec = res['loss_vec_history']
    
    n_epochs = len(agg)
    n_obj = len(vec[0])
    epochs = np.arange(1, n_epochs+1)

    fig, axes = plt.subplots(1, n_obj, figsize=(6 * n_obj, 5), sharex=True)

    if n_obj == 1:
        axes = [axes]

    for obj_idx in range(n_obj):
        ax = axes[obj_idx]
        ax.plot(epochs, vec[:, obj_idx], label=f'Loss $L_{{{obj_idx+1}}}$')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Objectif {obj_idx+1}')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_aggregated_loss(res):
    agg = res['loss_history']
    n_epochs = len(agg)
    epochs = np.arange(1, n_epochs+1)

    # Plot aggregated loss across all preferences
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, agg)
    plt.xlabel('Epoch')
    plt.ylabel('Aggregated Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    
    solver_pickle_file = os.path.join(save_dir, 'fmnist_solver_40_epoch_agg_LS.pkl')
    res_training_pickle_file = os.path.join(save_dir, 'fmnist_res_training_40_epoch_agg_LS.pkl')
    res_eval_pickle_file = os.path.join(save_dir, 'fmnist_res_eval_40_epoch_agg_LS.pkl')

    epoch = 200
    batch_size = 128
    step_size = 1e-4
    n_obj = 2
    solver_name = 'agg_LS'
    # problem_name: ['mnist', 'fashion', 'fmnist', 'adult', 'credit', 'compass']
    problem_name = 'credit'
    device = 'mps'


    print('Device:{}'.format(device))
    print('Running MTL PSL {} on {}'.format(solver_name, problem_name))
    
    prefs = get_mtl_prefs(problem_name=problem_name, n_prob=10)
    
    device = torch.device(device) if device in ['mps', 'cuda'] else torch.device('cpu')
    if problem_name in  ['adult', 'credit','compas'] :
        solver = TabPSLSolver(problem_name=problem_name, batch_size=batch_size,
                                    step_size=step_size, epoch=epoch, device=device, solver_name=solver_name)
        solver, res = solver.solve()
        eval = solver.eval(prefs=prefs)
    elif problem_name in ['mnist', 'fmnist', 'fashion'] :
        solver = GradBasePSLMTLSolver(problem_name=problem_name, batch_size=batch_size,
                                    step_size=step_size, epoch=epoch, device=device, solver_name=solver_name)
        res = solver.solve()
        eval = solver.eval(n_eval=10)

    
    #save the model/sovler
    with open(solver_pickle_file, 'wb') as f:
        pickle.dump(solver, f)

    #save the results of training (loss)
    with open(res_training_pickle_file, 'wb') as f:
        pickle.dump(res, f)

    #save the results of eval (loss)
    with open(res_eval_pickle_file, 'wb') as f:
        pickle.dump(eval, f)


    indicators = compute_indicators(objs=eval['eval_loss'],prefs=prefs, problem_name='credit')

    for name, value in indicators.items():
        print(f"{name}: {value:.4f}")

    plot_loss_by_objective(res)
    plot_aggregated_loss(res)
    plot_fig_last_loss(eval['eval_loss'], eval['prefs'])
