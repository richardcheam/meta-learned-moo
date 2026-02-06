#This code is used to analyse the results saved in our pickels
import sys
sys.path
sys.path.append("/Users/macbookpro/Desktop/M2DS/stageMOO/libmoon-enhanced/libmoon")

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from libmoon.metrics.metrics import compute_indicators
from libmoon.util.mtl import get_mtl_prefs
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#Plot whith a zoom if needed : 
"""
def plot_fig_loss(last_loss, prefs, problem, method):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()  # Récupère l'axe principal

    prefs_unit = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    
    # Main layout
    for loss in last_loss:
        rho = np.max([np.linalg.norm(elem) for elem in loss])
        for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, loss)):
            ax.plot([0, rho * p[0]], [0, rho * p[1]], '--', linewidth=1)
            ax.scatter(l1, l2, alpha=0.2)

    # Average
    rho = np.max([np.linalg.norm(elem) for elem in last_loss.mean(axis=0)])
    for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, last_loss.mean(axis=0))):
        ax.plot([0, rho * p[0]], [0, rho * p[1]], '--', linewidth=1, label=f'p{i}')
        ax.scatter(l1, l2, alpha=1)

    # Zoom
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    
    # Re-trace elements in the zoom
    for loss in last_loss:
        rho = np.max([np.linalg.norm(elem) for elem in loss])
        for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, loss)):
            axins.plot([0, rho * p[0]], [0, rho * p[1]], '--', linewidth=1)
            axins.scatter(l1, l2, alpha=0.2)

    rho = np.max([np.linalg.norm(elem) for elem in last_loss.mean(axis=0)])
    for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, last_loss.mean(axis=0))):
        axins.plot([0, rho * p[0]], [0, rho * p[1]], '--', linewidth=1)
        axins.scatter(l1, l2, alpha=1)

    # Define zoom area
    axins.set_xlim(0.0, 1)
    axins.set_ylim(0.0, 1)
    axins.grid(True)

    ax.grid(True)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./Resultats/MultiFashion/Last_loss_MultiFashion_{method}.png')
"""

def plot_fig_loss(last_loss, prefs, problem, method):
    plt.figure(figsize=(12, 10))
    prefs_unit = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    for loss in last_loss :
        rho = np.max([np.linalg.norm(elem) for elem in loss])
        for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, loss)):
            plt.plot([0, rho*p[0]], [0, rho*p[1]], '--', linewidth=1)
            plt.scatter(l1, l2, alpha= 0.2)

    rho = np.max([np.linalg.norm(elem) for elem in last_loss.mean(axis=0)])
    for i, (p, (l1, l2)) in enumerate(zip(prefs_unit, last_loss.mean(axis=0))):
        plt.plot([0, rho*p[0]], [0, rho*p[1]], '--', linewidth=1, label=f'p{i}')
        plt.scatter(l1, l2, alpha= 1)

    plt.xlabel('$L_1$'); plt.ylabel('$L_2$')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    #plt.show()
    plt.savefig(f'results/{problem}/{method}/last_loss_{problem}_{method}.png')


def plot_fig_history_loss(loss_history, n_prob, problem, method):
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
    #plt.show()
    plt.savefig(f'results/{problem}/{method}/each_loss_{problem}_{method}.png')



indicators_all = {'hv':[], 'spacing':[], 'span':[],'lmin':[],'soft_lmin':[], 'FD' :[],'pbi':[], 
                   'inner_product':[], 'cross_angle':[],'sparsity':[]}

last_loss = []
loss_per_epoch = []

# results_dir = '/multitask_learning_problem_test/results'
#Agg - [LS, Tche, PBI, COSMOS, STche, AASF, PNorm, mTche, invagg, SmTche]

#PNorm, mTche, invagg
results_dir = 'results'
problem_name = 'electricity_demand'
method = 'pmtl-velo'

dossier = f"{results_dir}/{problem_name}/{method}"

for fichier in os.listdir(dossier):
    if fichier.endswith(".pkl"):
        chemin_fichier = os.path.join(dossier, fichier)
        with open(chemin_fichier, 'rb') as f:
            res = pickle.load(f)

        n_prob = 10
        prefs = get_mtl_prefs(problem_name=problem_name, n_prob=n_prob)

        loss = res['loss']
        loss_history = res['loss_history']

        last_loss.append(loss)
        loss_per_epoch.append(loss_history)

        indicators = compute_indicators(objs=loss,prefs=prefs, problem_name=problem_name)

        for name, value in indicators.items():
            indicators_all[name].append(value)
            #print(f"{name}: {value:.4f}")

print_res = ''
for keys in indicators_all.keys() :
     metrics = np.array(indicators_all[keys])
     metrics_mean = metrics.mean()
     metrics_std = metrics.std(ddof=1)
     standard_error = metrics_std / np.sqrt(len(metrics))
     print(f'{keys}, mean : {metrics_mean:.4f} ± {standard_error:.4f}' )
     print_res += f'& {metrics_mean:.4f} ± {standard_error:.4f} '

print(print_res)

loss_per_epoch = np.array(loss_per_epoch)
last_loss = np.array(last_loss)

loss_history = loss_per_epoch.mean(axis=0)
print(loss_history.shape)

plot_fig_history_loss(loss_history, n_prob, problem_name, method)
plot_fig_loss(last_loss, prefs,problem_name, method)