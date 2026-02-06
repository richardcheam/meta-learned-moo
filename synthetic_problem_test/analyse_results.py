import sys
sys.path
sys.path.append("/Users/macbookpro/Desktop/M2DS/stageMOO/libmoon-enhanced/libmoon")

from libmoon.util.synthetic import synthetic_init
from libmoon.util.prefs import get_uniform_pref
from libmoon.util.problems import get_problem
from libmoon.solver.gradient.methods import EPOSolver, MGDAUBSolver, RandomSolver, PMGDASolver, MOOSVGDSolver, PMTLSolver, GradHVSolver, GradAggSolver, PCGradSolver, NashMTLSolver
from libmoon.solver.gradient.methods.velo_solver import VeLOSolver
from libmoon.metrics.metrics import compute_indicators
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

indicators_all = {'hv':[], 'spacing':[], 'span':[],'lmin':[],'soft_lmin':[], 'FD' :[],'pbi':[], 
                   'inner_product':[], 'cross_angle':[],'sparsity':[]}

objs = []

_solver = 'velo'
_problem = 'ZDT4'
_scalarization = 'invagg'

for k in range(5) : 

    save_dir = f'./results/{_problem}/{_solver}-{_scalarization}'
    os.makedirs(save_dir, exist_ok=True)
    # make figures dir
    os.makedirs(f'./results/{_problem}/figures', exist_ok=True)

    pickle_file = os.path.join(f'./results/{_problem}/{_solver}-{_scalarization}', f'res_{_problem}_{_solver}-{_scalarization}_{k}.pkl')

    problem = get_problem(problem_name=_problem)
    prefs = get_uniform_pref(n_prob=10, n_obj=problem.n_obj, clip_eps=1e-2)

    #Save the object res
    with open(pickle_file, 'rb') as f:
        res = pickle.load(f)

    Y = res['y'] 

    indicators = compute_indicators(objs=Y, prefs=prefs, problem_name=_problem, pareto_points=problem.get_pf(n_pareto_points=2000))

    for name, value in indicators.items():
        indicators_all[name].append(value)
        #print(f"{name}: {value:.4f}")

    objs.append(Y)


for keys in indicators_all.keys() :
    metrics = np.array(indicators_all[keys])
    metrics_mean = metrics.mean()
    metrics_std = metrics.std(ddof=1)
    standard_error = metrics_std / np.sqrt(len(metrics))
    print(f'{keys}, mean : {metrics_mean:.4f} ± {standard_error:.4f}' )


pareto_front = problem.get_pf(n_pareto_points=2000) 
            
PF = pareto_front
PREFS = prefs
PREFS_NORM = PREFS / np.linalg.norm(PREFS, axis=1, keepdims=True)
PREFS_NORM_4 = 0.4 * PREFS_NORM

"""
#History of the last run
HISTORY_Y = np.array(res['y_history'])
for k in range(len(Y)):
    plt.plot(HISTORY_Y[:,k,0], HISTORY_Y[:,k,1], alpha = 0.3)
"""

"""
# Preferences plot
for k in range(len(PREFS_NORM_4)):
    if k == 0:  # Add caption only for the first point
        plt.scatter(PREFS_NORM_4[k, 0], PREFS_NORM_4[k, 1], label='Preference', marker='+')
    else:
        plt.scatter(PREFS_NORM_4[k, 0], PREFS_NORM_4[k, 1], marker='+')


# Lines linking preferences to their solution
for i in range(len(Y)):
    plt.plot([PREFS_NORM_4[i, 0], Y[i, 0]], [PREFS_NORM_4[i, 1], Y[i, 1]], color='red', linestyle=':', linewidth=0.8)
""" 

fig = plt.figure()

rho = np.max([np.linalg.norm(elem) for elem in Y])
for i, (p, (l1, l2)) in enumerate(zip(PREFS_NORM, Y)):
    if i ==0 : 
        plt.plot([0, rho*p[0]], [0, rho*p[1]], '--', linewidth=1, label='Preference')
    else : 
        plt.plot([0, rho*p[0]], [0, rho*p[1]], '--', linewidth=1)


# Plot of solutions (for all the runs)
for i, obj in enumerate(objs) :
    for k in range(len(obj)):
        if (k,i) == (0,0) :  # Add caption only for the first point
            plt.scatter(obj[k, 0], obj[k, 1], alpha=0.15, label='Solutions')
        else : 
            plt.scatter(obj[k, 0], obj[k, 1], alpha=0.15)

# Plot of solutions (average)
obj_average = np.mean(objs, axis=0)
for k in range(len(obj_average)):
    if k == 0:  # Add caption only for the first point
        plt.scatter(obj_average[k, 0], obj_average[k, 1], alpha=1, label='Average')
    else : 
        plt.scatter(obj_average[k, 0], obj_average[k, 1], alpha=1)


# Theoretical Pareto front
plt.scatter(PF[:, 0], PF[:, 1], color='red', s=2, label='True PF')

plt.xlabel(r'$f_1$', fontsize=14)
plt.ylabel(r'$f_2$', fontsize=14)
plt.legend()
plt.grid(True)
#plt.axis('equal')
plt.xlim(-0.2, 6)
plt.ylim(-0.2, 8)
#plt.tight_layout()
plt.show()
# fig.savefig(f'results/{_problem}/figures/{_solver}-{_scalarization}/fig_{_problem}_{_solver}-{_scalarization}.png')
fig.savefig(f'results/{_problem}/figures/fig_{_problem}_{_solver}-{_scalarization}.png')