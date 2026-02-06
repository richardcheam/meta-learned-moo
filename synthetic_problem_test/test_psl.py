from libmoon.solver.psl.core_psl import BasePSLSolver
from libmoon.util.prefs import get_uniform_pref
from libmoon.util.problems import get_problem
from libmoon.metrics.metrics import compute_indicators
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import numpy as np
problem = get_problem(problem_name='VLMOP2')
prefs = get_uniform_pref(n_prob=10, n_obj=problem.n_obj, clip_eps=1e-2)

#[epo, pmgda, agg_LS, agg_Tche, agg_PBI, agg_COSMOS, agg_STche, agg_AASF, agg_PNorm, agg_mTche, agg_invagg, agg_SmTche]
solver = BasePSLSolver(problem, solver_name='epo', device='mps', use_es=False)
model, _ = solver.solve()

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
prefs_tensor = Tensor(prefs).to(device)
model = model.to(device)
eval_y = problem.evaluate(model(prefs_tensor))

pareto_front = problem.get_pf(n_pareto_points=2000) 

Y =  eval_y.detach().cpu().numpy()            
PF = pareto_front
PREFS = prefs

indicators = compute_indicators(objs=Y,prefs=prefs, problem_name='VLMOP2')

for name, value in indicators.items():
    print(f"{name}: {value:.4f}")

PREFS_NORM = PREFS / np.linalg.norm(PREFS, axis=1, keepdims=True)
PREFS_NORM = 0.4 * PREFS_NORM    

# Preferences plot
plt.scatter(PREFS_NORM[:, 0], PREFS_NORM[:, 1], color='blue', label='Preference')

# Lines linking preferences to their solution
for i in range(len(Y)):
    plt.plot([PREFS_NORM[i, 0], Y[i, 0]], [PREFS_NORM[i, 1], Y[i, 1]], color='red', linestyle=':', linewidth=0.8)


# Plot of solutions
plt.scatter(Y[:, 0], Y[:, 1], color='orange', label='Solutions')

# Theoretical Pareto front
plt.plot(PF[:, 0], PF[:, 1], color='red', linewidth=2, label='True PF')

plt.xlabel(r'$f_1$', fontsize=14)
plt.ylabel(r'$f_2$', fontsize=14)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.tight_layout()
plt.show()

