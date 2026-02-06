from libmoon.solver.mobo.methods.pslmobo_solver import PSLMOBOSolver
from libmoon.solver.mobo.methods.dirhvego_solver import DirHVEGOSolver
from libmoon.solver.mobo.methods.psldirhvei_solver import PSLDirHVEISolver
from libmoon.solver.mobo.utils.lhs import lhs

from libmoon.util.prefs import get_uniform_pref
from libmoon.util.problems import get_problem
from libmoon.metrics.metrics import compute_indicators
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import numpy as np
import pickle
import os

problem = get_problem(problem_name='VLMOP2')

#initialization as in their example
n_init = 11*problem.n_var-1
x_init = torch.from_numpy(lhs(problem.n_var, samples=n_init))

solver = DirHVEGOSolver(problem, x_init, MAX_FE=200, BATCH_SIZE=5)
res = solver.solve()

save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

# Saving pickle name
pickle_file = os.path.join(save_dir, 'res_dirhvgo.pkl')

# Save the object
with open(pickle_file, 'wb') as f:
    pickle.dump(res, f)

# Open the saved object
"""
with open('./results/res_dirhvgo.pkl', 'rb') as f:
    res = pickle.load(f)
"""

pareto_front = problem.get_pf(n_pareto_points=2000) 
PF = pareto_front

Y = res['y']
idx_nds = res['idx_nds'][0] if isinstance(res['idx_nds'], list) else res['idx_nds']

plt.figure()
plt.scatter(Y[idx_nds, 0], Y[idx_nds, 1], label='Solutions ND')
plt.plot(PF[:, 0], PF[:, 1], color='red', linewidth=2, label='True PF')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Estimated Pareto front')
plt.legend()
plt.grid(True)
plt.show()