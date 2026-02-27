import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.solver.gradient.methods.core.min_norm_solvers_numpy import MinNormSolver  # optional, kept for get_d_moomtl
from libmoon.util.constant import solution_eps

# -------------------- utils --------------------
def _row_normalize(x, axis=1, eps=1e-20):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def _project_to_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    v = v.ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * (np.arange(1, n + 1)) > (cssv - z))[0]
    if rho_idx.size == 0:
        return np.ones(n) * (z / n)
    rho = rho_idx[-1]
    theta = (cssv[rho] - z) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return np.ones(n) * (z / n) if (not np.isfinite(s) or s <= 0) else (w / s)

def solve_mgda_simplex(V: torch.Tensor,
                       eps: float = 1e-12,
                       ridge: float = 1e-9,
                       max_iter: int = 500) -> np.ndarray:
    """
    Minimize 0.5 * a^T (V V^T) a  s.t.  a >= 0, 1^T a = 1
    via accelerated projected gradient (FISTA). Returns numpy (k,) weights.
    """
    V = V.detach().cpu().double()
    if V.ndim != 2:
        V = V.reshape(V.shape[0], -1)
    k, _ = V.shape
    if k == 1:
        return np.array([1.0], dtype=np.float64)

    norms = V.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    Vn = V / norms

    Q = (Vn @ Vn.t()).numpy()
    Q = 0.5 * (Q + Q.T) + ridge * np.eye(k)

    try:
        L = float(np.max(np.linalg.eigvalsh(Q)))
    except Exception:
        L = None
    if (L is None) or (not np.isfinite(L)) or (L <= 0):
        return np.ones(k, dtype=np.float64) / k
    step = 1.0 / L

    a = np.ones(k, dtype=np.float64) / k
    y = a.copy()
    t = 1.0

    def f(x): return 0.5 * float(x @ (Q @ x))

    prev = f(a)
    for _ in range(max_iter):
        g = Q @ y
        a_next = _project_to_simplex(y - step * g)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = a_next + ((t - 1.0) / t_next) * (a_next - a)
        t = t_next

        curr = f(a_next)
        if not np.isfinite(curr):
            a = np.ones(k, dtype=np.float64) / k
            break
        if np.linalg.norm(a_next - a) < 1e-8 or abs(prev - curr) < 1e-10:
            a = a_next
            break
        a = a_next
        prev = curr

    a = np.clip(a, 0.0, 1.0)
    s = a.sum()
    return (np.ones(k) / k) if (s <= 0 or not np.isfinite(s)) else (a / s)

# -------------------- PMTL core helpers --------------------
def get_d_moomtl(grads):
    """
    calculate the gradient direction for MOO-MTL (kept from your original)
    """
    nobj, dim = grads.shape
    sol, nd = MinNormSolver.find_min_norm_element(grads)
    return sol

def get_d_paretomtl(grads, value, weights, i):
    """
    Calculate ParetoMTL direction as linear scalarization weights (no CVXOPT).
    Inputs:
      grads   : (nobj, dim) np.ndarray
      value   : (nobj,)     np.ndarray
      weights : (n_prob, nobj) np.ndarray
      i       : current sample index
    Returns:
      weight  : (nobj,) np.ndarray on the simplex
    """
    nobj, dim = grads.shape

    w_i = weights[i]
    ncw = w_i / (np.linalg.norm(w_i) + 1e-20)
    rest = np.delete(weights, i, axis=0)
    nrest = _row_normalize(rest, axis=1)

    w = nrest - ncw  # (n_prob-1, nobj)

    vnorm = np.linalg.norm(value) + 1e-20
    gx = (w @ (value / vnorm))
    idx = gx > 0

    if np.any(idx):
        Wg = w[idx] @ grads    # (k_act, dim)
        vec = np.concatenate([grads, Wg], axis=0)
    else:
        vec = grads

    # sanitize rows
    mask_finite = np.isfinite(vec).all(axis=1)
    vec = vec[mask_finite]
    if vec.shape[0] == 0:
        return np.ones(nobj) / nobj
    row_norms = np.linalg.norm(vec, axis=1)
    vec = vec[row_norms > 1e-12]
    if vec.shape[0] == 0:
        return np.ones(nobj) / nobj

    sol_np = solve_mgda_simplex(torch.tensor(vec, dtype=torch.float32))

    k_act = int(np.sum(idx))
    if k_act > 0:
        base  = sol_np[:nobj]
        extra = sol_np[nobj:]
        w_idx = w[idx]                     # (k_act, nobj)
        weight = base + (w_idx.T @ extra)  # (nobj,)
    else:
        weight = sol_np[:nobj]

    weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    s = weight.sum()
    return (np.ones(nobj) / nobj) if s <= 1e-12 else (weight / s)

def get_d_paretomtl_init(grads, value, weights, i):
    """
    Initialization stage for ParetoMTL (no CVXOPT).
    Returns:
      (weight, finish)
    """
    nobj, dim = grads.shape

    w_i = weights[i]
    ncw = w_i / (np.linalg.norm(w_i) + 1e-20)
    rest = np.delete(weights, i, axis=0)
    nrest = _row_normalize(rest, axis=1)

    w_diff = nrest - ncw
    vnorm = np.linalg.norm(value) + 1e-20
    gx = w_diff @ (value / vnorm)
    idx = gx > 0
    if np.sum(idx) <= 0:
        return np.zeros(nobj), True

    if np.sum(idx) == 1:
        sol_np = np.ones(1, dtype=np.float64)
    else:
        vecs = w_diff[idx] @ grads
        # sanitize
        mask_finite = np.isfinite(vecs).all(axis=1)
        vecs = vecs[mask_finite]
        if vecs.shape[0] == 0:
            return np.zeros(nobj), True
        vecs = vecs[np.linalg.norm(vecs, axis=1) > 1e-12]
        if vecs.shape[0] == 0:
            return np.zeros(nobj), True
        sol_np = solve_mgda_simplex(torch.tensor(vecs, dtype=torch.float32))

    w_index = w_diff[idx]  # (k_act, nobj)
    weight0 = float(sol_np @ w_index[:, 0])
    weight1 = float(sol_np @ w_index[:, 1])
    weight  = np.array([weight0, weight1], dtype=np.float64)
    return weight, False

# -------------------- PMTL Core & Solver --------------------
class PMTLCore():
    def __init__(self, n_obj, n_var, n_epoch, prefs):
        """
        prefs: (n_prob, n_obj)
        """
        self.core_name = 'PMTLCore'
        self.n_obj, self.n_var = n_obj, n_var
        self.n_epoch = n_epoch
        self.warmup_epoch = n_epoch // 5
        self.prefs_np = prefs.numpy() if isinstance(prefs, torch.Tensor) else prefs
        self.stage1_finish = False
        self.has_print = False

    def get_alpha_array(self, Jacobian_array, losses, epoch_idx):
        """
        Jacobian_array: (n_prob, n_obj, n_var) torch.Tensor
        losses:         (n_prob, n_obj)        torch.Tensor or np.ndarray
        returns:        (n_prob, n_obj)        torch.Tensor (same device as Jacobian_array)
        """
        losses_np = losses.detach().cpu().numpy() if isinstance(losses, torch.Tensor) else losses
        Jacobian_array_np = Jacobian_array.detach().cpu().numpy()
        n_prob = losses_np.shape[0]

        if not self.stage1_finish:
            res = [get_d_paretomtl_init(Jacobian_array_np[i], losses_np[i], self.prefs_np, i)
                   for i in range(n_prob)]
            weights = [res[i][0] for i in range(n_prob)]
            self.stage1_finish = all([res[i][1] for i in range(n_prob)])
        else:
            if not self.has_print:
                print('Begin second stage')
                self.has_print = True
            weights = [get_d_paretomtl(Jacobian_array_np[i], losses_np[i], self.prefs_np, i)
                       for i in range(n_prob)]

        return torch.tensor(np.array(weights), dtype=torch.float32, device=Jacobian_array.device)

class PMTLSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size, n_epoch, tol, folder_name=None):
        self.solver_name = 'PMTL'
        self.problem = problem
        self.prefs = prefs
        self.folder_name = folder_name
        self.core_solver = PMTLCore(n_obj=problem.n_obj,
                                    n_var=problem.n_var,
                                    n_epoch=n_epoch,
                                    prefs=prefs)
        self.warmup_epoch = n_epoch // 5
        super().__init__(step_size, n_epoch, tol, self.core_solver)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)
