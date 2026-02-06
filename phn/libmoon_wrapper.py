import sys
sys.path
sys.path.append("/Users/macbookpro/Desktop/M2DS/stageMOO/libmoon-enhanced/libmoon")
from libmoon.solver.gradient.methods.epo_solver import EPO_LP, solve_epo
import torch 

class Combiner:
    """Returns a scalar loss given per-task losses (and optionally grads)."""
    requires_grads: bool = False
    def __call__(self, losses: torch.Tensor, ray: torch.Tensor,
                 params: list[torch.nn.Parameter]) -> torch.Tensor:
        raise NotImplementedError

class LSCombiner(Combiner):
    requires_grads = False
    def __call__(self, losses, ray, params):
        return (losses * ray).sum()

class EPOCombiner(Combiner):
    requires_grads = True
    def __init__(self): self._last_shapes = None

    def __call__(self, losses, ray, params):
        # 1) Build Jacobian rows wrt *params* (hypernet or target — your choice)
        grads = []
        for li in losses:
            g = torch.autograd.grad(li, params, retain_graph=True)
            grads.append(torch.cat([gi.reshape(-1) for gi in g]).detach())
        G = torch.stack(grads)  # [m, n]

        # 2) LibMOON’s EPO LP expects r = 1/ray
        r = (1.0 / ray.clamp_min(1e-8)).detach().cpu().numpy()
        epo_lp = EPO_LP(m=G.shape[0], n=G.shape[1], r=r)

        # 3) Solve for alpha, return scalar weighted loss
        _, alpha_np = solve_epo(G, losses, pref=ray, epo_lp=epo_lp)
        alpha = torch.as_tensor(alpha_np, dtype=losses.dtype, device=losses.device)
        return torch.sum(alpha * losses)

def make_combiner(name: str) -> Combiner:
    name = name.lower()
    if name in ["ls", "linear", "agg-ls"]: return LSCombiner()
    if name in ["epo", "libmoon-epo"]:    return EPOCombiner()
    # add MGDA, NashMTL, PCGrad adapters later with same signature
    raise ValueError(f"Unknown combiner {name}")
