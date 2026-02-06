import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.objectives import from_name

from libmoon.util.mtl import model_from_dataset, mtl_dim_dict, mtl_setting_dict
from libmoon.util.mtl import get_dataset
from torch.autograd import Variable

import torch
import numpy as np
from tqdm import tqdm
from libmoon.util.constant import get_agg_func, root_name
from libmoon.util.gradient import calc_gradients_mtl, flatten_grads, calc_gradients_mtl_mnist, calc_gradients_mtl_2_logits, calc_gradients_mtl_2_logits_v2
from libmoon.model.hypernet import HyperNet, LeNetTarget
from libmoon.util.prefs import get_random_prefs, get_uniform_pref
from libmoon.util.mtl import numel
from libmoon.model.simple import SimplePSLModel, SimplePSLModel2
from libmoon.util.gradient import get_moo_Jacobian_batch, get_moo_Jacobian
from libmoon.solver.gradient.methods.epo_solver import EPOCore
from libmoon.solver.gradient.methods.pmgda_solver import PMGDACore
import time

from pylo import VeLO

#Used to solve Images problem  with PSL
class GradBasePSLMTLSolver:
    def __init__(self, problem_name, batch_size, step_size, epoch, device, solver_name):
        self.step_size = step_size
        self.problem_name = problem_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.solver_name = solver_name
        train_dataset = get_dataset(self.problem_name, type='train')
        test_dataset = get_dataset(self.problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=0)
        # For hypernetwork model, we have the hypernet and target network.
        self.hnet = HyperNet(kernel_size=(9, 5)).to(self.device)
        self.net = LeNetTarget(kernel_size=(9, 5)).to(self.device)

        num_param_hnet, num_param_net = numel(self.hnet), numel(self.net)
        print('Number of parameters in hnet: {:.2f}M, net: {:.2f}K'.format(num_param_hnet/1e6, num_param_net/1e3))

        self.optimizer = torch.optim.Adam(self.hnet.parameters(), lr=self.step_size)
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.is_agg = self.solver_name.startswith('agg')
        self.agg_name = self.solver_name.split('_')[-1] if self.is_agg else None

    def solve(self):
        loss_epoch = []
        loss_vec_history = [] 
        for epoch_idx in tqdm(range(self.epoch)):
            loss_batch = []
            loss_vec_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                ray = torch.from_numpy(
                    np.random.dirichlet((1, 1), 1).astype(np.float32).flatten()
                ).to(self.device)  # ray.shape (1,2), everytime, only sample one preference.
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                # batch['data'].shape: (batch_size, 1, 36, 36)
                self.hnet.train()
                self.optimizer.zero_grad()
                weights = self.hnet(ray)      # len(weights) = 10
                num_target = numel(weights)    # numel(weights) = 31910
                
                logits_l, logits_r = self.net(batch['data'], weights)
                logits_array = [logits_l, logits_r]

                loss_vec = torch.stack([obj(logits, **batch) for (logits,obj) in zip(logits_array, self.obj_arr)])
                if self.is_agg:
                    loss_vec_agg = torch.atleast_2d(loss_vec)
                    ray_agg = torch.atleast_2d(ray)
                    loss = torch.sum( get_agg_func(self.agg_name)(loss_vec_agg, ray_agg) )
                elif self.solver_name in ['epo', 'pmgda']:
                    # Here, we also need the Jacobian matrix.

                    grads = []
                    for i, loss_i in enumerate(loss_vec):
                        weights_list = list(weights.values())
                        g = torch.autograd.grad(loss_i, weights_list, retain_graph=True)
                        flat_grads = []
                        for gg in g :
                            flat_grads.append(gg.contiguous().view(-1))
                        flat_grad = torch.cat(flat_grads, dim=0)
                        grads.append(flat_grad.data)

                    grads = torch.stack(grads)

                    if self.solver_name == 'epo':
                        core_solver = EPOCore(n_var=num_target, prefs=ray.unsqueeze(0))
                    else:
                        core_solver = PMGDACore(n_var=num_target, prefs=ray.unsqueeze(0))

                    alpha_arr = torch.stack([core_solver.get_alpha(grads, loss_vec, 0)]).to(self.device)
                    loss = torch.mean(alpha_arr * loss_vec)

                else:
                    assert False, 'Unknown solver_name'
                loss_batch.append( loss.cpu().detach().numpy() )
                loss_vec_batch.append(loss_vec.cpu().detach().numpy())
                loss.backward()
                self.optimizer.step()
            loss_epoch.append( np.mean(np.array(loss_batch)) )
            mean_vec = np.mean(np.stack(loss_vec_batch, axis=0), axis=0)  # shape [n_obj]
            loss_vec_history.append(mean_vec)
        res = ({
        'loss_history':    np.array(loss_epoch),  # [epoch]
        'loss_vec_history':    np.stack(loss_vec_history, axis=0),  # [epoch, n_obj]
        })
        return res

    def eval(self, n_eval):
        uniform_prefs = torch.Tensor(get_uniform_pref(n_eval)).to(self.device)
        loss_pref = []
        for pref_idx, pref in tqdm(enumerate(uniform_prefs)):
            loss_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                weights = self.hnet(pref)
                logits_l, logits_r = self.net(batch['data'], weights)
                logits_array = [logits_l, logits_r]
                loss_vec = torch.stack([obj(logits, **batch) for logits, obj in zip(logits_array, self.obj_arr)])
                loss_batch.append(loss_vec.cpu().detach().numpy())
            loss_pref.append(np.mean(np.array(loss_batch), axis=0))

        res = {}
        res['eval_loss'] = np.array(loss_pref)
        res['prefs'] = uniform_prefs.cpu().detach().numpy()
        return res


#Used to solve tabular problem with MOO
class GradBaseMTLSolver:
    def __init__(self, problem_name, batch_size, step_size, epoch, core_solver, prefs):
        self.step_size = step_size
        self.problem_name = problem_name
        self.epoch = epoch
        self.n_prob = len(prefs)
        self.batch_size = batch_size
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.prefs = prefs
        self.core_solver = core_solver

        train_dataset = get_dataset(self.problem_name, type='train')
        test_dataset = get_dataset(self.problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0)

        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.model_arr = [model_from_dataset(self.problem_name) for _ in range( self.n_prob )]
        # self.optimizer_arr = [ torch.optim.Adam(model.parameters(), lr=self.step_size)
        #                        for model in self.model_arr ]
        self.optimizer_arr = [ VeLO(model.parameters(), lr=1.0)
                                for model in self.model_arr ]
        self.update_counter = 0
        self.solver_name = core_solver.core_name
        self.is_agg = self.solver_name.startswith('Agg')
        self.agg_name = core_solver.agg_name if self.is_agg else None

    def solve(self):
        prefs = self.prefs
        n_prob = len(prefs)
        loss_history = []
        for epoch_idx in tqdm( range(self.epoch) ):
            loss_mat_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                # Step 1, get Jacobian_array and fs.
                loss_mat = [0] * n_prob
                Jacobian_array = [0] * n_prob
                for pref_idx, pref in enumerate(self.prefs):
                    # model input: data
                    logits = self.model_arr[pref_idx](batch['data'])
                    loss_vec = torch.stack( [obj(logits['logits'], **batch) for obj in self.obj_arr] )
                    loss_mat[pref_idx] = loss_vec
                    if not self.is_agg:
                        Jacobian_ = calc_gradients_mtl(batch['data'], batch, self.model_arr[pref_idx], self.obj_arr)
                        Jacobian = torch.stack([flatten_grads(elem) for elem in Jacobian_])
                        Jacobian_array[pref_idx] = Jacobian
                if not self.is_agg:
                    Jacobian_array = torch.stack(Jacobian_array)
                    # shape: (n_prob, n_obj, n_param)
                loss_mat = torch.stack(loss_mat)
                loss_mat_detach = loss_mat.detach()
                loss_mat_np = loss_mat.detach().numpy()
                # shape: (n_prob, n_obj)
                loss_mat_batch.append(loss_mat_np)
                for idx in range(n_prob):
                    self.optimizer_arr[idx].zero_grad()
                # Step 2, get alpha_array
                if self.is_agg:
                    agg_func = get_agg_func(self.agg_name)
                    agg_val = agg_func(loss_mat, torch.Tensor(prefs).to(loss_mat.device))
                    # shape: (n_prob)
                    loss = torch.sum(agg_val)
                    loss.backward() # loss
                else:
                    if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                        alpha_array = torch.stack(
                            [self.core_solver.get_alpha(Jacobian_array[idx], loss_mat_detach[idx], idx) for idx in
                             range(self.n_prob)])
                    elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'GradHVCore']:
                        if self.core_solver.core_name == 'GradHVCore':
                            alpha_array = self.core_solver.get_alpha_array(loss_mat_detach)
                        elif self.core_solver.core_name == 'PMTLCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_np, epoch_idx)
                        elif self.core_solver.core_name == 'MOOSVGDCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_detach)
                        else:
                            assert False, 'Unknown core_name'
                    else:
                        assert False, 'Unknown core_name'
                    loss = torch.sum(alpha_array * loss_mat)
                    loss.backward() # loss
                for idx in range(n_prob):
                    self.optimizer_arr[idx].step(loss)
            loss_mat_batch_mean = np.mean(np.array(loss_mat_batch), axis=0)
            loss_history.append(loss_mat_batch_mean)
        res = {'loss_history': loss_history,
               'loss' : loss_history[-1]}
        return res
    
#I added this class to solve mtl problem for images with MOO (like MultiMNIST, MultiFashion, FashionMNIST and dSpritres)
class GradBaseMTLSolverMnist:
    def __init__(self, problem_name, batch_size, step_size, epoch, core_solver, prefs, device):
        self.step_size = step_size
        self.problem_name = problem_name
        self.epoch = epoch
        self.n_prob = len(prefs)
        self.batch_size = batch_size
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.prefs = torch.tensor(prefs, device=device)
        self.core_solver = core_solver
        self.device = device

        train_dataset = get_dataset(self.problem_name, type='train')
        test_dataset = get_dataset(self.problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0)

        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.model_arr = [model_from_dataset(self.problem_name).to(self.device) for _ in range( self.n_prob )]
        self.optimizer_arr = [ VeLO(model.parameters(), lr=1.0) for model in self.model_arr ]
        # self.optimizer_arr = [ torch.optim.Adam(model.parameters(), lr=self.step_size) for model in self.model_arr ]

        self.update_counter = 0
        self.solver_name = core_solver.core_name
        self.is_agg = self.solver_name.startswith('Agg')
        self.agg_name = core_solver.agg_name if self.is_agg else None

    def solve(self):
        prefs = self.prefs
        n_prob = len(prefs)
        loss_history = []
        for epoch_idx in tqdm( range(self.epoch) ):
            loss_mat_batch = []
            for batch_idx, batch in enumerate(self.train_loader):

                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                
                # Step 1, get Jacobian_array and fs.
                loss_mat = [0] * n_prob
                Jacobian_array = [0] * n_prob
                for pref_idx, pref in enumerate(self.prefs):
                    # model input: data
                    logits = self.model_arr[pref_idx](batch)
                    logits_array = [logits['logits_l'], logits['logits_r']]
                    loss_vec = torch.stack([obj(logits, **batch) for (logits,obj) in zip(logits_array, self.obj_arr)]).to(self.device)
                    loss_mat[pref_idx] = loss_vec
                    
                    if not self.is_agg:
                        Jacobian_ = calc_gradients_mtl_mnist(batch, self.model_arr[pref_idx], self.obj_arr)
                        Jacobian = torch.stack([flatten_grads(elem) for elem in Jacobian_])
                        Jacobian_array[pref_idx] = Jacobian

                if not self.is_agg:
                    Jacobian_array = torch.stack(Jacobian_array)
                    # shape: (n_prob, n_obj, n_param)
                loss_mat = torch.stack(loss_mat) 
                loss_mat_detach = loss_mat.detach()
                loss_mat_np = loss_mat.detach().cpu().numpy()
                # shape: (n_prob, n_obj)
                loss_mat_batch.append(loss_mat_np)
                for idx in range(n_prob):
                    self.optimizer_arr[idx].zero_grad()

                if self.core_solver.core_name in ['PCGradCore', 'NashMTLCore'] :

                    for i, model in enumerate(self.model_arr) :
                        grads = self.core_solver.get_shared_gradients(Jacobian_array[i])
                        ptr = 0
                        for p in model.shared.parameters():
                            n = p.numel()
                            p.grad = grads[ptr:ptr+n].view_as(p).clone()
                            ptr += n

                        # tâche 0 —> private_left
                        loss0 = loss_mat[i][0]
                        loss0.backward(inputs=list(model.private_left.parameters()), retain_graph=True)

                        # tâche 1 —> private_right
                        loss1 = loss_mat[i][1]
                        loss1.backward(inputs=list(model.private_right.parameters()))

                elif self.is_agg:
                    agg_func = get_agg_func(self.agg_name)
                    agg_val = agg_func(loss_mat, prefs)
                    # shape: (n_prob)
                    loss=torch.sum(agg_val)
                    loss.backward()
                    
                else:
                    if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                        alpha_array = torch.stack(
                            [self.core_solver.get_alpha(Jacobian_array[idx], loss_mat_detach[idx], idx) for idx in
                             range(self.n_prob)])
                    elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'GradHVCore']:
                        if self.core_solver.core_name == 'GradHVCore':
                            alpha_array = self.core_solver.get_alpha_array(loss_mat_detach)
                        elif self.core_solver.core_name == 'PMTLCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_np, epoch_idx)
                        elif self.core_solver.core_name == 'MOOSVGDCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_detach)
                        else:
                            assert False, 'Unknown core_name'
                    else:
                        assert False, 'Unknown core_name'
                    loss=torch.sum(alpha_array.to(self.device) * loss_mat.to(self.device))
                    loss.backward()

                for idx in range(n_prob):
                    self.optimizer_arr[idx].step(loss)
            loss_mat_batch_mean = np.mean(np.array(loss_mat_batch), axis=0)
            loss_history.append(loss_mat_batch_mean)
        res = {'loss_history': loss_history,
               'loss' : loss_history[-1]}
        return res
    

#I added this class to solve mtl problem for tabular problem with PSL (like MultiMNIST, MultiFashion, FashionMNIST and dSpritres)
class TabPSLSolver:
    def __init__(self, problem_name, batch_size, step_size, epoch, solver_name, device):
        self.step_size      = step_size
        self.problem_name   = problem_name
        self.epoch          = epoch
        self.batch_size     = batch_size
        self.device         = device
        self.solver_name    = solver_name

        train_dataset       = get_dataset(self.problem_name, type='train')
        test_dataset        = get_dataset(self.problem_name, type='test')
        self.dataset        = get_dataset(self.problem_name)
        self.train_loader   = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=True,     num_workers=2,  pin_memory=True )
        self.test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,num_workers=0)

        self.settings       = mtl_setting_dict[self.problem_name]
        self.obj_arr        = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.n_obj          = len(self.obj_arr)
        self.is_agg         = self.solver_name.startswith('agg')
        self.agg_name       = solver_name.split('_')[-1] if self.is_agg else None

        # base model for the problem
        self.base           = model_from_dataset(problem_name).to(device)
        self.shapes         = [p.shape for p in self.base.parameters()]

        # PSL model
        self.n_var               = int(sum(np.prod(s) for s in self.shapes))
        self.psl            = SimplePSLModel2(self.n_obj, self.n_var).to(device)

        self.optimizer      = torch.optim.Adam(self.psl.parameters(), lr=step_size)

    def solve(self):
        loss_history = [] 
        loss_vec_history = []  
        for epoch_idx in tqdm(range(self.epoch)):
            loss_batch = []
            loss_vec_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                B = batch['data'].size(0)

                # we draw a preference, and repeat it B times to match the batch size (we repeat it B times because we only need one preference)
                pref = torch.from_numpy(
                    np.random.dirichlet(np.ones(self.n_obj), 1).astype(np.float32).flatten()
                ).to(self.device) 
                prefs_batch = pref.repeat(B, 1)  # [B, n_obj]

                # PSL model
                self.psl.train()
                flat_weights = self.psl(prefs_batch)     # [B, n_var]

                # reshape our weights to obtain the weights associated with each layer of the FullyConnected model of the problem
                weights = []
                idx = 0
                for shape in self.shapes:
                    n = int(np.prod(shape))
                    intermediate_weights = flat_weights[:, idx : idx+n]       # [B, n]
                    weights.append(intermediate_weights.view(B, *shape).contiguous())
                    idx += n

                weights_dictionnary = {name: w[0] for (name,_), w in zip(self.base.named_parameters(), weights)}

                # forward and loss calculation
                logits = self.base.forward_with_weights(batch['data'], weights_dictionnary)
                loss_vec = torch.stack([obj(logits, **batch) for obj in self.obj_arr]).to(self.device)

                if self.is_agg:
                    loss = get_agg_func(self.agg_name)(loss_vec.unsqueeze(0), pref.unsqueeze(0))
                elif self.solver_name in ['epo', 'pmgda']:

                    grads = []
                    for i, loss_i in enumerate(loss_vec):
                        g = torch.autograd.grad(loss_i, weights, retain_graph=True)
                        flat_grads = []
                        for gg in g :
                            flat_grads.append(gg.contiguous().view(-1))
                        flat_grad = torch.cat(flat_grads, dim=0)
                        grads.append(flat_grad.data)

                    grads = torch.stack(grads)

                    if self.solver_name == 'epo':
                        core_solver = EPOCore(n_var=self.n_var, prefs=pref.unsqueeze(0))
                    else:
                        core_solver = PMGDACore(n_var=self.n_var, prefs=pref.unsqueeze(0))

                    alpha_arr = torch.stack([core_solver.get_alpha(grads, loss_vec, 0)]).to(self.device)
                    loss = torch.mean(alpha_arr * loss_vec)
                                 
                # backward & step
                loss_batch.append( loss.cpu().detach().numpy() )
                loss_vec_batch.append(loss_vec.cpu().detach().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss_history.append( np.mean(np.array(loss_batch)) )
            mean_vec = np.mean(np.stack(loss_vec_batch, axis=0), axis=0)  # shape [n_obj]
            loss_vec_history.append(mean_vec)
        res = ({
        'loss_history':    np.array(loss_history),  # [epoch]
        'loss_vec_history':    np.stack(loss_vec_history, axis=0),  # [epoch, n_obj]
        })
        return self, res

    def eval(self, prefs): 
        prefs = prefs.to(self.device)
        loss_pref = []
        for pref_idx, pref in tqdm(enumerate(prefs)):
            loss_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                B = batch['data'].size(0) 
                prefs_batch = pref.repeat(B, 1)  # [B, n_obj]
                flat_weights = self.psl(prefs_batch)     # [B, n_var]
                weights = []
                idx = 0
                for shape in self.shapes:
                    n = int(np.prod(shape))
                    intermediate_weights = flat_weights[:, idx : idx+n]       # [B, n]
                    weights.append(intermediate_weights.view(B, *shape).contiguous())
                    idx += n
                weights_dictionnary = {name: w[0] for (name,_), w in zip(self.base.named_parameters(), weights)}

                # forward and loss calculation
                logits = self.base.forward_with_weights(batch['data'], weights_dictionnary)
                loss_vec = torch.stack([obj(logits, **batch) for obj in self.obj_arr]).to(self.device)
                loss_batch.append(loss_vec.cpu().detach().numpy())
            loss_pref.append(np.mean(np.array(loss_batch), axis=0))

        res = {}
        res['eval_loss'] = np.array(loss_pref)
        res['prefs'] = prefs.cpu().detach().numpy()
        return res
    

#I added this class to solve mtl problem for temporal problem with MOO (electricity, covid) 
class GradBaseMTLSolverTemporal:
    def __init__(self, problem_name, batch_size, step_size, epoch, core_solver, prefs, device, args=None):
        self.step_size = step_size
        self.problem_name = problem_name
        self.epoch = epoch
        self.n_prob = len(prefs)
        self.batch_size = batch_size
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.prefs = prefs
        self.core_solver = core_solver
        self.args = args
        self.device = device

        train_dataset = get_dataset(self.problem_name, type='train')
        test_dataset = get_dataset(self.problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0)
        

        self.obj_arr = from_name(self.settings['objectives'], self.dataset.task_names())
        self.model_arr = [model_from_dataset(self.problem_name, self.args).to(self.device) for _ in range( self.n_prob )]
        #self.optimizer_arr = [ torch.optim.Adam(model.parameters(), lr=self.step_size)
        #                       for model in self.model_arr ]
        self.optimizer_arr = [ VeLO(model.parameters(), lr=1.0) for model in self.model_arr ]
        self.update_counter = 0
        self.solver_name = core_solver.core_name
        self.is_agg = self.solver_name.startswith('Agg')
        self.agg_name = core_solver.agg_name if self.is_agg else None

        params = [model.parameters() for model in self.model_arr]
        self.params = [model.parameters() for model in self.model_arr]

    def solve(self):
        prefs = self.prefs.to(self.device)
        n_prob = len(prefs)
        loss_history = []
        for epoch_idx in tqdm( range(self.epoch) ):
            loss_mat_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Step 1, get Jacobian_array and fs.
                loss_mat = [0] * n_prob
                Jacobian_array = [0] * n_prob
                for pref_idx, pref in enumerate(self.prefs):
                    # model input: data
                    logits = self.model_arr[pref_idx](batch['data'])
                    logits_array = [logits['logits_l'], logits['logits_r']]
                    loss_vec = torch.stack([obj(logits, **batch) for (logits,obj) in zip(logits_array, self.obj_arr)]).to(self.device)
                    loss_mat[pref_idx] = loss_vec
                    if not self.is_agg:
                        Jacobian_ = calc_gradients_mtl_2_logits_v2(batch, self.model_arr[pref_idx], self.obj_arr, batch['data'])
                        Jacobian = torch.stack([flatten_grads(elem) for elem in Jacobian_]).to(self.device)
                        Jacobian_array[pref_idx] = Jacobian
                if not self.is_agg:
                    Jacobian_array = torch.stack(Jacobian_array).to(self.device)
                    # shape: (n_prob, n_obj, n_param)
                loss_mat = torch.stack(loss_mat)
                loss_mat_detach = loss_mat.detach()
                loss_mat_np = loss_mat.cpu().detach().numpy()
                # shape: (n_prob, n_obj)
                loss_mat_batch.append(loss_mat_np)
                for idx in range(n_prob):
                    self.optimizer_arr[idx].zero_grad()
                # Step 2, get alpha_array
                if self.is_agg:
                    agg_func = get_agg_func(self.agg_name)
                    agg_val = agg_func(loss_mat, torch.Tensor(prefs).to(loss_mat.device)).to(self.device)
                    # shape: (n_prob)
                    loss=torch.sum(agg_val)
                    loss.backward()
                else:
                    if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                        alpha_array = torch.stack(
                            [self.core_solver.get_alpha(Jacobian_array[idx], loss_mat_detach[idx], idx) for idx in
                             range(self.n_prob)])
                    elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'GradHVCore']:
                        if self.core_solver.core_name == 'GradHVCore':
                            alpha_array = self.core_solver.get_alpha_array(loss_mat_detach).to(self.device)
                        elif self.core_solver.core_name == 'PMTLCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_np, epoch_idx).to(self.device)
                        elif self.core_solver.core_name == 'MOOSVGDCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_detach).to(self.device)
                        else:
                            assert False, 'Unknown core_name'
                    else:
                        assert False, 'Unknown core_name'
                    loss=torch.sum(alpha_array.to(self.device) * loss_mat.to(self.device)).to(self.device)
                    loss.backward()
                for idx in range(n_prob):
                    self.optimizer_arr[idx].step(loss)
            loss_mat_batch_mean = np.mean(np.array(loss_mat_batch), axis=0)
            loss_history.append(loss_mat_batch_mean)
        res = {'loss_history': loss_history,
               'loss' : loss_history[-1]}
        return res
    

from pylo import VeLO

class VeLOMTLSolver:
    def __init__(self, problem_name, prefs, batch_size=128, epoch=50, device='cpu', scalarization='linear'):
        self.problem_name = problem_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.prefs = prefs
        self.n_prob = len(prefs)
        self.scalarization = scalarization
        assert self.scalarization in ['linear', 'tchebycheff'], "Only 'linear' or 'tchebycheff' supported"

        # Setup dataset and loaders
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.train_loader = torch.utils.data.DataLoader(get_dataset(self.problem_name, type='train'),
                                                        batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(get_dataset(self.problem_name, type='test'),
                                                       batch_size=self.batch_size, shuffle=True)

        # Setup objectives and models
        self.obj_arr = from_name(self.settings['objectives'], self.dataset.task_names())
        self.model_arr = [model_from_dataset(self.problem_name).to(self.device) for _ in range(self.n_prob)]
        self.optimizer_arr = [VeLO(model.parameters(), lr=1.0) for model in self.model_arr]

    def solve(self):
        loss_history = []
        if self.scalarization == 'tchebycheff':
            ideal_point = torch.ones(len(self.obj_arr), device=self.device) * float('inf')

        for epoch_idx in tqdm(range(self.epoch)):
            loss_mat_batch = []
            for batch in self.train_loader:
                data = batch['data'].to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss_mat = []
                for pref_idx, pref in enumerate(self.prefs):
                    model = self.model_arr[pref_idx]
                    optimizer = self.optimizer_arr[pref_idx]

                    logits = model(data)
                    loss_vec = torch.stack([obj(logits['logits'], **batch) for obj in self.obj_arr])

                    # update ideal point
                    if self.scalarization == 'tchebycheff':
                        ideal_point = torch.min(ideal_point, loss_vec.detach())
                        scalar_loss = torch.max(pref * torch.abs(loss_vec - ideal_point))
                    elif self.scalarization == 'linear':
                        scalar_loss = torch.sum(torch.tensor(pref, device=self.device) * loss_vec)
                    
                    optimizer.zero_grad()
                    scalar_loss.backward()
                    optimizer.step(scalar_loss)

                    loss_mat.append(loss_vec.detach().cpu())
                
                loss_mat_batch.append(torch.stack(loss_mat))

            loss_mat_batch_mean = torch.mean(torch.stack(loss_mat_batch), dim=0).numpy()
            loss_history.append(loss_mat_batch_mean)

        return {
            'loss_history': loss_history,
            'loss': loss_history[-1],
            'prefs': self.prefs
        }

class VeLOMTLSolverMnist:
    def __init__(self, problem_name, batch_size, epoch, prefs, device, scalarization = 'linear'):
        self.problem_name = problem_name
        self.epoch = epoch
        self.n_prob = len(prefs)
        self.batch_size = batch_size
        self.dataset = get_dataset(problem_name)
        self.settings = mtl_setting_dict[problem_name]
        self.prefs = torch.tensor(prefs, device=device)
        self.device = device
        self.scalarization = scalarization
        assert self.scalarization in ['linear', 'tchebycheff'], "Only 'linear' or 'tchebycheff' supported"

        train_dataset = get_dataset(problem_name, type='train')
        test_dataset = get_dataset(problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # self.obj_arr = self.settings['objectives']
        self.task_names = self.dataset.task_names()
        # self.obj_arr = [obj_fn for obj_fn in self.settings['objectives']]
        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.model_arr = [model_from_dataset(problem_name).to(device) for _ in range(self.n_prob)]
        self.optimizer_arr = [VeLO(model.parameters(), lr=1.0) for model in self.model_arr]

    def solve(self):
        loss_history = []
        if self.scalarization == 'tchebycheff':
            ideal_point = torch.ones(len(self.obj_arr), device=self.device) * float('inf')

        for epoch_idx in tqdm(range(self.epoch)):
            loss_mat_batch = []

            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss_mat = []

                for i in range(self.n_prob):
                    model = self.model_arr[i]
                    optimizer = self.optimizer_arr[i]
                    pref = self.prefs[i]

                    # Forward pass
                    logits_dict = model(batch)
                    logits_array = [logits_dict['logits_l'], logits_dict['logits_r']]

                    # Compute losses for each task
                    losses = torch.stack([
                        obj_fn(logits, **batch) for logits, obj_fn in zip(logits_array, self.obj_arr)
                    ])
                    loss_mat.append(losses)
                    
                    # update ideal point
                    if self.scalarization == 'tchebycheff':
                        ideal_point = torch.min(ideal_point, losses.detach())
                        scalar_loss = torch.max(pref * torch.abs(losses - ideal_point))
                    elif self.scalarization == 'linear':
                        # Scalarize using weighted sum (preference vector)
                        scalar_loss = torch.sum(pref * losses)

                    optimizer.zero_grad()
                    scalar_loss.backward()
                    optimizer.step(scalar_loss)

                loss_mat = torch.stack(loss_mat)  # shape: (n_prob, n_obj)
                loss_mat_batch.append(loss_mat.detach().cpu().numpy())

            loss_mat_batch_mean = np.mean(np.array(loss_mat_batch), axis=0)
            loss_history.append(loss_mat_batch_mean)

        res = {
            'loss_history': loss_history,
            'loss': loss_history[-1],
        }
        return res



if __name__ == '__main__':
    print()
