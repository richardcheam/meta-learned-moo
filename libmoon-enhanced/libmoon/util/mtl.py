import numpy as np
from libmoon.problem.mtl.loaders import Adult, Credit, Compas, MultiMNISTData, DSpritesData, ElectricityDemandData
from libmoon.util.constant import nadir_point_dict, ideal_point_dict
from libmoon.model import MultiLeNetMnist, FullyConnected, MultiLeNetDSprites, ElectricityLSTM
from libmoon.problem.mtl.settings import adult_setting, credit_setting, compass_setting, mnist_setting, fashion_setting, fmnist_setting, dsprites_setting, electricity_demand_setting

import torch

mtl_dim_dict = {
    'adult' : (88,),
    'credit' : (90,),
    'compass' : (20,),
    'mnist' : (1,36,36),
    'fashion' : (1,36,36),
    'fmnist' : (1,36,36),
    'dsprites' : (1,64,64),
    'electricity_demand' : None #for the moment => must be updates
}

def get_dataset(dataset_name, type='train'):
    if dataset_name == 'adult':
        dataset_ = Adult(split=type)
    elif dataset_name == 'credit':
        dataset_ = Credit(split=type)
    elif dataset_name == 'compass':
        dataset_ = Compas(split=type)
    elif dataset_name == 'electricity_demand':
        dataset_ = ElectricityDemandData(split=type)
    elif dataset_name == 'mnist':
        dataset_ = MultiMNISTData(dataset='mnist', split=type)
    elif dataset_name == 'fashion':
        dataset_ = MultiMNISTData(dataset='fashion', split=type)
    elif dataset_name == 'fmnist':
        dataset_ = MultiMNISTData(dataset='fmnist', split=type)
    elif dataset_name == 'dsprites':
        dataset_ = DSpritesData(split=type)
    else:
        raise ValueError('Invalid dataset name')
    return dataset_

def model_from_dataset(dataset_name, args=None, architecture='M1', **kwargs):
    dim = mtl_dim_dict[dataset_name]
    # print('dim:',dim)
    if dataset_name == 'adult':
        return FullyConnected(dim, architecture, **kwargs)
    elif dataset_name == 'credit':
        return FullyConnected(dim, architecture, **kwargs)
    elif dataset_name == 'compass':
        return FullyConnected(dim, architecture, **kwargs)
    elif dataset_name in ['mnist','fashion','fmnist'] :
        return MultiLeNetMnist(dim=dim, **kwargs)
    elif dataset_name in ['dsprites'] :
        return MultiLeNetDSprites(dim=dim, **kwargs)
    elif dataset_name in ['electricity_demand'] :
        seq_length = args['seq_length']
        hidden_dim = args['hidden_dim']
        num_classes = args['num_classes']
        return ElectricityLSTM(seq_length=seq_length, hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name {}".format(dataset_name))

def get_angle_range(dataset, return_degrees=False):
    p1 = [nadir_point_dict[dataset][0], ideal_point_dict[dataset][1]]
    p2 = [ideal_point_dict[dataset][0], nadir_point_dict[dataset][1]]
    th1 = np.arctan2(p1[1], p1[0])
    th2 = np.arctan2(p2[1], p2[0])
    if return_degrees:
        th1 = np.rad2deg(th1)
        th2 = np.rad2deg(th2)
    return th1, th2

def get_mtl_prefs(problem_name, n_prob, type='Tensor'):
    '''
        Input: problem_name: str, n_prob: int.
        Return prefs: np.array of shape (n_prob, 2)
    '''
    p1 = [nadir_point_dict[problem_name][0], ideal_point_dict[problem_name][1]]
    p2 = [ideal_point_dict[problem_name][0], nadir_point_dict[problem_name][1]]
    th1 = np.arctan2(p1[1], p1[0])
    th2 = np.arctan2(p2[1], p2[0])
    theta_arr = np.linspace(th1, th2, n_prob)

    prefs = np.c_[np.cos(theta_arr), np.sin(theta_arr)]
    prefs = prefs / np.sum(prefs, axis=1)[:, None]
    if type == 'Tensor':
        prefs = torch.Tensor(prefs)
    return prefs

mtl_setting_dict = {
    'adult': adult_setting,
    'credit': credit_setting,
    'compass': compass_setting,
    'mnist' : mnist_setting,
    'fashion' : fashion_setting,
    'fmnist' : fmnist_setting,
    'dsprites' : dsprites_setting,
    'electricity_demand' : electricity_demand_setting
}

def numel(model):
    if type(model) == dict:
        return sum(p.numel() for p in model.values() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

