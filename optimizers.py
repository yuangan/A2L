#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW, Adam

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
                self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])
            
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]
            # _ = [print(self.schedulers[key].state_dict()) for key in self.keys]


def define_scheduler(optimizer, params):
    print(params)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=params.get('max_lr', 2e-4),
    #     epochs=params.get('epochs', 200),
    #     steps_per_epoch=params.get('steps_per_epoch', 1000),
    #     pct_start=params.get('pct_start', 0.0),
    #     div_factor=params.get('div_factor', 1),
    #     final_div_factor=params.get('final_div_factor', 1))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=params.get('T_0', 50)*params.get('steps_per_epoch', 1000), 
    #     T_mult=params.get('T_mult', 1), 
    #     eta_min=1e-5, 
    #     last_epoch=-1, 
    #     verbose=False)
    mls = []
    pm = params.get('milestones', [50, 500, 1000])
    for i in range(len(mls)):
        # mls.append(pm[i] * params.get('steps_per_epoch', 1000))
        mls.append(pm[i])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones = mls, 
        gamma=params.get('gamma', 0.1), 
        last_epoch=-1, 
        verbose=False)

    return scheduler

def build_optimizer(parameters_dict, scheduler_params_dict):
    k = list(scheduler_params_dict.keys())[0]
    optim = dict([(key, Adam(params, lr=scheduler_params_dict[k].get('lr', 0.01), weight_decay=scheduler_params_dict[k].get('wd', 0.01), betas=(0.0, 0.99), eps=1e-9))
                   for key, params in parameters_dict.items()])

    schedulers = dict([(key, define_scheduler(opt, scheduler_params_dict[key])) \
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim