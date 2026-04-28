import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as nnFun

class ParamCalculator(Optimizer):

    def __init__(self, params, amsgrad=False):
        defaults = dict(amsgrad=amsgrad)
        self.param_num = 0
        super(ParamCalculator, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ParamCalculator, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []

            for p in group['params']:
                if p.grad is not None:
                    
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    # grads.append(p.grad)
                    grad = p.grad.reshape(-1)
                    num, = grad.shape
                    self.param_num += num

                    
                       
            
        return self.param_num


