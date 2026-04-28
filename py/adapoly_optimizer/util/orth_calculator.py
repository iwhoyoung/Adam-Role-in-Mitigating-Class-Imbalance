import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as nnFun

class OrthCalculator(Optimizer):
    r"""Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: amsgrad                      \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.sims_epoch=[]
        super(OrthCalculator, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OrthCalculator, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def outer_step(self, closure=None):
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
            state_steps = []
            init_avg_abs_param_buffer=1

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # state['exp_avg'] = torch.zeros(1, device='cuda')
                        state['cur_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['cur_grad'].mul_(0).add_(p.grad, alpha=1)
                    # update the steps for each param group update
                    # state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])            
            
        return loss

    @torch.no_grad()
    def inner_step(self, closure=None):
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
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    if group['weight_decay'] != 0:
                        grad = grad.add(p, alpha=group['weight_decay'])

                    
        return loss

    @torch.no_grad()
    def calculate_orthogonality(self, path, saved=True):
        sims = []
        for group in self.param_groups:            
            for p in group['params']:
                state = self.state[p]
                if len(state) ==0:
                    continue
                vector_a = torch.abs(state['cur_grad'].reshape(-1))
                vector_b = torch.abs(p.grad.reshape(-1))
                sim = torch.sum(nnFun.normalize(vector_a,dim=0)*nnFun.normalize(vector_b,dim=0))
                sims.append(sim.item())
        self.sims_epoch.append(sims)
        if saved:
            mean_df_datas = pd.DataFrame(data=self.sims_epoch)
            mean_df_datas = mean_df_datas.mean(axis=0)
            mean_df_datas.to_csv(path + '.csv')

    @torch.no_grad()
    def calculate_mean_std(self, path, saved=True):
        learning_utility = []
        mean_grad = []
        sum_grad = []
        std_grad = []
        grad = []
        sims = []
        norm_values1 = []
        norm_values2 = []
        norm_values_rate = []
        for group in self.param_groups:            
            for p in group['params']:
                state = self.state[p]
                if len(state) ==0:
                    continue
                # avg_individual_weight_per_iter = torch.abs(state['avg_individual_weight'])/state['step_per_epoch']
                # mean_grad.append(torch.mean(avg_individual_weight_per_iter/(torch.abs(state['init_param_per_epoch'])+avg_individual_weight_per_iter)).item())
                # sum_grad.append((torch.mean(avg_individual_weight_per_iter)/(torch.mean(torch.abs(state['init_param_per_epoch'])+avg_individual_weight_per_iter))).item())
                # std_grad.append(torch.std(torch.log10(avg_individual_weight_per_iter/(torch.abs(state['init_param_per_epoch'])+avg_individual_weight_per_iter))).item())
                # grad.append(torch.mean(avg_individual_weight_per_iter).item())
                vector_a = state['init_param'].reshape(-1)
                vector_b = p.reshape(-1)
                norm_value1 = torch.sum(vector_a**2)**0.5
                norm_value2 = torch.sum(vector_b**2)**0.5
                sim = torch.sum(nnFun.normalize(vector_a,dim=0)*nnFun.normalize(vector_b,dim=0))
                sims.append(sim.item())
                norm_values1.append(norm_value1.item())
                norm_values2.append(norm_value2.item())
                norm_values_rate.append((norm_value2/(norm_value1+1e-5)).item())


                # state['init_param_per_epoch'] = p.clone()
                # state['avg_individual_weight'].mul_(0)
                # state['avg_std'].mul_(0)
                # state['step_per_epoch'] = 0
        #         avg_grad = state['exp_avg'] / state['step']
        #         # avg_grad = state['exp_avg'].mul_(beta1).add_(lr * p.grad, alpha=(1 - beta1)) / bias_correction1
        #         abs_avg_grad = torch.abs(avg_grad)
        #         lu = abs_avg_grad/(torch.abs(p)+abs_avg_grad+1e-8)
        #         learning_utility.append([torch.mean(lu).item(), torch.std(torch.log10(lu+1e-8)).item(), torch.std(lu).item()])
        self.sims_epoch.append(sims)
        self.init_norm_p_epoch = norm_values1
        self.norm_p_epoch.append(norm_values2)
        self.norm_p_rate_epoch.append(norm_values_rate)
        # self.grads.append(grad)
        if saved:
            mean_df_datas = pd.DataFrame(data=self.sims_epoch)
            sum_df_datas = pd.DataFrame(data=self.norm_p_epoch)
            std_df_datas = pd.DataFrame(data= self.init_norm_p_epoch)
            grad_df_datas = pd.DataFrame(data=self.norm_p_rate_epoch)
            mean_df_datas.to_csv(path + '_sim.csv')
            sum_df_datas.to_csv(path + '_normp.csv')
            std_df_datas.to_csv(path + '_initp.csv')
            grad_df_datas.to_csv(path + '_prate.csv')
        return learning_utility