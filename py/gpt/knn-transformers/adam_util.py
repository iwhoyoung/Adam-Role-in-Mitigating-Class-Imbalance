import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as nnFun
from torch.optim import Adam
class XAdaSSD(Optimizer):
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

    def __init__(self, params, lr=1e-3, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.sims_epoch=[]
        self.act_ovelap=[]
        self.init_norm_p_epoch=[]
        self.norm_p_epoch=[]
        self.norm_p_rate_epoch=[]
        self.convegence_factor = 0
        super(XAdaSSD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(XAdaSSD, self).__setstate__(state)
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
                        state['denom'] = torch.zeros(1,device='cuda')
                        state['exp_avg'] = torch.ones_like(p, memory_format=torch.preserve_format)*group['eps']
                        state['mean_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of gradient values
                        # state['cur_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['init_param'] = p.clone()
                        state['init_avg_abs_param'] = torch.mean(torch.abs(state['init_param']))
                        if state['init_avg_abs_param'] < group['eps']:
                            # state['init_avg_abs_param'] = torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                            state['init_avg_abs_param'] = 0.1*init_avg_abs_param_buffer*torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                        # state['init_smooth'] =torch.sum(torch.abs(p.grad*state['init_avg_abs_param']*group['lr']*p))/(torch.mean(p.grad*p.grad.conj()).sqrt()*torch.sum(p*p.conj()).sqrt()).add_(group['eps'])
                        init_avg_abs_param_buffer = state['init_avg_abs_param']

                    # update the steps for each param group update
                   
                   
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                # calculate mean mommentum
                
            self.adam(params_with_grad,
                   grads,
                   lr=group['lr'],#/10000 * state['step']
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
            
        return loss

    def adam(self,
         params: List[Tensor],
         grads: List[Tensor],
         lr: float,
         weight_decay: float,
         eps: float):

        for i, param in enumerate(params):

            grad = grads[i]

            state = self.state[param]

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # grad -= grad * torch.abs(avg_grad) / torch.sum(avg_grad*avg_grad.conj()).sqrt()
            # avg_grad = state['exp_avg']/(1 - 0.9 ** state['step'])
            # grad = torch.abs(grad)/torch.sum(grad*grad.conj()).sqrt() - avg_grad / (torch.sum(avg_grad*avg_grad.conj())+eps)
            # grad = ((1+torch.sign(grad))*grad)
            # mean_abs = torch.mean(torch.abs(grad))
            
            # ema_grad = state['mean_exp_avg'].mul_(0.9).add_(grad, alpha=(0.1))/(1 - 0.9 ** state['step'])
            # avg_grad = state['exp_avg'].mul_(0.9).add_(grad/torch.mean(torch.abs(grad)), alpha=(0.1))/(1 - 0.9 ** state['step'])
            # avg_grad = state['mean_exp_avg'].mul_(0.9).add_(grad, alpha=(0.1))/(1 - 0.9 ** state['step'])
            # avg_grad = state['exp_avg']
            # grad = grad / (avg_grad+0.02*torch.abs(grad)+eps)
            # avg_grad = state['exp_avg'].mul_(0.5).add_(torch.abs(grad), alpha=(0.5))/(1 - 0.5 ** state['step'])                      
            poly = state['exp_avg'].mul_(0.999).add_(grad, alpha=(0.001))/(1 - 0.999 ** state['step'])
            # avg_grad = state['exp_avg'].mul_((state['step']-1)/state['step']).add_(torch.abs(grad), alpha=(1/state['step']))
            # avg_grad = state['exp_avg'].mul_(1/torch.sqrt(torch.sum(state['exp_avg']**2)).add_(eps)).mul_(torch.abs(grad))
            # grad = (grad/torch.mean(torch.abs(grad))+torch.sign(grad)) / (avg_grad**0.5+eps)
            # mean_abs = torch.mean(avg_grad)
            # mean_abs = torch.mean(torch.abs(grad))
            # grad = (grad+torch.sign(grad)*mean_abs) / (avg_grad**1+0.0001*state['init_mean_grad'])
            # grad = grad * mean_abs/avg_grad.add_(eps)
            # act_overlapping = torch.sum(avg_grad/torch.sqrt(torch.sum(avg_grad**2))*torch.abs(grad)/torch.sqrt(torch.sum(grad**2)))
            # grad = grad * (1+torch.sign(act_overlapping-0.5)*torch.sign(avg_grad - mean_abs))
            # if not torch.sign(act_overlapping-0.5).any():
                # print('act_overlapping',state['step'])
            # grad = torch.sign(grad) * (torch.abs(torch.abs(grad) - avg_grad)/avg_grad.add_(eps))
            # grad = grad * torch.abs(torch.abs(grad) - avg_grad)/avg_grad.add_(eps)
            # grad = (grad) / (avg_grad + eps)
            # grad = grad * avg_grad
            # grad = (grad) / (avg_grad + mean_abs)
            # grad = torch.sign(grad)*torch.abs(torch.abs(grad)-avg_grad)/(avg_grad + eps)
            # grad =  torch.sign(grad)*avg_grad/(torch.abs(grad)+eps)
            # avg_grad = state['exp_avg'].mul_(0.9).add_(torch.abs(grad)**1+mean_abs, alpha=(0.1))
            # grad = (torch.sign(grad)*avg_grad/(avg_grad+torch.mean(avg_grad)))
            # avg_grad = state['exp_avg'].add_(torch.abs(grad), alpha=(0.5)).mul_(0.5)
            # grad = grad * avg_grad
            mean_exp_avg = state['mean_exp_avg'].mul_(0.999).add_(grad**2, alpha=(0.001))/(1 - 0.999 ** state['step'])
            # a = torch.sum((torch.abs(grad*avg_grad)/(torch.sum(grad*grad.conj())*torch.sum(avg_grad*avg_grad.conj())).sqrt()))
            # mean_grad = torch.mean((grad*grad.conj())).sqrt()
            # convergence = torch.mean((grad*grad.conj())).sqrt()/mean_grad
            # cfactor =100/(100+convergence)
            # mask = 0.5+0.5*torch.sign(mean_grad-torch.abs(grad))
            # mean_grad = torch.mean((mask*grad*grad.conj())).sqrt()
            # dynamics = grad/mean_grad.add_(eps)
            dynamics = grad/mean_exp_avg.sqrt().add_(eps)
            # dynamics = grad
            # param.add_(dynamics*state['init_avg_abs_param']+0.001*param, alpha=-lr)#*cfactor
            # param.add_(dynamics, alpha=-lr)#*cfactor
            scfactor = 10**(torch.sum(poly/torch.sqrt(torch.sum(poly**2)).add_(eps)*grad/torch.sqrt(torch.sum(grad**2)).add_(eps))-1)
            a = state['denom'].mul_(0.99).add_(1-scfactor, alpha=(0.01))
            param.add_(dynamics, alpha=-lr*(0.1+scfactor).item())#*cfactor
            # param.add_(dynamics*state['init_avg_abs_param'], alpha=-lr*(0.1+0.5*(1+torch.sign(0.8-scfactor))))#*cfactor
            # add_lr=1+5*(1+torch.sign(state['denom'].mul_(0.99).add_(1-scfactor, alpha=(0.01))-0.2))
            # param.add_(dynamics*state['init_avg_abs_param'], alpha=-lr*add_lr.item())#*cfactor


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
        overlap = []
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
                # overlap.append(torch.sum(state['exp_avg']/torch.sqrt(torch.sum(state['exp_avg']**2))*torch.abs(grad)/torch.sqrt(torch.sum(grad**2)).add_(1e-8)))
                overlap.append(state['denom'].item())
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
        self.act_ovelap.append(overlap)
        # self.grads.append(grad)
        if saved:
            mean_df_datas = pd.DataFrame(data=self.sims_epoch)
            sum_df_datas = pd.DataFrame(data=self.norm_p_epoch)
            std_df_datas = pd.DataFrame(data= self.init_norm_p_epoch)
            grad_df_datas = pd.DataFrame(data=self.norm_p_rate_epoch)
            overlap_df_datas = pd.DataFrame(data=self.act_ovelap)
            mean_df_datas.to_csv(path + '_sim.csv')
            sum_df_datas.to_csv(path + '_normp.csv')
            std_df_datas.to_csv(path + '_initp.csv')
            grad_df_datas.to_csv(path + '_prate.csv')
            overlap_df_datas.to_csv(path + '_overlap.csv')
        return learning_utility

import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as nnFun

class Adam_bn(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.weight = Parameter(torch.zeros(1,device='cuda'))
        super(Adam_bn, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Adam_bn, self).__setstate__(state)
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
            exp_avgs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    grads.append(p.grad)


                    # update the steps for each param group update
                    exp_avgs.append(state['exp_avg'])
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
            
        return loss

    def adam(self,
         params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         lr: float,
         weight_decay: float,
         eps: float):

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            # if amsgrad:
            #     # Maintains the maximum of all 2nd moment running avg. till now
            #     torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            #     # Use the max. for normalizing running avg. of gradient
            #     denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            # else:
            #     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            denom = torch.mean((exp_avg*exp_avg.conj())).sqrt().add_(eps)
            step_size = lr
            # step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)

class Sgd_m(Optimizer):


    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.weight = Parameter(torch.zeros(1,device='cuda'))
        super(Sgd_m, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Sgd_m, self).__setstate__(state)
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
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    grads.append(p.grad)

                    # update the steps for each param group update
                    exp_avgs.append(state['exp_avg'])
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
            
        return loss

    def adam(self,
         params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            step_size = lr/bias_correction1
            param.add_(exp_avg, alpha=-step_size)

import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as nnFun

class Adam_ini(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.weight = Parameter(torch.zeros(1,device='cuda'))
        super(Adam_ini, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_ini, self).__setstate__(state)
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
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            initial_norm = []
            beta1, beta2 = group['betas']
            init_avg_abs_param_buffer = 1

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['init_avg_abs_param'] = torch.mean(torch.abs(p.clone()))
                        if state['init_avg_abs_param'] < group['eps']:
                            state['init_avg_abs_param'] = 0.1*init_avg_abs_param_buffer*torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                        init_avg_abs_param_buffer = state['init_avg_abs_param']
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    grads.append(p.grad)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    initial_norm.append(state['init_avg_abs_param'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   initial_norm,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],#/10000 * state['step']
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
            
        return loss

    def adam(self,
         params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         initial_norm,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            # denom = torch.mean((exp_avg*exp_avg.conj())).sqrt().add_(eps)
            step_size = lr
            # step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size*initial_norm[i])

class Adam_Sbn(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.weight = Parameter(torch.zeros(1,device='cuda'))
        super(Adam_Sbn, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Adam_Sbn, self).__setstate__(state)
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
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            initial_norm = []
            beta1,beta2 = group['betas']
            init_avg_abs_param_buffer = 1

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['init_avg_abs_param'] = torch.mean(torch.abs(p.clone()))
                        if state['init_avg_abs_param'] < group['eps']:
                            state['init_avg_abs_param'] = 0.1*init_avg_abs_param_buffer*torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                        init_avg_abs_param_buffer = state['init_avg_abs_param']
                    grads.append(p.grad)

                    # update the steps for each param group update
                    exp_avgs.append(state['exp_avg'])
                    initial_norm.append(state['init_avg_abs_param'])
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   state_steps,
                    initial_norm=initial_norm,
                   beta1=beta1,             
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
            
        return loss

    def adam(self,
         params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         state_steps: List[int],
         initial_norm,
         *,
         beta1: float,
         lr: float,
         weight_decay: float,
         eps: float):

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            # if amsgrad:
            #     # Maintains the maximum of all 2nd moment running avg. till now
            #     torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            #     # Use the max. for normalizing running avg. of gradient
            #     denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            # else:
            #     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            denom = torch.mean((exp_avg*exp_avg.conj())).sqrt().add_(eps)
            step_size = lr
            # step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size*initial_norm[i])
