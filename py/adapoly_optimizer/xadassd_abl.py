import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as nnFun

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
    def pre(self, closure=None):
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
                        state['prefer'] = 0
                        state['denom'] = torch.zeros(1,device='cuda')
                        state['exp_avg'] = torch.ones_like(p, memory_format=torch.preserve_format)*group['eps']
                        state['mean_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['init_param'] = p.clone()
                        state['l2_grad'] = torch.mean(p.grad**2).sqrt().add_(group['eps'])
                        state['init_avg_abs_param'] = torch.mean(torch.abs(state['init_param']))
                        # state['init_avg_abs_param'] = torch.mean((state['init_param']**2)).sqrt_()
                        if state['init_avg_abs_param'] < group['eps']:
                            # state['init_avg_abs_param'] = torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                            state['init_avg_abs_param'] = 0.01*init_avg_abs_param_buffer*torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                    
                        # state['init_smooth'] =torch.sum(torch.abs(p.grad*state['init_avg_abs_param']*group['lr']*p))/(torch.mean(p.grad*p.grad.conj()).sqrt()*torch.sum(p*p.conj()).sqrt()).add_(group['eps'])
                        init_avg_abs_param_buffer = state['init_avg_abs_param']

                    # update the steps for each param group update
                    state['step'] += 1
                    state['exp_avg'].mul_(0.99).add_(p.grad, alpha=1 - 0.99)
                    state['mean_exp_avg'].mul_(0.99).add_(p.grad*p.grad.conj(), alpha=1 - 0.99)
                    certainty = torch.abs(state['exp_avg']/state['mean_exp_avg'].sqrt().add(group['eps']))
                    # max = torch.max((certainty)).item()
                    # norm_mean = torch.mean((certainty/max))
                    cer_mean = torch.mean((certainty))
                    # weight_num = len(grads)
                    state['prefer'] = cer_mean
                    # self.convegence_factor = self.convegence_factor * (weight_num-1)/weight_num + norm_mean/weight_num
                    pass

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
                        # cur_convegence_factor = torch.mean(torch.relu(torch.mean(p.grad)-p.grad)/torch.mean((p.grad*p.grad.conj())).sqrt())
                        # self.convegence_factor = self.convegence_factor * state['step']/(1+state['step'])+cur_convegence_factor/(1+state['step'])
                        # max = torch.max(torch.abs(p.grad)).item()
                        # norm_mean = torch.mean(torch.log10(torch.abs(p.grad/max).add(group['eps'])))
                        # weight_num = len(grads)
                        # self.convegence_factor = self.convegence_factor * (weight_num-1)/weight_num + norm_mean/weight_num
                        state['denom'] = torch.zeros(1,device='cuda')
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)#*group['eps']
                        state['mean_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of gradient values
                        # state['cur_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['init_param'] = p.clone()
                        state['l2_grad'] = torch.mean(p.grad**2).sqrt().add_(group['eps'])
                        state['init_avg_abs_param'] = torch.mean(torch.abs(state['init_param']))
                        state['mask'] = self.set_random_to_zero(torch.abs(p.grad),20)
                        # state['mask'] = self.set_last_percent_to_zero(torch.abs(p.grad),20)
                        # state['init_avg_abs_param'] = torch.mean((state['init_param']**2)).sqrt_()
                        if state['init_avg_abs_param'] < group['eps']:
                            # state['init_avg_abs_param'] = torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                            state['init_avg_abs_param'] = 0.01*init_avg_abs_param_buffer*torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                    
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

    def set_random_to_zero(self, tensor, percent=10):
        # 创建一个与原张量形状相同的掩码张量，初始化为1（表示不修改）
        mask = torch.rand_like(tensor, memory_format=torch.preserve_format)
        if tensor.dim() > 2:
            # 将张量展平为一维以便排序
            flattened = mask.view(-1)
            # 计算需要置零的元素数量（总元素数量的百分之10）
            num_to_zero = int(flattened.numel() * (percent / 100.0))
            
            # 获取排序后的索引
            sorted_indices = torch.sort(flattened)[1]
            
            # 计算需要置零的元素的索引范围（从小到大排序后的最后百分之10）
            start_index = 0
            end_index = num_to_zero
            
            # 将掩码张量中对应最小百分之10元素的位置置为0（表示需要修改）
            mask.view(-1)[sorted_indices[start_index:end_index]] = 0
            mask.view(tensor.shape)
        
        return mask
        
    def set_last_percent_to_zero(self, tensor, percent=10):
        # 创建一个与原张量形状相同的掩码张量，初始化为1（表示不修改）
        mask = torch.ones_like(tensor, memory_format=torch.preserve_format)
        if tensor.dim() > 2:
            # 将张量展平为一维以便排序
            flattened = tensor.view(-1)
            # 计算需要置零的元素数量（总元素数量的百分之10）
            num_to_zero = int(flattened.numel() * (percent / 100.0))
            
            # 获取排序后的索引
            sorted_indices = torch.sort(flattened)[1]
            
            # 计算需要置零的元素的索引范围（从小到大排序后的最后百分之10）
            start_index = 0
            end_index = num_to_zero
            
            # 将掩码张量中对应最小百分之10元素的位置置为0（表示需要修改）
            mask.view(-1)[sorted_indices[start_index:end_index]] = 0
            mask.view(tensor.shape)
        
        return mask

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
                grad = grad.add(param, alpha=lr*weight_decay)

            # grad -= grad * torch.abs(avg_grad) / torch.sum(avg_grad*avg_grad.conj()).sqrt()
            # avg_grad = state['exp_avg']/(1 - 0.9 ** state['step'])
            # grad = torch.abs(grad)/torch.sum(grad*grad.conj()).sqrt() - avg_grad / (torch.sum(avg_grad*avg_grad.conj())+eps)
            # grad = ((1+torch.sign(grad))*grad)
            # mean_abs = torch.mean(torch.abs(grad))
            
            # ema_grad = state['mean_exp_avg'].mul_(0.9).add_(grad, alpha=(0.1))/(1 - 0.9 ** state['step'])
            # avg_grad = state['exp_avg'].mul_(0.9).add_(grad/torch.mean(torch.abs(grad)), alpha=(0.1))/(1 - 0.9 ** state['step'])
            # avg_grad = state['exp_avg']
            # grad = grad / (avg_grad+0.02*torch.abs(grad)+eps)
            # avg_grad = state['exp_avg'].mul_(0.5).add_(torch.abs(grad), alpha=(0.5))/(1 - 0.5 ** state['step'])                      
            # factor = 1-torch.abs(torch.sum(state['exp_avg']/torch.sum(state['exp_avg']**2).sqrt().add_(eps)*grad/torch.sum(grad**2).sqrt().add_(eps)))
            # avg_grad = state['exp_avg'].mul_(0.99).add_(grad+torch.sign(grad)*torch.mean(torch.abs(grad)), alpha=(0.01))
            # avg_grad = state['exp_avg'].mul_(0.9).add_(torch.sign(grad), alpha=(0.1))#*(1 - 0.9 ** state['step'])
            # mean_exp_avg = state['mean_exp_avg'].mul_(0.99).add_(torch.abs(grad), alpha=(0.01))
            # mean_exp_avg = state['mean_exp_avg'].mul_(0.99).add_(grad*grad.conj(), alpha=(0.01))#/(1 - 0.999 ** state['step'])
            # certainty = avg_grad/mean_exp_avg.sqrt().add_(eps)
            # certainty = avg_grad/mean_exp_avg.add_(eps)
            # self.convegence_factor = self.convegence_factor*0.999 + torch.mean(torch.abs(certainty))*0.001
            # weigh_grad = certainty
            # weigh_grad = certainty*grad + torch.sign(grad) * torch.mean(torch.abs(certainty*grad))
            # mean_certainty = state['prefer'] * torch.mean(torch.abs(certainty))
            # mean_certainty = torch.mean(torch.abs(certainty))
            # weigh_grad = (certainty + torch.sign(grad)*mean_certainty)#
            # weigh_grad = (avg_grad + torch.sign(grad)*torch.mean(torch.abs(avg_grad)))#
            weigh_grad = grad
            # sign = (0.5-0.5*torch.sign(torch.abs(certainty)-mean_certainty))
            # sign = (0.5-0.5*torch.sign(torch.abs(certainty)-state['prefer']))
            # a = torch.sum(sign)/torch.numel(sign)
            # weigh_grad = torch.abs(grad) / (torch.sign(certainty)*torch.abs(certainty).add_(0.001))
            # weigh_grad = ((weigh_grad) / torch.mean(weigh_grad**2).sqrt().add_(eps))
            # weigh_grad = weigh_grad/mean_exp_avg
            # weigh_grad = (sign * grad/torch.mean(grad**2).sqrt().add_(eps))
            # ratio = state['prefer']/(mean_certainty+0.01*state['prefer'])*0.01
            # weigh_grad = ((ratio) * weigh_grad / torch.mean(weigh_grad**2).sqrt().add_(eps)
                            #    + (1-ratio) * grad/torch.mean(grad**2).sqrt().add_(eps)) #
            # weigh_grad = (sign * weigh_grad / torch.mean(weigh_grad**2).sqrt().add_(eps)
                            #    + 0.01 * (1-sign) * grad/torch.mean(grad**2).sqrt().add_(eps)) #
            # weigh_grad = certainty
            # avg_grad = state['exp_avg'].mul_(0.9).add_(grad, alpha=(0.1))
            # avg_grad = state['exp_avg'].mul_(0.9).add_(grad, alpha=(0.1))
            
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
            # mean_exp_avg = state['mean_exp_avg'].mul_(0.999).add_(grad**2, alpha=(0.001))/(1 - 0.999 ** state['step'])
            # a = torch.sum((torch.abs(grad*avg_grad)/(torch.sum(grad*grad.conj())*torch.sum(avg_grad*avg_grad.conj())).sqrt()))
            # mean_grad = torch.mean((weigh_grad*weigh_grad.conj())).sqrt()
            # mean_grad1 = torch.mean((certainty*certainty.conj())).sqrt()
            #mean_grad2 = torch.mean((grad*grad.conj())).sqrt()
            # mean_grad = torch.mean((weigh_grad*weigh_grad.conj())).sqrt()
            # convergence = torch.mean((grad*grad.conj())).sqrt()/mean_grad
            # cfactor =100/(100+convergence)
            # mask = 0.5+0.5*torch.sign(mean_grad-torch.abs(grad))
            # mean_grad = torch.mean((mask*grad*grad.conj())).sqrt()
            # factor = 3**torch.sum(-avg_grad/torch.sum(avg_grad**2).sqrt().add_(eps)*grad/torch.sum(grad**2).sqrt().add_(eps))
            dynamics = weigh_grad#/mean_grad.add_(eps)#+grad/mean_grad2
            # self.convegence_factor = self.convegence_factor * 0.999+torch.sum(mean_exp_avg).sqrt()/torch.sum((grad*grad.conj())).sqrt().add_(eps)*0.001
            # self.convegence_factor = self.convegence_factor * 0.999+(torch.mean(torch.abs((grad**2/mean_exp_avg.add(eps)).sqrt()-torch.mean(grad**2/mean_exp_avg.add(eps)).sqrt())))*0.001
            # self.convegence_factor = self.convegence_factor * 0.999+(torch.mean(torch.abs(torch.log10((grad**2/mean_exp_avg.add(eps)).add(eps)))))*0.001
            # dynamics = avg_grad/mean_exp_avg.sqrt().add_(eps)
            # dynamics = grad
            # param.add_(dynamics*state['init_avg_abs_param']+0.001*param, alpha=-lr)#*cfactor
            # param.add_(dynamics*state['init_avg_abs_param'], alpha=-lr/self.convegence_factor)#*cfactor
            param.add_(dynamics*state['init_avg_abs_param'], alpha=-lr)#*cfactor
            # param.add_(dynamics*state['init_avg_abs_param'], alpha=-lr/state['prefer'])#*cfactor
            # param.add_(dynamics*state['init_avg_abs_param'], alpha=-lr)#*cfactor
            # param.add_(grad*state['mask'], alpha=-lr)#*cfactor
            # scfactor = (torch.sum(avg_grad/torch.sqrt(torch.sum(avg_grad**2))*torch.abs(grad)/torch.sqrt(torch.sum(grad**2)).add_(eps)))
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