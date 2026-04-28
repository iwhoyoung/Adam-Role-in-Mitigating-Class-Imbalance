import math
from matplotlib import pyplot as plt
import torch
import pandas as pd
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from torch.nn.parameter import Parameter

def get_param_mean_and_std(path, model):
    """
    calculate the mean and std. of the parameters.
    :path: where the file is saved.
    :model: which model the param. belong to.
    """
    data = []
    for param in model.parameters():
        std, mean = torch.std_mean(param)
        data.append([mean.item(), std.item()])
    df_datas = pd.DataFrame(data=data, columns=['mean', 'std'])
    df_datas.to_csv(path + '.csv')

def get_param_learning_utility(path, model, datalodaer, criterion, lr):
    """
    read csv file.
    :param path: where the file is.
    :return: return the result of reading file.
    """
    data = []
    calculator = Learning_utility_calculator(model.parameters(), lr)
    for i, (input, target) in enumerate(datalodaer):
        input = input.cuda()
        target = target.cuda()
        classes = model(input)
        loss = criterion(classes, target)
        calculator.zero_grad()
        loss.backward()
        calculator.step()   
    data = calculator.calculate_learning_utility()
    df_datas = pd.DataFrame(data=data, columns=['mean_lu','std_log_lu','std_lu'])
    df_datas.to_csv(path + '.csv')

class Learning_utility_calculator(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.learning_utility = Parameter(torch.zeros(1,device='cuda'))
        super(Learning_utility_calculator, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Learning_utility_calculator, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]      
                    if len(state) == 0:
                        state['step'] = 0
                        state['grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    state['step'] += 1
                    grad = lr * p.grad
                    state['grad'].add_(grad)

    @torch.no_grad()
    def calculate_learning_utility(self, closure=None):
        learning_utility = []
        for group in self.param_groups:            
            for p in group['params']:
                state = self.state[p]
                avg_grad = state['grad'] / state['step']
                abs_avg_grad = torch.abs(avg_grad)
                lu = abs_avg_grad/(torch.abs(p)+abs_avg_grad)
                learning_utility.append([torch.mean(lu).item(), torch.std(torch.log10(lu)).item(), torch.std(lu).item()])
        return learning_utility

    def zero_grads(self):
        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    p.grads = []

def sum(bin,data):
    # num = torch.sum(torch.sign(torch.abs(data)+1e-5),dim=1,keepdim=True)
    min = torch.min(data,dim=1,keepdim=True).values
    max = torch.max(data,dim=1,keepdim=True).values
    stride = (max - min)/bin
    # axis_x = torch.linspace(torch.min(data).item()+stride*0.5, torch.max(data).item()-stride*0.5, bin)
    axis_x = torch.linspace(max.item()/bin, max.item(), bin).unsqueeze(1)
    axis_y = torch.linspace(0, min.size(0)-1, min.size(0)).unsqueeze(0)
    axis_x = axis_x.repeat(1,min.size(0))
    axis_y = axis_y.repeat(bin,1)
    pre = 0
    result = None
    for i in range(bin):
        cur = torch.sum(0.5-0.5*torch.sign(data - (min+(i+1)*stride)),dim=1,keepdim=True)
        if result is None:
            result = cur
        else:
            result = torch.cat([result,cur - pre],dim=1) 
        pre = cur
    return result, axis_x, axis_y

class distribution_calculator(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.99, 0.99), eps=1e-8,
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
        super(distribution_calculator, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(distribution_calculator, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # fig=plt.figure()
        # ax1=plt.axes(projection='3d')
        for group in self.param_groups:
            eps = group['eps']
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            beta1, beta2 = group['betas']
            df_grads = pd.DataFrame()
            grad_maxxz = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]     
                    if len(state) == 0:
                        state['step'] = 0
                        # state['grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_abs'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['init_avg_abs_param'] = torch.mean(torch.abs(p.clone()))
                        if state['init_avg_abs_param'] < group['eps']:
                            state['init_avg_abs_param'] = 0.01*torch.ones_like(state['init_avg_abs_param'], memory_format=torch.preserve_format)
                    
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state['step'] += 1

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    beta1 = beta2 = 0.99
                    state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad.conj(), value=1 - beta2)
                    # state['exp_avg_sq'].mul_(beta2).add_(torch.abs(p.grad), alpha=1 - beta2)
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    # denom = (state['exp_avg_sq'] / bias_correction2).add_(eps)
                    # state['exp_avg'].mul_(beta1).add_(torch.sign(p.grad)*torch.log(1+torch.exp((p.grad/denom)**2-1)).sqrt()*denom, alpha=1 - beta1)
                    # redistribution = (p.grad/denom)/(1+torch.exp(-8*(torch.abs(p.grad/denom)-1)))
                    # state['exp_avg_abs'].mul_(beta1).add_(redistribution, alpha=1 - beta1)
                    # state['exp_avg'].mul_(beta2).addcmul_(redistribution, redistribution.conj(), value=1 - beta2)

                    if state['step'] == 16:
                        # reshaped_model_output = p.grad.reshape(1,-1)
                        # reshaped_model_output = (p.grad/denom).reshape(1,-1)
                        # reshaped_model_output = (p.grad/denom).reshape(1,-1)
                        # reshaped_model_output = (redistribution/(state['exp_avg'].sqrt()/math.sqrt(bias_correction2)).add_(eps)).reshape(1,-1)
                        # reshaped_model_output = (torch.sign(p.grad)*torch.log(1+torch.exp(torch.abs(p.grad/denom)-1))).reshape(1,-1)        
                        # reshaped_model_output = (torch.sign(p.grad)*5/(1+torch.exp(-8*(p.grad/denom-1.2)))).reshape(1,-1)        
                        
                        # reshaped_model_output = ((p.grad/denom)/(1+torch.exp(-10*(torch.abs(p.grad/denom)-1)))).reshape(1,-1)        
                        # reshaped_model_output = ((state['exp_avg']/bias_correction1)).reshape(1,-1)
                        # reshaped_model_output = ((state['exp_avg']/state['exp_avg_sq'].sqrt().add_(eps))+torch.sign(p.grad)*torch.mean(torch.abs(state['exp_avg']/state['exp_avg_sq'].sqrt().add_(eps)))).reshape(1,-1)
                        # reshaped_model_output = ((state['exp_avg']/state['exp_avg_sq'].sqrt().add_(eps))).reshape(1,-1)
                        reshaped_model_output = ((state['exp_avg']/state['exp_avg_sq'].sqrt().add_(eps))+torch.sign(p.grad)*torch.mean(torch.abs(state['exp_avg'])/state['exp_avg_sq'].sqrt().add_(eps))).reshape(1,-1)
                        # reshaped_model_output = (torch.sign(state['exp_avg'])*(torch.abs(state['exp_avg']/bias_correction1)/denom)**1).reshape(1,-1)
                        # reshaped_model_output = (p.grad/torch.mean((p.grad*p.grad.conj())).sqrt().add_(eps)).reshape(1,-1)
                        # dynamics = state['exp_avg']/torch.mean((p.grad*p.grad.conj())).sqrt().add_(eps)/bias_correction1
                        # reshaped_model_output = (dynamics*state['init_avg_abs_param']+torch.mean(torch.abs(dynamics))*p).reshape(1,-1)
                        bins, axis_x, axis_y = sum(30,torch.abs(reshaped_model_output))
                        # ax1.plot_wireframe(axis_x.T, axis_y.T, bins.detach().cpu(), rstride=10, cstride=0)
                        # # ax1.set_title("Column (x) stride set to 0")
                        maxz = int(torch.max(bins.detach()).item())
                        # normlize the frequency
                        bins_sum = torch.sum(bins,dim=1)
                        bins = bins / bins_sum
                        # ax1.set_zlim(0, maxz)
                        # # ax1.set_ylim(0, 8)
                        # plt.tight_layout()    
                        # plt.savefig('/lichenghao/huY/ada_optimizer/submit/wireframe_flower17_sgdwnorm_batch256_maxz%d_maxx%f.png' % (maxz, torch.max(axis_x.detach()).item()), dpi=400, bbox_inches = 'tight')
                        df_grads = pd.concat([df_grads, pd.Series(torch.flatten(bins).detach().cpu().numpy()).to_frame().T], ignore_index=True)
                        # df_grads = df_grads.append(pd.DataFrame(bins.detach().cpu().numpy()), ignore_index=True)
                        grad_maxxz.append([maxz, torch.max(axis_x.detach()).item()])
                        pass
                    # dynamics = ((state['exp_avg']/state['exp_avg_sq'].sqrt().add_(eps)))
                    # dynamics = ((state['exp_avg']/bias_correction1)/denom)
                    # dynamics = p.grad
                    # p.add_(dynamics, alpha=-group['lr'])
                    # df_params.to_csv('/lichenghao/huY/ada_optimizer/submit/csv_flower17_resnet18_all_sgdwnorm_batch16_maxz%d_maxx%f.csv' % (maxz, torch.max(axis_x.detach()).item()))
            if len(grad_maxxz) > 0:
                df_grads.to_csv('/lichenghao/huY/ada_optimizer/submit/csv_%s.csv'%'cifar10_untrained_vits_certaintywpenalty_batch256')
                pd.DataFrame(data=grad_maxxz, columns=['maxz','maxx']).to_csv('/lichenghao/huY/ada_optimizer/submit/maxxz_%s.csv'%'cifar10_untrained_vits_certaintywpenalty_batch256')#adassdwnorm2unbias
                pass
        
def plot_wireframe3D(path,max_path):
    data = pd.read_csv(path, index_col=0)
    maxxz = pd.read_csv(max_path, index_col=0)['maxx']
    tensor = torch.Tensor(data.to_numpy())
    maxxz = torch.Tensor(maxxz.to_numpy())
    selected_layers = {'vgg16bn':[0,14,28,42,56],
                       'res18':[0,12,27,42,60],
                       'res50':[0,39,81,123,159],
                       'swint':[0,48,93,131,154],
                       'vits':[2,54,124,192,198],
                       'vitb':[2,54,124,192,198]}
    index =  selected_layers.get('vits')
    tensor = tensor[index].T
    # cifar100_res18v_adacertainty [0.4337, 0.4065, 0.4272, 0.4737, 0.5676] 
    # cifar100_vits_adacertainty [0.4356, 0.4404, 0.4324, 0.4120, 0.4521]
    maxxz = maxxz[index].T 
    plt.rcParams.update({'font.size': 14})
    fig=plt.figure()
    ax1=plt.axes(projection='3d')
    bin = tensor.size(0)
    axis_x = torch.linspace(1/bin, 1, bin).unsqueeze(1)
    axis_y = torch.linspace(0, tensor.size(1)-1, tensor.size(1)).unsqueeze(0)
    axis_x = axis_x.repeat(1,tensor.size(1))
    # sgd
    # axis_x = torch.Tensor([[0.00108108890708535,0.000510641664732247,0.000527678814250975
                            # ,0.000264126603724434,0.00966393388807773]])*axis_x
    # axis_x = torch.Tensor([[0.0252779964357614,0.0139988483861088,0.00943799689412117
                            # ,0.00908558070659637,0.155230745673179]])*axis_x
    # axis_x = torch.Tensor([[3.15845584869384,1.0704401731491,1.21274173259735
                            # ,2.08395981788635,63.8488883972167]])*axis_x
    # axis_x = torch.Tensor([[0.00498544983565807,0.00299232080578804,0.00183933519292622
                            # ,0.00123918405734002,0.507948815822601]])*axis_x
    # adassdwoscale resnet18 vit
    # axis_x = torch.Tensor([[4.71193695068359,9.91311454772949,12.3599405288696
                            # ,14.1519060134887,6.78339910507202]])*axis_x
    # axis_x = torch.Tensor([[4.4642014503479,14.5310335159301,16.1173458099365
                            # ,7.03437662124633,10.0304269790649]])*axis_x
    # adassdnorm2woscale resnet18 
    # axis_x = torch.Tensor([[2.61335277557373,3.80792760848999,6.02343702316284
                            # ,7.29957914352416,2.52504181861877]])*axis_x
    # axis_x = torch.Tensor([[2.66829586029052,3.68825554847717,4.15382480621337
    #                         ,3.96915364265441,2.86666011810302]])*axis_x
    
    axis_x = torch.Tensor([[1,1,1,1,1]])*axis_x
    # axis_x = maxxz*axis_x
    axis_y = axis_y.repeat(bin,1)        
    ax1.plot_wireframe(axis_x.T, axis_y.T, tensor.T, rstride=1, cstride=0)
    # maxz = int(torch.max(tensor).item())
    maxz = torch.max(tensor).item()
    # ax1.set_zlim(0, maxz)
    ax1.set_zlim(0, 0.6)
    # ax1.set_ylim(0, 8)
    selected_depth = {'vgg16bn':['1','8','15','22','29'],
                       'res18':['1','6','11','16','21'],
                       'res50':['1','14','28','42','54'],
                       'swint':['1','17','32','45','54'],
                       'vits':['1','14','31','48','50'],
                       'vitb':['1','14','31','48','50']}
    plt.yticks(torch.linspace(0, tensor.size(1)-1, tensor.size(1)),selected_depth.get('vits'))#'1','5','9','13','18'
    # plt.yticks(torch.linspace(0, tensor.size(1)-1, tensor.size(1)),)#'1','14','26','39','50'
    plt.xlabel('Norm. Abs. Dynamics')#Normalized Absolute Dynamics
    plt.ylabel('Depth')
    ax1.set_zlabel('Frequency')
    # plt.gcf().set_size_inches(6,6)
    # 调整位置 bbox_inches=None
    # fig.subplots_adjust(left=0.1)
    plt.tight_layout()    
    plt.savefig('/lichenghao/huY/ada_optimizer/log/wireframe_%s_16i.png' % 'cifar10_untrained_vits_certaintywpenalty_batch256', dpi=200, bbox_inches = 'tight',pad_inches=0.2)

# demo
if __name__ == '__main__':
    plot_wireframe3D('/lichenghao/huY/ada_optimizer/submit/csv_%s.csv'%'cifar10_untrained_vits_certaintywpenalty_batch256',
                     '/lichenghao/huY/ada_optimizer/submit/maxxz_%s.csv'%'cifar10_untrained_vits_certaintywpenalty_batch256')