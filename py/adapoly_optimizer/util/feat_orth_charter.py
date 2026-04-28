import torch
from torch import nn
from typing import Optional
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
import einops
from tqdm import tqdm
import time
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib import colors  as mcolors
from matplotlib import collections  as mc
from dataclasses import dataclass, replace
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.patches import Rectangle

seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

@dataclass
class Config:
  n_features: int
  n_hidden: int

  # We optimize n_instances models in a single training loop
  # to let us sweep over sparsity or importance curves 
  # efficiently.

  # We could potentially use torch.vmap instead.
  n_instances: int
 
class Model(nn.Module):
  def __init__(self, 
               config, 
               feature_probability: Optional[torch.Tensor] = None,
               importance: Optional[torch.Tensor] = None,               
               device='cuda'):
    super().__init__()
    self.config = config
    self.W = nn.Parameter(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device))
    self.W2 = nn.Parameter(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device))
    nn.init.xavier_normal_(self.W)
    nn.init.xavier_normal_(self.W2)
    self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features), device=device))

    if feature_probability is None:
      feature_probability = torch.ones(())
    # feature_probability = 0.9*torch.ones(())
    self.feature_probability = feature_probability.to(device)
    if importance is None:
        importance = torch.ones(())
    self.importance = importance.to(device)

  def forward(self, features):
    # features: [..., instance, n_features]
    # W: [instance, n_features, n_hidden]
    hidden = torch.einsum("...if,ifh->...ih", features, self.W)
    out = torch.einsum("...ih,ifh->...if", hidden, self.W2)
    out = out + self.b_final
    # out = out.reshape(10000,5)
    out = F.relu(out)
    return out

  def generate_batch(self, n_batch, has_noise=False):
    feat = torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
    # feat = torch.rand((1, 5, self.config.n_instances, self.config.n_features), device=self.W.device)*torch.tensor([1,1,0,0,0],device=self.W.device).reshape()
    # feat = feat.repeat(200,1,1,1).reshape(1000,self.config.n_instances, self.config.n_features)

    # label = torch.arange(0, 5, device=self.W.device).reshape(1,5,1)
    # label = label.repeat(200,1,self.config.n_instances).reshape(10000)

    if has_noise:
       feat += 0.1*torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
       batch = feat
    batch = torch.where(
        torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device) <= self.feature_probability,
        feat,
        torch.zeros((), device=self.W.device),
    )
    return batch

def linear_lr(step, steps):
  return (1 - (step / steps))

def constant_lr(*_):
  return 1.0

def cosine_decay_lr(step, steps):
  return np.cos(0.5 * np.pi * step / (steps - 1))

def optimize(model, batch,
             render=False, 
             n_batch=1000,
             steps=10_000,
             print_freq=100,
             lr=1e-3,
             lr_scale=constant_lr,
             hooks=[]):
    cfg = model.config
    
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    # start = time.time()
    # 创建进度条
    progress = tqdm(range(steps), desc="Training")
    # batch = model.generate_batch(n_batch,has_noise=True)
    # criterion = nn.CrossEntropyLoss(model.importance.reshape(5))
    for step in progress:
        step_lr = lr * lr_scale(step, steps)
        for group in opt.param_groups:
            group['lr'] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(n_batch)
            out = model(batch)
            # error = (model.importance*(batch.abs() - out)**2)
            
            error = (model.importance*(batch.abs() - out)**2)
            loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
            # loss = criterion(out,label)
            loss.backward()
            opt.step()

        if hooks:
            hook_data = dict(model=model,
                                step=step, 
                                opt=opt,
                                error=error,
                                loss=loss,
                                lr=step_lr)
            for h in hooks:
                h(hook_data)
        if step % print_freq == 0 or (step + 1 == steps):
            progress.set_postfix(
                loss=loss.item() / cfg.n_instances,
                lr=step_lr,
            )

def cal_nao(model, batch,
             render=False, 
             n_batch=100,
             print_freq=40,
             hooks=[]
             ):
    cfg = model.config

    naos=[]
    group_num=10
    for _ in range(group_num):
        naos.append(0)
    model.eval()
    total_num = 0
    avg=0
    # 创建进度条
    progress = tqdm(range(n_batch), desc="Training")
    # batch = model.generate_batch(n_batch,has_noise=True)
    batch= batch[0:100]
    for i in progress:
        # batch = model.generate_batch(n_batch)
        out = model(batch[i])
        error = (model.importance*(batch.abs() - out)**2)
        loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
        loss.backward()
        init_grads=[]
        for param in model.parameters():
            init_grads.append(param.grad.clone().reshape(-1))
            param.grad.zero_()
        for j in range(i+1,n_batch):
          total_num += 1
          out = model(batch[j])
          error = (model.importance*(batch.abs() - out)**2)
          loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
          loss.backward()
          nao=0
          num=0
          for k,(param) in enumerate(model.parameters()):
              if k>1:
                 break
              g = param.grad.reshape(-1)
              next_num = num+g.size()[0]
              nao = nao*num/next_num+ g.size()[0]/next_num*torch.sum(torch.abs(init_grads[k]/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()))*torch.abs(g/torch.sum((g*g.conj())).sqrt()))
              param.grad.zero_()
              num = next_num    
          index = 0
          if not (torch.isinf(nao).any() or torch.isnan(nao).any()):
              avg += nao.item()
          while nao>1/group_num and index<9:
              index += 1
              nao -= 1/group_num
          naos[index] += 1
        if i % print_freq == 0 or (i + 1 == n_batch):
            progress.set_postfix(
                loss=loss.item() / cfg.n_instances,
            )
    for i in range(len(naos)):
        naos[i] = naos[i] / total_num
    naos.append(avg/ total_num)
    pd.DataFrame(data=naos).to_csv('/lichenghao/huY/adapoly_optimizer/submit/'+'W1w2_noimp_%dbin_naodis_seed%d.csv'% (group_num,seed))

def plot_intro_diagram(model, batch):

    cfg = model.config
    # model.eval()
    # out = model(batch[0])
    # error = (model.importance*(batch.abs() - out)**2)
    # loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
    # loss.backward()
    WA = model.W2.detach()
    # WA = 0.5*(model.W.detach()+model.W2.detach())
    N = len(WA[:,0])
    sel = range(config.n_instances) # can be used to highlight specific sparsity levels
    importance = (0.8**torch.arange(config.n_features))[None, :]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(importance[0].cpu().numpy()))
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(model.importance[0].cpu().numpy()))
    plt.rcParams['figure.dpi'] = 200
    fig, axs = plt.subplots(1,len(sel), figsize=(2*len(sel),2))
    for i, ax in zip(sel, axs):
        W = WA[i].cpu().detach().numpy()
        colors = [mcolors.to_rgba(c)
                for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        ax.scatter(W[:,0], W[:,1], c=colors[0:len(W[:,0])])
        ax.set_aspect('equal')
        ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W),W), axis=1), colors=colors))
        
        z = 1.5
        ax.set_facecolor('gray')
        ax.set_xlim((-z,z))
        ax.set_ylim((-z,z))
        ax.tick_params(left = True, right = False , labelleft = False ,
                    labelbottom = False, bottom = True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_position('center')

        # 调整刻度位置
        ax.xaxis.set_ticks_position('bottom')  # x刻度在底部轴线显示
        # ax.yaxis.set_ticks_position('left')    # y刻度在左侧轴线显示

        ax.patch.set(
        linewidth=1.5, 
        edgecolor='gray',
        linestyle='-',
        facecolor='#FCFBF8',
        alpha=0.7,
        fill=True 
        )

    
    # fig.set_size_inches(6, 3)
    # plt.tight_layout()

    plt.savefig('/lichenghao/huY/adapoly_optimizer/submit/w2data_noimp_6feat_orth_10000i_seed%d.png'% seed, dpi=400, bbox_inches = 'tight')

def render_features(model, which=np.s_[:]):
    cfg = model.config
    W = model.W.detach()
    W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

    interference = torch.einsum('ifh,igh->ifg', W_norm, W)
    interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

    polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
    net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
    norms = torch.linalg.norm(W, 2, dim=-1).cpu()

    WtW = torch.einsum('sih,soh->sio', W, W).cpu()

    # width = weights[0].cpu()
    # x = torch.cumsum(width+0.1, 0) - width[0]
    x = torch.arange(cfg.n_features)
    width = 0.9

    which_instances = np.arange(cfg.n_instances)[which]
    fig = make_subplots(rows=len(which_instances),
                        cols=2,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        horizontal_spacing=0.1)
    for (row, inst) in enumerate(which_instances):
        fig.add_trace(
            go.Bar(x=x, 
                y=norms[inst],
                marker=dict(
                    color=polysemanticity[inst],
                    cmin=0,
                    cmax=1
                ),
                width=width,
            ),
            row=1+row, col=1
        )
        data = WtW[inst].numpy()
        fig.add_trace(
            go.Image(
                z=plt.cm.coolwarm((1 + data)/2, bytes=True),
                colormodel='rgba256',
                customdata=data,
                hovertemplate='''\
    In: %{x}<br>
    Out: %{y}<br>
    Weight: %{customdata:0.2f}
    '''            
            ),
            row=1+row, col=2
        )

    fig.add_vline(
        x=(x[cfg.n_hidden-1]+x[cfg.n_hidden])/2, 
        line=dict(width=0.5),
        col=1,
    )
        
    # fig.update_traces(marker_size=1)
    fig.update_layout(showlegend=False, 
                        width=600,
                        height=100*len(which_instances),
                        margin=dict(t=0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

@torch.no_grad()
def compute_dimensionality(W):
    norms = torch.linalg.norm(W, 2, dim=-1) 
    W_unit = W / torch.clamp(norms[:, :, None], 1e-6, float('inf'))

    interferences = (torch.einsum('eah,ebh->eab', W_unit, W)**2).sum(-1)

    dim_fracs = (norms**2/interferences)
    return dim_fracs.cpu()

if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    config = Config(
        n_features = 5,
        n_hidden = 2,
        n_instances = 6,
    )

    model = Model(
        config=config,
        device=DEVICE,
        # Exponential feature importance curve from 1 to 1/100
        importance = (0.8**torch.arange(config.n_features))[None, :],
        # importance = (torch.ones(config.n_features))[None, :],
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None]
        # feature_probability = (1-0.2*torch.arange(config.n_features))[:, None]
    )
    batch = model.generate_batch(1000,has_noise=True)
    optimize(model, batch)
    # cal_nao(model, batch)
    plot_intro_diagram(model, batch)


    # config = Config(
    # n_features = 200,
    # n_hidden = 20,
    # n_instances = 20,
    # )

    # model = Model(
    #     config=config,
    #     device=DEVICE,
    #     # For this experiment, use constant importance.

    #     # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
    #     feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None]
    # )
    # batch = model.generate_batch(1024,has_noise=True)
    # optimize(model, batch)
    # fig = px.line(
    #     x=1/model.feature_probability[:, 0].cpu(),
    #     y=(model.config.n_hidden/(torch.linalg.matrix_norm(model.W.detach(), 'fro')**2)).cpu(),
    #     log_x=True,
    #     markers=True,
    # )
    # fig.update_xaxes(title="1/(1-S)")
    # fig.update_yaxes(title=f"m/||W||_F^2")

    # dim_fracs = compute_dimensionality(model.W)
    # fig = go.Figure()

    # density = model.feature_probability[:, 0].cpu()
    # W = model.W.detach()

    # for a,b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
    #     val = a/b
    #     fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))

    # for a,b in [(5,6), (4,5), (3,4), (3,8), (3,12), (3,20)]:
    #     val = a/b
    #     fig.add_hline(val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05))

    # for i in range(len(W)):
    #     fracs_ = dim_fracs[i]
    #     N = fracs_.shape[0]
    #     xs = 1/density
    #     if i!= len(W)-1:
    #         dx = xs[i+1]-xs[i]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=1/density[i]*np.ones(N)+dx*np.random.uniform(-0.1,0.1,N),
    #             y=fracs_,
    #             marker=dict(
    #                 color='black',
    #                 size=1,
    #                 opacity=0.5,
    #             ),
    #             mode='markers',
    #         )
    #     )

    # fig.update_xaxes(
    #     type='log', 
    #     title='1/(1-S)',
    #     showgrid=False,
    # )
    # fig.update_yaxes(
    #     showgrid=False
    # )
    # fig.update_layout(showlegend=False)
    # fig.write_image('/lichenghao/huY/adapoly_optimizer/submit/feat_orth2_08_seed%d.png'% seed, scale=3)

    # config = Config(
    # n_features = 100,
    # n_hidden = 20,
    # n_instances = 20,
    # )

    # model = Model(
    #     config=config,
    #     device=DEVICE,
    #     # Exponential feature importance curve from 1 to 1/100
    #     importance = (100 ** -torch.linspace(0, 1, config.n_features))[None, :],
    #     # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
    #     feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None]
    # )

    # fig = render_features(model, np.s_[::2])
    # fig.update_layout()
    # fig.write_image('/lichenghao/huY/adapoly_optimizer/submit/feat_orth2_08_seed%d.png'% seed, scale=3)