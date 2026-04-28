import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
# import seaborn as sns  # 可选，用于更美观的样式

def compute_grad_vector(model, inputs, criterion):
    model.zero_grad()  # 清空过往梯度
    outputs = model(** inputs)  # 前向传播获取输出
    logits = outputs.logits  # 形状: [batch_size, seq_len, vocab_size]
    labels = inputs["input_ids"]  # 标签序列: [batch_size, seq_len]
    attention_mask = inputs.get("attention_mask", None)  # 注意力掩码: [batch_size, seq_len]
    batch_size, seq_len = labels.shape[:2]

    # 1. 计算每个样本的实际文本长度（非padding部分的长度）
    if attention_mask is None:
        # 如果没有attention_mask，默认全部是有效文本（兼容无掩码场景）
        actual_lengths = torch.full((batch_size,), seq_len, device=labels.device, dtype=torch.long)
    else:
        # 实际长度 = 掩码中1的数量（有效token数）
        actual_lengths = attention_mask.sum(dim=1).long()  # 形状: [batch_size]

    # 生成每个样本的随机位置（排除首尾）
    # 确保位置范围：1 <= pos <= seq_len-2（0是第一个，seq_len-1是最后一个）
    random_positions = torch.randint(low=2, high=seq_len-1, size=(batch_size,), device=labels.device)
    
    # 收集每个样本的随机位置对应的logits和labels
    selected_logits = []
    selected_labels = []
    for i in range(batch_size):
        pos = int(actual_lengths//2)
        # 取第pos位置的logits（预测下一个token）
        selected_logits.append(logits[i, pos, :])
        # 对应的标签是pos+1位置的token
        selected_labels.append(labels[i, pos + 1])

    # 转换为张量并计算损失
    selected_logits = torch.stack(selected_logits)  # 形状: [batch_size, vocab_size]
    selected_labels = torch.stack(selected_labels)  # 形状: [batch_size]
    loss = criterion(selected_logits, selected_labels)

    loss.backward()  # 反向传播计算梯度

    # 收集并合并所有参数的梯度向量
    grad_vector = []
    for p in model.parameters():
        if p.grad is not None:
            grad_vector.append(p.grad.detach().flatten().cpu())
    #grad_vector = torch.cat(grad_vector)
    return grad_vector

def compute_nao(g1, g2):
    norm1 = torch.norm(g1)
    norm2 = torch.norm(g2)
    if norm1 == 0 or norm2 == 0:
        return torch.tensor(0.0)
    return torch.sum(torch.abs(g1 * g2)) / (norm1 * norm2)

def main(
    dataset_path="diverse_tokens_dataset",
    save_csv_path="nao_distribution_gpt_adam_sbn.csv",
    model_name="/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_Sbn_0.05",
    step="0",
    opt="adam",
    device="cuda"
):
    # 1. 路径设置（包含层信息）
    save_csv_path = f'layer_nao_distribution_gpt_{opt}_{step}.csv'
    save_plot_path = f'layer_nao_plot_gpt_{opt}_{step}.png'
    if os.path.exists(save_csv_path) and os.path.exists(save_plot_path):
        print("Files already exist. Skipping.")
        return
    
    # 1. 加载数据和模型
    model_name=model_name+f"/checkpoint-{step}"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = load_from_disk(dataset_path)
   
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    param_names = [name for name, _ in model.named_parameters()]
    print(f"Loaded model with {len(param_names)} parameters.")

    i=0
    # 只输出前5个样本，因为后面代码中取了encoded=encoded[:5]
    # 2. 构造输入
    encoded = [torch.tensor(sample["input_ids"]) for sample in dataset]
    encoded = [tokenizer.pad({'input_ids': [e.tolist()]}, padding='max_length', max_length=512, return_tensors='pt') for e in encoded]
    encoded = [{k: v.to(device) for k, v in sample.items()} for sample in encoded]
    encoded=encoded[:100]

    # 3. 获取梯度向量
    grads = []
    criterion = torch.nn.CrossEntropyLoss()
    for example in tqdm(encoded, desc="Computing gradients"):
        grad = compute_grad_vector(model, example, criterion)
        grads.append(grad)

    # 6. 计算每层的平均NAO（所有样本对的统计）
    layer_nao_stats = {name: {'sum': 0.0, 'count': 0} for name in param_names}  # 层统计字典
    total_pairs = 0
    

    # for i in range(len(grads)):
    for i in tqdm(range(len(grads)), desc="Processing gradients", unit="grad"):
        for j in range(i+1, len(grads)):
            grad_i = grads[i]  # 样本i的梯度列表（每层一个张量）
            grad_j = grads[j]  # 样本j的梯度列表
            # 遍历每一层k（参数）
            # for k, g in enumerate(grads[i]):
            for k in range(len(grad_i)):
                g = grad_i[k].to(device)
                ig = grad_j[k].to(device)
                layer_nao = torch.sum(torch.abs(ig/(torch.sum((ig*ig.conj())).sqrt()))*torch.abs(g/torch.sum((g*g.conj())).sqrt()))  
                # 更新统计（累加NAO总和与样本对数量）
                layer_name = param_names[k]
                layer_nao_stats[layer_name]['sum'] += layer_nao
                layer_nao_stats[layer_name]['count'] += 1
                # if torch.isnan(nao).any() or torch.isinf(nao).any():
                #     continue
            total_pairs += 1
                       
    print(f"Total sample pairs: {total_pairs}")

    layer_nao_list = []
    for name, stats in layer_nao_stats.items():
        if stats['count'] == 0:
            continue
        avg_nao = stats['sum'] / stats['count']  # 每层的平均NAO
        layer_nao_list.append({
            'Layer Name': name,
            'Simplified Layer Name': name.replace('transformer.', '').replace('.', '_'),  # 简化层名称（方便绘图）
            'Average NAO': avg_nao.detach().cpu().item()
        })

    df_layer_nao = pd.DataFrame(layer_nao_list)
    # df_layer_nao = df_layer_nao.sort_values(by='Average NAO', ascending=False)  # 按NAO降序排列（可选）

    # 8. 保存CSV（包含层名称与平均NAO）
    df_layer_nao.to_csv(save_csv_path, index=False, columns=['Layer Name', 'Simplified Layer Name', 'Average NAO'])
    print(f"Saved layer-wise NAO to {save_csv_path}")

    # 9. 绘制折线图（展示每层平均NAO）
    plt.figure(figsize=(16, 8))
    plt.plot(
        df_layer_nao['Simplified Layer Name'],
        df_layer_nao['Average NAO'],
        marker='o',
        linestyle='-',
        color='#2ecc71'  # 绿色系（符合ENFP人格的活泼风格）
    )
    # 图表美化
    plt.xticks(rotation=90, fontsize=10)  # 旋转x轴标签（避免重叠）
    plt.yticks(fontsize=10)
    plt.ylim(-0.01,0.82)
    plt.xlabel('Simplified Layer Name', fontsize=12, fontweight='bold')
    plt.ylabel('Average NAO', fontsize=12, fontweight='bold')
    plt.title(f'Layer-wise Average NAO Distribution\n(opt: {opt}, step: {step})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)  # 网格线（增强可读性）
    plt.tight_layout()  # 自动调整布局（避免标签截断）
    # 保存图片（高分辨率）
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer-wise NAO plot to {save_plot_path}")


def plot_layer_nao_from_csv(
    csv_path="layer_nao_distribution_gpt_adam_0.csv",
    save_plot_path="layer_nao_distribution_gpt_adam_0.png",
    sort_by_layer=False  # 是否按层顺序排序（推荐开启）
):
    opt = 'adam'
    step = '0'
    # 1. 加载CSV数据
    try:
        df_layer_nao = pd.read_csv(csv_path)
        print(f"成功加载CSV文件：{csv_path}（共{len(df_layer_nao)}层）")
    except FileNotFoundError:
        print(f"错误：未找到文件{csv_path}，请检查路径是否正确。")
        return


    plt.figure(figsize=(21, 9))
    plt.plot(
        df_layer_nao['Simplified Layer Name'],
        df_layer_nao['Average NAO'],
        marker='o',
        linestyle='-',
        color='#2ecc71'  # 绿色系（符合ENFP人格的活泼风格）
    )
    # 图表美化
    plt.xticks(rotation=90, fontsize=10)  # 旋转x轴标签（避免重叠）
    plt.yticks(fontsize=10)
    plt.ylim(-0.01,0.82)
    plt.xlabel('Simplified Layer Name', fontsize=12, fontweight='bold')
    plt.ylabel('Average NAO', fontsize=12, fontweight='bold')
    plt.title(f'Layer-wise Average NAO Distribution\n(opt: {opt}, step: {step})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)  # 网格线（增强可读性）
    plt.tight_layout()  # 自动调整布局（避免标签截断）
    # 保存图片（高分辨率）
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    opt = ['adam']# 'adam_ini','adam_Sbn'
    i=0
    #"/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_bn_0.00005",
    # for each in ["/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_ini_0.005","/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_Sbn_0.05"]:
    plot_layer_nao_from_csv()
    # for each in ["/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_0.00005"]:
        
    #     # for step in ['0','5000', '10000', '15000',  '20000',  '25000', '30000',  '35000',  '40000',  '45000',  '50000']:
    #     for step in ['50000']:
    #         main(
    #         model_name=each,
    #         step=step,
    #         opt=opt[i],
    #         )
    #     i=i+1
    
