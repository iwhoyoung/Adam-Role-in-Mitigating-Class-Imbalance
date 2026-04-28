import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import os
from tqdm import tqdm
import argparse

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
    dataset_path="/home/wangjzh/adam_optimizer/py/gpt2/diverse_tokens_dataset",
    save_csv_path="/home/wangjzh/adam_optimizer/py/gpt2/nao_distribution_gpt_adam_sbn.csv",
    model_name="/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_Sbn_0.05",
    step="0",
    opt="adam",
    device="cuda"
):
    save_csv_path=f'/home/wangjzh/adam_optimizer/py/gpt2/nao_distribution_gpt_{opt}_{step}.csv'
    if os.path.exists(save_csv_path):return
    
    # 1. 加载数据和模型
    model_name=model_name+f"/checkpoint-{step}"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = load_from_disk(dataset_path)
   
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    print("数据集样本文本内容：")
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

    # 4. 计算所有 pair 的 NAO 值并分桶（10 bin）
    nao_bins = [0] * 10
    total_pairs = 0
    nao_sum = 0.0
    group_num=10
    # for i in range(len(grads)):
    for i in tqdm(range(len(grads)), desc="Computing nao values"):
        for j in range(i+1, len(grads)):
            nao=0
            num=0
            #nao = compute_nao(grads[i], grads[j]).item()
            init_grads = grads[j]
            for k, g in enumerate(grads[i]):
                g = g.to("cuda")
                ig = init_grads[k].to("cuda")
                next_num = num+g.size()[0]
                # nao = nao*num/next_num+ g.size()[0]/next_num*torch.sum(torch.abs(init_grads[k]/(torch.sum((init_grads[k]*init_grads[k].conj())).sqrt()))*torch.abs(g/torch.sum((g*g.conj())).sqrt()))
                nao = nao*num/next_num+ g.size()[0]/next_num*torch.sum(torch.abs(ig/(torch.sum((ig*ig.conj())).sqrt()))*torch.abs(g/torch.sum((g*g.conj())).sqrt()))
                num = next_num    
            index = 0
            
            if torch.isnan(nao).any() or torch.isinf(nao).any():
                continue
            total_pairs += 1
            nao_sum += nao.item()
            # print('----')
            # print(i)
            # print(j)
            # print(nao)
            while nao>1/group_num and index<9:
                index += 1
                nao -= 1/group_num
            nao_bins[index] += 1
            
            
                

    # 5. 归一化频率 + 添加平均值 + 保存
    nao_bins = [v / total_pairs for v in nao_bins]
    mean_nao = nao_sum / total_pairs
    nao_bins.append(mean_nao)

    df = pd.DataFrame(nao_bins)
    df.to_csv(save_csv_path, index=False, header=False)
    print(f"Saved NAO distribution to {save_csv_path}")

if __name__ == "__main__":
    opt = ['adam']# 'adam_ini','adam_Sbn'
    i=0
    #"/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_bn_0.00005",
    # for each in ["/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_ini_0.005","/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_Sbn_0.05"]:
    for each in ["/home/wangjzh/adam_optimizer/py/gpt/knn-transformers/finetune_gpt2_wikitext103_4gpu_adam_0.00005"]:
        
        for step in ['0','5000', '10000', '15000',  '20000',  '25000', '30000',  '35000',  '40000',  '45000',  '50000']:
            main(
            model_name=each,
            step=step,
            opt=opt[i],
            )
        i=i+1
