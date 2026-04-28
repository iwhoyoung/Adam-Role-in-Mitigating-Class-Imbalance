from transformers import GPT2TokenizerFast
from datasets import Dataset, DatasetDict
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import random

def build_diverse_tokens_dataset(save_path="diverse_tokens_dataset", target_count=1000):
    # 初始化 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    end_token = tokenizer.eos_token_id  # 获取结束token
    
    # 加载训练集的前 10%
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10%]")
    print(f"Loaded dataset with {len(dataset)} samples")
    # 分词
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["input_ids"]) >= 3)  # 确保序列足够长
    print(f"Tokenized dataset with {len(tokenized_dataset)} valid samples")
    
    # 按结尾token分组
    end_token_groups = defaultdict(list)
    middle_token_groups = defaultdict(list)  # 用于存储按中间token分组的序列
    
    for seq in tokenized_dataset["input_ids"]:
        # 获取结尾token（排除结束token）
        last_token = seq[-1]
        if last_token != end_token:
            end_token_groups[last_token].append(seq)
        
        # 同时收集中间token的信息（为可能的补充做准备）
        # 取序列中间位置的token
        mid_pos = len(seq) // 2
        mid_token = seq[mid_pos]
        middle_token_groups[mid_token].append(seq)
    
    print(f"Found {len(end_token_groups)} unique ending tokens (excluding EOS)")
    
    # 收集具有不同结尾token的序列
    selected_sequences = []
    used_end_tokens = set()
    
    # # 首先从结尾token组中收集
    # for token, seqs in end_token_groups.items():
    #     if len(selected_sequences) >= target_count:
    #         break
    #     if token not in used_end_tokens:
    #         # 从该组随机选择一个序列
    #         selected_sequences.append(random.choice(seqs))
    #         used_end_tokens.add(token)
    
    # 如果还不够，使用中间token补充
    remaining = target_count - len(selected_sequences)
    if remaining > 0:
        print(f"Need {remaining} more samples, using middle tokens to supplement")
        used_mid_tokens = set()
        
        for token, seqs in middle_token_groups.items():
            if remaining <= 0:
                break
            if token not in used_mid_tokens and token not in used_end_tokens:
                selected_sequences.append(random.choice(seqs))
                used_mid_tokens.add(token)
                remaining -= 1
    
    # 如果仍然不够，放宽条件（允许部分重复但尽量多样化）
    if len(selected_sequences) < target_count:
        print(f"Still need {target_count - len(selected_sequences)} samples, using best effort")
        all_sequences = tokenized_dataset["input_ids"]
        random.shuffle(all_sequences)
        
        for seq in all_sequences:
            if len(selected_sequences) >= target_count:
                break
            selected_sequences.append(seq)
    
    # 确保我们有足够的样本
    final_count = len(selected_sequences)
    print(f"Collected {final_count} samples with diverse ending/middle tokens")
    
    # 构造新数据集
    new_dataset = Dataset.from_dict({"input_ids": selected_sequences})

    # 保存数据集到本地
    os.makedirs(save_path, exist_ok=True)
    new_dataset.save_to_disk(save_path)
    print(f"Saved {final_count} diverse token sequences to {save_path}")

    # 统计最终数据集的token多样性
    end_tokens = [seq[-1] for seq in selected_sequences]
    unique_end_count = len(set(end_tokens))
    print(f"Final dataset has {unique_end_count} unique ending tokens")

build_diverse_tokens_dataset()
    