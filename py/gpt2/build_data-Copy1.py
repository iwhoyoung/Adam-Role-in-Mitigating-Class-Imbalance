from transformers import GPT2TokenizerFast
from datasets import Dataset, DatasetDict
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import os

def build_one_token_per_group_dataset(save_path="one_token_per_group_dataset"):
    # 初始化 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 加载训练集的前 10%
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10%]")

    # 分词
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["input_ids"]) > 0)

    # 统计 token 频率
    all_tokens = [token for seq in tokenized_dataset["input_ids"] for token in seq]
    token_freq = Counter(all_tokens)
    sorted_tokens = [token for token, _ in token_freq.most_common()]
    total_tokens = sum(token_freq.values())
    num_groups = 10
    group_size = total_tokens // num_groups

    # 分组
    token_to_group, token_sum, group_id = {}, 0, 0
    for token in sorted_tokens:
        token_to_group[token] = group_id
        token_sum += token_freq[token]
        if token_sum > (group_id + 1) * group_size and group_id < num_groups - 1:
            group_id += 1

    # 每组只取一个代表 token（频率最高的）
    group_representative_tokens = {}
    for token in sorted_tokens:
        group = token_to_group[token]
        if group not in group_representative_tokens:
            group_representative_tokens[group] = token

    # 构造新数据集
    token_sequences = [[tok] for tok in group_representative_tokens.values()]
    new_dataset = Dataset.from_dict({"input_ids": token_sequences})

    # 保存数据集到本地
    os.makedirs(save_path, exist_ok=True)
    new_dataset.save_to_disk(save_path)
    print(f"Saved 1-token-per-group dataset to {save_path}")

build_one_token_per_group_dataset()
