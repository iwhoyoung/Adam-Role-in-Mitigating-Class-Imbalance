#!/usr/bin/env python
"""
Pure PyTorch evaluation loop to compute group losses without OOM.
"""

from __future__ import annotations
from transformers import AutoConfig
import argparse
import logging
import os
import re
import pandas as pd
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def build_token_groups(datasets_dict: DatasetDict, num_groups: int = 10, col: str = "input_ids"):
    freq = Counter()
    logger.info("Counting token frequencies to create %d groups …", num_groups)
    for split in datasets_dict:
        freq.update(chain.from_iterable(datasets_dict[split][col]))

    sorted_items = freq.most_common()
    total = sum(freq.values())
    step = total / num_groups

    token_to_group: Dict[int, int] = {}
    group_token_lists: List[List[int]] = [[] for _ in range(num_groups)]

    acc = 0
    gid = 0
    for tok, f in sorted_items:
        token_to_group[tok] = gid
        group_token_lists[gid].append(tok)
        acc += f
        if acc >= (gid + 1) * step and gid < num_groups - 1:
            gid += 1

    group_token_ids = [torch.tensor(toks, dtype=torch.long) for toks in group_token_lists]
    return group_token_ids, token_to_group


def tokenize_corpus(dataset, tokenizer, block_size):
    column_names = list(dataset.features)
    text_col = "text" if "text" in column_names else column_names[0]

    def _tokenize(batch):
        return tokenizer(batch[text_col])

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing",
    )

    def _group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = total_len // block_size * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_len, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(
        _group_texts,
        batched=True,
        desc="Grouping into blocks",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_png", type=str, default="group_losses.png")
    parser.add_argument("--output_csv", type=str, default="group_losses.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_root = Path(args.ckpt_root).expanduser().resolve()
    ckpt_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and re.match(r"checkpoint-\d+", p.name)],
        key=lambda p: int(p.name.split("-")[1]),
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dirs[0])
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    elif args.eval_file:
        extension = os.path.splitext(args.eval_file)[1].lstrip(".")
        extension = "text" if extension == "txt" else extension
        dataset = load_dataset(extension, data_files=args.eval_file, split="train")
    else:
        raise ValueError("Provide either --dataset_name or --eval_file.")

    if args.max_eval_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_eval_samples)))

    tokenized = tokenize_corpus(dataset, tokenizer, args.block_size)
    group_token_ids, token_to_group = build_token_groups(DatasetDict({"train": tokenized}))

    dataloader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)

    all_metrics: Dict[str, List[float]] = defaultdict(list)
    steps: List[int] = []
    i=0
    for ckpt_dir in ckpt_dirs:
        step = int(ckpt_dir.name.split("-")[1])
        steps.append(step)
        logger.info("Evaluating %s (step %d)", ckpt_dir.name, step)
        if i==0:
            config = AutoConfig.from_pretrained(
                "gpt2",
                vocab_size=50257,
                n_layer=12,
                n_head=12,
                n_embd=768,
            )
            model = AutoModelForCausalLM.from_config(config).cuda().eval()
            i=1
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt_dir).cuda().eval()

        total_loss = 0.0
        total_count = 0
        group_losses = [0.0 for _ in range(len(group_token_ids))]
        group_counts = [0 for _ in range(len(group_token_ids))]

        # 创建token到组的映射张量，用于GPU上的快速查找
        max_token_id = max(token_to_group.keys())
        token_to_group_tensor = torch.zeros(max_token_id + 1, dtype=torch.long).cuda()
        for tok, gid in token_to_group.items():
            token_to_group_tensor[tok] = gid

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()

                outputs = model(input_ids=input_ids)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                token_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.size())

                total_loss += token_loss.sum().item()
                total_count += token_loss.numel()

                # 扁平化标签和损失，便于按组统计
                flat_labels = shift_labels.view(-1)
                flat_loss = token_loss.view(-1)

                # 按组累加损失
                for gid in range(len(group_token_ids)):
                    mask = (token_to_group_tensor[flat_labels] == gid)
                    if mask.any():
                        group_losses[gid] += flat_loss.masked_select(mask).sum().item()
                        group_counts[gid] += mask.sum().item()

        all_metrics["total_loss"].append(total_loss / total_count)
        for gid in range(len(group_token_ids)):
            if group_counts[gid] > 0:
                all_metrics[f"group_{gid}_loss"].append(group_losses[gid] / group_counts[gid])
            else:
                all_metrics[f"group_{gid}_loss"].append(0.0)

        del model
        print(group_counts)
        print(all_metrics)
        torch.cuda.empty_cache()
        
    df = pd.DataFrame(all_metrics)
    df.to_csv(args.output_csv, index=False)
    logger.info("Saved metrics to %s", args.output_csv)
    #all_metrics = pd.read_csv('adam_ini_0.0001.csv')

    plt.figure(figsize=(10, 6))
    for metric_name, values in sorted(all_metrics.items()):
        plt.plot(steps, values, label=metric_name.replace("_", " "))

    plt.xlabel("Training step (checkpoint)")
    plt.ylabel("Loss")
    plt.title("Per-group cross-entropy loss vs. checkpoint")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150)
    logger.info("Saved plot to %s", args.output_png)
    
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(seed=1234)
    main()