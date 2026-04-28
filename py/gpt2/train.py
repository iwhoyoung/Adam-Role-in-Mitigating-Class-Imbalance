import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import torch.distributed as dist

from collections import Counter, defaultdict
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


def init_distributed():
    if dist.is_available() and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 自动传入
        torch.cuda.set_device(local_rank)           # 显式指定本进程使用哪块GPU
        dist.init_process_group(backend="nccl")
        dist.barrier()
        print(f"[Rank {dist.get_rank()}] Using GPU {local_rank}")




def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def smooth_and_plot_loss(csv_path, output_path, optimizer_name):
    if not is_main_process():
        return

    df = pd.read_csv(csv_path)
    group_cols = [col for col in df.columns if col.startswith("group_")]
    steps = df["step"]

    for col in group_cols:
        df[col] = (
            df[col]
            .interpolate(method="linear", limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

    def smooth_curve(values, window_size=25):
        return np.convolve(values, np.ones(window_size)/window_size, mode='same')

    plt.figure(figsize=(4.2, 4))
    colors = [cm.viridis(i / len(group_cols)) for i in range(len(group_cols))]

    for idx, col in enumerate(group_cols):
        smoothed = smooth_curve(df[col].values, window_size=25)
        plt.plot(steps, smoothed, color=colors[idx], linewidth=2)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.grid(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")


def count_avg_token_frequency_per_group(token_to_group, token_freq):
    group_token_sum = defaultdict(int)
    group_token_count = defaultdict(int)

    for token, freq in token_freq.items():
        group = token_to_group[token]
        group_token_sum[group] += freq
        group_token_count[group] += 1

    avg_freq = {
        group: group_token_sum[group] / group_token_count[group]
        for group in group_token_sum
    }

    if is_main_process():
        df = pd.DataFrame({
            "group": list(avg_freq.keys()),
            "avg_token_frequency": list(avg_freq.values()),
            "token_count": [group_token_count[g] for g in avg_freq],
            "total_token_frequency": [group_token_sum[g] for g in avg_freq]
        }).sort_values(by="group")
        df.to_csv("token_avg_frequency_per_group.csv", index=False)

    return avg_freq


def main(optimizer_choice, learning_rate,batch_size):
    init_distributed()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:]")
    dataset = dataset.select(range(1000))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            return_special_tokens_mask=True,
        )


    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["input_ids"]) > 0)

    all_tokens = [token for seq in tokenized_dataset["input_ids"] for token in seq]
    token_freq = Counter(all_tokens)
    sorted_tokens = [token for token, _ in token_freq.most_common()]
    total_tokens = sum(token_freq.values())
    num_groups = 10
    group_size = total_tokens // num_groups
    token_to_group, token_sum, group_id = {}, 0, 0
    for token in sorted_tokens:
        token_to_group[token] = group_id
        token_sum += token_freq[token]
        if token_sum > (group_id + 1) * group_size and group_id < num_groups - 1:
            group_id += 1

    count_avg_token_frequency_per_group(token_to_group, token_freq)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 主进程先加载，其他进程等待
    if is_main_process():
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        dist.barrier()  # 主进程下载完后，其他进程再继续
    else:
        dist.barrier()  # 等主进程先下载
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))


    output_dir = f"gpt2_{optimizer_choice}_lr{learning_rate}_ckpt"
    optim_type = "adamw_torch" if optimizer_choice == "adam" else "sgd"

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=1000,
        #num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        logging_steps=100,
        save_steps=5000,
        save_total_limit=1,
        report_to="none",
        optim=optim_type,
        learning_rate=learning_rate,
        dataloader_drop_last=True,
        fp16=True if torch.cuda.is_available() else False,
    )

    group_loss_log = []

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            token_losses = loss.detach().cpu().numpy()
            token_ids = shift_labels.detach().cpu().numpy().flatten()

            group_losses = defaultdict(list)
            for l, t in zip(token_losses, token_ids):
                if t != -100:
                    group = token_to_group.get(int(t), -1)
                    if group != -1:
                        group_losses[group].append(l)

            group_avg = [np.mean(group_losses[i]) if i in group_losses else np.nan for i in range(num_groups)]
            group_loss_log.append(group_avg)

            return (loss.mean(), outputs) if return_outputs else loss.mean()

        def log(self, logs, iterator_start_time=None):
            super().log(logs, iterator_start_time)
            if self.is_world_process_zero() and len(group_loss_log) > 0:
                df = pd.DataFrame(group_loss_log, columns=[f"group_{i}" for i in range(num_groups)])
                df["step"] = [(i + 1) * training_args.logging_steps for i in range(len(group_loss_log))]
                loss_path = f"group_loss_{optimizer_choice}_lr{learning_rate}.csv"
                df.to_csv(loss_path, index=False)
                #plot_path = f"plots/{optimizer_choice}_lr{learning_rate}_group_loss.png"
                #smooth_and_plot_loss(loss_path, plot_path, optimizer_choice)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["adam", "sgd"], required=True, help="Choose the optimizer type.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Learning rate")
    args = parser.parse_args()
    main(args.optimizer, args.lr,args.batch_size)
