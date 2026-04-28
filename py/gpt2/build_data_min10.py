from transformers import GPT2TokenizerFast
from datasets import Dataset, load_dataset
import os
import random

def build_random_100_tokens_dataset(save_path="random_100_tokens_dataset"):
    # 1. 初始化 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 添加pad token
    pad_token_id = tokenizer.pad_token_id  # 获取pad token的ID，解码时跳过

    # 2. 加载数据集（保留原始text字段）
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10%]")
    print(f"原始数据集共 {len(dataset)} 条样本，每条包含原始text\n")

    # 3. 分词：不删除text字段，保留原始句子
    def tokenize_function(examples):
        # 只分词，不删除原始text，同时返回special_tokens_mask
        return tokenizer(
            examples["text"], 
            return_special_tokens_mask=True,
            truncation=False  # 不截断（保持原始序列长度）
        )

    # 执行分词（batched=True加速），保留text字段
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # 过滤空序列（避免后续报错）
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["input_ids"]) > 0)
    print(f"分词后剩余 {len(tokenized_dataset)} 条有效样本（非空序列）\n")

    # 4. 随机选择样本（原代码num_samples是10，注释写100，这里统一为10条，可改）
    num_samples = min(10, len(tokenized_dataset))  # 若想选100条，改min(100, ...)
    random.seed(42)  # 固定随机种子，结果可复现

    # 关键：获取选中样本的索引（后续用索引取原始text）
    selected_indices = random.sample(range(len(tokenized_dataset)), num_samples)
    # 根据索引获取选中的input_ids和原始text
    selected_data = [
        {
            "text": tokenized_dataset[idx]["text"],  # 原始句子
            "input_ids": tokenized_dataset[idx]["input_ids"]  # 对应的token编码
        }
        for idx in selected_indices
    ]

    # 5. 打印选中的句子（原始文本 + 解码后的文本，双重验证）
    print("=" * 80)
    print(f"最终选中的 {num_samples} 条句子：")
    print("=" * 80)
    for i, data in enumerate(selected_data, 1):
        # 原始文本（可能包含换行、标题标记，如"= XXX = "）
        raw_text = data["text"].strip()
        # 将input_ids解码回文本（跳过pad token，避免显示[PAD]）
        decoded_text = tokenizer.decode(
            [token for token in data["input_ids"] if token != pad_token_id],
            skip_special_tokens=True  # 跳过特殊token（如[CLS]，GPT2默认没有，但保留保险）
        ).strip()

        # 打印结果
        print(f"\n【样本 {i}】")
        print(f"原始文本：\n{raw_text}")
        print(f"解码文本（input_ids → 文字）：\n{decoded_text}")
        print("-" * 60)

    # 6. 构造新数据集（包含input_ids，用于后续模型输入）
    selected_input_ids = [data["input_ids"] for data in selected_data]
    new_dataset = Dataset.from_dict({"input_ids": selected_input_ids})

    # 7. 保存数据集
    os.makedirs(save_path, exist_ok=True)
    new_dataset.save_to_disk(save_path)
    print(f"\n已保存 {num_samples} 条样本的input_ids到 {save_path}")

# 执行函数
build_random_100_tokens_dataset()