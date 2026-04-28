#!/usr/bin/env python
import logging
import time
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator
)
from datasets import Dataset

# ---------------------- 配置参数 ----------------------
TEST_SAMPLES = 3  # 测试样本数量
MODEL_NAME = "gpt2"
OUTPUT_DIR = "./metrics"  # 结果输出目录
LOG_FILE = os.path.join(OUTPUT_DIR, "test_metrics.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- 初始化准备 ----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------- 简化数据加载 ----------------------
def load_small_samples():
    """生成3个简单测试样本"""
    texts = [
        "Hello, my name is",
        "The quick brown fox",
        "Machine learning is"
    ]
    return Dataset.from_dict({"text": texts[:TEST_SAMPLES]})

# ---------------------- 时间/显存监控工具 ----------------------
class Timer:
    def __enter__(self):
        self.start = time.time()
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start
        self.peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0

# ---------------------- 核心测试流程 ----------------------
def main():
    # 1. 加载简化数据
    raw_datasets = load_small_samples()
    logger.info(f"Loaded {len(raw_datasets)} test samples")

    # 2. 初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train()  # 训练模式

    # 3. 简单预处理（直接tokenize）
    tokenized = raw_datasets.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=64, padding="max_length"),
        remove_columns=["text"]
    )
    dataloader = torch.utils.data.DataLoader(
        tokenized, batch_size=1, collate_fn=default_data_collator
    )

    # 4. 逐个样本训练并记录指标
    for i, batch in enumerate(dataloader):
        with Timer() as t:
            # 数据转移到设备
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 前向+反向传播
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # 模拟优化步骤（实际可添加optimizer.step()）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
        # 记录指标
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Training time: {t.duration:.4f}s")
        logger.info(f"  Peak memory: {t.peak_mem:.2f}MB")

    # 5. 简单测试（生成文本）
    model.eval()
    with torch.no_grad(), Timer() as t:
        inputs = tokenizer("Hello, my name is", return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info("\n--- Test Results ---")
    logger.info(f"Generation time: {t.duration:.4f}s")
    logger.info(f"Generated text: {generated}")

if __name__ == "__main__":
    main()