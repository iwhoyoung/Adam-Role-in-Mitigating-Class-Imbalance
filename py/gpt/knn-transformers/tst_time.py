#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0
"""
Time-bench friendly CLM/Seq2Seq training script with tokens/sec reporting.
This version hardcodes the 'time bench' settings (no CLI flags needed).
"""

from adam_util import Adam_bn, Adam_ini, Adam_Sbn
import logging
import math
import random
import os
import sys
from dataclasses import dataclass, field, replace
from itertools import chain
import numpy as np
from typing import Optional
from collections import Counter

import datasets
import torch
from datasets import load_dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import T5Tokenizer

# ======== HARD-CODED TIME BENCH SETTINGS ========
# Turn this on to greatly reduce load/processing for quick timing.
TIME_BENCH = True
TB_TRAIN_TEXTS = 50000   # number of original "train" texts to load (pre-grouping)
TB_EVAL_TEXTS  = 5000    # number of original "eval"  texts to load (pre-grouping)
TB_TRAIN_BLOCKS = 1024   # number of grouped train blocks to keep (post-grouping)
TB_EVAL_BLOCKS  = 1024    # number of grouped eval  blocks to keep (post-grouping)
# ================================================

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.54.0.dev0")
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """Which model/config/tokenizer to fine-tune, or train from scratch."""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization. Leave unset to train from scratch."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={"help": "Override default config when training from scratch, e.g. n_embd=10,resid_pdrop=0.2"},
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Config name or path"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer name or path"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache dir"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer"})
    model_revision: str = field(default="main", metadata={"help": "Model revision (branch/tag/commit)"})
    token: Optional[str] = field(default=None, metadata={"help": "HF token"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust code on the Hub"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override torch.dtype; 'auto', 'bfloat16', 'float16', 'float32'"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    """What data to input to the model for training/eval."""
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF datasets name"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "HF datasets config"})
    train_file: Optional[str] = field(default=None, metadata={"help": "Training data file (csv/json/txt)"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Eval data file (csv/json/txt)"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate #train examples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate #eval examples"})
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(default=None, metadata={"help": "Sequence length after tokenization"})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite datasets cache"})
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "Split if no val set"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "#processes for preprocessing"})
    keep_linebreaks: bool = field(default=True, metadata={"help": "Keep line breaks for TXT"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` must be csv/json/txt."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` must be csv/json/txt."


def build_token_groups(datasets_dict, num_groups: int = 10, col: str = "input_ids", streaming: bool = False):
    """
    Split vocabulary into num_groups by token frequency over the tokenized datasets.
    Returns: token_to_group, group_token_ids (list[Tensor]), group_stats
    """
    from itertools import chain as it_chain
    freq = Counter()
    if streaming:
        for split in datasets_dict:
            for example in datasets_dict[split]:
                freq.update(example[col])
    else:
        for split in datasets_dict:
            split_ids = datasets_dict[split][col]
            freq.update(it_chain.from_iterable(split_ids))

    sorted_items = freq.most_common()
    total_freq = sum(freq.values())
    step = total_freq / max(num_groups, 1)

    token_to_group = {}
    group_token_lists = [[] for _ in range(num_groups)]
    group_stats = []
    acc = 0
    gid = 0
    for tok, f in sorted_items:
        token_to_group[tok] = gid
        group_token_lists[gid].append(tok)
        acc += f
        if acc >= (gid + 1) * step and gid < num_groups - 1:
            group_stats.append((len(group_token_lists[gid]), acc, acc / total_freq))
            gid += 1
    group_stats.append((len(group_token_lists[gid]), acc, acc / total_freq))
    group_token_ids = [torch.tensor(lst, dtype=torch.long) for lst in group_token_lists]
    return token_to_group, group_token_ids, group_stats


def main():
    # Parse the standard three groups of args. (No new CLI flags added.)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Telemetry
    send_example_telemetry("run_clm", model_args, data_args)

    # Logging
    log_file = "gpt2_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this, change --output_dir or add --overwrite_output_dir."
            )

    # Seed
    set_seed(training_args.seed)

    # Load datasets (with hard-coded time bench slicing)
    if data_args.dataset_name is not None:
        if TIME_BENCH and not data_args.max_train_samples:
            t_n = TB_TRAIN_TEXTS
            v_n = TB_EVAL_TEXTS
            raw_datasets = DatasetDict({
                "train": load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{t_n}]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                ),
                "validation": load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{t_n}]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                ),
            })
        elif data_args.max_train_samples:
            raw_datasets = DatasetDict({
                "train": load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f'train[:{data_args.max_train_samples}]',
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                ),
                "validation": load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f'train[{data_args.max_train_samples}:{data_args.max_train_samples + (data_args.max_eval_samples or 0)}]',
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                ),
            })
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            print(len(raw_datasets))

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
        # If local files, apply post-load truncation for time bench.
        if TIME_BENCH:
            if "train" in raw_datasets:
                n = min(len(raw_datasets["train"]), TB_TRAIN_TEXTS)
                raw_datasets["train"] = raw_datasets["train"].select(range(n))
                logger.info(f"[time_bench] Truncated raw train texts to {n}.")
            if "validation" in raw_datasets:
                n = min(len(raw_datasets["validation"]), TB_EVAL_TEXTS)
                raw_datasets["validation"] = raw_datasets["validation"].select(range(n))
                logger.info(f"[time_bench] Truncated raw eval texts to {n}.")

    # Load tokenizer/model
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.model_name_or_path == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_embd=768,
        )
        model = AutoModelForCausalLM.from_config(config)
    elif model_args.model_name_or_path == 't5':
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        config = AutoConfig.from_pretrained("google-t5/t5-small")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
    else:
        torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
        )
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError("Tokenizer not specified. Use --tokenizer_name or --model_name_or_path.")
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **tokenizer_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in str(model_args.model_name_or_path)),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess / tokenize
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    # Optional: token group stats
    token_to_group, group_token_ids, group_stats = build_token_groups(
        tokenized_datasets,
        num_groups=10,
        col="input_ids",
        streaming=data_args.streaming,
    )
    for gid, (n_tok, cum_freq, ratio) in enumerate(group_stats):
        print(f"Group {gid}: {n_tok} tokens, cumulative freq {cum_freq:,} ({ratio:.1%})")

    # block_size
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead."
            )
            block_size = min(1024, max_pos_embeddings) if max_pos_embeddings > 0 else 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                  for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # post-grouping selection for time bench
    if TIME_BENCH:
        if training_args.do_train and "train" in lm_datasets:
            train_blocks = min(len(lm_datasets["train"]), TB_TRAIN_BLOCKS)
            lm_datasets["train"] = lm_datasets["train"].select(range(train_blocks))
            logger.info(f"[time_bench] Using {train_blocks} train blocks (post-group).")
        if training_args.do_eval and "validation" in lm_datasets:
            eval_blocks = min(len(lm_datasets["validation"]), TB_EVAL_BLOCKS)
            lm_datasets["validation"] = lm_datasets["validation"].select(range(eval_blocks))
            logger.info(f"[time_bench] Using {eval_blocks} eval blocks (post-group).")

    # Datasets to Trainer
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    else:
        eval_dataset = None

    # Preprocess logits for metrics (optional)
    preprocess_logits_for_metrics = None
    if training_args.do_eval and not is_torch_xla_available():
        def _preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        preprocess_logits_for_metrics = _preprocess_logits_for_metrics

    # world size (works with or without DDP)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = world_size_env if world_size_env > 0 else max(1, training_args.n_gpu)

    # Train with three optimizers in sequence (same as your code)
    for each in ['adam','sgd','adam_bn', 'adam_ini', 'adam_Sbn']:
        print(each)
        opt_name = each
        opt_output_dir = os.path.join(training_args.output_dir, f"{opt_name}_time")
        os.makedirs(opt_output_dir, exist_ok=True)

        if opt_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), training_args.learning_rate, weight_decay=0)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), training_args.learning_rate, weight_decay=0)
        elif opt_name == 'adam_bn':
            optimizer = Adam_bn(model.parameters(), training_args.learning_rate, weight_decay=0)
        elif opt_name == 'adam_ini':
            optimizer = Adam_ini(model.parameters(), training_args.learning_rate, weight_decay=0)
        elif opt_name == 'adam_Sbn':
            optimizer = Adam_Sbn(model.parameters(), training_args.learning_rate, weight_decay=0)
        else:
            raise ValueError(f"Unknown optimizer name: {opt_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Create a per-optimizer copy of TrainingArguments with a different output_dir
        opt_args = replace(training_args, output_dir=opt_output_dir)

        trainer = Trainer(
            model=model,
            args=opt_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            data_collator=default_data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # ===== Training with throughput timing =====
        if training_args.do_train:
            ckpt0 = os.path.join(opt_output_dir, "checkpoint-0")
            if not os.path.exists(ckpt0):
                model.save_pretrained(ckpt0)
                tokenizer.save_pretrained(ckpt0)
                logger.info(f"Initial checkpoint saved to {ckpt0}")

            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

            import time
            start_time = time.perf_counter()
            train_result = trainer.train()
            elapsed = time.perf_counter() - start_time

            global_step = train_result.global_step
            eff_bs = (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * world_size
            )
            tokens_seen = global_step * eff_bs * block_size #1024*1*4
            tokens_per_sec = tokens_seen / max(elapsed, 1e-9)

            logger.info(f"[{opt_name}][TRAIN Throughput] tokens_seen={tokens_seen:,} | time={elapsed:.2f}s | tokens/sec={tokens_per_sec:,.2f}")
            print(f"[{opt_name}][TRAIN Throughput] tokens_seen={tokens_seen:,} | time={elapsed:.2f}s | tokens/sec={tokens_per_sec:,.2f}| tokens/sec={tokens_per_sec:,.2f}")

            trainer.save_model()
            metrics = train_result.metrics
            max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # ===== Evaluation with throughput timing =====
        if training_args.do_eval:
            logger.info(f"*** Evaluate ({opt_name}) ***")
            import time
            start_time = time.perf_counter()
            metrics = trainer.evaluate()
            elapsed = time.perf_counter() - start_time

            # Approx tokens evaluated (post-grouping blocks): #examples * block_size * world_size
            eval_examples = len(eval_dataset)
            eval_tokens = eval_examples * block_size * world_size
            eval_tokens_per_sec = eval_tokens / max(elapsed, 1e-9)

            logger.info(f"[{opt_name}][EVAL Throughput] tokens={eval_tokens:,} | time={elapsed:.2f}s | tokens/sec={eval_tokens_per_sec:,.2f}")
            print(f"[{opt_name}][EVAL Throughput] tokens={eval_tokens:,} | time={elapsed:.2f}s | tokens/sec={eval_tokens_per_sec:,.2f}")

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics.get("eval_loss", float("inf")))
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Model card / push
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda_manual_seed_all = getattr(torch.cuda, "manual_seed_all", None)
        if torch.cuda_manual_seed_all is not None:
            torch.cuda.manual_seed_all(seed)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    seed_everything(11)
    main()
