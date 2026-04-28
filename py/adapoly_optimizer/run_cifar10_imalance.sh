#!/bin/bash
PROGRAM_PATH="./main_sup_base.py"
python "$PROGRAM_PATH" \
        --model_name "vits" \
        --opt_name "adam" \
        --lr 0.001 \
        --datapath "adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-10"
# python main_sup_base.py \
#         --model_name "vgg16bn" \
#         --opt_name "sgd" \
#         --lr 0.01 \
#         --datapath "adam_optimizer/data/cifar10_lt_outputs/cifar10-lt-r-10"
# CUDA_VISIBLE_DEVICES=1 nohup python "$PROGRAM_PATH" \
#         --model_name "vgg16bn" \
#         --opt_name "sgd" \
#         --lr 0.01 \
#         > "./log/vgg16_sgd_0_01.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python "$PROGRAM_PATH" \
#         --model_name "resnet18" \
#         --opt_name "adam" \
#         --lr 0.001 \
#         > "./log/resnet18_adam_0_01.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python "$PROGRAM_PATH" \
#         --model_name "resnet18" \
#         --opt_name "sgd" \
#         --lr 0.01 \
#         > "./log/resnet18_sgd_0_01.log" 2>&1 &