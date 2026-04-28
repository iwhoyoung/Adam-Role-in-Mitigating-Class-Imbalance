python main_imagenetv2.py \
    --model_name "resnet18" \
    --opt_name "adam" \
    --lr 0.001 \
    --batch_size 128 \
    --cuda_visible_devices 1,2,3 \
    --nThreads 8 \
    --dataset_name "imagenet" \
    --datapath "adam_optimizer/data/imagenet"
