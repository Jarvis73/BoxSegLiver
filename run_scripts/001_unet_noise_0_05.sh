#!/usr/bin/env bash

TASK=$1
GPU_ID=$2
shift 2

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

if [[ "$TASK" == "train" ]]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} python ./entry/main.py liver \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --model UNet \
        --classes Liver Tumor \
        --test_fold 2 \
        --im_height 256 --im_width 256 --im_channel 3 \
        --noise_scale 0.05 \
        --eval_num_batches_per_epoch 100 \
        --num_of_total_steps 600000 \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --loss_weight_type numerical \
        --loss_numeric_w 0.2 0.4 4.4 \
        --batches_per_epoch 2000 \
        --batch_size 8 \
        --weight_decay_rate 0.000001 \
        --learning_policy plateau \
        --learning_rate 0.001 \
        --lr_end 0 \
        --lr_decay_rate 0.2 \
        --eval_num_batches_per_epoch 200 \
        --eval_per_epoch \
        --evaluator Volume \
        $@
elif [[ "$TASK" == "eval" ]]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main.py liver \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model UNet \
        --classes Liver Tumor \
        --test_fold 2 \
        --im_height 256 --im_width 256 --im_channel 3 \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --batch_size 8 \
        $@
fi
