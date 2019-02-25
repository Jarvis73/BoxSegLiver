#!/usr/bin/env bash

TASK=$1
GPU_ID=$2

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

if [ "$TASK" == "train" ]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main.py \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --model UNet \
        --classes Liver Tumor \
        --dataset_for_train LiTS_Train_Tumor.json \
        --dataset_for_eval LiTS_Eval_Tumor.json \
        --zoom --noise --flip \
        --zoom_scale 1.2 \
        --im_height 256 \
        --im_width 256 \
        --resize_for_batch \
        --num_of_total_steps 200000 \
        --lr_decay_step 200001 \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --loss_weight_type "numerical" \
        --loss_numeric_w 0.2 0.4 4.4 \
        --batch_size 16 \
        --weight_init "xavier" \
        --weight_decay_rate 0 \
        --use_spatial_guide
fi
