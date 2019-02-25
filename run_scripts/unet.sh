#!/usr/bin/env bash

TASK=$1
GPU_ID=$2

PROJECT_DIR="$(dirname "$(dirname "$(realpath $0)")")"

if [ "$TASK" == "train" ]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main.py \
        --mode train \
        --tag 001_liver_unet \
        --model UNet \
        --model_config "UNet_Small.yml" \
        --classes Liver \
        --dataset_for_train LiTS_Train.json \
        --dataset_for_eval LiTS_Eval.json \
        --zoom --noise \
        --num_of_steps 5001 \
        --num_of_total_steps 80000 \
        --lr_decay_step 80001 \
        --primary_metric "Liver/Dice"
fi
