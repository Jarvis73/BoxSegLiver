#!/usr/bin/env bash

TASK=$1
GPU_ID=$2
NICE=$3
if [[ "$NICE" == "nice" ]]; then
    shift 3
else
    shift 2
    NICE=""
fi

IFS="," read -ra GPU_IDS <<< ${GPU_ID}
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

if [[ "$TASK" == "train" ]]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} eval ${NICE} python ./entry/main_g.py liver \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --model GUNet \
        --model_config GUNet_SP.yml \
        --classes Liver Tumor \
        --test_fold 2 \
        --im_height 256 --im_width 256 --im_channel 3 \
        --noise_scale 0.05 --random_flip 3 \
        --num_of_total_steps 1000000 \
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
        --distribution_strategy mirrored --num_gpus ${#GPU_IDS[@]} \
        --normalizer instance_norm \
        --use_spatial --spatial_random 1.0 \
        --eval_num_batches_per_epoch 300 \
        --eval_per_epoch \
        --evaluator Volume \
        --save_best \
        $@
elif [[ "$TASK" == "eval" ]]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} eval ${NICE} python ./entry/main_g.py liver \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model GUNet \
        --model_config GUNet_SP.yml \
        --classes Liver Tumor \
        --test_fold 2 \
        --im_height 256 --im_width 256 --im_channel 3 \
         --random_flip 3 \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --batch_size 8 \
        --evaluator Volume \
        --normalizer instance_norm \
        --load_status_file checkpoint_best \
        $@
elif [[ "$TASK" == "infer" ]]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} eval ${NICE} python ./entry/main_g.py liver \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model GUNet \
        --model_config GUNet_SP.yml \
        --classes Liver Tumor \
        --test_fold 2 \
        --im_height 256 --im_width 256 --im_channel 3 \
         --random_flip 3 \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --batch_size 8 \
        --evaluator Volume \
        --normalizer instance_norm \
        --load_status_file checkpoint_best \
        $@
fi
