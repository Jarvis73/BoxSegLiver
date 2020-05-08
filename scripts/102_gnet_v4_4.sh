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
   PYTHONPATH="${PROJECT_DIR}" PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} \
        eval ${NICE} $CONDA_PREFIX/bin/python ./entry/main_g.py nf_inter \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --model GUNet \
        --classes NF \
        --test_fold 0 \
        --im_height 256 --im_width 256 --im_channel 3 \
        --noise_scale 0 --random_flip 3 \
        --num_of_total_steps 999999 \
        --primary_metric "NF/Dice" \
        --loss_weight_type numerical \
        --loss_numeric_w 1 1 \
        --batches_per_epoch 1200 \
        --batch_size 16 \
        --weight_decay_rate 0.00005 \
        --learning_policy plateau \
        --learning_rate 0.0003 \
        --lr_end 0.0000005 \
        --lr_decay_rate 0.2 \
        --distribution_strategy mirrored --num_gpus ${#GPU_IDS[@]} \
        --normalizer instance_norm \
        --eval_num_batches_per_epoch 120 \
        --eval_per_epoch \
        --evaluator Volume \
        --save_best \
        --summary_prefix nf \
        --use_spatial \
        --save_interval 50000 \
        $@
elif [[ "$TASK" == "eval" ]]; then
    PYTHONPATH="${PROJECT_DIR}" PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} \
        eval ${NICE} $CONDA_PREFIX/bin/python ./entry/main_eval.py \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model GUNet \
        --classes NF \
        --test_fold 0 \
        --im_height 960 --im_width 320 --im_channel 3 \
        --random_flip 3 \
        --batch_size 1 \
        --normalizer instance_norm \
        --eval_mirror \
        --metrics_eval Dice VOE RVD \
        --use_spatial \
        $@
fi
