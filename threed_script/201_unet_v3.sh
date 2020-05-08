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
        eval ${NICE} $CONDA_PREFIX/bin/python ./entry/main.py nf_3d \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --model UNet3D \
        --model_config UNet3D_V2.yml \
        --classes NF \
        --test_fold 0 \
        --im_depth 10 --im_height 256 --im_width 256 --im_channel 1 \
        --random_flip 7 \
        --num_of_total_steps 250000 \
        --primary_metric "NF/Dice" \
        --loss_weight_type numerical \
        --loss_numeric_w 1 10 \
        --batches_per_epoch 400 \
        --batch_size 4 \
        --weight_decay_rate 0.00003 \
        --learning_policy plateau \
        --learning_rate 0.0003 \
        --lr_end 0.0000005 \
        --lr_decay_rate 0.2 \
        --distribution_strategy mirrored --num_gpus ${#GPU_IDS[@]} \
        --normalizer instance_norm \
        --eval_num_batches_per_epoch 40 \
        --eval_per_epoch \
        --evaluator Volume \
        --save_best \
        --summary_prefix nf \
        --tumor_percent 0.75 --log_step 125 \
        $@
elif [[ "$TASK" == "eval" ]]; then
    PYTHONPATH="${PROJECT_DIR}" PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} \
        eval ${NICE} $CONDA_PREFIX/bin/python ./entry/main_eval_3d.py \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model UNet3D \
        --model_config UNet3D_V2.yml \
        --classes NF \
        --test_fold 0 \
        --im_depth -1 --im_height 960 --im_width 320 --im_channel 1 \
        --random_flip 7 \
        --batch_size 1 \
        --normalizer instance_norm \
        --eval_mirror \
        --metrics_eval Dice VOE RVD \
        $@
fi
