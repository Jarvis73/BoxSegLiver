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
        --load_status_file _checkpoint \
        --summary_prefix nf \
        --save_best \
        --log_step 125 \
        --distribution_strategy off \
        --num_gpus 1 \
        --model VNet3D \
        --classes NF \
        --batch_size 1 \
        --weight_init xavier \
        --normalizer instance_norm \
        --batches_per_epoch 300 \
        --eval_per_epoch \
        --dropout 0.3 \
        --learning_rate 0.0003 \
        --learning_policy plateau \
        --num_of_total_steps 999999 \
        --lr_decay_rate 0.2 \
        --lr_end 0.0000005 \
        --weight_decay_rate 0.00003 \
        --loss_type xentropy \
        --loss_weight_type numerical \
        --loss_numeric_w 1 10 \
        --test_fold 0 \
        --im_depth 10 --im_height 256 --im_width 256 --im_channel 1 \
        --random_flip 7 \
        --eval_num_batches_per_epoch 30 \
        --primary_metric NF/Dice \
        --evaluator Volume \
        --tumor_percent 0.75 \
        $@
elif [[ "$TASK" == "eval" ]]; then
    PYTHONPATH="${PROJECT_DIR}" PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} \
        eval ${NICE} $CONDA_PREFIX/bin/python ./entry/main_eval_3d.py \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model VNet3D \
        --load_status_file checkpoint_best \
        --classes NF \
        --dropout 0 \
        --batch_size 1 \
        --test_fold 0 \
        --im_depth -1 --im_height 960 --im_width 320 --im_channel 1 \
        --random_flip 7 \
        --normalizer instance_norm \
        --eval_mirror \
        --metrics_eval Dice VOE RVD \
        $@
fi