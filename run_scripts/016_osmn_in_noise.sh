#!/usr/bin/env bash

TASK=$1
GPU_ID=$2
shift 2

IFS="," read -ra GPU_IDS <<< ${GPU_ID}
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

if [ "$TASK" == "train" ]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main_osmn.py \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --model OSMNUNet \
        --classes Liver Tumor \
        --dataset_for_train LiTS_Train_Tumor_Triplet.json \
        --dataset_for_eval_while_train LiTS_Eval_Tumor.json \
        --hist_dataset_for_train LiTS_Train_Hist.json \
        --hist_dataset_for_eval_while_train LiTS_Eval_Hist.json \
        --zoom --noise --flip --zoom_scale 1.2 \
        --im_height 256 --im_width 256 --im_channel 3 \
        --resize_for_batch \
        --num_of_total_steps 400000 \
        --lr_decay_step 400001 \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --loss_weight_type numerical \
        --loss_numeric_w 0.2 0.4 4.4 \
        --eval_steps 5000 \
        --batch_size 8 \
        --weight_decay_rate 0 \
        --learning_policy custom_step \
        --lr_decay_boundaries 200000 300000 --lr_custom_values 0.001 0.0003 0.0001 \
        --input_group 3 --eval_3d \
        --use_fake_guide \
        --distribution_strategy mirrored --num_gpus ${#GPU_IDS[@]} \
        --hist_noise \
        --normalizer instance_norm \
        $@
elif [ "$TASK" == "eval" ]; then
    PYTHONPATH=${PROJECT_DIR} CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main_osmn.py \
        --mode eval \
        --tag ${BASE_NAME%".sh"} \
        --model OSMNUNet \
        --classes Liver Tumor \
        --dataset_for_eval LiTS_Eval_Tumor.json \
        --hist_dataset_for_eval LiTS_Eval_Hist.json \
        --im_height 256 --im_width 256 --im_channel 3 \
        --resize_for_batch \
        --primary_metric "Tumor/Dice" \
        --secondary_metric "Liver/Dice" \
        --batch_size 8 \
        --input_group 3 \
        --eval_3d \
        --normalizer instance_norm \
        $@
fi
