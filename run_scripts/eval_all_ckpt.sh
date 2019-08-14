#!/usr/bin/env bash

TAG=$1
GPU_ID=$2
NICE=$3
if [[ "$NICE" == "nice" ]]; then
    shift 3
else
    shift 2
    NICE=""
fi

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
MODEL_DIR="${PROJECT_DIR}/model_dir/${TAG}"

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "${MODEL_DIR} not found!"
fi

for status_file in $(ls ${MODEL_DIR} | grep checkpoint_best_*)
do
    echo ${status_file} >> "${MODEL_DIR}/logs/eval_all_results"
    ./run_scripts/${TAG}.sh eval ${GPU_ID} ${NICE} \
        --load_status_file ${status_file} \
        --out_file eval_all_results \
        $@
done

cat "${MODEL_DIR}/logs/eval_all_results" | while read line
do
    if [[ ${line} =~ checkpoint_best_ && ${line} != *"Namespace"* || ${line} =~ ----Process ]]; then
        echo ${line} >> "${MODEL_DIR}/eval_trim_results"
    fi
done

