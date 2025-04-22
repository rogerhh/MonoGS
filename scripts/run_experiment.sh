#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
RESULTS_DIR="$PROJECT_DIR/results"
WANDB_DIR="$PROJECT_DIR/wandb"
SAVE_DIR="$PROJECT_DIR/saved_runs"

cd "$PROJECT_DIR"

# CONFIG="$PROJECT_DIR/configs/mono/tum/fr3_office.yaml"
CONFIG="$PROJECT_DIR/configs/mono/tum/fr1_desk.yaml"

num_iters=3

#expname="f20s5d32l0.01"
expname="fr1_desk_f30s5d1024l0.001huber0.01"
for i in $(seq 1 $num_iters); do
    # Run the experiment
    python3 slam.py --config $CONFIG --eval 2>&1 | tee run.log
    
    # Check the log to see where the results are saved
    grep_str="$(grep "saving results in" run.log)"
    
    # Split the grep_str by "/" and get the last part
    datetime="${grep_str##*/}"
    
    RESULTS_DIR_DATETIME="$(ls -d $RESULTS_DIR/*/$datetime)"
    
    mkdir -p "$SAVE_DIR"
    
    mv run.log "$RESULTS_DIR_DATETIME/run.log"

    SAVE_DIR_PATH="$SAVE_DIR/${datetime}_${expname}"

    echo "Copying $RESULTS_DIR_DATETIME to $SAVE_DIR_PATH"
    cp -r "$RESULTS_DIR_DATETIME" "$SAVE_DIR_PATH"
    
    result_wandb_path="$(ls -d $WANDB_DIR/*${datetime}*)"
    echo "Copying ${result_wandb_path} to $SAVE_DIR_PATH"
    cp -r ${result_wandb_path}* "$SAVE_DIR_PATH/"
done


