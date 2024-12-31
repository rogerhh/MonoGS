#!/bin/bash

export CUDA_PATH=/usr/local/cuda-12.6
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/bin/lib64:$LD_LIBRARY_PATH"

if [ -z "$CONDA_PREFIX" ]; then
    echo "No conda environment found. Please activate a conda environment."
    return
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
