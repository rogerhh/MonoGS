#!/bin/bash

export CUDA_PATH=/usr/local/cuda-11.3
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/bin/lib64:$LD_LIBRARY_PATH"
