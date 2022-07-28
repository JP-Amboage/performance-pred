#!/bin/bash

############################################################################################
# This script is a modification to the implementation suggest here:
# https://github.com/jpata/particleflow/blob/master/mlpf/raytune/start-head.sh
############################################################################################

echo "[start-head.sh]: starting ray head node"

ray start --head --node-ip-address=$1 --port=6379 --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block

sleep infinity
