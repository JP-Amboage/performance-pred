#!/bin/bash

############################################################################################
# This script is a modification to the implementation suggest here:
# https://github.com/jpata/particleflow/blob/master/mlpf/raytune/start-worker.sh
############################################################################################

echo "[start-worker.sh]: starting ray worker node"

ray start --address $1 --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block

sleep infinity
