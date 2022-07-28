#!/bin/bash

############################################################################################
# This script is a modification to the implementation suggest here:
# https://github.com/jpata/particleflow/blob/master/mlpf/flatiron/raytune.sh
############################################################################################

#SBATCH -N2 --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=0


echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


#module --force purge; module load modules/1.49-20211101
#module load slurm gcc nccl cuda/11.3.1 cudnn/8.2.0.53-11.3 openmpi/4.0.6
#nvidia-smi
module load gcc

source ~/venvs/venv1/bin/activate 
python --version


################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password
echo "Redis password: ${redis_password}"

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" start-head.sh "$ip" &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node

for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head &
  sleep 5
done
##############################################################################################

#### call your code below
python ray_test.py --cpus "${SLURM_CPUS_PER_TASK}" --gpus "${SLURM_GPUS_PER_TASK}"
exit
