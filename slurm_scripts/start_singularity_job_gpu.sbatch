#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --constraint="h200"

singularity exec --nv --overlay /scratch/yz12129/pytorch-env/my_pytorch.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash \
    -c "export SSL_CERT_FILE=/home/yz12129/cacert.pem && source /ext3/env.sh && conda activate world && cd world-model-eval-yixiang && torchrun --nproc_per_node=1 train.py"

# srun -c 4 -t 0-4:00 --mem=16G --pty bash -i
# srun --qos=h200_cds --mem=8GB --time=4:00:00 --cpus-per-task=4 --constraint="h200" --gres=gpu:1 --pty bash