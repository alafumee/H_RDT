#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint="h200|l40s"
#SBATCH --requeue

singularity exec --nv --overlay /scratch/yz12129/h-rdt-env/overlay-25GB-500K.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash \
    -c "export SSL_CERT_FILE=/home/yz12129/cacert.pem && source /ext3/env.sh && conda activate hrdt && cd H_RDT && python ./datasets/robotwin2/encode_lang_all.py /scratch/yz12129/robotwin/dataset"