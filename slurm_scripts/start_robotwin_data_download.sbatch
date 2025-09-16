#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --time=48:00:00

export HUGGINGFACE_TOKEN="xxxx" # fill in your huggingface token here

singularity exec --nv --overlay /scratch/yz12129/h-rdt-env/overlay-25GB-500K.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash \
    -c "export SSL_CERT_FILE=/home/yz12129/cacert.pem && source /ext3/env.sh && conda activate hrdt && huggingface-cli login --token $HUGGINGFACE_TOKEN && huggingface-cli download --repo-type dataset TianxingChen/RoboTwin2.0 --local-dir /scratch/yz12129/robotwin/"