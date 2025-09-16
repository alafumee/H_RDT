#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --partition=h200

singularity exec --nv --overlay /scratch/yz12129/h-rdt-env/overlay-25GB-500K.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash \
    -c "export SSL_CERT_FILE=/home/yz12129/cacert.pem && source /ext3/env.sh && conda activate hrdt && cd /scratch/yz12129/egodex \
    && curl -C - "https://ml-site.cdn-apple.com/datasets/egodex/part2.zip" -o part2.zip && unzip part2.zip \
    && rm part2.zip"