#!/bin/bash

# 240630

#SBATCH -J LLM_run
#SBATCH -o %A.out
#SBATCH -N 1
##SBATCH -n 1
#SBATCH -c 16
#SBATCH -p 6000ada_short
#SBATCH --gpus-per-node=3
#####################
source ~/miniconda3/bin/activate llama
python local_batch_hf.py
# source ~/miniconda3/bin/activate vllm
# python local_batch_vllm.py
#####################