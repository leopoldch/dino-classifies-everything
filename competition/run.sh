#!/bin/bash
#SBATCH --time=0-16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:4
#SBATCH --output=tp2-%j.out
#SBATCH --job-name=tp2-leo

source ../.venv/bin/activate

python "$@"