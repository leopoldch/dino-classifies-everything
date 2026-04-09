#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=sweep-%j.out
#SBATCH --job-name=tp2-wandb-sweep

source ../.venv/bin/activate
python models/dinov2-sweep.py
