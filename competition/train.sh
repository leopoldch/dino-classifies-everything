#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=train-%j.out
#SBATCH --job-name=tp2-competition-train

source .venv/bin/activate

TRAIN_FILE="$1"
cd competition && python models/"$TRAIN_FILE"