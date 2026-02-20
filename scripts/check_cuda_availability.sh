#!/bin/bash

#SBATCH --job-name=test_dppl
#SBATCH --output=/scratch/sandbox/sbenson/cuda_availability.out
#SBATCH --error=/scratch/sandbox/sbenson/cuda_availability.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source /home/sandbox/sbenson/snow_shell.sh

# Run the Python script to check CUDA availability
python -c "import torch; print('CUDA is available:', torch.cuda.is_available())"