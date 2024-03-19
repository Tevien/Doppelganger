#!/bin/bash

#Set job requirements
#SBATCH -J ETL_SB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=fat_rome
#SBATCH --time=04:00:00

cd $HOME/sw/Doppelganger/
conda activate synth2

python scripts/run_amc.py