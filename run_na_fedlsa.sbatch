#! /bin/bash
#SBATCH --job-name="run_rrlsa"
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-12:00:00
#SBATCH --output="logs/run_na_fedlsa.out"
#SBATCH --constraint="type_b"

python run_na_fedlsa.py