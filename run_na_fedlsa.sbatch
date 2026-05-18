#! /bin/bash
#SBATCH --job-name="run_rrlsa"
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --time=0-06:00:00
#SBATCH --output="logs/run_na_fedlsa_gamma_H=0.2_new.out"
#SBATCH --constraint="type_b"

python run_na_fedlsa.py