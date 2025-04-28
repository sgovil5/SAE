#!/bin/bash

#SBATCH -J batchtopk_sae                # Job name
#SBATCH -A gts-vfung3                       # Charge account
#SBATCH -N1 --gres=gpu:V100:1                # Number of nodes and GPUs requiredi
#SBATCH --mem-per-gpu=40G                    # Memory per gpu
#SBATCH -t2:00:00                            # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                            # QOS name
#SBATCH -o results/batchtopk_sae.out          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL           # Mail preferences
#SBATCH --mail-user=sgovil9@gatech.edu           # e-mail address for notifications

module load anaconda3
conda activate wta
python train_batchtopk.py