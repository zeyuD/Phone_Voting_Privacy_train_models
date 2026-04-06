#!/bin/bash
#SBATCH --job-name=vote_privacy
#SBATCH -A cwang6_111_0001
#SBATCH --partition=short
#SBATCH --nodelist=bcm-dgxa100-0001
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=job_%j_vote_privacy.out

# Load modules if needed
module load conda
module load gcc/11.2.0
module load cuda
module load spack
module load cmake

# Activate Conda environment
conda activate eyegaze

# Go to the target directory
cd ~/Projects/Vote_Privacy_train_models/mvts_transformer

# Run your script
python src/cust_run.py