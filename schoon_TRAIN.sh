#!/bin/bash
#SBATCH -p ai2es_a100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=04:30:00
#SBATCH --job-name="training"
#SBATCH --mail-user=mcy@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/mcy/conus2_ml/logs/R-%x.%j.out
#SBATCH --error=/home/mcy/conus2_ml/logs/R-%x.%j.err
#SBATCH -w c732
#SBATCH --chdir=/home/mcy/conus2_ml/

#need to source your bash script to access your python!
source /home/mcy/.bashrc
bash

#activate your env
mamba activate tf_gpu

date

python MAIN_TRAIN_and_SAVE_MODEL.py configure.txt

date