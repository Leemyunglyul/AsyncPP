#!/bin/sh

#SBATCH -J 20190980
#SBATCH -o 20190980.%j.out
#SBATCH -t 04:00:00

#SBATCH --gres=gpu:2

#SBATCH --nodes=1

