#!/bin/bash
#SBATCH --cpus-per-task=1
###SBATCH --nodelist=node[21-24]
###SBATCH --exclude=node[1-5]
#SBATCH -J fake_news_epidemic
#SBATCH --nodes=1
#SBATCH -p blcy
#SBATCH --output=./slurm_reallog/%j.out

# init tensorflow env
source activate tf1_13-atari

# MAIN BATCH COMMANDS
#
# echo "RUNNING $@ "
echo "python $@"
python "$@"

#cd -
