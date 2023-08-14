#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -p cpu
#SBATCH -a 0,1,2,3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=slurm-%A_%a_PC.out
#SBATCH --error=slurm-%A_%a_PC.out
#SBATCH --qos=nopreemption

source /pkgs/anaconda3/bin/activate stressrecov
  
declare -a stress_definitions=("shift_stress" "hrv_binary" "daily_stressed" "daily_shifts")

for j in {0..3}
  do
    let task_id=$j
    if [[ $task_id -eq $SLURM_ARRAY_TASK_ID ]]
    then
      stress_definition=${stress_definitions[$j]}
      source /pkgs/anaconda3/bin/activate stressrecov #activate environment
      python3 -u PC_results_reference.py --stress_definition $stress_definition
    fi
  done



