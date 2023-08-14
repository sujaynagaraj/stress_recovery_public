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

declare -a stress_windows=("daily_stressed" "daily_shifts" "shift_stress" "hrv_binary")

for j in {0..3}
  do
    let task_id=$j
    if [[ $task_id -eq $SLURM_ARRAY_TASK_ID ]]
    then
      stress_window=${stress_windows[$j]}
      source /pkgs/anaconda3/bin/activate stressrecov #activate environment
      python3 -u run_PC_individual_reference.py --features oura+surveys --method pearsonr --stress_window $stress_window --stress_cutoff 0.75 --no_stress_cutoff 0.25 --bootstraps 100 --buffer 3 --missing_node 0 --max_cond_vars 2 --significance_level 0.05
    fi
  done

