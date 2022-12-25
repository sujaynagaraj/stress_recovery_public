#!/usr/bin/env bash

declare -a features=("oura+surveys")

for i in "${features[@]}"
do
  feature="$i"

  #source /pkgs/anaconda3/bin/activate stressrecov #activate environment
  
  python3 -u run_PC_individual_reference.py --features $feature --method pearsonr --stress_window daily_stressed --stress_cutoff 0.75 --no_stress_cutoff 0.25 --bootstraps 10 --buffer 3 --missing_node 0 --max_cond_vars 2 --significance_level 0.05

done

for i in "${features[@]}"
do
  feature="$i"

  #source /pkgs/anaconda3/bin/activate stressrecov #activate environment
  
  python3 -u run_PC_individual_reference.py --features $feature --method pearsonr --stress_window daily_shifts --stress_cutoff 0.75 --no_stress_cutoff 0.25 --bootstraps 10 --buffer 3 --missing_node 0 --max_cond_vars 2 --significance_level 0.05

done

for i in "${features[@]}"
do
  feature="$i"

  #source /pkgs/anaconda3/bin/activate stressrecov #activate environment
  
  python3 -u run_PC_individual_reference.py --features $feature --method pearsonr --stress_window shift_stress --stress_cutoff 0.75 --no_stress_cutoff 0.25 --bootstraps 10 --buffer 3 --missing_node 0 --max_cond_vars 2 --significance_level 0.05

done

for i in "${features[@]}"
do
  feature="$i"

  #source /pkgs/anaconda3/bin/activate stressrecov #activate environment
  
  python3 -u run_PC_individual_reference.py --features $feature --method pearsonr --stress_window hrv_binary --stress_cutoff 0.75 --no_stress_cutoff 0.25 --bootstraps 10 --buffer 3 --missing_node 0 --max_cond_vars 2 --significance_level 0.05

done