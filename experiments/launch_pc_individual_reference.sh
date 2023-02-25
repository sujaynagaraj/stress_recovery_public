#!/usr/bin/env bash

declare -a stress_windows=("daily_stressed" "daily_shifts" "shift_stress" "hrv_binary")

for j in {0..3}
  do
    stress_window=${stress_windows[$j]}
    source /pkgs/anaconda3/bin/activate stressrecov #activate environment
    python3 -u run_PC_individual_reference.py --features oura+surveys --method pearsonr --stress_window $stress_window --stress_cutoff 0.75 --no_stress_cutoff 0.25 --bootstraps 100 --buffer 3 --missing_node 0 --max_cond_vars 2 --significance_level 0.05
  done

