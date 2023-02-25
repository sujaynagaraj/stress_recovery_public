#!/usr/bin/env bash

source /pkgs/anaconda3/bin/activate stressrecov
  
declare -a stress_definitions=("shift_stress" "hrv_binary" "daily_stressed" "daily_shifts")

for j in {0..3}
  do
    stress_definition=${stress_definitions[$j]}
    source /pkgs/anaconda3/bin/activate stressrecov #activate environment
    python3 -u PC_results_reference.py --stress_definition $stress_definition
  done



