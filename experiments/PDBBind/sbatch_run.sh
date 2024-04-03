#!/bin/bash

# Directory where the generated bash scripts are located
scripts_dir="/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/FACTORY/RUNS/TREAMID_PDBBIND"

# Counter to keep track of how many scripts have been submitted
counter=0

# Maximum number of scripts to submit
max_submissions=130

# Submit the scripts
for script_file in "${scripts_dir}"/*.sh; do
  if [ -f "$script_file" ] && [ $counter -lt $max_submissions ]; then
    echo "Submitted for $script_file..."
    sbatch "$script_file"
    ((counter++))
  fi
done

echo "Submitted $counter scripts."
