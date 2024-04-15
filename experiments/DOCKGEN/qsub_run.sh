#!/bin/bash

# Directory where the generated bash scripts are located
scripts_dir="/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/FACTORY/RUNS/TREAMID_DOCKGEN"

# Counter to keep track of how many scripts have been submitted
counter=0

# Maximum number of scripts to submit
max_submissions=7

# Submit the scripts
for script_file in "${scripts_dir}"/*.sh; do
  if [ -f "$script_file" ] && [ $counter -lt $max_submissions ]; then
    echo "Submitted for $script_file..."
    qsub "$script_file"
    ((counter++))
  fi
done

echo "Submitted $counter scripts."



