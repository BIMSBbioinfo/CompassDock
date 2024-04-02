#!/bin/bash

scripts_dir="/fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RUNS/TREAMID_HURI"

# Clear run.lst if it already exists, to start fresh
> run.lst

for script in "$scripts_dir"/*.sh; do
    if [[ -f "$script" ]]; then
        echo "$script" >> /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RUNS/TREAMID_HURI/run.lst
    fi
done