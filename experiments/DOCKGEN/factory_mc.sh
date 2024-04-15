#!/bin/bash

# Folder to save the scripts
SCRIPTS_FOLDER="/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/FACTORY/RUNS/TREAMID_DOCKGEN"

# Create the folder if it doesn't exist
if [ ! -d "$SCRIPTS_FOLDER" ]; then
  mkdir -p "$SCRIPTS_FOLDER"
fi

# Start and end values
START=0
END=50
MAX=330

# Script counter
COUNTER=1

# Loop until the MAX value
while [ $START -lt $MAX ]; do
  # Create a bash script for the current range, saving in the specified folder
  cat <<EOT > "$SCRIPTS_FOLDER/job_script_$COUNTER.sh"
#!/bin/bash
#$ -l gpu=1 -l cuda_memory=32G
#$ -cwd
#$ -V
#$ -N "diffdock_compass_dockgen_${COUNTER}"
#$ -l h_rt=12:00:00
#$ -pe smp 1
#$ -e /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_DOCKGEN/job_errors/
#$ -o /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_DOCKGEN/job_outputs/


# Change directory to the project root
cd /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS

# Activate the Python environment
source /home/asarigu/miniconda3/etc/profile.d/conda.sh
conda activate diffdock_compass


# Execute the Python script

python -W ignore -m main_multi_shot \
  --config DiffDock/default_inference_args.yaml \
  --protein_dir /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/DockGen/protein_pdb_files \
  --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" \
  --out_dir /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_DOCKGEN/results/ \
  --max_recursion_step 5 \
  --wandb_path /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_DOCKGEN \
  --start $START \
  --end $END


EOT

  # Make the script executable
  chmod +x "$SCRIPTS_FOLDER/job_script_$COUNTER.sh"

  # Update START and END for the next script
  START=$((END + 1))
  END=$((END + 50))
  if [ $END -gt $MAX ]; then
    END=$MAX
  fi

  # Increment the script counter
  COUNTER=$((COUNTER + 1))
done
