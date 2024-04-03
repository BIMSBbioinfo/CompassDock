#!/bin/bash

# Folder to save the scripts
SCRIPTS_FOLDER="/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/FACTORY/RUNS/TREAMID_PDBBIND"

# Create the folder if it doesn't exist
if [ ! -d "$SCRIPTS_FOLDER" ]; then
  mkdir -p "$SCRIPTS_FOLDER"
fi

# Start and end values
START=0
END=150
MAX=19119

# Script counter
COUNTER=1

# Loop until the MAX value
while [ $START -lt $MAX ]; do
  # Create a bash script for the current range, saving in the specified folder
  cat <<EOT > "$SCRIPTS_FOLDER/job_script_$COUNTER.sh"
#!/bin/bash
#SBATCH --job-name=compass_$COUNTER          # Job name
#SBATCH --nodes=1                         # Run all processes on a single node  
#SBATCH --ntasks=1                        # Run a single task       
#SBATCH --cpus-per-task=1                # Number of CPU cores per task # 1
#SBATCH --mem=128GB                       # Job memory request
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec
#SBATCH --gpus=1                          # Request GPU
#SBATCH --partition=booster               # Specify the partition
#SBATCH --account=hai_bimsb_dock_drug     # Specify the account
#SBATCH --gres=gpu:1                      # Generic Resource (GRES) request for one GPU
#SBATCH --ntasks-per-core=1
#SBATCH --output=/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_PDBBIND/job_outputs/%j.out # Standard output path
#SBATCH --error=/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_PDBBIND/job_errors/%j.err  # Standard error path


# Navigate to the project directory
cd //p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS

# Initialize Conda environment
source /p/project/hai_bimsb_dock_drug/miniconda3/etc/profile.d/conda.sh
conda activate diffdock


# Execute the Python script

python -W ignore -m main_multi_shot \
  --config DiffDock/default_inference_args.yaml \
  --protein_dir /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/PROTEIN_DB/PDBBind_processed2 \
  --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" \
  --out_dir /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_PDBBIND/results/ \
  --max_recursion_step 5 \
  --wandb_path /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/FACTORY/RESULTS/TREAMID_PDBBIND \
  --start $START \
  --end $END


EOT

  # Make the script executable
  chmod +x "$SCRIPTS_FOLDER/job_script_$COUNTER.sh"

  # Update START and END for the next script
  START=$((END + 1))
  END=$((END + 150))
  if [ $END -gt $MAX ]; then
    END=$MAX
  fi

  # Increment the script counter
  COUNTER=$((COUNTER + 1))
done
