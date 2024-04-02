#!/bin/bash

# Directories
main_dir="/fast/AG_Akalin/asarigun/Arcas_Stage_1/PROTEIN_DB/DockGen"
factor_dir="/fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RUNS/TREAMID_DOCKGEN"

# Ensure factor_dir exists
mkdir -p "$factor_dir"

echo "Checking for PDB files in: $main_dir"

# Loop over each PDB file in the directory
for pdb_file in "$main_dir"/*.pdb; do
  if [ -f "$pdb_file" ]; then
    pdb_id=$(basename "$pdb_file" .pdb)
    script_name="slurm_${pdb_id}.sh"

    echo "Creating script for: $pdb_id"

    # Create the SLURM script
    cat > "$factor_dir/$script_name" << EOF
#!/bin/bash
# SLURM directives for resource allocation
#$ -l gpu=1 -l cuda_memory=35G
#$ -cwd
#$ -V
#$ -N "diffdock_compass_dockgen_${pdb_id}"
#$ -l h_rt=00:45:00
#$ -pe smp 1
#$ -e /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_DOCKGEN/log_\$JOB_ID.err
#$ -o /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_DOCKGEN/log_\$JOB_ID.out

# Change directory to the project root
cd /fast/AG_Akalin/asarigun/Arcas_Stage_1/DiffDock_Compass

# Activate the Python environment
source /home/asarigu/miniconda3/etc/profile.d/conda.sh
conda activate diffdock_compass

# Notify start of process
echo "Processing ${pdb_id}..."

# Execute the main Python script with arguments, split for readability
python -W ignore -m main_multi_shot \
  --config DiffDock/default_inference_args.yaml \
  --complex_name ${pdb_id} \
  --protein_path ${main_dir}/${pdb_id}.pdb \
  --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" \
  --out_dir /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_DOCKGEN/results/ \
  --max_recursion_step 5 \
  --wandb_path /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_DOCKGEN

EOF


    # Make the script executable
    chmod +x "$factor_dir/$script_name"

    echo "Created script: $factor_dir/$script_name"
  fi
done
