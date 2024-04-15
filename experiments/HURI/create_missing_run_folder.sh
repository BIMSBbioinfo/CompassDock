#!/bin/bash

# Path to your text file containing Protein IDs
TEXT_FILE="/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/HURI/missing_runs.txt"

# Directory containing your original .pdb files
SOURCE_DIR="/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/HuRI"

# Target directory where you want to copy the .pdb files
TARGET_DIR="/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/HuRI_missing_RUNS"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Read each Protein ID and copy the corresponding .pdb file
while IFS= read -r PROTEIN_ID
do
  cp -r "${SOURCE_DIR}/${PROTEIN_ID}.pdb" "${TARGET_DIR}/"
done < "$TEXT_FILE"

echo "Copying completed."
