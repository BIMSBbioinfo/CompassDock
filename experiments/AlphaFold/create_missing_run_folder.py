import shutil
import os

# Path to your text file containing Protein IDs
text_file_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/HURI/missing_runs.txt'

# Directory containing your original .pdb files
source_directory = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/HuRI'

# Target directory where you want to copy the .pdb files
target_directory = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/HuRI_missing_RUNS'

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Read the Protein IDs from the text file
with open(text_file_path, 'r') as file:
    protein_ids = file.read().splitlines()

# Copy each corresponding .pdb file to the target directory
for protein_id in protein_ids:
    source_file = os.path.join(source_directory, f"{protein_id}.pdb")
    target_file = os.path.join(target_directory, f"{protein_id}.pdb")
    
    # Check if the source file exists before attempting to copy
    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)
        print(f"Copied: {source_file} to {target_file}")
    else:
        print(f"File not found: {source_file}")

print("Copying completed.")
