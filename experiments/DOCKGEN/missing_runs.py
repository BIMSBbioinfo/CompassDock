import pandas as pd
import os

# Load the filtered CSV file
filtered_csv_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/DOCKGEN/data/filtered_summary_new_rec_1.csv'  # Update this path
filtered_df = pd.read_csv(filtered_csv_path)

# Directory containing your .pdb files
pdb_directory = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/DockGen/protein_pdb_files/'

# List all .pdb files and extract Protein IDs
protein_ids_in_directory = [filename.split('.')[0] for filename in os.listdir(pdb_directory) if filename.endswith('.pdb')]

# Get the list of Protein IDs from the filtered CSV file
protein_ids_in_csv = filtered_df['Protein ID'].tolist()

# Find Protein IDs not in the filtered CSV file
protein_ids_not_in_csv = [protein_id for protein_id in protein_ids_in_directory if protein_id not in protein_ids_in_csv]

# Path to save the Protein IDs not in the filtered CSV file
output_file_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/DOCKGEN/missing_runs.txt'  # Update this path

# Save the Protein IDs to a text file
with open(output_file_path, 'w') as file:
    for protein_id in protein_ids_not_in_csv:
        file.write(protein_id + '\n')

print(f"Protein IDs not in the filtered CSV file have been saved to {output_file_path}")
