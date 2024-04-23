import pandas as pd

# Load the text file
data = pd.read_csv('/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/ECBL/eos_smiles.txt', sep='\t')

# Extract rows 5 through 9 (index 4 to 8)
selected_smiles = data.loc[5012:5015, 'smiles']
selected_eos = data.loc[5012:5015, 'eos']

# Print the selected "SMILES"
print(selected_smiles)
print(selected_eos)


for smile in selected_smiles:
    # Example processing: print each SMILES string
    print(smile)