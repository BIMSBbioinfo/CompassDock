import pandas as pd

# Load the Excel file
data = pd.read_excel('/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/ECBL/data/pilot_library.xlsx')

# Check columns and extract 'EOS' and 'SMILES' columns
if 'eos' in data.columns and 'smiles' in data.columns:
    eos_smiles = data[['eos', 'smiles']]

    # Save these columns to a txt file
    eos_smiles.to_csv('eos_smiles.txt', sep='\t', index=False)
    print("File saved successfully.")
else:
    print("Columns 'EOS' and/or 'SMILES' not found in the file.")
