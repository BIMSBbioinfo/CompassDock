import sys
import subprocess
from rdkit import Chem, RDLogger



sys.path.append('/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS/AA_Score_Tool/')
# Ensure the path is correctly added. This path should be to the directory containing the PoseCheck module.
sys.path.append('/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS/posecheck/')

# export PYTHONPATH="/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/diffdock_new_version:$PYTHONPATH"
path_to_add = "/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS"

# Prepend the path to sys.path
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)

from AA_Score_Tool.AA_Score import predict_dG
from posecheck import PoseCheck

def run_obabel(input_sdf_path, output_sdf_path):
    try:
        # Construct the command to run
        cmd = [
            'obabel', '-isdf', input_sdf_path, '-osdf',
            '-O', output_sdf_path, '-h'
        ]
        # Execute the command, capturing stdout and stderr
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        print("obabel output:", output.decode())
    except subprocess.CalledProcessError as e:
        print("Error running obabel:", e.output.decode())


def run_get_pocket(protein_path, ligand_output_path, pocket_path, energy_calc_path):
    try:
        # Command to activate conda environment and run the script
        cmd = f"""
        python "{energy_calc_path}/AA_Score_Tool/scripts/get_pocket_by_biopandas.py" "{protein_path}" "{ligand_output_path}" "{pocket_path}"
        """
        script_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, shell=True, executable="/bin/bash")
        if "Failed to generate pocket molecule." in script_output:
            print("Error: Failed to generate pocket molecule")
            # Handle the error as needed
    except subprocess.CalledProcessError as e:
        print("Error running Python script:", e.output)


# Function to process each ligand and protein pair
def binding_affinity(protein_pocket_file, ligand_file):
    mol_lig = next(Chem.SDMolSupplier(ligand_file, removeHs=False))
    mol_prot = Chem.MolFromPDBFile(protein_pocket_file, removeHs=False)
    mol_name, score = predict_dG(mol_prot, mol_lig, output_file=None)
    print(f"name: {mol_name}, Binding Affinity energy: {round(score, 2)} kcal/mol")
    return mol_name, score


def setup_rdkit_logging():
    # Disable RDKit warnings
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)


def posecheck_eval(protein_file, ligand_file):
    setup_rdkit_logging()

    pc = PoseCheck()

    # Load a protein from a PDB file (will run reduce in the background)
    pc.load_protein_from_pdb(protein_file)

    # Load ligands from an SDF file
    pc.load_ligands_from_sdf(ligand_file)

    # Check for clashes
    clashes = pc.calculate_clashes()
    print(f"Number of clashes in example molecule: {clashes[0]}")

    # Check for strain
    strain = pc.calculate_strain_energy()
    print(f"Strain energy of example molecule: {strain[0]}")

    # Check for interactions
    interactions = pc.calculate_interactions()
    print(f"Interactions of example molecule: {interactions}")

    # Processing interaction data
    flattened_columns = ['-'.join(col) if isinstance(col, tuple) else col for col in interactions.columns.tolist()]

    for index, row in interactions.iterrows():
        # Convert all row values to integers (assuming all interaction data can be represented as integers)
        int_row_values = [int(value) for value in row.tolist()]

    # Combine column names and row values into a dictionary
    inter_dict = dict(zip(flattened_columns, int_row_values))
    #print(inter_dict)

    return clashes[0], strain[0], inter_dict


