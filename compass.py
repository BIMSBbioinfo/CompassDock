import sys
import subprocess
from rdkit import Chem, RDLogger
from openbabel import pybel
import os



path_to_add = os.getcwd()

sys.path.append(f'{path_to_add}/AA_Score_Tool/')
# Ensure the path is correctly added. This path should be to the directory containing the PoseCheck module.
sys.path.append(f'{path_to_add}/posecheck/')


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
        '''# Open the output file
        with open(output_sdf_path, 'w') as outfile:
            # Read and process each molecule from the input file
            for mol in pybel.readfile('sdf', input_sdf_path):
                # Add hydrogens to the molecule
                mol.OBMol.AddHydrogens()
                # Write the processed molecule to the output file in SDF format
                outfile.write(mol.write('sdf'))
        print("obabel output: Conversion completed successfully.")
    except Exception as e:
        print("Error running obabel:", str(e))'''


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



def posecheck_for_preds(protein_file, ligand_file):
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

    return clashes[0], strain[0]




def process_docked_file(write_dir, sdf_file, protein_path_list):

    processed_sdf_directory = os.path.join(write_dir, "processed_sdf_files")
    energy_calc_path = os.getcwd()
    pocket_path = os.path.join(write_dir, "pockets/")
    protein_path = protein_path_list

    os.makedirs(processed_sdf_directory, exist_ok=True)
    os.makedirs(pocket_path, exist_ok=True)  # Ensure the pockets directory exists

    file_base_name = os.path.basename(sdf_file).replace('.sdf', '')
    
    protein_name = protein_path.split('/')[-1].replace('.pdb', '')

    input_sdf_path = sdf_file
    output_sdf_path = os.path.join(processed_sdf_directory, f"{file_base_name}_output_clean.sdf")
    
    pdb_name_with_extension = os.path.basename(protein_path)

    # Split the file name and extension
    pdb_base_name, _ = os.path.splitext(pdb_name_with_extension)

    try:
        clashes, strain = posecheck_for_preds(protein_path, input_sdf_path)

    except RuntimeError as e:
        if "Element '' not found" in str(e):
            print(f"Error encountered with posecheck_eval for {pdb_base_name}. Assigning default values.")
            # Assign default values or take alternative actions here
            clashes = 1000
            strain = 1000
        else:
            raise  # Reraise if it's a different RuntimeError
    except ValueError as e:  # Assuming ValueError can be raised
        print(f"Value error in posecheck_eval for {pdb_base_name}: {e}")
        # Assign default values or take alternative actions here
        clashes = 1000
        strain = 1000


    try:
        run_obabel(input_sdf_path, output_sdf_path)
    except Exception as e:
        print(f"Obabel error for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        # You might want to log this condition as well to WandB
        return score, clashes, strain


    try:
        pocket_path2 = os.path.join(pocket_path, f"{file_base_name}")
        # Assuming run_get_pocket would raise an exception if it fails
        run_get_pocket(protein_path, output_sdf_path, pocket_path2, energy_calc_path)
    except Exception as e:  # Catch a more specific exception if possible
        print(f"Pocket generation failed for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        # You might want to log this condition as well to WandB
        return score, clashes, strain  # Skip further processing for this molecule


    # Continue with binding affinity calculation
    try:
        pocket_path3 = os.path.join(pocket_path, f"{file_base_name}_pocket.pdb")
        mol_name, score = binding_affinity(pocket_path3, output_sdf_path)
    except Exception as e:  # Again, catch more specific exceptions if possible
        print(f"Error during binding affinity calculation for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
    except RuntimeError as e:
        if "Element '' not found" in str(e):
            print(f"Error encountered with binding_affinity for {pdb_base_name}. Assigning default values.")
            score = 1000  # Assign a default or indicative value for the binding affinity
        else:
            raise  # Reraise if it's a different RuntimeError 
    except ValueError as e:  # Assuming ValueError can be raised
        print(f"Value error in binding_affinity for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        

    movin_pckt_pdb = os.path.join(f"{protein_name}_pocket.pdb") 

    try:
        os.remove(movin_pckt_pdb)
        print(f"Removed file: {movin_pckt_pdb}")
    except FileNotFoundError:
        print(f"File not found: {movin_pckt_pdb}")

    return score, clashes, strain