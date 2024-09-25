import sys
from rdkit import Chem, RDLogger
from openbabel import pybel
import os

path_to_add = os.getcwd()
sys.path.append(f'{path_to_add}/AA_Score_Tool/')
sys.path.append(f'{path_to_add}/posecheck/')


# Prepend the path to sys.path
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)


from AA_Score_Tool.AA_Score import predict_dG
from posecheck.posecheck import PoseCheck
from AA_Score_Tool.utils.get_pocket_by_biopandas import GetPocket

def run_obabel(input_sdf_path, output_sdf_path):
    try:
        with open(output_sdf_path, 'w') as outfile:
                # Read and process each molecule from the input file
            for mol in pybel.readfile('sdf', input_sdf_path):
                # Add hydrogens to the molecule
                mol.OBMol.AddHydrogens()
                # Write the processed molecule to the output file in SDF format
                outfile.write(mol.write('sdf'))
    except Exception as e:
        print("Error running obabel:", str(e))


def run_get_pocket(protein_path, ligand_output_path, pocket_path):
    try:
        GetPocket(protein_file=protein_path, ligand_file=ligand_output_path, pdb_id=pocket_path)
    except RuntimeError as e:
        print("Error during Pocket Extraction:", e)


# Function to process each ligand and protein pair
def binding_affinity(protein_pocket_file, ligand_file):
    mol_lig = next(Chem.SDMolSupplier(ligand_file, removeHs=False))
    mol_prot = Chem.MolFromPDBFile(protein_pocket_file, removeHs=False)
    mol_name, score = predict_dG(mol_prot, mol_lig, output_file=None)
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

    # Check for strain
    strain = pc.calculate_strain_energy()

    # Check for interactions
    interactions = pc.calculate_interactions()

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

    # Check for strain
    strain = pc.calculate_strain_energy()

    return clashes[0], strain[0]




def process_docked_file(write_dir, sdf_file, protein_path_list):

    processed_sdf_directory = os.path.join(write_dir, "processed_sdf_files")
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
            # Assign error values 
            clashes = 1000
            strain = 1000
        else:
            raise  # Reraise if it's a different RuntimeError
    except ValueError as e:  # Assuming ValueError can be raised
        print(f"Value error in posecheck_eval for {pdb_base_name}: {e}")
        # Assign error values
        clashes = 1000
        strain = 1000


    try:
        run_obabel(input_sdf_path, output_sdf_path)
    except Exception as e:
        print(f"Obabel error for {pdb_base_name}: {e}")
        score = 1000  # Assign a error value for the binding affinity
        return score, clashes, strain


    try:
        pocket_path2 = os.path.join(pocket_path, f"{file_base_name}")
        # Assuming run_get_pocket would raise an exception if it fails
        run_get_pocket(protein_path, output_sdf_path, pocket_path2)
    except Exception as e:  # Catch a more specific exception if possible
        print(f"Pocket generation failed for {pdb_base_name}: {e}")
        score = 1000  # Assign a error value for the binding affinity
        return score, clashes, strain  # Skip further processing for this molecule


    # Continue with binding affinity calculation
    try:
        pocket_path3 = os.path.join(pocket_path, f"{file_base_name}_pocket.pdb")
        mol_name, score = binding_affinity(pocket_path3, output_sdf_path)
    except Exception as e:  # Again, catch more specific exceptions if possible
        print(f"Error during binding affinity calculation for {pdb_base_name}: {e}")
        score = 1000  # Assign a error value for the binding affinity
    except RuntimeError as e:
        if "Element '' not found" in str(e):
            print(f"Error encountered with binding_affinity for {pdb_base_name}. Assigning error values.")
            score = 1000  # Assign a error value for the binding affinity
        else:
            raise  # Reraise if it's a different RuntimeError 
    except ValueError as e:  # Assuming ValueError can be raised
        print(f"Value error in binding_affinity for {pdb_base_name}: {e}")
        score = 1000  # Assign a error value for the binding affinity
        

    movin_pckt_pdb = os.path.join(f"{protein_name}_pocket.pdb") 

    try:
        os.remove(movin_pckt_pdb)
    except FileNotFoundError:
        print(f"File not found: {movin_pckt_pdb}")

    return score, clashes, strain