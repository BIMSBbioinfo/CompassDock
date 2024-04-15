import os
import glob
from argparse import ArgumentParser, FileType, Namespace
import wandb
from pytorch_lightning.loggers import WandbLogger

from inference_wrap import run_docking
from compass import run_obabel, run_get_pocket, binding_affinity, posecheck_eval

from art import *

parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default='DiffDock/default_inference_args.yaml')
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
parser.add_argument('--complex_name', type=str, default=None, help='Name that the complex will be saved with')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')

parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=10, help='')
parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')

parser.add_argument('--old_score_model', action='store_true', default=False, help='')
parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0, help='Initial noise std proportion')
parser.add_argument('--choose_residue', action='store_true', default=False, help='')

parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
parser.add_argument('--temp_psi_tr', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
parser.add_argument('--temp_psi_rot', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
parser.add_argument('--temp_psi_tor', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
parser.add_argument('--gnina_path', type=str, default='gnina', help='')
parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')  # To redirect gnina subprocesses stdouts from the terminal window
parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)

parser.add_argument('--max_recursion_step', type=int, default=5, help='')
parser.add_argument('--wandb_path', type=str, default='/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS', help='')
parser.add_argument('--protein_dir', type=str, default=None, help='')
parser.add_argument('--start', type=int, default=None, help='Start index of pdb file range')
parser.add_argument('--end', type=int, default=None, help='End index of the pdb file range')

args = parser.parse_args()


def recursive_docking_and_processing(args, iteration=0, max_iterations=args.max_recursion_step, ligand_description=None):
    if iteration >= max_iterations:
        print("Maximum iterations reached.")
        return
    
    if ligand_description is None:
        results_summary, protein_path_list, write_dir, args, best_confidence_score = run_docking(args)
    else:
        results_summary, protein_path_list, write_dir, args, best_confidence_score = run_docking(args, ligand_description)

    sdf_files = glob.glob(os.path.join(write_dir, f"rank1_confidence{best_confidence_score}.sdf"))

    if not sdf_files:
        print(f"No .sdf files found matching the pattern for iteration {iteration}.")
        return

    sdf_file = sdf_files[0]  # Assume we're interested in the first file found
    binding_aff, clashes, strain, confidence_value = process_sdf_file(write_dir, sdf_file, args, protein_path_list, iteration, ligand_description)
    
    if binding_aff <= 0 and clashes <= 6 and strain <= 4 and confidence_value >= -1.5:
        print("Optimal docking conformation achieved with high confidence. Halting DiffDock-Compass execution...")
        return
    
    if binding_aff >= 0 or clashes >= 13 or strain >= 14 or confidence_value <= -1.5:
        print("DiffDock-Compass' scores are very low. Halting DiffDock-Compass execution for next recursion step...")
        return
    
    recursive_docking_and_processing(args, iteration + 1, max_iterations, ligand_description=sdf_file)


def process_sdf_file(write_dir, sdf_file, args, protein_path_list, iteration, ligand_description):

    os.environ["WANDB_MODE"] = "dryrun"
    # Set the WANDB_DIR environment variable
    os.environ['WANDB_DIR'] = args.wandb_path
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    wandb_logger = WandbLogger()

    processed_sdf_directory = os.path.join(write_dir, "processed_sdf_files")
    energy_calc_path = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS"
    pocket_path = os.path.join(write_dir, "pockets/")
    protein_path = protein_path_list[0]

    os.makedirs(processed_sdf_directory, exist_ok=True)
    os.makedirs(pocket_path, exist_ok=True)  # Ensure the pockets directory exists

    wandb.init(project="diffdock_41_genes_treamid", entity="asarigun", config=args, 
               settings=wandb.Settings(start_method="fork"))#, dir=args.wandb_path)

    
    recursion_step = f"{iteration+1}/{args.max_recursion_step}"
    print("Recursion Step Done:", recursion_step)
    wandb.log({"Recursion Step Done": recursion_step})
    
    print("Ligand Description:", ligand_description)
    wandb.log({"Ligand Description": ligand_description})
    
    file_base_name = os.path.basename(sdf_file).replace('.sdf', '')
    
    # Extracting confidence, if applicable
    if "confidence" in file_base_name:
        try:
            confidence_part = file_base_name.split("confidence")[-1]
            confidence_value = float(confidence_part)
            if confidence_value <= -1000:
                print(f"Skipping molecule with confidence value {confidence_value}")
                return  # Skip to the next molecule
        except ValueError:
            # Handle any conversion errors gracefully
            print("Error extracting confidence value; processing the molecule anyway")
            confidence_value = None
    else:
        # If "confidence" is not in the file name, process the molecule as usual
        confidence_value = None

    print("Confidence value:", confidence_value)
    wandb.log({"Confidence Score": confidence_value})
    

    protein_name = protein_path.split('/')[-1].replace('.pdb', '')

    print('protein_name:', protein_name)
    wandb.log({"Protein ID": protein_name})
    wandb.log({"Rank of sdf": file_base_name})

    input_sdf_path = sdf_file
    output_sdf_path = os.path.join(processed_sdf_directory, f"{file_base_name}_output_clean.sdf")
    
    pdb_name_with_extension = os.path.basename(protein_path)

    # Split the file name and extension
    pdb_base_name, _ = os.path.splitext(pdb_name_with_extension)

    try:
        clashes, strain, inter_dict = posecheck_eval(protein_path, input_sdf_path)
        wandb.log({"Number of clashes": clashes})
        wandb.log({"Strain Energy": strain})
        wandb.log(inter_dict)
    except RuntimeError as e:
        if "Element '' not found" in str(e):
            print(f"Error encountered with posecheck_eval for {pdb_base_name}. Assigning default values.")
            # Assign default values or take alternative actions here
            clashes = 1000
            strain = 1000
            inter_dict = {'error': 'Element not found, default values assigned'}
            # Log the default values to WandB as well
            wandb.log({"Number of clashes": clashes})
            wandb.log({"Strain Energy": strain})
            wandb.log(inter_dict)
        else:
            raise  # Reraise if it's a different RuntimeError
    except ValueError as e:  # Assuming ValueError can be raised
        print(f"Value error in posecheck_eval for {pdb_base_name}: {e}")
        # Assign default values or take alternative actions here
        clashes = 1000
        strain = 1000
        inter_dict = {'error': 'Element not found, default values assigned'}
        # Log the default values to WandB as well
        wandb.log({"Number of clashes": clashes})
        wandb.log({"Strain Energy": strain})
        wandb.log(inter_dict)


    try:
        run_obabel(input_sdf_path, output_sdf_path)
    except Exception as e:
        print(f"Obabel error for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        # You might want to log this condition as well to WandB
        wandb.log({"Binding Affinity (kcal/mol)": score})
        wandb.finish()
        return score, clashes, strain, confidence_value


    try:
        pocket_path2 = os.path.join(pocket_path, f"{file_base_name}")
        # Assuming run_get_pocket would raise an exception if it fails
        run_get_pocket(protein_path, output_sdf_path, pocket_path2, energy_calc_path)
    except Exception as e:  # Catch a more specific exception if possible
        print(f"Pocket generation failed for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        # You might want to log this condition as well to WandB
        wandb.log({"Binding Affinity (kcal/mol)": score})
        wandb.finish()
        return score, clashes, strain, confidence_value  # Skip further processing for this molecule


    # Continue with binding affinity calculation
    try:
        pocket_path3 = os.path.join(pocket_path, f"{file_base_name}_pocket.pdb")
        mol_name, score = binding_affinity(pocket_path3, output_sdf_path)
        wandb.log({"Binding Affinity (kcal/mol)": score})
    except Exception as e:  # Again, catch more specific exceptions if possible
        print(f"Error during binding affinity calculation for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        wandb.log({"Binding Affinity (kcal/mol)": score})
        wandb.finish()
    except RuntimeError as e:
        if "Element '' not found" in str(e):
            print(f"Error encountered with binding_affinity for {pdb_base_name}. Assigning default values.")
            score = 1000  # Assign a default or indicative value for the binding affinity
            wandb.log({"Binding Affinity (kcal/mol)": score})
            wandb.finish()
        else:
            raise  # Reraise if it's a different RuntimeError 
    except ValueError as e:  # Assuming ValueError can be raised
        print(f"Value error in binding_affinity for {pdb_base_name}: {e}")
        score = 1000  # Assign a default or indicative value for the binding affinity
        wandb.log({"Binding Affinity (kcal/mol)": score})
        wandb.finish()
        
    wandb.finish()

    movin_pckt_pdb = os.path.join(f"{protein_name}_pocket.pdb") 

    try:
        os.remove(movin_pckt_pdb)
        print(f"Removed file: {movin_pckt_pdb}")
    except FileNotFoundError:
        print(f"File not found: {movin_pckt_pdb}")

    return score, clashes, strain, confidence_value


def main():

    decor = None#"fancy20"
    welcome = text2art("|    Wellcome to DiffDock-Compass    |", font="small",decoration=decor)
    logo1=text2art("|                     Navigating the Future                     |", font="small",decoration=decor)
    logo2=text2art("|     Drugs with DiffDock-Compass      |", font="small",decoration=decor)
    art_decore = text2art("------------------------------", font="small",decoration=decor)

    print(art_decore)
    print(welcome)
    print(art_decore)
    print(logo1)
    print(logo2)
    print(art_decore)

    if args.protein_dir:
        # Get a list of all .pdb files in the directory and sort them alphabetically
        pdb_files = sorted([file for file in os.listdir(args.protein_dir) if file.endswith('.pdb')])

        # Process PDB files within the specified range
        for protein_file in pdb_files[args.start:args.end]:  # This slices the list to include only the first two elements
            protein_path = os.path.join(args.protein_dir, protein_file)
            
            # Create a copy of args to modify for each protein file
            # This avoids altering the original args object directly
            current_args = Namespace(**vars(args))
            current_args.protein_path = protein_path
            current_args.complex_name = protein_file[:-4]  # Remove the '.pdb' extension for the complex name

            try:
                recursive_docking_and_processing(current_args)
            except Exception as e:
                print(f"Error processing {protein_file}: {e}")
                # Optionally, log the error or handle it as needed
                continue  # This ensures the loop continues with the next file

    else:
        recursive_docking_and_processing(args)

if __name__ == "__main__":
    main()



"""
python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --complex_name TEST --protein_path /fast/AG_Akalin/asarigun/Arcas_Stage_1/PROTEIN_DB/AlphaFold_HUMAN_v3/AF-P14618-F1-model_v3.pdb --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir results --max_recursion_step 2

python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --complex_name TEST --protein_path /fast/AG_Akalin/asarigun/Arcas_Stage_1/PROTEIN_DB/AlphaFold_HUMAN_v3/AF-P14618-F1-model_v3.pdb --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_AF/results --max_recursion_step 5

python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --complex_name TEST --protein_path /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS/DiffDock/examples/1a46_protein_processed.pdb --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir results --max_recursion_step 2


python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --complex_name TEST --protein_path DiffDock/examples/1a46_protein_processed.pdb --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir results --max_recursion_step 2

python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --protein_dir /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS/DiffDock/examples --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir results --max_recursion_step 2


python -W ignore -m main_multi_shot   --config DiffDock/default_inference_args.yaml   --complex_name AF-Q2M1K9-F1-model_v3   --protein_path /fast/AG_Akalin/asarigun/Arcas_Stage_1/PROTEIN_DB/AlphaFold_HUMAN_v3/AF-Q2M1K9-F1-model_v3.pdb   --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2"   --out_dir /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_AF/results/   --max_recursion_step 5   --wandb_path /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_AF

if protein_dir:

python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --protein_dir /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS/DiffDock/examples --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir results --max_recursion_step 1 --start 0 --end 3

else:

python -W ignore -m main_multi_shot --config DiffDock/default_inference_args.yaml --complex_name TEST --protein_path /p/project/hai_bimsb_dock_drug/Arcas_Stage_1/ROOF/COMPASS/DiffDock/examples/1a46_protein_processed.pdb --ligand_description "C1=CN=C(N1)CCNC(=O)CCCC(=O)NCCC2=NC=CN2" --out_dir results --max_recursion_step 1

"""
