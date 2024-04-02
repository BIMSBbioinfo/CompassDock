import os
import glob
from argparse import ArgumentParser, FileType
import wandb

from inference_wrap import run_docking
from compass import run_obabel, run_get_pocket, binding_affinity, posecheck_eval

os.environ["WANDB_MODE"] = "dryrun"
# Set the WANDB_DIR environment variable
# os.environ['WANDB_DIR'] = '/path/to/your/directory'

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

args = parser.parse_args()

results_summary, protein_path_list, write_dir, args, best_confidence_score = run_docking(args)

sdf_directory = write_dir
processed_sdf_directory = os.path.join(write_dir, "processed_sdf_files")
energy_calc_path = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/DiffDock_Compass"
pocket_path = os.path.join(write_dir, "pockets/")
protein_path = protein_path_list[0]

os.makedirs(processed_sdf_directory, exist_ok=True)
os.makedirs(pocket_path, exist_ok=True)  # Ensure the pockets directory exists

sdf_files = glob.glob(os.path.join(sdf_directory, f"rank1_confidence{best_confidence_score}.sdf"))

for sdf_file in sdf_files:
    
    wandb.init(project="diffdock_41_genes_treamid", entity="asarigun", config=args, settings=wandb.Settings(start_method="fork"))

    file_base_name = os.path.basename(sdf_file).replace('.sdf', '')
    
    if "confidence" in file_base_name:
        try:
            confidence_part = file_base_name.split("confidence")[-1]
            confidence_value = float(confidence_part)
            if confidence_value <= -1000:
                print(f"Skipping molecule with confidence value {confidence_value}")
                continue  # Skip to the next molecule
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
    
    run_obabel(input_sdf_path, output_sdf_path)
    
    pocket_path2 = os.path.join(pocket_path, f"{file_base_name}")
    # Assuming run_get_pocket generates and saves the pocket.pdb in pocket_path directory
    run_get_pocket(protein_path, output_sdf_path, pocket_path2, energy_calc_path)

    # Assuming binding_affinity and posecheck_eval functions handle file paths correctly
    # Generate pocket file name based on the convention you want

    pdb_name_with_extension = os.path.basename(protein_path)

    # Split the file name and extension
    pdb_base_name, _ = os.path.splitext(pdb_name_with_extension)

    #pocket_file_path = os.path.join(pocket_path, f"{file_base_name}_pocket.pdb")
    clashes, strain, inter_dict = posecheck_eval(protein_path, input_sdf_path)
    wandb.log({"Number of clashes": clashes})
    wandb.log({"Strain Energy": strain})
    wandb.log(inter_dict)

    pocket_path3 = os.path.join(pocket_path, f"{file_base_name}_pocket.pdb")
    mol_name, score = binding_affinity(pocket_path3, output_sdf_path)
    wandb.log({"Binding Affinity (kcal/mol)": score})
    wandb.finish()

    movin_pckt_pdb = os.path.join(f"{protein_name}_pocket.pdb") 

    try:
        os.remove(movin_pckt_pdb)
        print(f"Removed file: {movin_pckt_pdb}")
    except FileNotFoundError:
        print(f"File not found: {movin_pckt_pdb}")


