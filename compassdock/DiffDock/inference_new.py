import copy
import os
import torch
from argparse import ArgumentParser, Namespace, FileType
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from rdkit.Chem import RemoveAllHs

from datasets.process_mols import write_mol_with_coords
from utils.download import download_and_extract
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning)


RDLogger.DisableLog('rdApp.*')
import yaml
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default='default_inference_args.yaml')
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

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")

if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value

# Download models if they don't exist locally
if not os.path.exists(args.model_dir):
    print(f"Models not found. Downloading")
    # TODO Remove the dropbox URL once the models are uploaded to GitHub release
    remote_urls = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
                   "https://www.dropbox.com/scl/fi/drg90rst8uhd2633tyou0/diffdock_models.zip?rlkey=afzq4kuqor2jb8adah41ro2lz&dl=1"]
    downloaded_successfully = False
    for remote_url in remote_urls:
        try:
            print(f"Attempting download from {remote_url}")
            files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
            if not files_downloaded:
                print(f"Download from {remote_url} failed.")
                continue
            print(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
            downloaded_successfully = True
        except Exception as e:
            pass

    if not downloaded_successfully:
        raise Exception(f"Models not found locally and failed to download them from {remote_urls}")

os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DiffDock will run on {device}")

if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list = set_nones(df['complex_name'].tolist())
    protein_path_list = set_nones(df['protein_path'].tolist())
    protein_sequence_list = set_nones(df['protein_sequence'].tolist())
    ligand_description_list = set_nones(df['ligand_description'].tolist())
else:
    complex_name_list = [args.complex_name if args.complex_name else f"complex_0"]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]

print('protein_path_list:', protein_path_list[0])
print('protein_path_list:', protein_path_list)


complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
for name in complex_name_list:
    write_dir = f'{args.out_dir}/{name}'
    os.makedirs(write_dir, exist_ok=True)

# preprocessing of complexes into geometric graphs
test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                lm_embeddings=True,
                                receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                atom_max_neighbors=score_model_args.atom_max_neighbors,
                                knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
    print('HAPPENING | confidence model uses different type of graphs than the score model. '
          'Loading (or creating if not existing) the data for the confidence model now.')
    confidence_test_dataset = \
        InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                         ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                         lm_embeddings=True,
                         receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                         c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                         all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                         atom_max_neighbors=confidence_args.atom_max_neighbors,
                         precomputed_lm_embeddings=test_dataset.lm_embeddings,
                         knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
else:
    confidence_test_dataset = None

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

if args.confidence_model_dir is not None:
    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                 confidence_mode=True, old=args.old_confidence_model)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None
    confidence_args = None

tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

failures, skipped = 0, 0
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))
for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    if not orig_complex_graph.success[0]:
        skipped += 1
        print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
        continue
    try:
        if confidence_test_dataset is not None:
            confidence_complex_graph = confidence_test_dataset[idx]
            if not confidence_complex_graph.success:
                skipped += 1
                print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                continue
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
        else:
            confidence_data_list = None
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
                           initial_noise_std_proportion=args.initial_noise_std_proportion,
                           choose_residue=args.choose_residue)

        lig = orig_complex_graph.mol[0]

        # initialize visualisation
        pdb = None
        if args.save_visualisation:
            visualization_list = []
            for graph in data_list:
                pdb = PDBFile(lig)
                pdb.add(lig, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)
        else:
            visualization_list = None

        # run reverse diffusion
        data_list, confidence = sampling(data_list=data_list, model=model,
                                         inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                         tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                         device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                         visualization_list=visualization_list, confidence_model=confidence_model,
                                         confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                         batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                         temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                        args.temp_sampling_tor],
                                         temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                         temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot,
                                                          args.temp_sigma_data_tor])

        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

        # reorder predictions based on confidence output
        if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]

        # save predictions
        write_dir = f'{args.out_dir}/{complex_name_list[idx]}'
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)
            if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

        # save visualisation frames
        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

    except Exception as e:
        print("Failed on", orig_complex_graph["name"], e)
        failures += 1

print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
print(f'Results are in {args.out_dir}')
        
import sys
import warnings
import subprocess
import os
import glob
import shutil
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Add the path to the directory containing the AA_Score.py file
sys.path.append('/fast/AG_Akalin/asarigun/Arcas_Stage_1/DiffDock_Compass/AA_Score_Tool/')
# Ensure the path is correctly added. This path should be to the directory containing the PoseCheck module.
sys.path.append('/fast/AG_Akalin/asarigun/Arcas_Stage_1/DiffDock_Compass/posecheck/')

# export PYTHONPATH="/p/project/hai_bimsb_dock_drug/Arcas_Stage_1/diffdock_new_version:$PYTHONPATH"
# Define the path you want to add
path_to_add = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/DiffDock_Compass"

# Prepend the path to sys.path
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)

from AA_Score_Tool.AA_Score import predict_dG
from posecheck import PoseCheck
import wandb

os.environ["WANDB_MODE"] = "dryrun"


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
    print(f"name: {mol_name}, energy: {round(score, 2)} kcal/mol")
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
    #print(flattened_columns)

    for index, row in interactions.iterrows():
        # Convert all row values to integers (assuming all interaction data can be represented as integers)
        int_row_values = [int(value) for value in row.tolist()]

    # Combine column names and row values into a dictionary
    inter_dict = dict(zip(flattened_columns, int_row_values))
    print(inter_dict)

    #protein_id = os.path.splitext(os.path.basename(protein_file))[0]
    #print(protein_id)
    return clashes[0], strain[0], inter_dict
    

sdf_directory = write_dir
processed_sdf_directory = os.path.join(write_dir, "processed_sdf_files")
energy_calc_path = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/DiffDock_Compass"
pocket_path = os.path.join(write_dir, "pockets/")
protein_path = protein_path_list[0]

os.makedirs(processed_sdf_directory, exist_ok=True)
os.makedirs(pocket_path, exist_ok=True)  # Ensure the pockets directory exists

sdf_files = glob.glob(os.path.join(sdf_directory, "rank1_confidence*.sdf"))

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


