from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
    parser.add_argument('--complex_name', type=str, default=None, help='Name that the complex will be saved with')
    parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
    parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
    parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')

    parser.add_argument('--max_redocking_step', type=int, default=5, help='')
    parser.add_argument('--wandb_path', type=str, default='', help='')
    parser.add_argument('--protein_dir', type=str, default=None, help='')
    parser.add_argument('--smiles_dir', type=str, default=None, help='')
    parser.add_argument('--molecule_name', type=str, default=None, help='')
    parser.add_argument('--protein_start', type=int, default=None, help='Start index of pdb file range')
    parser.add_argument('--protein_end', type=int, default=None, help='End index of the pdb file range')
    parser.add_argument('--smiles_start', type=int, default=None, help='Start index of smiles range')
    parser.add_argument('--smiles_end', type=int, default=None, help='End index of the smiles file range')

    parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
    parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

    parser.add_argument('--model_dir', type=str, default='workdir/v1.1/score_model', help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--confidence_model_dir', type=str, default='workdir/v1.1/confidence_model', help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--batch_size', type=int, default=10, help='')
    parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--actual_steps', type=int, default=19, help='Number of denoising steps that are actually performed')

    parser.add_argument('--old_score_model', action='store_true', default=False, help='')
    parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
    parser.add_argument('--initial_noise_std_proportion', type=float, default=1.4601642460337794, help='Initial noise std proportion')
    parser.add_argument('--choose_residue', action='store_true', default=False, help='')

    parser.add_argument('--temp_sampling_tr', type=float, default=1.170050527854316)
    parser.add_argument('--temp_psi_tr', type=float, default=0.727287304570729)
    parser.add_argument('--temp_sigma_data_tr', type=float, default=0.9299802531572672)
    parser.add_argument('--temp_sampling_rot', type=float, default=2.06391612594481)
    parser.add_argument('--temp_psi_rot', type=float, default=0.9022615585677628)
    parser.add_argument('--temp_sigma_data_rot', type=float, default=0.7464326999906034)
    parser.add_argument('--temp_sampling_tor', type=float, default=7.044261621607846)
    parser.add_argument('--temp_psi_tor', type=float, default=0.5946212391366862)
    parser.add_argument('--temp_sigma_data_tor', type=float, default=0.6943254174849822)

    parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
    parser.add_argument('--gnina_path', type=str, default='gnina', help='')
    parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')  # To redirect gnina subprocesses stdouts from the terminal window
    parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
    parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
    parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)

    args = parser.parse_args()

    return args


configs = {
    'samples_per_complex': 10, 
    'model_dir': 'workdir/v1.1/score_model', 
    'ckpt': 'best_ema_inference_epoch_model.pt', 
    'confidence_model_dir': 'workdir/v1.1/confidence_model', 
    'confidence_ckpt': 'best_model_epoch75.pt', 

    'batch_size': 10, 
    'no_final_step_noise': True, 
    'inference_steps': 20, 
    'actual_steps': 19, 

    'old_score_model': False, 
    'old_confidence_model': True, 
    'initial_noise_std_proportion': 1.4601642460337794, 
    'choose_residue': False, 

    'temp_sampling_tr': 1.170050527854316,
    'temp_psi_tr': 0.727287304570729, 
    'temp_sigma_data_tr': 0.9299802531572672, 
    'temp_sampling_rot': 2.06391612594481, 
    'temp_psi_rot': 0.9022615585677628, 
    'temp_sigma_data_rot': 0.7464326999906034, 
    'temp_sampling_tor': 7.044261621607846, 
    'temp_psi_tor': 0.5946212391366862,
    'temp_sigma_data_tor': 0.6943254174849822,
}