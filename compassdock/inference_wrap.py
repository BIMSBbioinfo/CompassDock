import copy
import os
import sys
import warnings
from argparse import Namespace
from functools import partial
from tqdm import tqdm
import yaml

import random
import torch
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from rdkit.Chem import RemoveAllHs

from compassdock.DiffDock.datasets.process_mols import write_mol_with_coords
from compassdock.DiffDock.utils.download import download_and_extract
from compassdock.DiffDock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from compassdock.DiffDock.utils.inference_utils import InferenceDataset, set_nones
from compassdock.DiffDock.utils.sampling import randomize_position, sampling
from compassdock.DiffDock.utils.utils import get_model
from compassdock.DiffDock.utils.visualise import PDBFile

warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the function that accepts an args Namespace
def run_docking(args, protein_path = None, protein_sequence = None,
                complex_name = None, ligand_description = None,
                out_dir = 'results/user_inference', save_visualisation = False,
                ligand_describe = None,
                protein_ligand_csv = None):
    """
    Runs the molecular docking process based on the provided configuration.

    Parameters:
    - args: An argparse.Namespace object containing all configuration options.
    """
    RDLogger.DisableLog('rdApp.*')

    set_seed(0)

    REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")

    # Download models if they don't exist locally
    if not os.path.exists(args['model_dir']):
        print(f"Models not found. Downloading")
        remote_urls = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
                       "https://www.dropbox.com/scl/fi/drg90rst8uhd2633tyou0/diffdock_models.zip?rlkey=afzq4kuqor2jb8adah41ro2lz&dl=1"]
        downloaded_successfully = False
        for remote_url in remote_urls:
            try:
                print(f"Attempting download from {remote_url}")
                files_downloaded = download_and_extract(remote_url, os.path.dirname(args['model_dir']))
                if not files_downloaded:
                    print(f"Download from {remote_url} failed.")
                    continue
                print(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
                downloaded_successfully = True
            except Exception as e:
                pass

        if not downloaded_successfully:
            raise Exception(f"Models not found locally and failed to download them from {remote_urls}")


    os.makedirs(out_dir, exist_ok=True)
    with open(f"{args['model_dir']}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if args['confidence_model_dir'] is not None:
        with open(f"{args['confidence_model_dir']}/model_parameters.yml") as f:
            confidence_args = Namespace(**yaml.full_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"DiffDock will run on {device}")

    if protein_ligand_csv is not None:
        df = pd.read_csv(protein_ligand_csv)
        complex_name_list = set_nones(df['complex_name'].tolist())
        protein_path_list = set_nones(df['protein_path'].tolist())
        protein_sequence_list = set_nones(df['protein_sequence'].tolist())
        ligand_description_list = set_nones(df['ligand_description'].tolist())
    else:
        complex_name_list = [complex_name if complex_name else f"complex_0"]
        protein_path_list = [protein_path]
        protein_sequence_list = [protein_sequence]
        if ligand_describe is None:
            print(f'Using initial ligand SMILES: {ligand_description}')
            ligand_description_list = [ligand_description]
        else:
            print(f'Using predocked ligand at {ligand_describe}')
            ligand_description_list = [ligand_describe]

    complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
    for name in complex_name_list:
        write_dir = f'{out_dir}/{name}'
        os.makedirs(write_dir, exist_ok=True)

    # preprocessing of complexes into geometric graphs
    test_dataset = InferenceDataset(out_dir=out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                    ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                    lm_embeddings=True,
                                    receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                    c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                    all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                    atom_max_neighbors=score_model_args.atom_max_neighbors,
                                    knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args['confidence_model_dir'] is not None and not confidence_args.use_original_model_cache:
        print('HAPPENING | confidence model uses different type of graphs than the score model. '
            'Loading (or creating if not existing) the data for the confidence model now.')
        confidence_test_dataset = \
            InferenceDataset(out_dir=out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
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

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args['old_score_model'])
    state_dict = torch.load(f"{args['model_dir']}/{args['ckpt']}", map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args['confidence_model_dir'] is not None:
        confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                    confidence_mode=True, old=args['old_confidence_model'])
        state_dict = torch.load(f"{args['confidence_model_dir']}/{args['confidence_ckpt']}", map_location=torch.device(device))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None

    tr_schedule = get_t_schedule(inference_steps=args['inference_steps'], sigma_schedule='expbeta')

    results_summary = {
            'failed': [],
            'skipped': [],
            'success': [],
            'output_files': []
        }
    
    failures, skipped = 0, 0
    N = args['samples_per_complex']
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
                            initial_noise_std_proportion=args['initial_noise_std_proportion'],
                            choose_residue=args['choose_residue'])

            lig = orig_complex_graph.mol[0]

            # initialize visualisation
            pdb = None
            if save_visualisation:
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
                                            inference_steps=args['actual_steps'] if args['actual_steps'] is not None else args['inference_steps'],
                                            tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                            device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                            visualization_list=visualization_list, confidence_model=confidence_model,
                                            confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                            batch_size=args['batch_size'], no_final_step_noise=args['no_final_step_noise'],
                                            temp_sampling=[args['temp_sampling_tr'], args['temp_sampling_rot'],
                                                            args['temp_sampling_tor']],
                                            temp_psi=[args['temp_psi_tr'], args['temp_psi_rot'], args['temp_psi_tor']],
                                            temp_sigma_data=[args['temp_sigma_data_tr'], args['temp_sigma_data_rot'],
                                                            args['temp_sigma_data_tor']])

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

            write_dir = f'{out_dir}/{complex_name_list[idx]}'
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)
                if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))
                if rank == 0:
                    best_confidence_score = f"{confidence[rank]:.2f}"

            # save visualisation frames
            if save_visualisation:
                if confidence is not None:
                    for rank, batch_idx in enumerate(re_order):
                        visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
                else:
                    for rank, batch_idx in enumerate(ligand_pos):
                        visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

        except Exception as e:
            print("Failed on", orig_complex_graph["name"], e)
            failures += 1

    results_summary['failed_count'] = failures
    results_summary['skipped_count'] = skipped
    results_summary['success_count'] = len(test_dataset) - failures - skipped

    return results_summary, protein_path_list, write_dir, best_confidence_score

