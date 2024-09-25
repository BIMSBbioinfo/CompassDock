import copy
import os
import numpy as np
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import itertools
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles
import pandas as pd


from DiffDock.datasets.process_mols import generate_conformer, read_molecule
from DiffDock.confidence.dataset import ListDataset
from DiffDock.utils import so3, torus
from DiffDock.utils.molecules_utils import get_symmetry_rmsd
from DiffDock.utils.sampling import randomize_position, sampling
from DiffDock.utils.diffusion_utils import get_t_schedule
from DiffDock.utils.diffusion_utils import modify_conformer, set_time, modify_conformer_batch
from DiffDock.datasets.process_mols import write_mol_with_coords
from compassdock.compassdock.compass import process_docked_file

def get_compass_precalculated_data(protein_id):
    # Load the CSV file into a DataFrame
    file_path = 'experiments/PDBBind_GT/data/pdbbind_gt_compass.csv'
    df = pd.read_csv(file_path)

    # Filter the DataFrame for the specified Protein ID
    protein_data = df[df['Protein ID'] == protein_id]

    # Check if the Protein ID is found
    if not protein_data.empty:
        # Extract and print the required data
        binding_affinity = protein_data['Binding Affinity (kcal/mol)'].values[0]
        strain_energy = protein_data['Strain Energy'].values[0]
        number_of_clashes = protein_data['Number of clashes'].values[0]
        
        print(f"Ground Truth Protein ID: {protein_id}, Binding Affinity (kcal/mol): {binding_affinity}, Strain Energy: {strain_energy}, Number of Clashes: {number_of_clashes}")
    else:
        print(f"No data found for Protein ID: {protein_id}")
    
    return binding_affinity, number_of_clashes, strain_energy

def LANMSE(pred, true):
    'Log Absolute Normalized - MSE'
    epsilon = 1e-5  # Small constant to avoid division by zero
    # Apply a logarithmic transformation with a shift to ensure positive values
    if torch.abs(true).item() < 1:
        buffer_true = 1.1
    else:
        buffer_true = 1.0
    
    if torch.abs(pred).item() < 1:
        buffer_pred = 1.1
    else:
        buffer_pred = 1.0
    
    pred_log = torch.log(torch.abs(pred) + buffer_pred)
    true_log = torch.log(torch.abs(true) + buffer_true)
    
    # Compute the mean squared error on the transformed values
    return torch.mean(((true_log - pred_log) / (torch.abs(true_log)*2 + epsilon)) ** 2)


def compass_constant(true_affinity, pred_affinity, true_strain, pred_strain, true_clashes, pred_clashes_probs):

    true_values = torch.stack([
        torch.tensor([true_affinity], dtype=torch.float32),
        torch.tensor([true_strain], dtype=torch.float32),
        torch.tensor([true_clashes], dtype=torch.float32)
    ])

    pred_values = torch.stack([
        torch.tensor([pred_affinity], dtype=torch.float32),
        torch.tensor([pred_strain], dtype=torch.float32),
        torch.tensor([pred_clashes_probs], dtype=torch.float32)
    ])

    # Compute losses for each pair of true and predicted values
    losses = []
    for true, pred in zip(true_values, pred_values):
        loss = LANMSE(pred, true)
        losses.append(loss)

    # Log losses for debugging or monitoring
    #for i, loss in enumerate(losses, start=1):
    #    print(f'loss_{i}:', loss)

    total_loss = sum(losses) / len(losses)
    return total_loss



def loss_function(tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, backbone_weight=0, sidechain_weight=0, apply_mean=True, no_torsion=False, epoch = None, args=None):
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score.cpu()) ** 2 * tr_sigma.cpu() ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score.cpu()) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.cpu() / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score.cpu()) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score.cpu() ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if backbone_weight > 0:
        backbone_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data], dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        backbone_vecs = backbone_vecs[:, 4:]
        backbone_pred = sidechain_pred[:, 4:]

        backbone_base_loss = (backbone_vecs ** 2).detach().mean(dim=1) + 0.0001
        backbone_loss = ((backbone_pred.cpu() - backbone_vecs) ** 2).mean(dim=1) / backbone_base_loss.mean()
        backbone_base_loss = backbone_base_loss / backbone_base_loss.mean()
        if apply_mean:
            backbone_loss, backbone_base_loss = backbone_loss.mean() * torch.ones(1, dtype=torch.float), backbone_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)], dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(backbone_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, backbone_loss)
            s_b_l.index_add_(0, index, backbone_base_loss)
            backbone_loss, backbone_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            backbone_loss, backbone_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            backbone_loss, backbone_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if sidechain_weight > 0:
        sidechain_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data],
                                  dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        chi_angles = sidechain_vecs[:, :4].to(device)
        chi_pred = sidechain_pred[:, :4].to(device)

        chi_pred = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_pred)
        chi_angles = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_angles)

        difference = torch.abs(chi_pred - chi_angles)
        difference = torch.min(difference, 1 - difference) # angles are circular and 360 degrees = 1

        sidechain_base_loss = (chi_angles ** 2).detach().mean(dim=1) + 0.0001
        sidechain_loss = (difference ** 2).mean(dim=1) / sidechain_base_loss.mean()
        sidechain_base_loss = sidechain_base_loss / sidechain_base_loss.mean()
        if apply_mean:
            sidechain_loss, sidechain_base_loss = \
                sidechain_loss.mean().cpu() * torch.ones(1, dtype=torch.float), \
                sidechain_base_loss.mean().cpu() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)],
                              dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sidechain_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, sidechain_loss.cpu())
            s_b_l.index_add_(0, index, sidechain_base_loss.cpu())
            sidechain_loss, sidechain_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            sidechain_loss, sidechain_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sidechain_loss, sidechain_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(
                len(rot_loss), dtype=torch.float)

    
    if args.compass_weight > 0:

        compass_constant_total = []
        
        for i in range(len(data)):
            try:
                data = [d.to(device) for d in data] if device.type == 'cuda' else data
                if i % 2 == 0:
                    if data[i].tor_score.shape[0] == 0:
                        data_pr = modify_conformer(data[i], tr_update=tr_pred[i], rot_update=rot_pred[i], 
                                                torsion_updates=None, pivot=None)
                    else:
                        data_pr = modify_conformer(data[i], tr_update=tr_pred[i], rot_update=rot_pred[i], 
                                                torsion_updates=tor_pred[:data[i].tor_score.shape[0]], pivot=None)
                        data_pr = data_pr.to(device)
                else:
                    if data[i].tor_score.shape[0] == 0:
                        data_pr = modify_conformer(data[i], tr_update=tr_pred[i], rot_update=rot_pred[i], 
                                                torsion_updates=None, pivot=None)
                    else:
                        data_pr = modify_conformer(data[i], tr_update=tr_pred[i], rot_update=rot_pred[i], 
                                                torsion_updates=tor_pred[-data[i].tor_score.shape[0]:], pivot=None)
                        data_pr = data_pr.to(device)

                #ligand_pos = np.asarray(data_pr['ligand'].pos.detach().cpu().numpy() + data[i].original_center.detach().cpu().numpy())
                ligand_pos = np.asarray(data_pr['ligand'].pos.detach().cpu().numpy() + data[i].original_center.detach().cpu().numpy())
                current_directory = os.getcwd()
                ligand_path = f'{current_directory}/DiffDock/data/PDBBind_processed/{data_pr.name}/{data_pr.name}_ligand.sdf'
                protein_path = f'{current_directory}/DiffDock/data/PDBBind_processed/{data_pr.name}/{data_pr.name}_protein_processed.pdb'
                mol = read_molecule(ligand_path, remove_hs=True, sanitize=True)
                mol.RemoveAllConformers()
                #mol = AddHs(mol)
                mol = RemoveAllHs(mol)
                generate_conformer(mol)
                mol_pred = copy.deepcopy(mol)
                output_dir = os.path.join(current_directory, f'generated_data/{data_pr.name}')

                os.makedirs(output_dir, exist_ok=True)
                ligand_path = f'{output_dir}/{data_pr.name}_generated_epoch_{epoch}_lr_{args.lr}_wd_{args.w_decay}_compass_w_{args.compass_weight}.sdf'
                write_mol_with_coords(mol_pred, ligand_pos, ligand_path)
                pre_bind_aff, pre_clash, pre_strain_en = process_docked_file(output_dir, ligand_path, protein_path)
                true_bind_aff, true_clash, true_strain_en = get_compass_precalculated_data(data_pr.name)

                comp_const = compass_constant(true_bind_aff, pre_bind_aff, true_strain_en, pre_strain_en, true_clash, pre_clash)
                compass_constant_total.append(comp_const.unsqueeze(-1))

            except Exception as e:
                print(f'An error occurred: {e}')
                pass
        
        if not compass_constant_total:  # This checks if the list is empty
            compass_constant_total = [torch.tensor([1], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)]
        #else:
            # Assuming you might have a list of tensors and you want to concatenate them
        #    compass_loss_total = torch.cat(compass_loss_total)


        compass_constant_total = torch.cat(compass_constant_total)
        compass_constant_total = torch.mean(compass_constant_total)
        compass_constant_total = compass_constant_total.unsqueeze(0)

        print('tr_loss:', tr_loss)
        print('rot_loss:', rot_loss)
        print('tor_loss:', tor_loss)
        print('compass_constant_total:', compass_constant_total)


        loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sidechain_loss * sidechain_weight + backbone_loss * backbone_weight + compass_constant_total * args.compass_weight


    else:

        loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sidechain_loss * sidechain_weight + backbone_loss * backbone_weight
    
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), backbone_loss.detach(), sidechain_loss.detach(), \
           tr_base_loss, rot_base_loss, tor_base_loss, backbone_base_loss, sidechain_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum().cpu() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out



def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights, subsample_size=None, epoch = None, args = None):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'])
    
    if subsample_size:

        data_iter = tqdm(itertools.islice(loader, subsample_size), total=subsample_size)

    else:

        data_iter = tqdm(loader, total=len(loader))

    for data in data_iter:
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
            continue
        optimizer.zero_grad()
        data = [d.to(device) for d in data] if device.type == 'cuda' else data

        
        try:
            tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, device=device, epoch = epoch, args=args)
            if loss_tuple is None:
                print("None loss tuple, skipping")
                continue
            loss = loss_tuple[0]

            if torch.any(torch.isnan(loss)):
                names = data.name if device.type == 'cpu' else [d.name for d in data]
                print("Nan loss, skipping batch with complexes", names)
                continue
            loss.backward()
            optimizer.step()
            if ema_weights is not None: ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), *loss_tuple[1:]])
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                #raise e
                print(e)
                continue
            
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False, subsample_size=None, epoch=None, args=None):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
             'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
            unpooled_metrics=True, intervals=11)
        
    if subsample_size:

        data_iter = tqdm(itertools.islice(loader, subsample_size), total=subsample_size)

    else:

        data_iter = tqdm(loader, total=len(loader))

    for data in data_iter:
        try:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device, epoch = epoch, args=args)
            if loss_tuple is None: continue
            meter.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                print(e)
                continue

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args, subsample_size=None):
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds, min_rmsds = [], []

    if subsample_size:

        data_iter = tqdm(itertools.islice(loader, subsample_size), total=subsample_size)

    else:

        data_iter = tqdm(loader, total=len(loader))

    for orig_complex_graph in data_iter: #tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         t_schedule=t_schedule)
            except Exception as e:
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print("Exception while running inference on complex:", e)
        if failed_convergence_counter > 5:
            rmsds.extend([100] * args.inference_samples)
            min_rmsds.append(100)
            continue

        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # if len(orig_complex_graph['ligand'].orig_pos.shape) == 3:
        #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        if len(orig_complex_graph['ligand'].orig_pos.shape) == 2:
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[None, :, :]
        try:
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()
        except Exception as e:
            print("problem with orig_pos which is of shape:", orig_complex_graph['ligand'].orig_pos.shape, e)
            continue
        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))

    rmsds = np.array(rmsds)
    min_rmsds = np.array(min_rmsds)
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),}
    return losses
