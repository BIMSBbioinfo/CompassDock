import os
import pathlib
import argparse
from argparse import FileType, ArgumentParser

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from Bio import SeqIO

import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

# Combining parsers
def create_combined_parser():
    parser = ArgumentParser(description="Process protein data and extract representations using ESM model")
    
    # Arguments for first part
    parser.add_argument('--out_file', type=str, default="data/prepared_for_esm.fasta")
    parser.add_argument('--protein_ligand_csv', type=str, default='data/protein_ligand_example_csv.csv', help='Path to a .csv specifying the input as described in the main README')
    parser.add_argument('--protein_path', type=str, default=None, help='Path to a single PDB file. If this is not None then it will be used instead of the --protein_ligand_csv')

    # Arguments for second part
    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="Output directory for extracted representations",
    )
    parser.add_argument("--toks_per_batch", type=int, default=4096, help="Maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="Layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="Specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="Truncate sequences longer than the given value",
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
        nargs='?',
        default="data/prepared_for_esm.fasta"
    )

    # Arguments for loading and saving embeddings
    parser.add_argument('--output_path', type=str, default='pdbbind_sequences_new.pt', help='Output path for saving specific embeddings')

    return parser

# Function to process protein data and generate FASTA file
def process_protein_data(args):
    biopython_parser = PDBParser()

    three_to_one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
                    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                    'MET': 'M', 'MSE': 'M', 'PHE': 'F', 'PRO': 'P', 'PYL': 'O', 'SER': 'S',
                    'SEC': 'U', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'ASX': 'B',
                    'GLX': 'Z', 'XAA': 'X', 'XLE': 'J'}

    if args.protein_path is not None:
        file_paths = [args.protein_path]
    else:
        df = pd.read_csv(args.protein_ligand_csv)
        file_paths = list(set(df['protein_path'].tolist()))

    sequences = []
    ids = []
    for file_path in tqdm(file_paths):
        structure = biopython_parser.get_structure('random_id', file_path)
        structure = structure[0]
        for i, chain in enumerate(structure):
            seq = ''
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    continue
                residue_coords = []
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == 'CA':
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N':
                        n = list(atom.get_vector())
                    if atom.name == 'C':
                        c = list(atom.get_vector())
                if c_alpha is not None and n is not None and c is not None:  # only append residue if it is an amino acid
                    try:
                        seq += three_to_one[residue.get_resname()]
                    except Exception as e:
                        seq += '-'
                        print("Encountered unknown AA: ", residue.get_resname(), ' in the complex ', file_path, '. Replacing it with a dash - .')
            sequences.append(seq)
            ids.append(f'{os.path.basename(file_path)}_chain_{i}')
    records = []
    for (index, seq) in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ''
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")

# Function to extract representations using ESM model
def extract_representations(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    if args.out_file is not None:
        args.fasta_file = args.out_file

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(result, args.output_file)

# Function to load and save specific embeddings
def load_and_save_embeddings(args):
    dict = {}
    for filename in tqdm(os.listdir(args.output_dir)):
        dict[filename.split('.')[0]] = torch.load(os.path.join(args.output_dir, filename))['representations'][33]
    torch.save(dict, args.output_path)


