import csv

def read_gene_info(txt_file):
    """ Read gene information from the TXT file and store gene names. """
    gene_info = {}
    with open(txt_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_info[row['gene_name']] = row['gene_id']
    return gene_info

def read_tsv_for_af_id(tsv_file):
    """ Read TSV file and create a mapping of gene_name to af_id. """
    gene_to_af_id = {}
    with open(tsv_file, mode='r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            gene_to_af_id[row['gene_name']] = row['af_id']
    return gene_to_af_id

def match_gene_to_af_id(gene_info, gene_to_af_id):
    """ Match gene names to af_ids using the mappings. """
    results = {}
    for gene_name, gene_id in gene_info.items():
        if gene_name in gene_to_af_id:
            results[gene_name] = gene_to_af_id[gene_name]
        else:
            results[gene_name] = None
    return results

# File paths (adjust these paths to your files)
txt_file_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/ECBL/data/targets_FMP_reduced.txt'
tsv_file_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/ECBL/data/table-af_id-gene_name.tsv'

# Read files and prepare data
gene_info = read_gene_info(txt_file_path)
gene_to_af_id = read_tsv_for_af_id(tsv_file_path)

# Get matches
matched_results = match_gene_to_af_id(gene_info, gene_to_af_id)

# Print results
for gene_name, af_id in matched_results.items():
    print(f"{gene_name}: {af_id}")



def read_and_extract_af_ids(file_path):
    """ Read the TXT file and extract only the AF IDs. """
    af_ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                af_ids.append(parts[1])
    return af_ids

# File path (adjust this path to your file)
txt_file_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/ECBL/af_files.txt'

# Extract AF IDs
af_ids = read_and_extract_af_ids(txt_file_path)

# Print extracted AF IDs
for af_id in af_ids:
    print(af_id)


## ZAR1: AF-A6NP61-F1-model_v3.pd