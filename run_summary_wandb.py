from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
import csv
import os
import glob
import traceback

# Base directory containing your WandB run subfolders
base_directory = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_AF/wandb"

# Prepare a dictionary to hold lists of values for each key of interest
summary_values = {
    "Protein ID": [],
    "Recursion Step Done":[],
    "Ligand Description": [],
    "Binding Affinity (kcal/mol)": [],
    "Number of clashes": [],
    "Strain Energy": [],
    "Confidence Score": [],
    "Rank of sdf": [],
    "_timestamp": [],
    "_runtime": [],
    "_step": [],
}

def process_wandb_file(data_path):
    # Initialize and open the datastore for scanning
    ds = datastore.DataStore()
    try:
        ds.open_for_scan(data_path)
    except Exception as e:
        print(f"Failed to open datastore for scanning {data_path}: {e}")
        return  # Skip this file and continue with the next one

    while True:
        try:
            data = ds.scan_record()
            if not data:  # Break the loop if no more data is available
                break
        except AssertionError as e:
            print(f"AssertionError processing record: {e} - Skipping corrupted record.")
            continue  # Skip this corrupted record and continue with the next
        except Exception as e:
            print(f"General error processing record: {e} - Skipping corrupted record.")
            continue  # Skip this corrupted record and continue with the next

        pb = wandb_internal_pb2.Record()
        try:
            pb.ParseFromString(data[1])  # Parse the binary data
        except Exception as e:
            print(f"Failed to parse record in {data_path}: {e}")
            continue  # Skip this record and continue with the next

        record_type = pb.WhichOneof("record_type")
        if record_type == "summary":
            for update in pb.summary.update:
                key = update.key
                value_json = update.value_json.strip('"')  # Removing quotes
                if key in summary_values:
                    summary_values[key].append(value_json)  # Append value to the corresponding list

    ds.close()

for run_folder in glob.glob(os.path.join(base_directory, "offline-*")):
    wandb_file = next(glob.iglob(os.path.join(run_folder, "*.wandb")), None)
    if wandb_file:
        try:
            process_wandb_file(wandb_file)
        except Exception as e:
            print(f"Failed to process {wandb_file}: {traceback.format_exc()}")

# Path to save the CSV file
csv_file_path = os.path.join(base_directory, "summary_new.csv")
max_length = max(len(values) for values in summary_values.values())

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(summary_values.keys())  # Writing the headers
    for i in range(max_length):
        row = [summary_values[key][i] if i < len(summary_values[key]) else '' for key in summary_values]
        writer.writerow(row)

print(f"Summary values saved to {csv_file_path}")
