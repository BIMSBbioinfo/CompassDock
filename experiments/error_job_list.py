import re

# Path to your text file with error logs
error_log_file_path = 'JobErrors.txt'
output_file_path = 'jobIDs.txt'  # Output file to write the job IDs

# Initialize an empty set to hold job numbers (sets automatically remove duplicates)
job_numbers = set()

# Open and read the error log file
with open(error_log_file_path, 'r') as file:
    for line in file:
        # Regular expression to match the pattern "array_launcher.sh.e7037591.xxx"
        match = re.search(r"array_launcher.sh.e7037591.(\d+)", line)
        if match:
            # Add the job number to the set
            job_numbers.add(int(match.group(1)))

# Write the unique job numbers to the output file
with open(output_file_path, 'w') as file:
    for job_id in sorted(job_numbers):  # Sort the job numbers before writing
        file.write(f"{job_id}\n")

print(f"Job IDs have been written to {output_file_path}")

