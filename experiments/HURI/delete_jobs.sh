# list of all jobs with 'hqw' status
jobs_to_delete=$(qstat | awk '$5 == "hqw" {print $1}')

# Delete the jobs
for job in $jobs_to_delete; do
  qdel $job
done

