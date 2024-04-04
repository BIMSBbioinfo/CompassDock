for job_id in $(seq 9647153 9647250); do
  scancel $job_id
done
