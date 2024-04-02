
#!/bin/bash
#jobIDs.txt contains all the job IDs you want to rerun, one per line
for id in $(cat jobIDs.txt); do
  qsub -t $id -hold_jid 7037591 HURI/array_launcher.sh
  # Optional: sleep a bit to avoid overloading the submission system
  sleep 1
done
