#!/bin/bash
#$ -l gpu=1 -l cuda_memory=35G
#$ -pe smp 1
#$ -o /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_DOCKGEN/job_outputs/
#$ -e /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RESULTS/TREAMID_DOCKGEN/job_errors/
#$ -t 1-330 # Adjust this range to the number of lines in "run.lst" - check "wc -l run.lst"
#              # for testing, only try a small range like "-t 1-10" - when this worky nicely,i
#              # then for full computation select the remaining range, e.g. "-t 11-118735"
#$ -tc 20 # Have not more than 100 tasks running at the same time, to leave resources for other
#          # cluster users

# Workflow :
# First create the "run.lst" file, containing all *.sh files in your script dir.
#        (Instead of immediately qsub'ing every script.)
# Then qsub THIS script file ("array_launcher.sh") ONCE to work on the "run.lst" line-by-line

echo "Current directory: $(pwd)"
ls -l

# Picking one file from "run.lst", pased on the Task-Number:
SCRIPT=$(sed "${SGE_TASK_ID}q;d" < /fast/AG_Akalin/asarigun/Arcas_Stage_1/FACTORY/RUNS/TREAMID_DOCKGEN/run.lst)


echo "SCRIPT to be executed: '$SCRIPT'"
if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: Script file not found: $SCRIPT"
    exit 1
fi

echo "Working on $SCRIPT at $(date)"
    cd "$(dirname "$SCRIPT")"
    "${SCRIPT}"
echo "done at $(date)"
