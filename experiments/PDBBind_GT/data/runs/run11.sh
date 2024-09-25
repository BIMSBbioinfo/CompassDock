
#!/bin/bash
#$ -l gpu=1 -l cuda_memory=40G
#$ -l m_mem_free=100G
#$ -cwd
#$ -V
#$ -N "compass_finetune_1"
#$ -l h_rt=120:00:00
#$ -pe smp 1
#$ -e /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS_DEV/workdir_new/job_errors/
#$ -o /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS_DEV/workdir_new/job_outputs/


# Change directory to the project root
cd /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS_DEV

# Activate the Python environment
source /home/asarigu/miniconda3/etc/profile.d/conda.sh
conda activate compass2


# Execute the Python script

python -W ignore -m finetune --config /fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS_DEV/workdir/v1.1/configs/model_parameters11.yml


