#!/bin/bash
#SBATCH --account=[CHANGME]                         # Account name for the job
#SBATCH --job-name="boltz_design_pipeline"          # Name of the job
#SBATCH --partition=[CHANGME TO GPU PARTITION]      # Partition to submit the job (using A100 GPUs)
#SBATCH --gres=gpu:1                                # Request 1 GPU resource
#SBATCH -N1                                         # Number of nodes
#SBATCH -n4                                         # Number of tasks
#SBATCH --mem=128GB                                 # Memory size
#SBATCH -t 48:00:00                                 # Maximum runtime of 48 hours (2 days)

# create 50 array tasks (0‑49); at most 30 may run at once
#SBATCH --array=0-49%10

# one log per array task: %A = array job‑ID, %a = task‑ID
#SBATCH --output="./log/BOLTZ_AF3.%A_%a.out"
#SBATCH --error="./log/BOLTZ_AF3.%A_%a.err"

date

module load cuda
module load miniconda3
module load alphafold/3.0.0
source activate boltz_design

#—‑ INPUTS —‑—————————————————————————
TARGET_NAME="$1"
RUN="$SLURM_ARRAY_TASK_ID"               # 0–49 from the array index
PDB_PATH="$2"
BOLTZDESIGN_SCRIPT="./boltzdesign.py"
#———————————————————————————————————————

#This will create 50 X 200 = 10,000 binders, adjust the array size or design_samples for more binders
#Note the queue time limit and number of jobs per queue

python "$BOLTZDESIGN_SCRIPT" \
    --target_name "$TARGET_NAME" \
    --pdb_path "$PDB_PATH" \
    --target_type protein \
    --pdb_target_ids A \
    --gpu_id 0 \
    --suffix "run${RUN}" \
    --run_alphafold False \
    --length_min 75 \
    --length_max 100 \
    --distogram_only False \
    --design_samples 200

date
