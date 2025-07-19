#!/bin/bash -l
#SBATCH -A [CHANGME]                            # Account name for the job
#SBATCH --partition=[CHANGME TO GPU PARTITION]  # Partition to submit the job (using A100 GPUs)
#SBATCH --job-name=boltz                        # Name of the job
#SBATCH --output=./log/BoltzL40S.%J.out         # Standard output file with job ID
#SBATCH --error=./log/BoltzL40S.%J.err          # Standard error file with job ID
#SBATCH -t 4:00:00                              # Maximum runtime of 4 hours
#SBATCH --mem=32GB                              # Allocate 32GB of memory
#SBATCH --ntasks-per-node=10                    # Number of tasks (using 10 CPU cores here)
#SBATCH --gres=gpu:1                            # Request 1 GPU resource

# Load necessary modules for the job
module load miniconda3                 # Load Miniconda module for environment management
module load cuda                       # Load CUDA for GPU support
module load python/3.12.5              # Load Python version 3.12.5

date                                   # Print the current date and time for logging

# Activate the ESMFold conda environment
source activate myboltz2

# Define input and output directories
INPUT_FILE=$1 # "./fasta_classical"          # Directory containing FASTA files
OUTPUT_DIR=$2 # "./pdb_classical_L100_timeout"           # Directory to store output results
FINAL_DIR=${OUTPUT_DIR}/PDB/ # "./pdb_classical_L100_timeout"           # Directory to store output results
CACHE_DIR="/home/carson.andorf/.boltz"

mkdir $OUTPUT_DIR
mkdir $FINAL_DIR

which boltz

~/.local/bin/boltz predict --use_msa_server "$INPUT_FILE" --out_dir "$OUTPUT_DIR" --cache "$CACHE_DIR" --output_format pdb --num_workers 4 --preprocessing-threads 4
# Deactivate the Conda environment after the script completes
conda deactivate

echo "$OUTPUT_DIR -type f -name "*.pdb" -exec cp {} ${FINAL_DIR}/ \;"
find $OUTPUT_DIR -type f -name "*.pdb" -exec cp {} ${FINAL_DIR}/ \;

date                                   # Print the date and time again to mark job end
