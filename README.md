# 🧬 Boltz-2 and BoltzDesign Tutorial (HPC + Slurm)

This tutorial provides step-by-step instructions for running [Boltz-2](https://github.com/jwohlwend/boltz) and [BoltzDesign](https://github.com/yehlincho/BoltzDesign1) on a High Performance Computing (HPC) environment using Slurm. These tools allow for GPU-accelerated prediction and design of protein structures.

---

## 📦 Environment Setup (Common for Both Tools)

Load required modules and create a clean conda environment.

```bash
# Load necessary modules
module load miniconda3
module load python/3.12.5
module load cuda

# Create and activate environment
conda create -n myboltz2 python=3.12 -y
source activate myboltz2
```

## ⚛️ Boltz-2: Protein Structure Prediction

### 1. Clone and Install Boltz

```bash
git clone https://github.com/jwohlwend/boltz.git
cd boltz

# Install with CUDA support
pip install -e .[cuda]

# (Optional) Ensure local bin is in your PATH
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

### 2. Submit a Slurm Job

```bash
# Example: run Boltz-2 on a single FASTA file
sbatch my_boltz2_single_protein_L40S.sh ./fasta/Zm00001eb361090_P002.fa ./maize/
```

### 3. Results
The final PDB will be in folder:

```bash
./maize/PDB/
```

---

## 🧬 BoltzDesign: Protein Design with Diffusion Models

### 1. Clone and Set Up BoltzDesign

```bash
git clone https://github.com/yehlincho/BoltzDesign1.git
cd BoltzDesign1

# Optional: create a directory for logs
mkdir log

# Load modules
module load miniconda3
module load cuda

# Run setup script
chmod +x setup.sh
./setup.sh
```

### 2. Activate the Conda Environment

```bash
conda activate boltz_design
```

### 3. Submit a Slurm Job

```bash
# Example: run BoltzDesign using an input PDB
sbatch my_boltz_design_array_L40S.sh Zm00001eb361090_P002 ./maize/PDB/Zm00001eb361090_P002_model_0.pdb
```

### 3. Results
The statistics of the binders will be found in the folder:

```bash
./outputs/protein_Zm00001eb361090_P002_run0/results_final/rmsd_results.csv
```

The structure of the binder will be in a folder like:

```bash
./outputs/protein_Zm00001eb361090_P002_run0/results_final/boltz_results_Zm00001eb361090_P002_results_itr1_length77/predictions/Zm00001eb361090_P002_results_itr1_length77/Zm00001eb361090_P002_results_itr1_length77_model_0.cif
```
To move all the CIF files to a single location use command:
```bash
mkdir ./final_cif/
find ./outputs/protein_Zm00001eb361090_P002_run0/ -type f -name "*_model_0.cif" -exec cp {} ./final_cif/ \; &
```

---


## 📚 References

- [Boltz GitHub](https://github.com/jwohlwend/boltz)
- [BoltzDesign GitHub](https://github.com/yehlincho/BoltzDesign1)

---
