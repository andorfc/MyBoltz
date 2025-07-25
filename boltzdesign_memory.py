import os
import sys
import argparse
import yaml
import json
import shutil
import glob
import numpy as np
import logging
import subprocess
import pandas as pd
from pathlib import Path
import warnings
import gc
import torch

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure local boltzdesign package is on path
sys.path.append(os.path.join(os.getcwd(), 'boltzdesign'))

from boltzdesign_utils import *
from ligandmpnn_utils import *
from alphafold_utils import *
from input_utils import *
from utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YamlConfig:
    """Configuration class for managing input directories."""
    def __init__(self, main_dir: str = None):
        if main_dir is None:
            self.MAIN_DIR = Path.cwd() / 'inputs'
        else:
            self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / 'PDB'
        self.MSA_DIR = self.MAIN_DIR / 'MSA'
        self.YAML_DIR = self.MAIN_DIR / 'yaml'

    def setup_directories(self):
        """Create necessary input directories if they don't exist."""
        for directory in [self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

def clear_gpu_memory():
    """Free unused GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()

def print_memory_usage():
    """Print current GPU memory usage."""
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved : {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ('yes', 'true', 't', 'y', '1'):
        return True
    if val in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_gpu_environment(gpu_id):
    """Set CUDA_VISIBLE_DEVICES and ordering."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BoltzDesign: Protein Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Required
    parser.add_argument('--target_name', type=str, required=True,
                        help='PDB code or custom name')
    parser.add_argument('--target_type', type=str,
                        choices=['protein','rna','dna','small_molecule','metal'],
                        default='protein', help='Type of target')
    parser.add_argument('--input_type', type=str,
                        choices=['pdb','custom'], default='pdb',
                        help='Input mode')
    parser.add_argument('--pdb_path', type=str, default='',
                        help='Local PDB file path (if custom input)')
    parser.add_argument('--pdb_target_ids', type=str, default='',
                        help='Chains to target, comma-separated (e.g. A,B)')
    parser.add_argument('--custom_target_input', type=str, default='',
                        help='Custom sequence or ligand inputs, comma-separated')
    parser.add_argument('--custom_target_ids', type=str, default='',
                        help='Custom target IDs, comma-separated')
    parser.add_argument('--binder_id', type=str, default='A',
                        help='Chain ID to assign to designed binder')
    parser.add_argument('--use_msa', type=str2bool, default=False,
                        help='Whether to use an MSA')
    parser.add_argument('--msa_max_seqs', type=int, default=4096,
                        help='Max sequences for MSA generation')
    parser.add_argument('--suffix', type=str, default='0',
                        help='Suffix for output directory')
    # Modifications & constraints
    parser.add_argument('--modifications', type=str, default='',
                        help='Residue modifications, comma-separated')
    parser.add_argument('--modifications_wt', type=str, default='',
                        help='WT residues for modifications, comma-separated')
    parser.add_argument('--modifications_positions', type=str, default='',
                        help='Positions for modifications, comma-separated')
    parser.add_argument('--modification_target', type=str, default='',
                        help='Target chain for modifications')
    parser.add_argument('--constraint_target', type=str, default='',
                        help='Chain for contact constraints')
    parser.add_argument('--contact_residues', type=str, default='',
                        help='Residues for contacts, comma-separated')
    # Design parameters
    parser.add_argument('--length_min', type=int, default=75,
                        help='Minimum binder length')
    parser.add_argument('--length_max', type=int, default=150,
                        help='Maximum binder length')
    parser.add_argument('--design_samples', type=int, default=1,
                        help='Number of design samples')
    parser.add_argument('--optimizer_type', type=str, choices=['SGD','AdamW'],
                        default='SGD', help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--learning_rate_pre', type=float, default=0.1,
                        help='LR for pre-iterations')
    parser.add_argument('--e_soft', type=float, default=0.8,
                        help='Softmax temp for 3stages')
    parser.add_argument('--e_soft_1', type=float, default=0.8,
                        help='Temp stage 1 for extra')
    parser.add_argument('--e_soft_2', type=float, default=1.0,
                        help='Temp stage 2 for extra')
    parser.add_argument('--distogram_only', type=str2bool, default=True,
                        help='Optimize only distogram')
    parser.add_argument('--design_algorithm', type=str,
                        choices=['3stages','3stages_extra'], default='3stages',
                        help='Design algorithm')
    # Iterations
    parser.add_argument('--pre_iteration', type=int, default=30)
    parser.add_argument('--soft_iteration', type=int, default=75)
    parser.add_argument('--temp_iteration', type=int, default=50)
    parser.add_argument('--hard_iteration', type=int, default=5)
    parser.add_argument('--semi_greedy_steps', type=int, default=2)
    parser.add_argument('--recycling_steps', type=int, default=0)
    # Loss weights
    parser.add_argument('--con_loss', type=float, default=1.0)
    parser.add_argument('--i_con_loss', type=float, default=1.0)
    parser.add_argument('--plddt_loss', type=float, default=0.1)
    parser.add_argument('--pae_loss', type=float, default=0.4)
    parser.add_argument('--i_pae_loss', type=float, default=0.1)
    parser.add_argument('--rg_loss', type=float, default=0.0)
    parser.add_argument('--helix_loss_max', type=float, default=0.0)
    parser.add_argument('--helix_loss_min', type=float, default=-0.3)
    # LigandMPNN
    parser.add_argument('--num_designs', type=int, default=2,
                        help='Designs per PDB for LigandMPNN')
    parser.add_argument('--cutoff', type=int, default=4,
                        help='Interface cutoff (Å)')
    parser.add_argument('--i_ptm_cutoff', type=float, default=0.5,
                        help='iPTM cutoff')
    parser.add_argument('--complex_plddt_cutoff', type=float, default=0.7,
                        help='pLDDT cutoff')
    # System & control flags
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--high_iptm', type=str2bool, default=True)
    parser.add_argument('--run_boltz_design', type=str2bool, default=True)
    parser.add_argument('--run_ligandmpnn', type=str2bool, default=True)
    parser.add_argument('--run_alphafold', type=str2bool, default=True)
    parser.add_argument('--run_rosetta', type=str2bool, default=True)
    parser.add_argument('--redo_boltz_predict', type=str2bool, default=False)
    parser.add_argument('--show_animation', type=str2bool, default=True)
    parser.add_argument('--save_trajectory', type=str2bool, default=False)
    # Paths
    parser.add_argument('--boltz_checkpoint', type=str,
                        default=os.path.expanduser('~/.boltz/boltz1_conf.ckpt'))
    parser.add_argument('--ccd_path', type=str,
                        default=os.path.expanduser('~/.boltz/ccd.pkl'))
    parser.add_argument('--alphafold_dir', type=str,
                        default=os.path.expanduser('~/alphafold3'))
    parser.add_argument('--af3_docker_name', type=str, default='alphafold3')
    parser.add_argument('--af3_database_settings', type=str,
                        default=os.path.expanduser('~/alphafold3/alphafold3_data_save'))
    parser.add_argument('--af3_hmmer_path', type=str,
                        default=os.path.expanduser('~/alphafold3/env'))
    return parser.parse_args()

def get_explicit_args():
    """Return the set of CLI argument names explicitly provided."""
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            if '=' in arg:
                explicit_args.add(arg.split('=')[0].lstrip('-').replace('-', '_'))
            else:
                explicit_args.add(arg.lstrip('-').replace('-', '_'))
    return explicit_args

def update_config_with_args(config, args):
    """Update configuration dict with only those args explicitly set."""
    basic_params = {
        'binder_chain': args.binder_id,
        'non_protein_target': args.target_type != 'protein',
        'pocket_conditioning': bool(args.contact_residues),
    }
    config.update(basic_params)
    explicit = get_explicit_args()

    advanced = {
        'mask_ligand': args.distogram_only,  # example; adjust as needed
        'optimize_contact_per_binder_pos': args.distogram_only,
        'distogram_only': args.distogram_only,
        'design_algorithm': args.design_algorithm,
        'learning_rate': args.learning_rate,
        'learning_rate_pre': args.learning_rate_pre,
        'e_soft': args.e_soft,
        'e_soft_1': args.e_soft_1,
        'e_soft_2': args.e_soft_2,
        'length_min': args.length_min,
        'length_max': args.length_max,
        'inter_chain_cutoff': args.cutoff,
        'intra_chain_cutoff': args.cutoff,
        'num_inter_contacts': 1,
        'num_intra_contacts': 2,
        'helix_loss_max': args.helix_loss_max,
        'helix_loss_min': args.helix_loss_min,
        'optimizer_type': args.optimizer_type,
        'pre_iteration': args.pre_iteration,
        'soft_iteration': args.soft_iteration,
        'temp_iteration': args.temp_iteration,
        'hard_iteration': args.hard_iteration,
        'semi_greedy_steps': args.semi_greedy_steps,
        'msa_max_seqs': args.msa_max_seqs,
        'recycling_steps': args.recycling_steps,
    }
    for name, val in advanced.items():
        if name in explicit:
            config[name] = val
    return config

def load_design_config(target_type, work_dir):
    """Load default YAML config based on target type."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, 'boltzdesign', 'configs')
    if target_type == 'small_molecule':
        cfg = 'default_sm_config.yaml'
    elif target_type == 'metal':
        cfg = 'default_metal_config.yaml'
    elif target_type in ('dna', 'rna'):
        cfg = 'default_na_config.yaml'
    elif target_type == 'protein':
        cfg = 'default_ppi_config.yaml'
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    path = os.path.join(config_dir, cfg)
    print(f"Loading config from: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

def setup_environment():
    """Parse args, cd to work_dir, set GPU, and print device."""
    args = parse_arguments()
    work_dir = args.work_dir or os.getcwd()
    os.chdir(work_dir)
    setup_gpu_environment(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args

def load_boltz_model(args, device):
    """Load Boltz model and switch to train mode."""
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    model = get_boltz_model(args.boltz_checkpoint, predict_args, device)
    model.train()
    return model, predict_args

def initialize_pipeline(args):
    """Initialize Boltz model and input directories."""
    model, _ = load_boltz_model(args, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    cfg_obj = YamlConfig(main_dir=f"{args.work_dir or os.getcwd()}/inputs/{args.target_type}_{args.target_name}_{args.suffix}")
    cfg_obj.setup_directories()
    return model, cfg_obj

def get_target_ids(args):
    """Parse chain IDs from args or raise if needed."""
    ids = args.pdb_target_ids if args.input_type == 'pdb' else args.custom_target_ids
    if (args.contact_residues or args.modifications) and not ids:
        raise ValueError("Target IDs must be specified when using contacts or modifications")
    return [x.strip() for x in ids.split(',')] if ids else []

def assign_chain_ids(target_ids_list, binder_chain='A'):
    """Map target IDs to unique chain IDs, skipping binder_chain."""
    letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c != binder_chain]
    return {tid: letters[i] for i, tid in enumerate(target_ids_list)}

def modification_to_wt_aa(modifications, modifications_wt):
    """Map modified residues back to wild-type."""
    if not modifications:
        return None
    return dict(zip(modifications.split(','), modifications_wt.split(',')))

def generate_yaml_config(args, config_obj):
    """Generate and write YAML config for BoltzDesign."""
    if args.contact_residues or args.modifications:
        tids = get_target_ids(args)
        tmap = assign_chain_ids(tids, args.binder_id)
        constraints, mods = process_design_constraints(
            tmap,
            args.modifications, args.modifications_positions,
            args.modification_target,
            args.contact_residues,
            args.constraint_target,
            args.binder_id
        )
    else:
        constraints, mods = None, None

    target = []
    if args.input_type == 'pdb':
        if args.pdb_path:
            pdb_file = Path(args.pdb_path)
            if not pdb_file.is_file():
                raise FileNotFoundError(f"Local PDB not found: {pdb_file}")
        else:
            download_pdb(args.target_name, config_obj.PDB_DIR)
            pdb_file = config_obj.PDB_DIR / f"{args.target_name}.pdb"

        if args.target_type in ('dna','rna'):
            nuc = get_nucleotide_from_pdb(pdb_file)
            for cid in args.pdb_target_ids.split(','):
                target.append(nuc[cid]['seq'])
        elif args.target_type == 'small_molecule':
            lig = get_ligand_from_pdb(args.target_name)
            for m in args.custom_target_input.split(','):
                target.append(lig[m])
        else:
            seqs = get_chains_sequence(pdb_file)
            for cid in args.pdb_target_ids.split(','):
                target.append(seqs[cid])
    else:
        target = args.custom_target_input.split(',') if args.custom_target_input else [args.target_name]

    return generate_yaml_for_target_binder(
        args.target_name,
        args.target_type,
        target,
        config=config_obj,
        binder_id=args.binder_id,
        constraints=constraints,
        modifications=(mods['data'] if mods else None),
        modification_target=(mods['target'] if mods else None),
        use_msa=args.use_msa
    )

def setup_pipeline_config(args):
    """Load default config and override with explicit CLI args."""
    base = load_design_config(args.target_type, args.work_dir or os.getcwd())
    return update_config_with_args(base, args)

def setup_output_directories(args):
    """Create outputs folder and return paths."""
    wd = args.work_dir or os.getcwd()
    main = Path(wd) / 'outputs'
    main.mkdir(parents=True, exist_ok=True)
    version = f"{args.target_type}_{args.target_name}_{args.suffix}"
    return {'main_dir': str(main), 'version': version}

def run_boltz_design_step(args, config, boltz_model, yaml_dir, main_dir, version):
    """Execute the BoltzDesign step."""
    print("Starting BoltzDesign step...")
    loss_scales = {
        'con_loss': args.con_loss,
        'i_con_loss': args.i_con_loss,
        'plddt_loss': args.plddt_loss,
        'pae_loss': args.pae_loss,
        'i_pae_loss': args.i_pae_loss,
        'rg_loss': args.rg_loss,
    }
    boltz_exec = shutil.which('boltz')
    if boltz_exec is None:
        raise FileNotFoundError("boltz executable not found in PATH")
    run_boltz_design(
        boltz_path=boltz_exec,
        main_dir=main_dir,
        yaml_dir=os.path.dirname(yaml_dir),
        boltz_model=boltz_model,
        ccd_path=args.ccd_path,
        design_samples=args.design_samples,
        version_name=version,
        config=config,
        loss_scales=loss_scales,
        show_animation=args.show_animation,
        save_trajectory=args.save_trajectory,
        redo_boltz_predict=args.redo_boltz_predict
    )
    print("BoltzDesign step completed")

def run_ligandmpnn_step(args, main_dir, version, ligandmpnn_dir, yaml_dir, work_dir):
    """Execute the LigandMPNN redesign step."""
    print("Starting LigandMPNN redesign step...")
    yaml_path = f"{work_dir}/LigandMPNN/run_ligandmpnn_logits_config.yaml"
    with open(yaml_path, "r") as f:
        mpnn_cfg = yaml.safe_load(f)
    for k, v in mpnn_cfg.items():
        if isinstance(v, str) and "${CWD}" in v:
            mpnn_cfg[k] = v.replace("${CWD}", work_dir)
    with open(yaml_path, "w") as f:
        yaml.dump(mpnn_cfg, f, default_flow_style=False)

    boltz_dir = f"{main_dir}/{version}/results_final"
    pdb_dir = f"{main_dir}/{version}/pdb"
    lmpnn_redesigned_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned')
    lmpnn_redesigned_fa_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_fa')
    lmpnn_redesigned_yaml_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_yaml')

    os.makedirs(ligandmpnn_dir, exist_ok=True)
    convert_cif_files_to_pdb(boltz_dir, pdb_dir, high_iptm=args.high_iptm, i_ptm_cutoff=args.i_ptm_cutoff)

    if not any(f.endswith('.pdb') for f in os.listdir(pdb_dir)):
        print("No successful designs from BoltzDesign")
        sys.exit(1)

    run_ligandmpnn_redesign(
        ligandmpnn_dir,
        pdb_dir,
        shutil.which("boltz"),
        os.path.dirname(yaml_dir),
        yaml_path,
        top_k=args.num_designs,
        cutoff=args.cutoff,
        non_protein_target=args.target_type != 'protein',
        binder_chain=args.binder_id,
        target_chains="all",
        out_dir=lmpnn_redesigned_fa_dir,
        lmpnn_yaml_dir=lmpnn_redesigned_yaml_dir,
        results_final_dir=lmpnn_redesigned_dir
    )

    filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir)
    print("LigandMPNN redesign step completed")
    return ligandmpnn_dir

def filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir):
    """Filter and save high confidence LigandMPNN designs."""
    print("Filtering high confidence designs...")
    success_base = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_high_iptm')
    yaml_out = os.path.join(success_base, 'yaml')
    cif_out = os.path.join(success_base, 'cif')
    os.makedirs(yaml_out, exist_ok=True)
    os.makedirs(cif_out, exist_ok=True)

    successful = 0
    for root in os.listdir(lmpnn_redesigned_dir):
        pred_dir = os.path.join(lmpnn_redesigned_dir, root, 'predictions')
        if not os.path.isdir(pred_dir):
            continue
        for sub in os.listdir(pred_dir):
            json_path = os.path.join(pred_dir, sub, f'confidence_{sub}_model_0.json')
            yaml_path = os.path.join(lmpnn_redesigned_yaml_dir, f'{sub}.yaml')
            cif_path = os.path.join(lmpnn_redesigned_dir, f'boltz_results_{sub}', 'predictions', sub, f'{sub}_model_0.cif')
            try:
                with open(json_path) as jf:
                    data = json.load(jf)
                iptm = data.get('iptm', 0)
                plddt = data.get('complex_plddt', 0)
                print(f"{sub}: iptm={iptm:.2f}, pLDDT={plddt:.2f}")
                if iptm > args.i_ptm_cutoff and plddt > args.complex_plddt_cutoff:
                    shutil.copy(yaml_path, os.path.join(yaml_out, f'{sub}.yaml'))
                    shutil.copy(cif_path, os.path.join(cif_out, f'{sub}.cif'))
                    print(f"✅ {sub} copied")
                    successful += 1
            except Exception as e:
                print(f"Skipping {sub}: {e}")
    if successful == 0:
        print("Error: No LigandMPNN designs passed thresholds")
        sys.exit(1)

def calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_chain):
    """Compute RMSD between holo and apo structures and update CSV."""
    csv_path = os.path.join(af_pdb_dir, 'high_iptm_confidence_scores.csv')
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    for fname in os.listdir(af_pdb_dir):
        if not fname.endswith('.pdb'):
            continue
        holo = os.path.join(af_pdb_dir, fname)
        apo = os.path.join(af_pdb_dir_apo, fname)
        xyz_h, _ = get_CA_and_sequence(holo, chain_id=binder_chain)
        xyz_a, _ = get_CA_and_sequence(apo, chain_id='A')
        rmsd = np_rmsd(np.array(xyz_h), np.array(xyz_a))
        df.loc[df['file'] == fname.replace('.pdb', '.cif'), 'rmsd'] = rmsd
        print(f"{fname} RMSD: {rmsd:.3f}")
    df.to_csv(csv_path, index=False)

def run_alphafold_step(args, ligandmpnn_dir, work_dir, mod_to_wt_aa):
    """Execute the AlphaFold validation step."""
    print("Starting AlphaFold validation step...")
    af_dir = os.path.expanduser(args.alphafold_dir)
    db_dir = os.path.expanduser(args.af3_database_settings)
    hmmer = os.path.expanduser(args.af3_hmmer_path)
    print(f"AF dir: {af_dir}, DB: {db_dir}, HMMER: {hmmer}")

    in_holo = f"{ligandmpnn_dir}/02_design_json_af3"
    out_holo = f"{ligandmpnn_dir}/02_design_final_af3"
    in_apo = f"{ligandmpnn_dir}/02_design_json_af3_apo"
    out_apo = f"{ligandmpnn_dir}/02_design_final_af3_apo"
    for d in (in_holo, out_holo, in_apo, out_apo):
        os.makedirs(d, exist_ok=True)

    yaml_src = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_high_iptm', 'yaml')
    process_yaml_files(
        yaml_src,
        in_holo,
        in_apo,
        target_type=args.target_type,
        binder_chain=args.binder_id,
        mod_to_wt_aa=mod_to_wt_aa,
        afdb_dir=db_dir,
        hmmer_path=hmmer
    )

    for inp, out in ((in_holo, out_holo), (in_apo, out_apo)):
        subprocess.run(
            [f"{work_dir}/boltzdesign/alphafold.sh", inp, out, str(args.gpu_id), af_dir, args.af3_docker_name],
            check=True
        )

    print("AlphaFold validation step completed!")
    pdb_out = f"{ligandmpnn_dir}/03_af_pdb_success"
    pdb_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"
    convert_cif_files_to_pdb(out_holo, pdb_out, af_dir=True, high_iptm=args.high_iptm)
    if not any(f.endswith('.pdb') for f in os.listdir(pdb_out)):
        print("No successful AF designs")
        sys.exit(1)
    convert_cif_files_to_pdb(out_apo, pdb_apo, af_dir=True)
    calculate_holo_apo_rmsd(pdb_out, pdb_apo, args.binder_id)
    return out_holo, out_apo, pdb_out, pdb_apo

def run_rosetta_step(args, ligandmpnn_dir, af_out, af_apo, pdb_out, pdb_apo):
    """Execute the Rosetta energy calculation step."""
    if args.target_type != 'protein':
        print("Skipping Rosetta (non-protein target)")
        return
    print("Starting Rosetta energy calculation...")
    from pyrosetta_utils import measure_rosetta_energy
    success_dir = os.path.join(ligandmpnn_dir, 'af_pdb_rosetta_success')
    measure_rosetta_energy(
        pdb_out, pdb_apo, success_dir,
        binder_holo_chain=args.binder_id, binder_apo_chain='A'
    )
    print("Rosetta energy calculation completed!")

def run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir):
    """Run each pipeline step according to flags."""
    results = {
        'ligandmpnn_dir': f"{output_dir['main_dir']}/{output_dir['version']}/ligandmpnn_cutoff_{args.cutoff}",
        'af_output_dir': None,
        'af_output_apo_dir': None,
        'af_pdb_dir': None,
        'af_pdb_dir_apo': None
    }

    if args.run_boltz_design:
        run_boltz_design_step(
            args, config, boltz_model, yaml_dir,
            output_dir['main_dir'], output_dir['version']
        )
        clear_gpu_memory()

    if args.run_ligandmpnn:
        run_ligandmpnn_step(
            args,
            output_dir['main_dir'],
            output_dir['version'],
            results['ligandmpnn_dir'],
            yaml_dir,
            args.work_dir or os.getcwd()
        )

    if args.run_alphafold:
        mod_map = modification_to_wt_aa(args.modifications, args.modifications_wt)
        af_out, af_apo, pdb_out, pdb_apo = run_alphafold_step(
            args, results['ligandmpnn_dir'],
            args.work_dir or os.getcwd(),
            mod_map
        )
        results.update({
            'af_output_dir': af_out,
            'af_output_apo_dir': af_apo,
            'af_pdb_dir': pdb_out,
            'af_pdb_dir_apo': pdb_apo
        })

    if args.run_rosetta:
        run_rosetta_step(
            args,
            results['ligandmpnn_dir'],
            results['af_output_dir'],
            results['af_output_apo_dir'],
            results['af_pdb_dir'],
            results['af_pdb_dir_apo']
        )

    print_memory_usage()
    return results

def main():
    """Main entrypoint for the BoltzDesign pipeline."""
    args = setup_environment()
    boltz_model, cfg_obj = initialize_pipeline(args)
    yaml_dict, yaml_dir = generate_yaml_config(args, cfg_obj)

    print("Generated YAML configuration:")
    for k, v in yaml_dict.items():
        print(f"  {k}: {v}")

    config = setup_pipeline_config(args)
    output_dir = setup_output_directories(args)

    results = run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir)
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
