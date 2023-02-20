"""
GradPose: PDB Superimposition using Gradient Descent
Authors: Daniel Rademaker, Kevin van Geemen
"""
import warnings
import glob
import time
import os
import math
from itertools import repeat
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm, trange
#from gradpose import util
import util
from PDBdataset import *
from Rotator import *


# TODO: Consistent bar lengths (bar_format=)
# TODO: Argument to overwrite original files
# TODO: Check all of the docstrings, some arguments are be different!

def superpose(pdbs_list, template, output=None, residues=None, chain=None, cores=mp.cpu_count(),
    batch_size=50000, gpu=False, rmsd_path=None, verbosity=1):
    """Runs all the steps to superpose a list of PDBs.

    Args:
        pdbs_list (list): List of paths to PDB files.
        template (str): Template PDB path.
        output (str, optional): Folder to write the rotated PDBs. Defaults to None.
        residues (list, optional): List of residue indices to align to. Defaults to None.
        chain (str, optional): Chain to align to.. Defaults to None.
        cores (int, optional): The number of CPU cores to use for multiprocessing.
            Defaults to the amount of cores on the system.
        batch_size (int, optional): The batch size of the alignment. Defaults to 50000.
        gpu (bool, optional): Flag to indicate whether GPU should be utilized. Defaults to False.
        rmsd_path (str, optional): If set, will write RMSDs of all models
            compared to the template. Defaults to None.
        verbosity (int, optional): Verbosity level. 0 = Silent, 1 = Normal, 2 = Verbose.
            Defaults to 1.
    """
    if verbosity > 0:
        start_time = time.perf_counter()
        print("\n=== Superimpose ===")

    torch.set_num_threads(cores)

    # TODO: Do we keep this?
    # for REPRODUCIBILITY
    warnings.filterwarnings("ignore")

    # Make output folder
    os.makedirs(output, exist_ok=True)

    if gpu:
        device = torch.device("cuda")
        mp.set_start_method('spawn')
    else:
        device = torch.device("cpu")

    if rmsd_path and os.path.exists(rmsd_path):
        os.remove(rmsd_path)

    num_batches = math.ceil(len(pdbs_list)/batch_size)
    for i in range(num_batches):
        batch_start = i * batch_size
        if verbosity > 0:
            print(f"Processing batch {i+1}/{num_batches}")
        data_processor = PDBdataset(
            pdbs_list[batch_start:batch_start+batch_size],
            template,
            residues,
            chain,
            cores,
            output,
            device,
            verbosity
        )
        data_processor.optimize()
        data_processor.rotate_all_pool()
        if rmsd_path:
            data_processor.calc_rmsd_with_template(rmsd_path, first = i==0)

    if verbosity > 0:
        end_time = time.perf_counter()
        print(f"=== Superimposing completed in {end_time - start_time:.1f}s ===\n")


def main():
    """Runs the PDB superposition using command line arguments."""
    arguments = util.parse_args()

    pdbs_list = []
    if arguments.subfolders:
        for root, _, files in os.walk(arguments.input):
            for file in [file for file in files if file.endswith('.pdb')]:
                pdbs_list.append(str(os.path.join(root, file)))
    else:
        pdbs_list += glob.glob(str(os.path.join(arguments.input, '*.pdb')))

    if not pdbs_list:
        raise Exception("Specified folder does not contain .pdb files.")

    # Sort the files alphabetically
    pdbs_list.sort(key=str.lower)

    if not arguments.template:
        arguments.template = pdbs_list[0]
        if arguments.verbosity > 0:
            print(f"No template assigned. Using: {arguments.template}")

    if arguments.rmsd:
        arguments.rmsd = os.path.join(arguments.output, "rmsd.tsv")

    superpose(
        pdbs_list,
        arguments.template,
        arguments.output,
        arguments.residues,
        arguments.chain,
        arguments.n_cores,
        arguments.batch_size,
        arguments.gpu,
        arguments.rmsd,
        arguments.verbosity
    )



if __name__ == '__main__':
    main()
