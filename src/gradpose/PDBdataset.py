import warnings
import glob
import os
import math
from itertools import repeat
import torch
from tqdm import tqdm, trange
import torch.multiprocessing as mp
import torch
import time
from Rotator import *
class PDBdataset():
    """Class that processes PDBs.
    """
    def __init__(self, pdbs, template_pdb, residues, chain,
            cores=mp.cpu_count(), output='result', device='cpu', verbosity=1):
        """_summary_

        Args:
            pdbs (str): Folder where the PDB files are located.
            template_pdb (str): Path to the template PDB.
            residues (list[int]): List of residue IDs to be used for alignment.
            chain (str): Chain ID to be used for alignment.
            cores (int, optional): Amount of CPU cores to use for multiprocessing.
                Defaults to the amount of cores on the system.
            output (str, optional): Path to output folder. Defaults to 'result'.
            device (str, optional): Device to store the tensors on. Defaults to 'cpu'.
            verbosity (int, optional): Verbosity level. Defaults to 1.
        """
        self.device = device
        self.verbosity = verbosity
        self.chain = chain

        if not residues or not chain:
            with open(template_pdb, 'r', encoding='utf-8') as template_file:
                template_file_data = template_file.read().split('\n')

                if not chain:
                    if self.verbosity > 0:
                        print("No chain selected, using longest chain: ", end='')

                    chain_dict = {}
                    for line in template_file_data:
                        if not line.startswith('ATOM  '):
                            continue
                        chain_id = line[21]
                        if not chain_id in chain_dict:
                            chain_dict[chain_id] = 0
                        chain_dict[chain_id] += 1
                    self.chain = sorted([
                        (count, chain_id)
                        for chain_id, count in chain_dict.items()
                    ], reverse=True)[0][1]

                    if self.verbosity > 0:
                        print(f"{self.chain}")

                if not residues:
                    if self.verbosity > 0:
                        print("No residues selected, determining from template chain: ", end='')
                    residues = list(set([
                        int(line[22:26])
                        for line in template_file_data
                        if line.startswith('ATOM ') and line[21] == self.chain
                    ]))

                    if self.verbosity > 0:
                        print(f"{residues[0]}:{residues[-1]}")


        self.residues = [int(i) for i in residues]
        self.residues_dict = {i:self.residues.index(i) for i in self.residues}
        self.pdbs = [template_pdb] + pdbs
        self.cores = cores
        self.output = output

        # Load CA coordinates for each PDB
        if self.verbosity > 1:
            print(f"Number of 3D Models: {len(self.pdbs)}")
            print('Start loading data')
        t0 = time.perf_counter()
        xyz_ca, presence_mask = self.load_ca_atom_positions()
        if self.verbosity > 1:
            print(f'Retrieved data, time: {time.perf_counter()-t0:.2f} seconds')

        # Instantiate a Rotator
        self.rotator = Rotator(xyz_ca, presence_mask, device=self.device)

    def _extract_pdb_atoms(self, file_name):
        """Extract all atom xyz coordinates

        Args:
            file_name (str): Path to PDB file.

        Returns:
            list: XYZ coordinates for each atom.
        """
        with open(file_name, encoding='utf-8') as pdb_file:
            return [[float(line[30:38]),float(line[38:46]),float(line[46:54])] \
                    for line in pdb_file.read().split('\n') \
                            if line.startswith('ATOM  ') or line.startswith('HETATM')]

    def _extract_pdb_ca(self, file_name):
        """Extract coordianates of all CA in specified chains and residues.

        Args:
            file_name (str): Path to PDB file.

        Returns:
            numpy.ndarray: Array that contains the XYZ coordinates of the CA atom per residue.
        """
        with open(file_name, encoding='utf-8') as pdb_file:
            pxyz = np.array(
                [
                    [
                        self.residues_dict[int(line[22:26])],
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54])
                    ]
                    for line in pdb_file.read().split('\n') \
                    if line.startswith('ATOM ') and line[13:15] == 'CA' \
                        and line[21] == self.chain \
                        and int(line[22:26]) in self.residues
                ]
            )
        xyz = np.zeros((1, len(self.residues), 3))
        xyz[0][pxyz[:, 0].astype(int)] = pxyz[:, 1:]
        return xyz

    def load_ca_atom_positions(self):
        """Parse the PDBs to obtain the data needed for calculating the rotation matrices.

        Returns:
            torch.Tensor: CA atom coordinates for each PDB.
            torch.Tensor: The loss mask to exclude missing residues.
        """
        with mp.Pool(self.cores) as pool:
            ca_xyz = torch.tensor(
                np.concatenate(list(tqdm(
                    pool.imap(
                        self._extract_pdb_ca,
                        self.pdbs,
                        max(2, round(len(self.pdbs) / (self.cores * 16)))
                    ),
                    total=len(self.pdbs), desc="Extracting backbones",
                    disable=self.verbosity!=1
                )))).float().to(self.device)
        presence_mask = (ca_xyz.abs().sum(2) != 0).unsqueeze(2).float().to(self.device)
        return ca_xyz, presence_mask

    def _rotate_single_pdb(self, pdb_index, rotation_matrix):
        """Rotate a PDB by the quaternions of the Rotator and save the resulting PDB to a file.

        Args:
            pdb_index (int): The index of the PDB to be rotated.
            rotation_matrix (torch.Tensor): The rotation matrix to use for the rotator.
        """
        # TODO: Update docstring
        with torch.no_grad():
            pdb_file = self.pdbs[pdb_index]
            xyz = torch.Tensor(self._extract_pdb_atoms(pdb_file)).reshape(1, -1, 3)
            xyz = xyz - self.rotator.center[pdb_index]
            rotated_xyz = self.rotator.get_rotated_xyz(pdb_index, xyz, rotation_matrix).squeeze() \
                + self.rotator.center[0]

            with open(pdb_file, encoding='utf-8') as in_file:
                with open(os.path.join(self.output, os.path.basename(pdb_file)), 'w',
                        encoding='utf-8') as out_file:
                    if self.verbosity > 1:
                        print(f"Saving {out_file.name}")
                    coord_strings = (
                        '{:>8.3f}{:>8.3f}{:>8.3f}'.format(*atom.tolist()) for atom in rotated_xyz
                    )
                    pdb = [
                        line if not line.startswith('ATOM') or line.startswith('HETATM ')
                        else f'{line[:30]}{next(coord_strings)}{line[54:]}'
                        for line in in_file.readlines()
                    ]
                    out_file.write(''.join(pdb))


    def _rotate_single_pdb_pool(self, variables):
        """Intermediate function that unpacks the single tuple variable from imap
        as arguments to the true rotate function.

        Args:
            variables (Iterable): Iterable to be unpacked.
        """
        self._rotate_single_pdb(*variables)

    def rotate_all_pool(self):
        """Pool the rotating of PDBs."""
        with torch.no_grad():
            t0 = time.perf_counter()
            self.rotator.use_numpy = True  # Enable numpy for Rotator
            self.rotator.quaternions = self.rotator.quaternions.cpu()
            self.rotator.center = self.rotator.center.cpu()
            rotation_matrix = self.rotator.get_matrix()

            with mp.Pool(self.cores) as pool:
                list(tqdm(pool.imap(
                                self._rotate_single_pdb_pool,
                                zip(range(1, len(self.pdbs)), repeat(rotation_matrix)),
                                chunksize=max(2, round(len(self.pdbs) / (self.cores * 16)))
                                    ),
                        total=len(self.pdbs) - 1, desc="Rotating PDBs",
                        disable=self.verbosity == 0
                            ))
        if self.verbosity > 1:
            print(f'Rotating and saving data took: {time.perf_counter() - t0:.2f} seconds')

    def calc_rmsd_with_template(self, output_file, first=False):
        """Calculates the RMSD of each aligned PDB with the template.
        Only takes into account the selected residues and ignores any deletions.

        Args:
            output_file (str): Path to write to.
        """
        with torch.no_grad():
            if self.verbosity > 0:
                print("Saving RMSDs...")
            rotated_xyz = self.rotator.get_rotated_xyz()
            pdb_range = torch.arange(start=1, end=len(self.pdbs))
            combined_masks = self.rotator.presence_mask[0] * self.rotator.presence_mask[pdb_range]
            rmsds = ((rotated_xyz[0] - rotated_xyz[pdb_range]) * combined_masks)\
                .pow(2).reshape(len(self.pdbs)-1, -1).sum(1)\
                .div(combined_masks.sum(1).squeeze()).sqrt()

            with open(output_file, 'a', encoding='utf-8') as rmsd_file:
                if first:
                    rmsd_file.write(f"Template\t{self.pdbs[0]}\n")
                rmsd_file.write("\n".join([
                    f"{os.path.basename(pdb)}\t{rmsds[i].item()}"
                    for i, pdb in enumerate(self.pdbs[1:])
                ]))
                rmsd_file.write("\n")

    def optimize(self):
        """Runs the optimization loop.
        Attempts to find the optimal rotation to superpose all PDBs to the template."""
        if self.verbosity > 1:
            print("Aligning backbones")
        t0 = time.perf_counter()
        label = self.rotator.xyz[0:1]
        shake_steps = max(75, -4*label.shape[1]+400)
        finetune_steps = 250
        finetune_steps = 1000 #xue
        step = 0
        for step in trange(0, shake_steps+finetune_steps, desc="Aligning backbones",
                disable=self.verbosity != 1):
            # Take a step
            loss = self.rotator.step() #update self.rotator.quaternion
            if step <= shake_steps:
                self.rotator.normalize_q()
            if step == shake_steps:
                self.rotator.optimizer = torch.optim.SGD([self.rotator.quaternions], lr=0.00001)
            if self.verbosity > 1:
                print(f'Step: {step+1} \t Loss: {loss}')
        if self.verbosity > 1:
            print(f'Steps: {step+1}, alignment-time: {time.perf_counter()-t0:.2f}')

