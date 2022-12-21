"""
PDB Superimposition
"""
import warnings
import glob
import time
import os
import math
import sys
from itertools import repeat
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm, trange
import pdb2sql
from superpose import util

# TODO: Don't forget to change chain from 72 to 21
# TODO: Update pyproject.toml dependencies and everything else
# TODO: Consistent bar lengths (bar_format=)
# TODO: Argument to overwrite original files
# TODO: Check all of the docstrings, some arguments are be different!
# TODO: RMSD of each model vs template, write to file. Enable as a parameter.

class Rotator():
    """The Rotation module with all learnable parameters.
    """
    def __init__(self, xyz, del_mask, center=None, quaternions=None, device='cpu', use_numpy=False):
        """Initializes the Rotator object.

        Args:
            xyz (torch.Tensor): Coordinates for each atom for each PDB.
            del_mask (torch.Tensor): A mask to zero out missing atoms when optimizing.
            center (torch.Tensor, optional): Alternative center coordinates to use for each PDB.
                Shape=(No. of PDBs, 3) Defaults to None.
            quaternions (torch.Tensor, optional): Alternative quaternions to set for each PDB.
                Shape=(No. of PDBs, 4) Defaults to None.
        """
        self.device = device
        # Rotation is always over the center of the pdb structures
        self.center = self.get_center(xyz, del_mask) \
            if center is None else center.to(self.device)
        self.xyz = xyz - self.center
        self.del_mask = del_mask
        self.nmb_samples  = self.xyz.shape[0]
        if use_numpy:
            self.quaternions = np.random.rand(self.nmb_samples, 4) \
                if quaternions is None else quaternions
        else:
            self.quaternions = torch.rand(self.nmb_samples, 4).to(self.device) \
                if quaternions is None else quaternions.to(self.device)
        self.quaternions.requires_grad = True

        # Normalize the scale of the proteins
        scale_factor = self.xyz[0].norm(dim=1).std()/100
        self.xyz = self.xyz.div(scale_factor)
        self.learning_rate = 100

        self.optimizer = torch.optim.SGD([self.quaternions], lr=self.learning_rate)
        self.use_numpy = use_numpy
        # TODO, nadat je klaar bent met experimenteren zorgen dat de
        # self.history niet meer in de code staat
        # self.history = []

    def get_center(self, xyz, del_mask):
        """Get the centers of each PDB. Ignoring missing residues.

        Args:
            xyz (torch.Tensor): PDB coordinates.
            del_mask (torch.Tensor): Loss mask for each PDB.

        Returns:
            torch.Tensor: Tensor with the center of each PDB.
        """
        return (xyz*del_mask).sum(1).div(del_mask.sum(1)).reshape(-1,1,3)

    def get_matrix(self):
        """Get the rotation matrix using the learned quaternions

        Returns:
            torch.Tensor: Rotation matrix.
        """
        # Normalize the q-values. The norm should equal to 1.
        q = self._normalize(self.quaternions)
        # init an empty matrix to fill
        if self.use_numpy:
            matrix = np.zeros((q.shape[0], 3, 3))
        else:
            matrix = torch.zeros(q.shape[0], 3, 3).to(self.device)
        # Fill the matrix
        matrix[:,0,0] = (2 * (q[:,0]**2     + q[:,1]**2)) - 1
        matrix[:,0,1] =  2 * (q[:,1]*q[:,2] - q[:,0]*q[:,3])
        matrix[:,0,2] =  2 * (q[:,1]*q[:,3] + q[:,0]*q[:,2])
        matrix[:,1,0] =  2 * (q[:,1]*q[:,2] + q[:,0]*q[:,3])
        matrix[:,1,1] = (2 * (q[:,0]**2     + q[:,2]**2)) - 1
        matrix[:,1,2] =  2 * (q[:,2]*q[:,3] - q[:,0]*q[:,1])
        matrix[:,2,0] =  2 * (q[:,1]*q[:,3] - q[:,0]*q[:,2])
        matrix[:,2,1] =  2 * (q[:,2]*q[:,3] + q[:,0]*q[:,1])
        matrix[:,2,2] = (2 * (q[:,0]**2     + q[:,3]**2)) - 1
        return matrix

    def get_rotated_xyz(self, pdb_index=None, xyz=None, rotation_matrix=None):
        """Returns the PDB coordinates after rotating by the Rotator's quaternions.

        Args:
            pdb_index (int, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: Rotated PDB coordinates.
        """
        if rotation_matrix is None:
            # Retrieve rotation matrix
            rotation_matrix = self.get_matrix()

        # Apply the rotation matrix
        if pdb_index:
            if self.use_numpy:
                return np.matmul(xyz, rotation_matrix[pdb_index:pdb_index+1])
            return torch.matmul(xyz, rotation_matrix[pdb_index:pdb_index+1])
        if self.use_numpy:
            return np.matmul(self.xyz, rotation_matrix)
        return torch.matmul(self.xyz, rotation_matrix)

    def _normalize(self, quaternions):
        """Normalize vector (vector.pow(2).sum()==1)"""
        quaternions = quaternions / quaternions.norm(dim=1).reshape(-1,1)
        return quaternions

    def _loss(self):
        """Calculate the loss by comparing the rotated coordinates with the template's."""
        new_xyz = self.get_rotated_xyz()
        loss =  F.mse_loss(new_xyz, self.xyz[0], reduction='none') * self.del_mask
        loss = loss.mean(2)
        loss = loss.mean(1)
        # self.history += [[i.item() for i in loss]]
        return loss.sum()

    def normalize_q(self):
        """Normalize the Rotator's quaternions."""
        self.quaternions.data = self._normalize(self.quaternions.data)

    # Take a gradient step
    def step(self):
        """Optimize the rotation matrix of the Rotator by one step.

        Returns:
            Float: The loss of the step.
        """
        loss = self._loss()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

class PDBdataset():
    """Class that processes PDBs.
    """
    def __init__(self, pdbs, template_pdb, residues, chain, cores=mp.cpu_count(), output='result', device='cpu', verbosity=1):
        """Initializes the PDBdataset object.

        Args:
            pdb_input_folder (str): Folder where the PDB files are located.
            template_pdb (str): Path to the template PDB.
            residues (list[int]): List of residue IDs to be used for alignment.
            chain (str): Chain ID to be used for alignment.
            cores (int, optional): Amount of CPU cores to use for multiprocessing. Defaults to 1.
            subfolders (bool, optional): Look for .pdb files in subfolders of pdb_input_folder.
                Defaults to False.
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
        xyz, del_mask = self.load_ca_atom_positions()
        if self.verbosity > 1:
            print(f'Retrieved data, time: {time.perf_counter()-t0:.2f} seconds')

        # Instantiate a Rotator
        self.rotator = Rotator(xyz, del_mask, device=self.device)

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
        del_mask = (ca_xyz.abs().sum(2) != 0).unsqueeze(2).float().to(self.device)
        return ca_xyz, del_mask

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

    def calc_rmsd_with_template(self, output_file):
        """Calculates the RMSD of each aligned PDB with the template.
        Only takes into account the selected residues and ignores any deletions.
        """
        with torch.no_grad():
            rotated_xyz = self.rotator.get_rotated_xyz()
            pdb_range = torch.arange(start=1, end=len(self.pdbs))
            combined_masks = self.rotator.del_mask[0] * self.rotator.del_mask[pdb_range]
            rmsds = ((rotated_xyz[0] - rotated_xyz[pdb_range]) * combined_masks)\
                .pow(2).reshape(len(self.pdbs)-1, -1).sum(1)\
                .div(combined_masks.sum(1).squeeze()).sqrt()

            with open(output_file, 'a', encoding='utf-8') as rmsd_file:
                rmsd_file.write(f"Template\t{self.pdbs[0]}\n")
                rmsd_file.write("\n".join([
                    f"{os.path.basename(pdb)}\t{rmsds[i].item()}"
                    for i, pdb in enumerate(self.pdbs[1:])
                ]))

    def optimize(self):
        """Runs the optimization loop.
        Attempts to find the optimal rotation to superpose all PDBs to the template."""
        if self.verbosity > 1:
            print("Aligning backbones")
        t0 = time.perf_counter()
        label = self.rotator.xyz[0:1]
        shake_steps = max(75, -4*label.shape[1]+400)
        finetune_steps = 25
        step = 0
        for step in trange(0, shake_steps+finetune_steps, desc="Aligning backbones",
                disable=self.verbosity != 1):
            # Take a step
            loss = self.rotator.step()
            if step <= shake_steps:
                self.rotator.normalize_q()
            if step == shake_steps:
                self.rotator.optimizer = torch.optim.SGD([self.rotator.quaternions], lr=0.00001)
            if self.verbosity > 1:
                print(f'Step: {step+1} \t Loss: {loss}')
        if self.verbosity > 1:
            print(f'Steps: {step+1}, alignment-time: {time.perf_counter()-t0:.2f}')


def superpose(pdbs_list, template, output=None, residues=None, chain=None, cores=mp.cpu_count(),
    batch_size=50000, gpu=False, rmsd_path=None, verbosity=1):
    """Runs all the steps to superpose a list of PDBs.

    Args:
        pdbs_list (list): List of paths to PDB files.
        output (str): Output folder.
        template (str): Template PDB path.
        residues (list): List of residue indices to align to.
        chain (str): Chain to align to.
        cores (int, optional): The number of CPU cores to use for multiprocessing. Defaults to 4.
        batch_size (int, optional): The batch size of the alignment. Defaults to 50000.
        device (str, optional): The device to use for PyTorch tensors. Defaults to 'cpu'.
        rmsd_path (str, optional): The path to write RMSDs to (compared to template).
            Leave None to skip RMSD calculation. Defaults to None.
        verbosity (int, optional): Verbosity level. 0 = Silent, 1 = Normal, 2 = Verbose.
            Defaults to 1.
    """
    if verbosity > 0:
        start_time = time.perf_counter()
        print("\n=== Superimpose ===")
    
    torch.set_num_threads(cores)

    # for REPRODUCIBILITY
    warnings.filterwarnings("ignore")

    # Make output folder
    os.makedirs(output, exist_ok=True)

    if gpu:
        device = torch.device("cuda")
        mp.set_start_method('spawn')
    else:
        device = torch.device("cpu")

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
        # import cProfile
        # import pstats
        # with cProfile.Profile() as p:
        data_processor.rotate_all_pool()
        # torch.save(data_processor, "data_processor.pt")
        # quit()
        if rmsd_path:
            data_processor.calc_rmsd_with_template(rmsd_path)
        # stats = pstats.Stats(p)
        # stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
        # import pylab as plt
        # plt.plot(data_processor.rotator.history)
        # plt.show()
        # Free memory
        # del data_processor

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
