import torch
import torch.nn.functional as F
import numpy as np
import pdb
class Rotator():
    """The Rotation module with all learnable parameters.
    """
    def __init__(self, xyz, presence_mask, center=None, quaternions=None,
        device='cpu', use_numpy=False):
        """Initializes the Rotator object.

        Args:
            xyz (torch.Tensor): Coordinates for each atom for each PDB.
            presence_mask (torch.Tensor): A mask to zero out missing atoms when optimizing.
            center (torch.Tensor, optional): Alternative center coordinates to use for each PDB.
                Shape=(No. of PDBs, 3) Defaults to None.
            quaternions (torch.Tensor, optional): Alternative quaternions to set for each PDB.
            device (str, optional): The device tensors will be stored on. Defaults to 'cpu'.
            use_numpy (bool, optional): Whether to use numpy instead of pytorch. Defaults to False.
        """

        self.device = device
        # Rotation is always over the center of the pdb structures
        self.center = self.get_center(xyz, presence_mask) \
            if center is None else center.to(self.device)
        self.xyz = xyz - self.center
        self.presence_mask = presence_mask
        self.nmb_samples  = self.xyz.shape[0]
        if use_numpy:
            self.quaternions = np.random.rand(self.nmb_samples, 3) \
                if quaternions is None else quaternions
        else:
            self.quaternions = torch.rand(self.nmb_samples, 3).to(self.device) \
                if quaternions is None else quaternions.to(self.device)
        self.quaternions.requires_grad = True

        # Normalize the scale of the proteins
        #scale_factor = self.xyz[0].norm(dim=1).std()/100
        #self.xyz = self.xyz.div(scale_factor)

        #pdb.set_trace()
        self.xyz_normed = F.normalize(self.xyz, dim = 2)


        #quaternions = quaternions / quaternions.norm(dim=1).reshape(-1,1)
        self.learning_rate = 100

        self.optimizer = torch.optim.SGD([self.quaternions], lr=self.learning_rate)
        self.use_numpy = use_numpy


    def get_center(self, xyz, presence_mask):
        """Get the centers of each PDB. Ignoring missing residues.

        Args:
            xyz (torch.Tensor): PDB coordinates.
            presence_mask (torch.Tensor): Loss mask for each PDB.

        Returns:
            torch.Tensor: Tensor with the center of each PDB.
        """
        return (xyz*presence_mask).sum(1).div(presence_mask.sum(1)).reshape(-1,1,3)

    def get_matrix(self):
        """Get the rotation matrix using the learned quaternions

        Returns:
            torch.Tensor: Rotation matrix.
        """
        # # Normalize the q-values. The norm should equal to 1.
        # q = self._normalize(self.quaternions)

        q = self.quaternions

        # Add fourth quaternion, always 1.
        q = torch.concat((torch.ones(q.shape[0], 1), q), dim=1)

        # Normalize
        q = self._normalize(q)

        a = q[:,0]
        b = q[:,1]
        c = q[:,2]
        d = q[:,3]

        # init an empty matrix to fill
        if self.use_numpy:
            matrix = np.zeros((q.shape[0], 3, 3))
        else:
            matrix = torch.zeros(q.shape[0], 3, 3).to(self.device)
        # Fill the matrix
        matrix[:,0,0] = a**2 + b**2 - c**2 - d**2
        matrix[:,0,1] = 2*b*c - 2*a*d
        matrix[:,0,2] = 2*b*d + 2*a*c
        matrix[:,1,0] = 2*b*c + 2*a*d
        matrix[:,1,1] = a**2 - b**2 + c**2 - d**2
        matrix[:,1,2] = 2*c*d - 2*a*b
        matrix[:,2,0] = 2*b*d - 2*a*c
        matrix[:,2,1] = 2*c*d + 2*a*b
        matrix[:,2,2] = a**2 - b**2 - c**2 + d**2
        return matrix

    def get_rotated_xyz(self, pdb_index=None, xyz=None, rotation_matrix=None):
        """Returns the PDB coordinates after rotating by the Rotator's quaternions.

        Args:
            pdb_index (int, optional): If manually supplying xyz and rotation_matrix,
                the index to use. Defaults to None.
            xyz (torch.Tensor, optional): Override the coordinates matrix. Defaults to None.
            rotation_matrix (torch.Tensor, optional): Override the rotation matrix.
                Defaults to None.

        Returns:
            torch.Tensor: Rotated coordinates.
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
        """Normalize quaternions.

        Args:
            quaternions (torch.Tensor): Matrix of quaternions.

        Returns:
            torch.Tensor: Normalized quaternions.
        """
        quaternions = quaternions / quaternions.norm(dim=1).reshape(-1,1)
        return quaternions

    def _loss(self):
        """Calculate the loss by comparing the rotated coordinates with the template's."""
        new_xyz = self.get_rotated_xyz()
        loss =  F.mse_loss(new_xyz, self.xyz[0], reduction='none') * self.presence_mask
        loss = loss.mean(2)
        loss = loss.mean(1)
        # self.history += [[i.item() for i in loss]]
        return loss.sum()

    def normalize_q(self):
        """Normalize the Rotator's quaternions."""
        q = self.quaternions.data
        q = torch.concat((torch.ones(q.shape[0], 1), q), dim=1)
        q = self._normalize(q)
        self.quaternions.data = q[:,1:]
        # self.quaternions.data = self._normalize(self.quaternions.data)

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

