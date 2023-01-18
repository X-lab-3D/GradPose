# Example scripts

This folder contains scripts to show how to use GradPose.
All scripts perform the same superimposition.

* `linux.sh` is a bash script for Linux.
* `windows.bat` is a batch file for Windows.
* `python.py` is a Python script.

## Alignment details
The `PDBs` folder contains 10 protein-MHC complex structures.
The alignment is performed on **residues 1 through 180** of the **M chain**.
`1AGD.pdb` is used as the template.
The aligned PDBs are saved in a folder called `aligned_PDBs`.

## PDB source
The PDBs provided in this example are taken from the [PANDORA database](https://github.com/X-lab-3D/PANDORA_database).