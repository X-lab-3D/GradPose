# GradPose

Welcome to the GitHub repository for the gradPose tool.

This tool is based on the tool paper titled **Eliminating Bottlenecks in Protein Structure Superimposition with: GradPose**, published in **Oxford Bioinformatics** in **2022**. GradPose is a novel structural superimposition command line tool and Python package for PDB files. GradPose uses gradient descent, to incrementally approach the optimal rotation matrices (via quaternions) for alignment. It is efficient in terms of memory usage and allows fast alignment of thousands to millions of protein structures to a template structure, while also offering precise control over which chain and specific residues to align.The tool is designed to overcome the limitations of classical superimposition software, which are not equipped to handle the large number of PDB files produced by modern researchers. Our method scales linearly with the number of residues and can also effectively utilize batch matrix operations even in the presence of amino-acid insertions or deletions. This makes it more efficient than traditional methods, which tend to scale exponentially with the number of residues and process the pbds individually. Additionally, it will automatically make use of CUDA acceleration using a GPU if available. 

This repository contains the source code and documentation for the gradPose tool. Please refer to the documentation for instructions on how to use the tool and for more information about its features and capabilities.

We hope that this tool will be useful to researchers and practitioners in the field of bioinformatics. If you have any questions or suggestions, please don't hesitate to open an issue or submit a pull request. We welcome contributions and feedback from the community.

## Install

From pip:
    
```bash
pip install gradPose
```

From repository:
    
```bash
pip install git+https://github.com/X-lab-3D/fast-superimposition
```

## Examples of usage:

A folder containing N pdbs names 'example_pdb_folder'

```bash
gradPose -i example_pdb_folder
```
or
```
gradPose example_pdb_folder
```
The aligned proteins are automatically stored in the folder 'output'. Using another folder name, or overwriting the current folder without creating a second is possible using the -o option. 


If the pdbs need to be aligned to a specific template:
```
gradPose -i example_pdb_folder -t template_example.pdb
```

If only a spedific part of the proten needs to be aligned, for exmaple, the first 10 amino-acids and the the aminoacids ranging between 20 and 30:

```
gradPose -i example_pdb_folder -r 1:10 20:30
```

etc... Kevin can yo make other examples? or list the options with explanations?
