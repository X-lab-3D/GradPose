# GradPose

Welcome to the GitHub repository for the gradPose tool.

This tool is based on the research paper titled **Eliminating Bottlenecks in Protein Structure Superimposition with: GradPose**, published in **Oxford Bioinformatics** in **2022**. The tool is designed to do fast and effient superimpositions of thousands to millions of pdbs (Protein Data Bank files) capable of runnning on standard hardware. Our software is able to overcome the botlenecks of classic strucutral superimpositions algorithms by being able to scale linearly with the number of residues (instead of exponentially) and being able to effiently use multocore matrix operations that can tolerate insertions and deletions intruced by homology moddeling and prediction software. The software supports both CPU and GPU via cuda operations.

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
