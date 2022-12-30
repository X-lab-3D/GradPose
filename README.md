# GradPose

Welcome to the GitHub repository for the GradPose tool.

GradPose is a novel structural superimposition command-line tool and Python package for PDB files. GradPose uses gradient descent to incrementally approach the optimal rotation matrices (via quaternions) for alignment.  It is memory efficient and enables for the quick alignment of thousands to millions of protein structures to a template structure while also providing exact control over which chain and specific residues to align. The tool is designed to overcome the limitations of classical superimposition algorithms, which are not equipped to handle the large number of PDB files produced by researchers today. Our method scales linearly with the number of residues and can also use batch matrix operations effectively even when there are amino-acid insertions or deletions. This makes it more efficient than traditional methods, which tend to scale exponentially with the number of residues and process the pbds individually. Furthermore, if a GPU is available, it can use CUDA acceleration.

This repository contains the source code and documentation for the GradPose tool. Please refer to the documentation for instructions on how to use the tool and for more information about its features and capabilities.

We hope that this tool will be useful to researchers and practitioners in the field of bioinformatics. If you have any questions or suggestions, please don't hesitate to open an issue or submit a pull request. We welcome contributions and feedback from the community.


## Dependencies
GradPose requires [Python 3](https://www.python.org/downloads/) to be installed on your system.


## Install

Install GradPose using Python's package installer pip:
```sh
pip install gradpose
```


## Usage

```sh
gradpose [-h] [-i INPUT] [-s] [-t TEMPLATE] [-o OUTPUT] [-c CHAIN] [-r RESIDUES [RESIDUES ...]] [-n N_CORES] [-g] [-b BATCH_SIZE] [--silent] [--verbose] [--rmsd]
```
Help and defaults for each argument can be viewed by executing GradPose with the help argument: ``gradpose -h``.
Alternatively, you can examine the example usages listed below.

## Example usages

### Input
To use GradPose, specify a folder folder containing any amount of PDBs to be used for alignment using the ``-i`` argument.
For example, let's use a folder named 'example_folder'.
```sh
gradpose -i example_folder
```
or
```sh
gradpose example_folder
```
Note: Omitting ``-i`` is only possible if no other arguments are used.
Note: The aligned proteins are automatically stored in the folder 'output'.

### Output
Using another folder name, or overwriting the current folder without creating a second is possible using the ``-o`` argument.
```sh
gradpose -i example_folder -o example_output_folder
```
Note: To overwrite the current files, provide the same input and output folder.


### Template
If the PDBs in folder 'example_folder' need to be aligned to a specific template, use the ``-t`` argument:
```sh
gradpose -i example_folder -t example_folder/template_example.pdb
```
Note: The template **does not** need to be in the same folder as the PDBs used for alignment.

### Chain
By default, GradPose aligns to the longest chain in the template PDB. You can choose a chain ID with the ``-c`` argument.
In this example, the alignment is done on all residues of chain B:
```sh
gradpose -i example_folder -c B
```

### Residues
By default, GradPose aligns to all residues of the selected chain. If a finer selection can be made with the ``-r`` argument.
For exmaple, to align on the first 10 amino-acids, animo-acids 12 and 14, and the the aminoacids ranging between (and including) 20 and 30 of chain B:
```sh
gradpose -i example_folder -c B -r 1:10 12 14 20:30
```

### CPU Cores
By default, GradPose utilizes all CPU cores on the system. To manually specify the amount of cores, use ``-n``.
```sh
gradpose -i example_folder -n 4
```

### Batch Size
By default, the batch size is set to 50,000 to limit memory usage. To set a lower batch size, use the ``-b`` argument.
```sh
gradpose -i example_folder -b 1000
```

### CUDA Acceleration
GPU acceleration is disabled by default. If you have a PyTorch installation with CUDA enabled, simply add the ``--gpu`` flag to the command to enable GPU acceleration.
```sh
gradpose -i example_folder --gpu
```

### RMSD Calculation
GradPose can automatically calculate the RMSD of the residues on which it aligns compared to the template for every PDB. This feature can be enabled using the ``--rmsd`` flag. The results will be saved as 'rmsd.tsv' in the output folder.
```sh
gradpose -i example_folder --rmsd
```

### Verbosity Levels
GradPose allows the user to choose to run the tool silently, without generating any output in the console, with the ``--silent`` flag.
```sh
gradpose -i example_folder --silent
```

Alternatively, more verbose console output may be enabled with ``--verbose``.
```sh
gradpose -i example_folder --verbose
```