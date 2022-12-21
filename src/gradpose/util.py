"""
Utility function(s) moved here to declutter the main file.
"""
import sys
import os
from argparse import ArgumentParser
import torch.multiprocessing as mp
import torch.cuda
import torch._C

def parse_args():
    """Parses and processes the command line arguments.

    Returns:
            Namespace: Argument namespace.
    """
    arg_parser = ArgumentParser(
        description="Aligns PDB structures in bulk against a template."
        # TODO: Improve description
    )

    if len(sys.argv) == 2 and sys.argv[1] not in ["-h", "--help"]:
        sys.argv.insert(1, "-i")
    arg_parser.add_argument("-i", "--input",
        help="""
        Folder containing the PDB models. -i or --input can be left out if calling superpose with only an input folder.
        """,
        type=str
    )
    arg_parser.add_argument("-s", "--subfolders",
        help="""
        (Optional) Add this option to enable searching for PDBs recursively in subfolders.
        Warning: Files with identical names will be overwritten in the output folder.
        """,
        action='store_true'
    )
    arg_parser.add_argument("-t", "--template",
        help="""
        (Optional) Path to a PDB file to use as template for alignment.
        It does not have to be part of the input files.
        The template will not be written to the output folder unless it is also an input file.
        Defaults to the first input file alphabetically.
        """,
        type=str,
        default=None
    )
    arg_parser.add_argument("-o", "--output",
        help="""
        (Optional) Set the output folder.
        Defaults to ./output.
        """,
        type=str,
        default='output'
    )
    arg_parser.add_argument("-c", "--chain",
        help="""
        (Optional) The chain ID to use for alignment. Defaults to the longest chain of the template.
        """,
        type=str,
        default=None
    )
    arg_parser.add_argument("-r", "--residues",
        help="""
        (Optional) List of residue numbers to use for alignment. Separated by spaces.
        Indicate ranges as: start:stop (Including start and stop.) Defaults to all residues in the selected chain of the template.
        """,
        nargs='+',
        type=str,
        default=None
    )
    # arg_parser.add_argument("-f", "--full-backbone",
    #     help="""
    #     (Optional) Add this option to align using all backbone atoms.
    #     By default, the aligning algorithm only uses the CA atoms of the backbone.
    #     """,
    #     action='store_true'
    # )
    arg_parser.add_argument("-n", "--n-cores",
        help="""
        (Optional) Number of CPU cores to use for multiprocessing.
        Defaults to 1.
        """,
        type=int,
        default=mp.cpu_count()
    )

    arg_parser.add_argument("-g", "--gpu",
        help="""
        (Optional) Enable CUDA accelleration for alignment.
        Requires a PyTorch installation that supports CUDA.
        """,
        action='store_true'
    )

    arg_parser.add_argument("-b", "--batch-size",
        help="""
        (Optional) Number of PDB files to align per batch.
        Defaults to 50,000 (maximum).
        """,
        type=int,
        default=50000
    )
    arg_parser.add_argument("--silent",
        help="""
        (Optional) Do not print anything.
        """,
        action='store_true'
    )
    arg_parser.add_argument("--verbose",
        help="""
        (Optional) Print extra information.
        """,
        action='store_true'
    )

    arg_parser.add_argument("--rmsd",
        help="""
        (Optional) Calculate RMSD between template and models (only at selected residues) and save the results as rmsd.tsv in the output folder.
        """,
        action="store_true"
    )

    # Parse the arguments
    parsed_args = arg_parser.parse_args()

    # Print help if no arguments are used
    if len(sys.argv) < 2:
        arg_parser.print_help()
        sys.exit()

    # Determine verbosity level
    # 0 = silent
    # 1 = default
    # 2 = verbose
    parsed_args.verbosity = 1
    if parsed_args.verbose:
        parsed_args.verbosity = 2
    elif parsed_args.silent:
        parsed_args.verbosity = 0

    if parsed_args.n_cores > mp.cpu_count():
        parsed_args.n_cores = mp.cpu_count()
        print(f"Argument Warning: CPU cores limited to {parsed_args.n_cores}.")

    if parsed_args.batch_size > 50000:
        parsed_args.batch_size = 50000
        print(f"Argument Warning: Batch size limited to {parsed_args.batch_size}")

    if parsed_args.gpu and not hasattr(torch._C, '_cuda_getDeviceCount'):
        print("Argument Warning: Attempted to enable CUDA but PyTorch is not compiled with CUDA enabled. Continuing on CPU.")
        parsed_args.gpu = False
    
    parsed_args.input = str(os.path.normpath(parsed_args.input))

    # Process residue selection
    if parsed_args.residues:
        try:
            residue_ids = []
            for segment in parsed_args.residues:
                if ":" in segment:
                    # Convert range notation to list of ints.
                    # Allowing reverse order.
                    start, stop = segment.split(":", 1)
                    if not start.isnumeric() or not stop.isnumeric():
                        # Do not trust that a ValueError gets thrown automatically,
                        # because it might be interpreted as a base10 number.
                        raise TypeError()
                    start, stop = map(int, (start, stop))
                    if stop < start:  # Reverse order if needed
                        stop, start = start, stop
                    residue_ids += list(range(int(start), int(stop)+1))
                else:
                    residue_ids.append(int(segment))
            parsed_args.residues = list(set(residue_ids))  # Remove duplicates
            if parsed_args.verbosity > 1:
                print("Selected residues:")
                print(parsed_args.residues)
            if len(parsed_args.residues) < 3:
                raise Exception("Please specify at least 3 residues for alignment.")
        except TypeError as exc:
            raise TypeError("Could not parse a residue (range) as integers.") from exc

    return parsed_args
