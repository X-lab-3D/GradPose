"""
Utility function(s) moved here to declutter the main file.
"""
import sys
from argparse import ArgumentParser

def parse_args():
    """Parses and processes the command line arguments.

    Returns:
            Namespace: Argument namespace.
    """
    arg_parser = ArgumentParser(
        description="Aligns PDB structures in bulk against a template."
        # TODO: Improve description
    )

    if len(sys.argv) == 2:
        sys.argv.insert(1, "-i")
    arg_parser.add_argument("-i", "--input",
        help="""
        Folder containing the PDB models. -i or --input can be left out if calling superpose with only an input folder.
        """,
        required=True
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
    arg_parser.add_argument("-f", "--full-backbone",
        help="""
        (Optional) Add this option to align using all backbone atoms.
        By default, the aligning algorithm only uses the CA atoms of the backbone.
        """,
        action='store_true'
    )
    arg_parser.add_argument("-n", "--n-cores",
        help="""
        (Optional) Number of CPU cores to use for multiprocessing.
        Defaults to 1.
        """,
        type=int,
        default=4
    )
    arg_parser.add_argument("-b", "--batch-size",
        help="""
        (Optional) Number of PDB files to align per batch.
        Defaults to 50,000.
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

    arg_parser.add_argument("--rmsd-test",
        action="store_true"
    )

    # Print help if no arguments are used
    if len(sys.argv) < 2:
        arg_parser.print_help()
        sys.exit()

    # Parse the arguments
    parsed_args = arg_parser.parse_args()

    # Determine verbosity level
    # 0 = silent
    # 1 = default
    # 2 = verbose
    parsed_args.verbosity = 1
    if parsed_args.verbose:
        parsed_args.verbosity = 2
    elif parsed_args.silent:
        parsed_args.verbosity = 0

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

    print(parsed_args)
    return parsed_args
