import glob
import gradpose

def main():
    # Make a list of all PDB files to align.
    pdb_files = glob.glob("PDBs/*.pdb")

    # Superimposition with GradPose.
    gradpose.superpose(
        pdbs_list=pdb_files,
        template="PDBs/1AGD.pdb",
        output="aligned_PDBs",
        chain="M",
        residues=range(1, 181)  # 1 through 180
    )

if __name__ == "__main__":
    main()
