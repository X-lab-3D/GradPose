import glob
import os
import unittest
import tempfile
import subprocess
import gradpose
from gradpose import util

DOCKING_PATH = "./test/data/docking"
DOCKING_MODELS = glob.glob(os.path.join(DOCKING_PATH, "*.pdb"))

TEST_ENV = os.environ.copy()

class TestGradPose(unittest.TestCase):
    """Testcases for the usage of GradPose."""

    def assert_docking(self, tmp_path, rmsd_path):
        """Generic check if superimposing of docking models succeeded.
        Because the receptor chain is identical in all PDBs,

        Args:
            tmp_path (str): Path to folder with aligned PBDs.
            rmsd_path (str): Path to what should be the RMSD file.
        """
        # Ensure we have the right amount of output pdbs.
        self.assertTrue(len(DOCKING_MODELS) == len(glob.glob(os.path.join(tmp_path, "*.pdb"))))

        # Confirm an rmsd file was made.
        self.assertTrue(os.path.exists(rmsd_path))

        # Confirm that all RMSDs are there.
        with open(rmsd_path, "r", encoding='utf-8') as rmsd_file:
            rmsds = rmsd_file.read()
        rmsds = [float(rmsd.split("\t")[1]) for rmsd in rmsds.split("\n")[1:] if rmsd]
        self.assertTrue(len(rmsds) == len(DOCKING_MODELS))

        # Confirm that aligning worked.
        for rmsd in rmsds:
            self.assertTrue(rmsd < 1e-5)


    def test_docking_package(self):
        """Test GradPose on docking models through the Python package.
        """
        with tempfile.TemporaryDirectory() as tmp_path:

            rmsd_path = os.path.join(tmp_path, "rmsd.tsv")
            gradpose.superpose(DOCKING_MODELS, DOCKING_MODELS[0], tmp_path, rmsd_path=rmsd_path)

            # Confirm that superposition worked.
            self.assert_docking(tmp_path, rmsd_path)


    def test_docking_cmd(self):
        """Test GradPose on docking models through the command line.
        """
        with tempfile.TemporaryDirectory() as tmp_path:

            rmsd_path = os.path.join(tmp_path, "rmsd.tsv")
            subprocess.run(["gradpose", "-i", DOCKING_PATH, "-o", tmp_path, "--rmsd"], check=True)

            # Confirm that superposition worked.
            self.assert_docking(tmp_path, rmsd_path)


    def test_params_cmd(self):
        """Test GradPose on docking models through the command line,
        using all available parameters (excluding --silent).
        """
        with tempfile.TemporaryDirectory() as tmp_path:

            rmsd_path = os.path.join(tmp_path, "rmsd.tsv")
            subprocess.run(["gradpose", "-i", DOCKING_PATH, "-o", tmp_path, "-r", "10:50",
            "-c", "A", "-n", "1", "-b", "100", "--gpu", "--verbose", "--rmsd"], check=True)

            # Confirm that superposition worked.
            self.assert_docking(tmp_path, rmsd_path)

    def test_silent_cmd(self):
        """Test if silent parameter works as intended.
        """
        with tempfile.TemporaryDirectory() as tmp_path:

            rmsd_path = os.path.join(tmp_path, "rmsd.tsv")
            process = subprocess.Popen(["gradpose", "-i", DOCKING_PATH,
            "-o", tmp_path, "--silent", "--rmsd"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Confirm no output was created on stdout or stderr.
            stdout, stderr = process.communicate()
            self.assertTrue(len(stdout) == 0)
            self.assertTrue(len(stderr) == 0)

            # Confirm that superposition worked.
            self.assert_docking(tmp_path, rmsd_path)
    
    def test_arg_parse(self):
        """Test all the arguments in the argparser.
        """
        arguments = ["-i", DOCKING_PATH, "-o", "out", "-r", "10:50",
            "-c", "A", "-n", "1", "-b", "100", "--gpu", "--verbose", "--rmsd"]
        parsed = util.parse_args(arguments)
        self.assertEqual(os.path.relpath(parsed.input), os.path.relpath(DOCKING_PATH))
        self.assertEqual(parsed.subfolders, False)
        self.assertEqual(parsed.template, None)
        self.assertEqual(parsed.output, "out")
        self.assertEqual(parsed.chain, "A")
        self.assertEqual(parsed.residues, list(range(10, 51)))
        self.assertEqual(parsed.subfolders, False)
        self.assertEqual(parsed.n_cores, 1)
        self.assertEqual(parsed.batch_size, 100)
        self.assertEqual(parsed.rmsd, True)
        self.assertEqual(parsed.silent, False)
        self.assertEqual(parsed.verbose, True)
        self.assertEqual(parsed.verbosity, 2)


if __name__ == "__main__":
    unittest.main()
