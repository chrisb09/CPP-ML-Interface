"""Compile the production client/runtime against tiny MPI/PhyDLL/Score-P mocks."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ProfileWarmupsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="dl-profile-")
        cls.addClassCleanup(cls.temp.cleanup)
        tests = Path(__file__).resolve().parent
        clients = tests.parent
        compiler = shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("g++ unavailable")
        cls.binaries = {}
        for scorep in (False, True):
            binary = Path(cls.temp.name) / ("scorep" if scorep else "plain")
            command = [compiler, "-std=c++17", "-O0", "-I", str(tests / "profile_mocks"),
                       "-I", str(clients.parent / "extern/phydll/src/core")]
            if scorep:
                command.append("-DUSE_SCOREP")
            command += [str(clients / "dl_client.cpp"), str(clients / "phydll_dl_runtime.cpp"),
                        str(tests / "profile_mocks.cpp"), "-o", str(binary)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            cls.binaries[scorep] = binary

    def run_client(self, frames, warmups, scorep=True):
        env = os.environ.copy()
        for name in ("CMI_SCOREP_WARMUP_CALLS", "PHYDLL_DL_FIELD_COUNT", "PHYDLL_DL_COUNT"):
            env.pop(name, None)
        env["TEST_FRAME_COUNT"] = str(frames)
        env["PHYDLL_MPMD_SHUTDOWN_BARRIER"] = "1"
        if warmups is not None:
            env["CMI_SCOREP_WARMUP_CALLS"] = str(warmups)
        return subprocess.run([str(self.binaries[scorep])], env=env, capture_output=True,
                              text=True, timeout=10)

    def test_frame_counts_and_shutdown(self):
        for frames, warmups in ((7, 2), (3, 2), (2, 2), (0, 2), (3, 0), (3, 1),
                               (3, 4), (7, None), (3, None)):
            with self.subTest(frames=frames, warmups=warmups):
                result = self.run_client(frames, warmups)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_without_scorep(self):
        for warmups in (None, 2):
            result = self.run_client(3, warmups, scorep=False)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_configuration(self):
        for warmups in ("", "-1", "two", "2extra", "999999999999999999999999"):
            with self.subTest(warmups=warmups):
                result = self.run_client(3, warmups)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("must be a nonnegative integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
