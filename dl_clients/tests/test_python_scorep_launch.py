"""Exercise the mini-app's Python Score-P launch contract without MPI jobs."""
import ast
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "mini_app/het_hires_job.sh"
CLIENT = ROOT / "CPP-ML-Interface/dl_clients/phydll_dl_client.py"


class PythonScorepLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = LAUNCHER.read_text()
        start = text.index('if [[ "${USE_SCOREP}" == "1" && "${USE_CPP_ML_INTERFACE}" -eq 1 &&')
        end = text.index('if [[ -n "${COMPILE_OUTPUT_PATH_ENV:-}" ]]', start)
        cls.guard = text[start:end]
        cls.command = next(line.strip() for line in text.splitlines()
                           if line.strip().startswith('DL_CLIENT_CMD=("env"'))

    def shell(self, script, **variables):
        env = os.environ.copy()
        for key in ("PHYDLL_NP_DL", "PHYDLL_NP_DL_ENV", "PHYDLL_PY_SCOREP_WRAPPER",
                    "USE_PYTHON_DL_CLIENT", "SCOREP_ENABLE_TRACING_ENV",
                    "SCOREP_SHARED_PROFILE_ENV"):
            env.pop(key, None)
        env.update(variables)
        return subprocess.run(["zsh", "-c", script], env=env, text=True,
                              capture_output=True, timeout=10)

    def test_guard(self):
        base = dict(USE_SCOREP="1", USE_CPP_ML_INTERFACE="1",
                    CPP_ML_INTERFACE_PROVIDER="PHYDLL", USE_PYTHON_DL_CLIENT="1")
        cases = [
            ({"SCOREP_MPP": "mpi", "PHYDLL_PY_SCOREP_WRAPPER": wrapper}, False)
            for wrapper in ("0", "1")
        ] + [
            ({"SCOREP_MPP": "none", "PHYDLL_PY_SCOREP_WRAPPER": "1"}, True),
            ({"SCOREP_MPP": "none", "PHYDLL_PY_SCOREP_WRAPPER": "1", "PHYDLL_NP_DL_ENV": "2"}, False),
            ({"SCOREP_MPP": "none", "PHYDLL_PY_SCOREP_WRAPPER": "0", "PHYDLL_NP_DL_ENV": "2"}, True),
            ({"SCOREP_MPP": "none", "SCOREP_ENABLE_TRACING_ENV": "true"}, False),
            ({"SCOREP_MPP": "none", "SCOREP_SHARED_PROFILE_ENV": "1"}, False),
            ({"SCOREP_MPP": "mpi", "USE_PYTHON_DL_CLIENT": "0"}, True),
            ({"SCOREP_MPP": "mpi", "USE_SCOREP": "0"}, True),
            ({"SCOREP_MPP": "mpi", "CPP_ML_INTERFACE_PROVIDER": "SMARTSIM"}, True),
        ]
        for overrides, allowed in cases:
            with self.subTest(overrides=overrides):
                result = self.shell(self.guard, **(base | overrides))
                self.assertEqual(result.returncode == 0, allowed, result.stderr)

    def test_wrapper_command(self):
        script = self.command + '\nprint -rl -- "${DL_CLIENT_CMD[@]}"'
        result = self.shell(script, SCOREP_MPP="mpi", _py_exp_dir="/tmp/exp",
                            _py_overwrite="SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true",
                            SMARTSIM_PAPI_ROOT="/papi", SMARTSIM_SCOREP_ROOT="/scorep",
                            SCOREP_BIN_DIR="/bin", SCOREP_PYTHONPATH="/site",
                            PY_ENV="/python", MINI_APP_DIR="/mini")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("SCOREP_MPP=none", args)
        self.assertIn("--mpp=none", args)
        self.assertIn("--instrumenter-type=profile", args)
        self.assertIn("--noinstrumenter", args)
        self.assertNotIn("--instrumenter-type=dummy", args)

    def test_client_finalization_contract(self):
        tree = ast.parse(CLIENT.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
        # Execute the real teardown using stand-ins, without importing MPI or Torch.
        calls = []

        class MPI:
            @staticmethod
            def Is_initialized(): return True
            @staticmethod
            def Is_finalized(): return False
            @staticmethod
            def Finalize(): calls.append("mpi")

        class User:
            @staticmethod
            def force_finalize(): calls.append("scorep")

        class Scorep:
            user = User()

        class Env:
            environ = {"SCOREP_MPP": "none", "PHYDLL_PY_FORCE_EXIT": "0"}

        namespace = dict(__name__="__main__", main=lambda: None, MPI=MPI, os=Env(),
                         HAS_SCOREP=True, scorep=Scorep(), PY_SCOREP_WRAPPER=True)
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(CLIENT), "exec"), namespace)
        self.assertEqual(calls, ["scorep", "mpi"])
        calls.clear()
        Env.environ["SCOREP_MPP"] = "mpi"
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(CLIENT), "exec"), namespace)
        self.assertEqual(calls, ["mpi"])


if __name__ == "__main__":
    unittest.main()
