import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReproCliTest(unittest.TestCase):
    def _base_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        env.pop("CEA_MI_DATASET", None)
        env.pop("CEA_MI_NANOBOT_PROJECT", None)
        return env

    def test_agent_interface_module_import_does_not_require_nanobot(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import agent_interface; print('ok')"],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_natural_attack_requires_explicit_dataset_path(self):
        proc = subprocess.run(
            [sys.executable, "natural_attack.py", "--help"],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)

        proc = subprocess.run(
            [sys.executable, "natural_attack.py", "--access", "blackbox"],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("dataset", (proc.stderr + proc.stdout).lower())


if __name__ == "__main__":
    unittest.main()
