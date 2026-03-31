import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunAllAttacksV4Test(unittest.TestCase):
    def _base_env(self):
        env = os.environ.copy()
        env.pop("CEA_MI_DATASET", None)
        env.pop("CEA_MI_LOG_DIR", None)
        env.pop("CEA_MI_NANOBOT_DB_PATH", None)
        return env

    def test_requires_dataset_path_before_launching(self):
        env = self._base_env()
        proc = subprocess.run(
            ["bash", "run_all_attacks_v4.sh"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("CEA_MI_DATASET", proc.stderr + proc.stdout)

    def test_passes_dataset_and_uses_configured_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "benchmark.json"
            dataset_path.write_text("[]")
            log_dir = tmp_path / "logs"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            args_log = tmp_path / "python_args.log"

            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/bash\n"
                "echo \"$@\" >> \"$ARGS_LOG\"\n"
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

            env = self._base_env()
            env["CEA_MI_DATASET"] = str(dataset_path)
            env["CEA_MI_LOG_DIR"] = str(log_dir)
            env["ARGS_LOG"] = str(args_log)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            proc = subprocess.run(
                ["bash", "run_all_attacks_v4.sh"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            invocations = args_log.read_text().strip().splitlines()
            self.assertEqual(len(invocations), 3)
            for invocation in invocations:
                self.assertIn("natural_attack.py", invocation)
                self.assertIn(f"--dataset {dataset_path}", invocation)
            self.assertTrue((log_dir / "attack_v4_blackbox.log").exists())
            self.assertTrue((log_dir / "attack_v4_graybox.log").exists())
            self.assertTrue((log_dir / "attack_v4_whitebox.log").exists())


if __name__ == "__main__":
    unittest.main()
