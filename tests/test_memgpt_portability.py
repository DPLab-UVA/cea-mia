import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MemGPTPortabilityTest(unittest.TestCase):
    def _base_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        env.pop("CEA_MI_DATASET", None)
        env.pop("CEA_MI_LOG_DIR", None)
        env.pop("CEA_MI_MEMGPT_MEMORY_FILE", None)
        env.pop("CEA_MI_INSTALL_MEMGPT_DEPS", None)
        return env

    def test_memgpt_attack_requires_explicit_dataset_path(self):
        proc = subprocess.run(
            [sys.executable, "memgpt_target/memgpt_attack.py", "--help"],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

        proc = subprocess.run(
            [sys.executable, "memgpt_target/memgpt_attack.py", "--access", "blackbox"],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("dataset", (proc.stderr + proc.stdout).lower())
        self.assertNotIn("sentence_transformers", proc.stderr + proc.stdout)

    def test_setup_memgpt_standalone_help_mentions_memory_file(self):
        proc = subprocess.run(
            [sys.executable, "memgpt_target/setup_memgpt.py", "standalone", "--help"],
            cwd=REPO_ROOT,
            env=self._base_env(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--memory-file", proc.stdout)

    def test_run_memgpt_attack_uses_dataset_memory_file_and_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "benchmark.json"
            dataset_path.write_text("[]")
            memory_file = tmp_path / "memories.json"
            log_dir = tmp_path / "logs"
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            args_log = tmp_path / "python_args.log"
            pip_log = tmp_path / "pip.log"

            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/bash\n"
                "echo \"$@\" >> \"$ARGS_LOG\"\n"
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

            fake_pip = fake_bin / "pip"
            fake_pip.write_text(
                "#!/bin/bash\n"
                "echo \"$@\" >> \"$PIP_LOG\"\n"
            )
            fake_pip.chmod(fake_pip.stat().st_mode | stat.S_IEXEC)

            env = self._base_env()
            env["CEA_MI_DATASET"] = str(dataset_path)
            env["CEA_MI_MEMGPT_MEMORY_FILE"] = str(memory_file)
            env["CEA_MI_LOG_DIR"] = str(log_dir)
            env["ARGS_LOG"] = str(args_log)
            env["PIP_LOG"] = str(pip_log)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            proc = subprocess.run(
                ["bash", "memgpt_target/run_memgpt_attack.sh"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(pip_log.exists(), "runner should not install dependencies implicitly")

            invocations = args_log.read_text().strip().splitlines()
            self.assertEqual(len(invocations), 4)
            self.assertIn(
                f"memgpt_target/setup_memgpt.py standalone --dataset {dataset_path} --memory-file {memory_file}",
                invocations[0],
            )
            for invocation in invocations[1:]:
                self.assertIn("memgpt_target/memgpt_attack.py", invocation)
                self.assertIn(f"--dataset {dataset_path}", invocation)
                self.assertIn(f"--memory-file {memory_file}", invocation)
            self.assertTrue((log_dir / "attack_memgpt_blackbox.log").exists())
            self.assertTrue((log_dir / "attack_memgpt_graybox.log").exists())
            self.assertTrue((log_dir / "attack_memgpt_whitebox.log").exists())


if __name__ == "__main__":
    unittest.main()
