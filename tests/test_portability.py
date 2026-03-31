import unittest
from pathlib import Path

from config import Config
from data_loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]


class PortabilityDefaultsTest(unittest.TestCase):
    def test_config_defaults_use_repo_local_paths(self):
        cfg = Config()

        self.assertEqual(cfg.nanobot_project, REPO_ROOT)
        self.assertEqual(cfg.data_dir, REPO_ROOT / "data")
        self.assertEqual(cfg.output_dir, REPO_ROOT / "results")
        self.assertFalse(str(cfg.data_dir).startswith("/bigtemp"))
        self.assertFalse(str(cfg.output_dir).startswith("/bigtemp"))

    def test_data_loader_uses_repo_local_data_dir_by_default(self):
        loader = DataLoader(seed=7)

        self.assertEqual(loader.data_dir, REPO_ROOT / "data")
        self.assertTrue(loader.data_dir.exists())


if __name__ == "__main__":
    unittest.main()
