import tempfile
import unittest
from pathlib import Path

from experiment_db import prepare_isolated_memory_db


class PrepareIsolatedMemoryDbTest(unittest.TestCase):
    def test_copies_source_db_into_run_scoped_location(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "pmc.db"
            source.write_bytes(b"source-db")

            isolated = prepare_isolated_memory_db(
                source_db_path=source,
                output_dir=root / "results",
                seed=42,
                access_level="blackbox",
            )

            self.assertNotEqual(isolated, source)
            self.assertTrue(isolated.exists())
            self.assertEqual(isolated.read_bytes(), b"source-db")
            self.assertEqual(source.read_bytes(), b"source-db")

    def test_raises_when_source_db_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaises(FileNotFoundError):
                prepare_isolated_memory_db(
                    source_db_path=root / "missing.db",
                    output_dir=root / "results",
                    seed=7,
                    access_level="graybox",
                )


if __name__ == "__main__":
    unittest.main()
