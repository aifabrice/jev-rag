import os
import subprocess
import sys
import unittest
from pathlib import Path


class SmokeCliEncodingTests(unittest.TestCase):
    def test_dry_run_writes_unicode_under_legacy_console_encoding(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "cp1252"
        result = subprocess.run(
            [sys.executable, "jev_test.py", "--dry-run"],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        output = result.stdout.decode("utf-8")
        self.assertIn("questions", output)
        self.assertIn("客户说", output)


if __name__ == "__main__":
    unittest.main()
