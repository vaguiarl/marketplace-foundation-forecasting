"""Public packaging must not leave a literal private-home placeholder in paths."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pressure_benchmark import SNAPSHOT


class PublicPathTests(unittest.TestCase):
    def test_snapshot_resolves_from_current_user_home(self):
        self.assertTrue(SNAPSHOT.is_absolute())
        self.assertTrue(SNAPSHOT.is_relative_to(Path.home()))

    def test_snapshot_revision_is_pinned(self):
        self.assertEqual(SNAPSHOT.name, "29ec3766d36d6f73f0696f85560a422f50e8498c")


if __name__ == "__main__":
    unittest.main()
