"""Validate paths crossing the desktop portal boundary."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'desktop'))
from directory_picker import initial_directory, local_directory, PickerError


class DirectoryPaths(unittest.TestCase):
    def test_unicode_space_percent_and_hash_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Результаты 100% #1'
            path.mkdir()
            self.assertEqual(local_directory(path.as_uri()), str(path))

    def test_missing_output_starts_in_existing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(initial_directory(str(Path(directory) / 'new' / 'nested')), Path(directory))

    def test_reject_nonlocal_and_missing_paths(self):
        for uri in ('https://example.com/', 'file://remote/tmp', 'file:relative', 'file:///does-not-exist-heat'):
            with self.subTest(uri=uri), self.assertRaises(PickerError):
                local_directory(uri)
