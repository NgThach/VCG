"""Distribution checks independent of the author's data directory."""
import json
from pathlib import Path
import tempfile
import unittest

from project_integrity import file_digest
import project_setup


class DistributionTests(unittest.TestCase):
    def test_manifest_has_all_signal_files(self):
        manifest = json.loads((project_setup.ROOT / 'data/ptbdb-files.json').read_text())
        self.assertEqual(len(manifest), 1651)
        for suffix in ['.hea', '.dat', '.xyz']:
            self.assertEqual(sum(name.endswith(suffix) for name in manifest), 549)
        for name in manifest:
            self.assertFalse(Path(name).is_absolute())
            self.assertNotIn('..', Path(name).parts)

    def test_output_does_not_change_notebook_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.ipynb'
            notebook = {'cells': [{'cell_type': 'code', 'source': ['x = 1'],
                'outputs': [], 'execution_count': None}]}
            path.write_text(json.dumps(notebook))
            original = file_digest(path)
            notebook['cells'][0]['outputs'] = [{'output_type': 'stream', 'text': 'example'}]
            notebook['cells'][0]['execution_count'] = 7
            path.write_text(json.dumps(notebook))
            self.assertEqual(original, file_digest(path))
            notebook['cells'][0]['source'] = ['x = 2']
            path.write_text(json.dumps(notebook))
            self.assertNotEqual(original, file_digest(path))

    def test_every_notebook_prepares_inputs(self):
        paths = sorted(project_setup.ROOT.glob('[0-9][0-9]_*.ipynb'))
        self.assertEqual(len(paths), 9)
        for number, path in enumerate(paths, 1):
            notebook = json.loads(path.read_text())
            first_code = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
            self.assertIn(f'prepare_notebook({number})', ''.join(first_code['source']))


if __name__ == '__main__':
    unittest.main()
