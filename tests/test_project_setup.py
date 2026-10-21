"""Distribution checks independent of the author's data directory."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project_integrity import file_digest
import project_setup


class DistributionTests(unittest.TestCase):
    def test_release_parts_cover_full_archive(self):
        resource = json.loads((project_setup.ROOT / 'data/resources.json').read_text())['ptbdb']
        self.assertEqual(sum(part['bytes'] for part in resource['parts']), resource['bytes'])
        self.assertEqual(len(resource['parts']), 41)
        self.assertTrue(all(part['bytes'] <= 32 * 1024 * 1024 for part in resource['parts']))

    def test_download_parts_join_in_order_and_check_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            contents = [b'first', b'second', b'third']
            combined = b''.join(contents)
            parts = [{'filename': f'part{i}', 'url': f'https://example.org/{i}',
                      'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                     for i, content in enumerate(contents)]
            resource = {'parts': parts, 'bytes': len(combined),
                        'sha256': hashlib.sha256(combined).hexdigest()}
            def fake_download(url, path, expected):
                path.write_bytes(contents[int(url.rsplit('/', 1)[1])])
            with patch.object(project_setup, 'download', side_effect=fake_download):
                project_setup.download_bundle(resource, cache, cache / 'bundle.tar.gz')
            self.assertEqual((cache / 'bundle.tar.gz').read_bytes(), combined)
            self.assertFalse(any(cache.glob('part*')))

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
