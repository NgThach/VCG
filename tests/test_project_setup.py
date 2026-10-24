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
        self.assertEqual(len(paths), 11)
        for number, path in enumerate(paths, 1):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            first_code = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
            if number <= 9:
                self.assertIn(f'prepare_notebook({number})', ''.join(first_code['source']))
            elif number == 10:
                # V3 reads raw independently; it must not force V1/V2 output generation.
                code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
                self.assertIn('CX_DATA_ROOT', code)
                self.assertIn('CX_DOWNLOAD_MISSING=False', code)
                self.assertIn('load_xyz(CX_DATA_ROOT', code)
            else:
                # V4 is a self-contained J-point -> T-end cohort experiment.
                code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
                self.assertIn('V4_DATA', code)
                self.assertIn('V4_DOWNLOAD_MISSING = False', code)
                self.assertIn('V4_RUN_TEST = True', code)
                self.assertIn('load_xyz(V4_DATA', code)

    def test_prerequisite_process_uses_utf8(self):
        with patch.object(project_setup.subprocess, 'run') as run:
            project_setup.run('example.py')
        env = run.call_args.kwargs['env']
        self.assertEqual(env['PYTHONUTF8'], '1')
        self.assertEqual(env['PYTHONIOENCODING'], 'utf-8')

    def test_verified_artifact_cache_does_not_resume_historical_evaluation(self):
        from dataclasses import asdict
        import numpy as np
        from vcg_preprocessing import Config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'outputs/vcg_full_evaluation'
            artifacts = folder / 'artifacts'
            artifacts.mkdir(parents=True)
            (root / 'data').mkdir()
            manifest = 'record_id,patient_id,split,label_name\npatient001/example,patient001,train,HC\n'
            (root / 'data/split_manifest.csv').write_text(manifest, encoding='utf-8')
            (folder / 'split_manifest.csv').write_text(manifest, encoding='utf-8')
            names = ('vcg_pipeline.py', 'vcg_preprocessing.py')
            for name in names:
                (root / name).write_text('# fixture numerical source\n', encoding='utf-8')
            config = dict(config=asdict(Config()), baseline_hashes={
                name: project_setup.digest(root / name) for name in names})
            (folder / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            artifact = dict(record_id='patient001/example', fs=1000,
                time_reference='median_relative_ms', median_anchor_sample=40, boundaries={})
            jp = artifacts / 'patient001_example.json'
            jp.write_text(json.dumps(artifact), encoding='utf-8')
            np.savez_compressed(artifacts / 'patient001_example.npz',
                median_xyz=np.zeros((120, 3)), time_ms=np.arange(120)-40.)
            with patch.object(project_setup, 'ROOT', root):
                self.assertFalse(project_setup.full_v1_ready())
                self.assertTrue(project_setup.cached_v1_inputs_ready())
                with patch.object(project_setup, 'run') as run:
                    project_setup.ensure_v1()
                    run.assert_not_called()
                artifact['record_id'] = 'wrong/identity'
                jp.write_text(json.dumps(artifact), encoding='utf-8')
                self.assertFalse(project_setup.cached_v1_inputs_ready())
                artifact['record_id'] = 'patient001/example'
                jp.write_text(json.dumps(artifact), encoding='utf-8')
                (folder / 'split_manifest.csv').write_text(manifest.replace('train', 'test'), encoding='utf-8')
                self.assertFalse(project_setup.cached_v1_inputs_ready())
                (folder / 'split_manifest.csv').write_text(manifest, encoding='utf-8')
                (root / names[0]).write_text('# changed numerical code\n', encoding='utf-8')
                self.assertFalse(project_setup.cached_v1_inputs_ready())


if __name__ == '__main__':
    unittest.main()
