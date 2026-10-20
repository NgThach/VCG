"""Regression checks for frozen-V1 evaluation, denominators, and review exports.

Run directly with the notebook environment; no pytest or batch evaluation is
required. Only two real recordings are processed, without plots or artifacts.
"""
from collections import Counter
from dataclasses import asdict, FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import run_full_vcg_evaluation as runner
import vcg_pipeline as v1
from vcg_preprocessing import Config

DATA_ROOT = REPO / 'content/ptbdb/physionet.org/files/ptbdb/1.0.0'
LEGACY_FEATURES = REPO / 'vcg_12_features.csv'
HISTORICAL_FAILURES = ['patient007/s0029lre', 'patient010/s0042lre']
REFERENCE_FIELDS = ['qrs_on_ms', 'qrs_off_ms', 't_on_ms', 't_off_ms']


def metadata(record_id='patient104/s0306lre', label_name='HC'):
    return dict(patient_id=record_id.split('/')[0], record_id=record_id,
                split='train', diagnosis='Healthy control' if label_name == 'HC'
                else 'Myocardial infarction', label=int(label_name == 'MI'),
                label_name=label_name)


def review_fixture(manifest):
    """All attempted records, including failures and flagged candidate features."""
    frame = manifest.copy().reset_index(drop=True)
    index = np.arange(len(frame))
    frame['processing_success'] = index % 11 != 0
    frame['extraction_success'] = index % 13 != 0
    frame['qc_status'] = np.where(~frame.processing_success, 'failed',
                                 np.where(index % 3, 'needs_review', 'auto_candidate'))
    frame['qrs_qc_flags'] = np.where(index % 7 == 0, 'qrs_duration_or_anchor_requires_review', '')
    frame['t_qc_flags'] = np.where(index % 5 == 0, 't_tangent_threshold_disagreement', '')
    frame['qc_flags'] = [('; '.join(filter(None, [q, t])))
                         for q, t in zip(frame.qrs_qc_flags, frame.t_qc_flags)]
    frame.loc[~frame.processing_success, 'qc_flags'] = 'processing_failed; features_incomplete'
    frame['eligible_for_features'] = frame.qc_status.eq('auto_candidate')
    for key, value in zip(REFERENCE_FIELDS, [-60., 80., 180., 400.]):
        frame[key] = value
        frame.loc[~frame.processing_success, key] = np.nan
    frame['plot_path'] = 'plots/example_delineation.png'
    frame['loop_plot_path'] = 'plots/example_loops.png'
    for feature in v1.FEATURE_NAMES:
        frame[feature] = 1.0
        frame.loc[~frame.extraction_success, feature] = np.nan
    return frame


class TestFrozenV1Equivalence(unittest.TestCase):
    def test_staged_runner_matches_original_on_real_hc_and_mi(self):
        if not DATA_ROOT.exists():
            self.skipTest('Local PTB data is unavailable')
        cfg = Config()
        config_before = asdict(cfg)
        global_before = asdict(v1.CFG)
        hashes_before = {name: runner.sha256(REPO / name)
                         for name in ['vcg_pipeline.py', 'vcg_preprocessing.py']}
        for record_id in ['patient104/s0306lre', 'patient001/s0010_re']:
            with self.subTest(record=record_id):
                expected = v1.process_record(DATA_ROOT / record_id, cfg)
                trace = {}
                actual = runner.run_staged(DATA_ROOT / record_id, cfg, trace)
                self.assertEqual(trace['stage'], 'complete')
                self.assertEqual(actual['raw']['channels'], ['vx', 'vy', 'vz'])
                self.assertEqual(list(actual['features']), v1.FEATURE_NAMES)
                np.testing.assert_allclose(
                    [actual['features'][f] for f in v1.FEATURE_NAMES],
                    [expected['features'][f] for f in v1.FEATURE_NAMES],
                    rtol=1e-12, atol=1e-12, equal_nan=True)
                for key in ['status', 'eligible_for_features', 'flags', 'config']:
                    self.assertEqual(actual[key], expected[key], key)
                for key in ['morph', 'detection_xyz', 'qrs_loop', 't_loop']:
                    if expected[key] is None:
                        self.assertIsNone(actual[key], key)
                    else:
                        np.testing.assert_allclose(actual[key], expected[key],
                                                   rtol=1e-12, atol=1e-12, err_msg=key)
                for key in ['anchors', 'rr', 'envelope', 'threshold']:
                    np.testing.assert_array_equal(actual['detector'][key],
                                                  expected['detector'][key], err_msg=key)
                for key in ['xyz', 'time_ms', 'used_samples']:
                    np.testing.assert_array_equal(actual['rep'][key], expected['rep'][key], err_msg=key)
                for key in ['anchor', 'n_used', 'n_excluded', 'pq', 'quality_flags']:
                    self.assertEqual(actual['rep'][key], expected['rep'][key], key)
                for key in runner.BOUNDARIES + ['t_method', 't_tangent_on', 't_tangent_off',
                                                't_threshold_on', 't_threshold_off']:
                    self.assertEqual(actual['boundaries'][key], expected['boundaries'][key], key)
                for key in ['speed', 'smooth_speed', 'magnitude', 't_magnitude']:
                    np.testing.assert_array_equal(actual['boundaries'][key],
                                                  expected['boundaries'][key], err_msg=key)
                # Preserve the native full-template SG derivative before loop cutting.
                # Recomputing a derivative on a cropped loop changes its edge values.
                b = actual['boundaries']
                for name, on, off in [('QRS', 'qrs_on', 'qrs_off'), ('T', 't_on', 't_off')]:
                    if actual['qrs_loop' if name == 'QRS' else 't_loop'] is None:
                        continue
                    speed = b['speed'][b[on]:b[off] + 1]
                    for suffix, value in [('MaxVel', speed.max()), ('MeanVel', speed.mean()),
                                          ('StdVel', speed.std())]:
                        np.testing.assert_allclose(actual['features'][name + '_' + suffix], value,
                                                   rtol=1e-12, atol=1e-12)
                native_qc = v1.qc_row(expected)
                row = runner.describe_result(metadata(record_id,
                    'HC' if 'patient104/' in record_id else 'MI'), trace, actual)
                for key in runner.BOUNDARIES:
                    np.testing.assert_allclose(row[key + '_ms'], native_qc[key + '_ms'],
                                               equal_nan=True)
                self.assertEqual(row['n_qrs_detected'], native_qc['n_qrs_detected'])
                self.assertEqual(row['n_beats_used'], native_qc['n_beats_used'])
                self.assertEqual(row['qc_status'], native_qc['status'])
                self.assertEqual(row['qc_flags'], native_qc['flags'])
        self.assertEqual(asdict(cfg), config_before)
        self.assertEqual(asdict(v1.CFG), global_before)
        self.assertEqual(hashes_before, {name: runner.sha256(REPO / name) for name in hashes_before})
        with self.assertRaises(FrozenInstanceError):
            cfg.energy_window_ms = 60.


class TestCohortAndReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_ROOT.exists() or not LEGACY_FEATURES.exists():
            raise unittest.SkipTest('Local PTB data and historical feature table are required')
        cls.inventory = runner.discover_records(DATA_ROOT)
        cls.manifest, cls.details = runner.make_split(cls.inventory, LEGACY_FEATURES)

    def test_original_test_assignment_and_all_448_records_are_retained(self):
        legacy = pd.read_csv(LEGACY_FEATURES)
        patients = legacy[['patient_id', 'label']].drop_duplicates()
        _, legacy_test = train_test_split(patients, test_size=.25, random_state=42,
                                          stratify=patients.label)
        manifest = self.manifest
        self.assertEqual(len(self.inventory), 549)
        self.assertEqual(len(manifest), 448)
        self.assertEqual(manifest.patient_id.nunique(), 200)
        self.assertEqual(manifest.label_name.value_counts().to_dict(), {'MI': 368, 'HC': 80})
        self.assertEqual(set(manifest.loc[manifest.split.eq('test'), 'patient_id']),
                         set(legacy_test.patient_id))
        self.assertEqual(manifest.groupby('split').size().to_dict(),
                         {'test': 115, 'train': 263, 'val': 70})
        self.assertEqual(manifest.groupby('split').patient_id.nunique().to_dict(),
                         {'test': 50, 'train': 120, 'val': 30})
        self.assertTrue(manifest.groupby('patient_id').split.nunique().eq(1).all())
        sets = {k: set(g.patient_id) for k, g in manifest.groupby('split')}
        self.assertFalse(sets['train'] & sets['val'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['val'] & sets['test'])
        failures = manifest.set_index('record_id').loc[HISTORICAL_FAILURES]
        self.assertTrue(failures.split.eq('train').all())
        self.assertTrue(failures.label_name.eq('MI').all())
        self.assertEqual(set(manifest.record_id),
                         set(self.inventory.loc[self.inventory.included, 'record_id']))

    def test_qc_flags_and_failures_do_not_change_split_or_export_denominator(self):
        flagged = self.inventory.copy()
        flagged['qc_status'] = 'failed'
        flagged['processing_success'] = False
        flagged['eligible_for_features'] = False
        flagged['qrs_boundary_valid'] = False
        flagged['t_boundary_valid'] = False
        manifest, _ = runner.make_split(flagged, LEGACY_FEATURES)
        pd.testing.assert_frame_equal(
            manifest[['record_id', 'patient_id', 'split']].reset_index(drop=True),
            self.manifest[['record_id', 'patient_id', 'split']].reset_index(drop=True))
        frame = review_fixture(self.manifest)
        with tempfile.TemporaryDirectory(prefix='vcg_export_test_') as temp:
            out = Path(temp)
            for folder in ['features', 'evaluation', 'annotations']:
                (out / folder).mkdir()
            exported = runner.write_tables(out, frame)
            stored = pd.read_csv(out / 'features/all_features.csv')
            self.assertEqual(len(exported), 448)
            self.assertEqual(set(stored.record_id), set(self.manifest.record_id))
            self.assertEqual(int((~stored.processing_success).sum()),
                             int((~frame.processing_success).sum()))
            self.assertEqual(len(pd.read_csv(out / 'evaluation/failed_records.csv')),
                             int((~frame.processing_success).sum()))
            for split in ['train', 'val', 'test']:
                part = pd.read_csv(out / f'features/{split}_features.csv')
                self.assertEqual(set(part.record_id), set(frame.loc[frame.split.eq(split), 'record_id']))
            self.assertEqual(sum(len(pd.read_csv(out / f'features/{s}_features.csv'))
                                 for s in ['train', 'val', 'test']), 448)
            self.assertTrue(pd.read_csv(out / 'evaluation/boundary_reference_comparison.csv').empty)

    def assert_review_is_unlabeled(self, sample, source):
        self.assertFalse(sample.record_id.duplicated().any())
        self.assertTrue(sample.groupby('label_name').size().ge(30).all())
        self.assertEqual(set(sample.label_name), {'HC', 'MI'})
        self.assertTrue(sample[REFERENCE_FIELDS].isna().all().all())
        self.assertTrue(sample.reference_anchor_sample.isna().all())
        self.assertTrue(sample.reviewer.fillna('').eq('').all())
        self.assertTrue(sample.review_status.eq('pending').all())
        self.assertTrue(sample.independent_reference.eq(False).all())
        self.assertTrue(sample.time_reference.eq('median_relative_ms').all())
        self.assertTrue(set(sample.record_id).issubset(set(source.record_id)))
        indexed = source.set_index('record_id')
        for reference in REFERENCE_FIELDS:
            expected = indexed.loc[sample.record_id, reference].to_numpy(dtype=float)
            np.testing.assert_allclose(sample['pred_' + reference].to_numpy(dtype=float), expected,
                                       equal_nan=True)

    def test_review_sample_keeps_reference_blank_and_predictions_separate(self):
        frame = review_fixture(self.manifest)
        sample = runner.review_sample(frame, per_label=30)
        self.assert_review_is_unlabeled(sample, frame)
        # At least 30 patients are available per label; diversity should precede repeats.
        self.assertTrue(sample.groupby('label_name').patient_id.nunique().ge(30).all())
        sample_again = runner.review_sample(frame, per_label=30)
        pd.testing.assert_frame_equal(sample, sample_again)

    def test_full_cohort_review_artifact_if_available(self):
        output = REPO / 'outputs/vcg_full_evaluation'
        features = output / 'features/all_features.csv'
        template = output / 'annotations/manual_review_template.csv'
        if not features.exists() or not template.exists():
            self.skipTest('Full-cohort output is not yet available; fixture checks still run')
        frame = pd.read_csv(features)
        if len(frame) != 448:
            self.skipTest('Output currently contains a smoke subset rather than the full cohort')
        self.assert_review_is_unlabeled(pd.read_csv(template), frame)


class TestPartialFailureReporting(unittest.TestCase):
    def test_representative_failure_retains_detection_and_input_diagnostics(self):
        raw = {'xyz': np.zeros((5000, 3)), 'fs': 1000.}
        prep = {'input_quality': {'flags': ['flat_input_review']},
                'notch_applied': False, 'notch_mode': 'auto',
                'powerline_status': 'inconclusive', 'powerline_score_db': np.nan}
        detector = {'anchors': np.array([500, 700, 1700, 4200]),
                    'rr': np.array([.2, 1., 2.5]),
                    'flags': ['rr_extreme_or_missed_beat', 'rr_irregular'],
                    'envelope': np.zeros(5000), 'threshold': np.zeros(5000)}
        trace = {}
        with patch.object(v1, 'load_xyz', return_value=raw), \
             patch.object(v1, 'preprocess_xyz', return_value=(raw['xyz'], raw['xyz'], prep)), \
             patch.object(v1, 'detect_qrs_anchors', return_value=detector), \
             patch.object(v1, 'representative_beat', side_effect=ValueError('No representative beat')):
            with self.assertRaisesRegex(ValueError, 'No representative beat'):
                runner.run_staged(Path('mock-record'), Config(), trace)
        row = runner.describe_result(metadata(), trace, error='ValueError: No representative beat')
        self.assertEqual(row['failure_stage'], 'representative_beat')
        self.assertFalse(row['processing_success'])
        self.assertFalse(row['extraction_success'])
        self.assertFalse(row['eligible_for_features'])
        self.assertEqual(row['n_qrs_detected'], 4)
        self.assertEqual(row['median_rr_ms'], 1000.)
        self.assertEqual(row['rr_short_count'], 1)
        self.assertEqual(row['rr_long_count'], 1)
        self.assertEqual(row['suspected_missed_rr_count'], 1)
        self.assertEqual(row['suspected_extra_rr_count'], 1)
        self.assertEqual(row['qrs_detection_flags'], '; '.join(detector['flags']))
        self.assertIn('rr_irregular', row['qc_flags'])
        self.assertIn('processing_failed', row['qc_flags'])
        self.assertEqual(row['input_qc_flags'], 'flat_input_review')
        self.assertEqual(row['powerline_status'], 'inconclusive')
        self.assertEqual(row['feature_completeness'], 0)
        self.assertTrue(np.isnan([row[f] for f in v1.FEATURE_NAMES]).all())
        reasons = json.loads(row['feature_missing_reasons'])
        self.assertEqual(set(reasons), {'QRS', 'T'})
        self.assertTrue(all('representative_beat' in reason for reason in reasons.values()))

    def test_real_failed_record_preserves_native_beat_exclusions(self):
        if not DATA_ROOT.exists():
            self.skipTest('Local PTB data is unavailable')
        record_id = 'patient033/s0157lre'
        cfg = Config()
        native = None
        # An uncaught V1 failure retains its original frame in the traceback.
        # This independent route checks the staged observer against actual V1
        # diagnostics, rather than constructing the expected exclusion rows.
        try:
            v1.process_record(DATA_ROOT / record_id, cfg)
        except ValueError as exc:
            tb = exc.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code is v1.representative_beat.__code__:
                    local = tb.tb_frame.f_locals
                    native = dict(beat_qc=[dict(r) for r in local['rows']],
                                  n_used=len(local['beats']),
                                  n_excluded=len(local['rows']) - len(local['beats']),
                                  used_samples=list(local['kept']))
                    break
                tb = tb.tb_next
        self.assertIsNotNone(native, 'Expected this frozen-V1 recording to fail at representative beat')
        self.assertEqual(native['n_used'], 0)
        self.assertEqual(native['n_excluded'], 58)
        self.assertEqual(Counter(r['reason'] for r in native['beat_qc']),
                         {'rr_outlier': 56, 'edge': 2})
        trace = {}
        previous_trace = sys.gettrace()
        with self.assertRaisesRegex(ValueError, 'Không đủ 3 beat'):
            runner.run_staged(DATA_ROOT / record_id, cfg, trace)
        self.assertIs(sys.gettrace(), previous_trace)
        partial = trace['representative_failure_diagnostics']
        self.assertEqual(partial['n_used'], native['n_used'])
        self.assertEqual(partial['n_excluded'], native['n_excluded'])
        self.assertEqual(partial['used_samples'], native['used_samples'])
        pd.testing.assert_frame_equal(pd.DataFrame(partial['beat_qc']),
                                      pd.DataFrame(native['beat_qc']))
        row = runner.describe_result(metadata(record_id, 'MI'), trace,
                                     error='ValueError: Không đủ 3 beat')
        self.assertEqual(row['failure_stage'], 'representative_beat')
        self.assertEqual(row['n_qrs_detected'], 58)
        self.assertEqual(row['n_beats_used'], 0)
        self.assertEqual(row['n_beats_excluded'], 58)
        self.assertEqual(row['excluded_beat_fraction'], 1.)
        self.assertFalse(row['processing_success'])
        self.assertFalse(row['eligible_for_features'])
        with tempfile.TemporaryDirectory(prefix='vcg_failure_trace_test_') as temp:
            out = Path(temp)
            (out / 'artifacts').mkdir()
            runner.save_trace(out, row, trace, None)
            saved = json.loads((out / 'artifacts/patient033_s0157lre.json').read_text())
            self.assertEqual(saved['beat_qc'], runner.jsonable(native['beat_qc']))
            self.assertEqual(saved['n_used'], 0)
            self.assertEqual(saved['n_excluded'], 58)
            self.assertEqual(saved['used_samples'], [])

    def test_partial_features_are_missing_not_zero_and_legitimate_zero_counts(self):
        features = {name: float(i) if i < 7 else np.nan
                    for i, name in enumerate(v1.FEATURE_NAMES)}
        result = {'features': features, 'status': 'needs_review',
                  'flags': ['features_incomplete'], 'eligible_for_features': False}
        row = runner.describe_result(metadata(), {}, result)
        self.assertTrue(row['processing_success'])
        self.assertFalse(row['extraction_success'])
        self.assertEqual(row['feature_completeness'], 7)
        self.assertEqual(row[v1.FEATURE_NAMES[0]], 0.)
        self.assertTrue(np.isnan([row[f] for f in v1.FEATURE_NAMES[7:]]).all())
        self.assertEqual(set(json.loads(row['feature_missing_reasons'])), {'T'})
        serialized = runner.jsonable(row)
        self.assertEqual(serialized[v1.FEATURE_NAMES[0]], 0.)
        self.assertTrue(all(serialized[f] is None for f in v1.FEATURE_NAMES[7:]))


class TestCheckpointAndResume(unittest.TestCase):
    def test_json_checkpoint_replacement_is_atomic_and_nan_becomes_null(self):
        with tempfile.TemporaryDirectory(prefix='vcg_atomic_test_') as temp:
            path = Path(temp) / 'record.json'
            runner.write_json(path, {'attempt': 1})
            with patch.object(Path, 'replace', side_effect=OSError('Simulated rename failure')):
                with self.assertRaisesRegex(OSError, 'Simulated rename failure'):
                    runner.write_json(path, {'attempt': 2})
            self.assertEqual(json.loads(path.read_text()), {'attempt': 1})
            runner.write_json(path, {'attempt': 3, 'missing': np.nan, 'array': np.array([np.inf, 0.])})
            self.assertEqual(json.loads(path.read_text()),
                             {'attempt': 3, 'missing': None, 'array': [None, 0.]})
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_resume_accepts_identical_run_and_blocks_source_config_inventory_or_split_change(self):
        with tempfile.TemporaryDirectory(prefix='vcg_resume_test_') as temp:
            root = Path(temp)
            source_root = root / 'tiny_source'
            source_root.mkdir()
            (source_root / 'scripts').mkdir()
            source = source_root / 'frozen_v1.py'
            source.write_text('# tiny source used solely for resume guard testing\n')
            engine = source_root / 'scripts/tiny_evaluator.py'
            engine.write_text('# tiny evaluation engine used only for resume guard testing\n')
            inventory = pd.DataFrame([dict(record_id='patient001/example', included=True)])
            manifest = pd.DataFrame([dict(record_id='patient001/example', patient_id='patient001',
                                          label_name='MI', split='train')])
            args = SimpleNamespace(output_dir=root / 'run', data_root=root / 'data',
                                   seconds=30., resume=False)
            with patch.object(runner, 'REPO', source_root), \
                 patch.object(runner, 'SOURCES', ['frozen_v1.py']), \
                 patch.object(runner, 'ENGINE_SOURCES', ['scripts/tiny_evaluator.py']):
                out, original = runner.initialize_run(args, inventory, manifest, {'method': 'test only'})
                self.assertEqual((out / 'baseline_v1/frozen_v1.py').read_text(), source.read_text())
                self.assertEqual((out / 'baseline_v1/scripts/tiny_evaluator.py').read_text(), engine.read_text())
                with self.assertRaises(FileExistsError):
                    runner.initialize_run(args, inventory, manifest, {})
                args.resume = True
                resumed_out, resumed = runner.initialize_run(args, inventory, manifest, {})
                self.assertEqual(resumed_out, out)
                self.assertEqual(runner.jsonable(original), runner.jsonable(resumed))
                for change in ['config', 'source', 'engine', 'inventory', 'split']:
                    with self.subTest(change=change):
                        changed_inventory, changed_manifest = inventory.copy(), manifest.copy()
                        original_source = source.read_text()
                        original_engine = engine.read_text()
                        if change == 'config':
                            args.seconds = 31.
                        elif change == 'source':
                            source.write_text(original_source + '# changed\n')
                        elif change == 'engine':
                            engine.write_text(original_engine + '# changed\n')
                        elif change == 'inventory':
                            changed_inventory.loc[0, 'included'] = False
                        else:
                            changed_manifest.loc[0, 'split'] = 'test'
                        try:
                            with self.assertRaisesRegex(ValueError, 'Resume blocked'):
                                runner.initialize_run(args, changed_inventory, changed_manifest, {})
                        finally:
                            args.seconds = 30.
                            source.write_text(original_source)
                            engine.write_text(original_engine)
                real_version = runner.importlib.metadata.version
                with patch.object(runner.importlib.metadata, 'version', side_effect=lambda package:
                                  'changed-for-test' if package == 'numpy' else real_version(package)):
                    with self.assertRaisesRegex(ValueError, 'Resume blocked'):
                        runner.initialize_run(args, inventory, manifest, {})
                self.assertEqual(json.loads((out / 'config.json').read_text()), runner.jsonable(original))


if __name__ == '__main__':
    unittest.main(verbosity=2)
