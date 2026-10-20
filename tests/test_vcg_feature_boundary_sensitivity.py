"""Numerical/denominator regression checks for the Notebook 07 experiment."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import vcg_pipeline as v1
import vcg_feature_boundary_sensitivity as sensitivity
from vcg_preprocessing import Config


def fixture(root, record_id='patient001/synthetic', missing_t=False, no_median=False,
            boundaries_override=None):
    fs, anchor = 1000.0, 140
    tt = np.arange(401) / fs
    xyz = np.column_stack([np.sin(2 * np.pi * 17 * tt) + .2 * np.cos(2 * np.pi * 71 * tt),
                           .7 * np.sin(2 * np.pi * 11 * tt + .3),
                           .4 * np.cos(2 * np.pi * 23 * tt + .1)])
    bounds = dict(qrs_on=90, qrs_off=180, t_on=220, t_peak=260, t_off=300)
    bounds.update(boundaries_override or {})
    if missing_t:
        bounds.update(t_on=None, t_peak=None, t_off=None)
    row = dict(record_id=record_id, patient_id=record_id.split('/')[0],
               label_name='HC', label=0, split='train', fs=fs,
               qc_status='needs_review', qc_flags='features_incomplete' if
               missing_t or no_median else '', feature_completeness=0 if no_median else
               7 if missing_t else 12)
    if no_median:
        features = dict.fromkeys(v1.FEATURE_NAMES, np.nan)
        bounds = dict.fromkeys(bounds)
    else:
        qrs = v1.cut_loop(xyz, bounds['qrs_on'], bounds['qrs_off'])
        tloop = v1.cut_loop(xyz, bounds['t_on'], bounds['t_off'])
        full_speed = v1.spatial_speed(xyz, fs)
        speeds = dict(QRS=full_speed[bounds['qrs_on']:bounds['qrs_off'] + 1],
                      T=None if missing_t else full_speed[bounds['t_on']:bounds['t_off'] + 1])
        features = v1.extract_features(qrs, tloop, fs, speeds=speeds)
    row.update(features)
    for key, ix in bounds.items():
        row[key + '_ms'] = np.nan if ix is None else (ix - anchor) * 1000 / fs
    artifact = dict(record_id=record_id, fs=fs, time_reference='median_relative_ms',
                    boundaries=bounds, failure_stage='representative' if no_median else '')
    if not no_median:
        artifact['median_anchor_sample'] = anchor
    safe = record_id.replace('/', '_')
    (root / (safe + '.json')).write_text(json.dumps(artifact))
    if not no_median:
        np.savez_compressed(root / (safe + '.npz'), median_xyz=xyz,
                            time_ms=(np.arange(len(xyz)) - anchor) * 1000 / fs)
    return row, xyz, bounds, fs, anchor


class TestFeatureSensitivity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.output = self.root / 'output'

    def tearDown(self):
        self.temp.cleanup()

    def test_baseline_and_shift_use_full_median_derivative(self):
        row, xyz, bounds, fs, _ = fixture(self.artifacts)
        result = sensitivity.run_feature_sensitivity(pd.DataFrame([row]), self.artifacts,
                                                     self.output)
        self.assertEqual(len(result['scenarios']), 16)
        self.assertEqual(len(result['long']), 192)
        self.assertTrue(result['baseline_check'].matches_csv.all())
        shifted = dict(bounds, qrs_on=bounds['qrs_on'] + 20)
        full_speed = v1.spatial_speed(xyz, fs)
        loop = v1.cut_loop(xyz, shifted['qrs_on'], shifted['qrs_off'])
        crop_speed = v1.spatial_speed(loop, fs)
        correct_speed = full_speed[shifted['qrs_on']:shifted['qrs_off'] + 1]
        # The fixture makes SG endpoint treatment consequential.
        self.assertGreater(abs(crop_speed[0] - correct_speed[0]), .01)
        selected = result['long'].query('boundary == "qrs_on" and shift_ms == 20')
        got = selected.set_index('feature').perturbed
        self.assertAlmostEqual(got['QRS_MeanVel'], correct_speed.mean(), places=12)
        self.assertAlmostEqual(got['QRS_StdVel'], correct_speed.std(), places=12)
        self.assertNotAlmostEqual(got['QRS_MeanVel'], crop_speed.mean(), places=5)
        self.assertEqual(got['T_MaxVec'], row['T_MaxVec'])
        rank = result['rank'].query('scope == "all" and grouping == "all"')
        self.assertTrue(rank.n_scenarios_total.eq(8).all())

    def test_invalid_order_and_range_are_rejected_without_repair(self):
        _, _, b, fs, anchor = fixture(self.artifacts)
        invalid = [
            (dict(b, qrs_on=anchor), 'qrs_on', 'qrs_on_anchor_qrs_off_order'),
            (dict(b, qrs_on=-1), 'qrs_on', 'index_out_of_range:qrs_on'),
            (dict(b, qrs_off=b['t_on']), 'qrs_off', 'qrs_off_t_on_order'),
            (dict(b, t_on=b['t_peak']), 't_on', 't_on_t_peak_order'),
            (dict(b, t_off=b['t_peak']), 't_off', 't_peak_t_off_order')]
        for shifted, boundary, expected in invalid:
            with self.subTest(boundary=boundary, expected=expected):
                valid, reason, _ = sensitivity.validate_scenario(shifted, 401, anchor,
                                                                fs, boundary)
                self.assertFalse(valid)
                self.assertEqual(reason, expected)
        # 20 ms is valid technically: no unrequested 60 ms minimum is imposed.
        valid, _, _ = sensitivity.validate_scenario(dict(b, t_on=250, t_off=270),
                                                     401, anchor, fs, 't_on')
        self.assertTrue(valid)
        valid, reason, _ = sensitivity.validate_scenario(dict(b, t_on=256, t_off=264),
                                                          401, anchor, fs, 't_on')
        self.assertFalse(valid)
        self.assertEqual(reason, 't_loop_too_short_for_v1_features')

    def test_zero_and_near_zero_keep_absolute_change_and_declared_epsilon(self):
        absolute, relative, near_zero = sensitivity.feature_change(0, 2e-9)
        self.assertEqual(absolute, 2e-9)
        self.assertEqual(relative, 2)
        self.assertTrue(near_zero)
        absolute, relative, near_zero = sensitivity.feature_change(-.5e-9, 1.5e-9)
        self.assertEqual(absolute, 2e-9)
        self.assertAlmostEqual(relative, 4 / 3)
        self.assertTrue(near_zero)
        absolute, relative, near_zero = sensitivity.feature_change(None, 2e-9)
        self.assertTrue(np.isnan(absolute))
        self.assertTrue(np.isnan(relative))
        self.assertFalse(near_zero)

    def test_invalid_shift_keeps_other_boundaries_and_rejected_denominator(self):
        row, _, b, _, _ = fixture(self.artifacts, boundaries_override={'t_on': 195})
        result = sensitivity.run_feature_sensitivity(pd.DataFrame([row]), self.artifacts,
                                                     self.output)
        invalid = result['scenarios'].query('boundary == "qrs_off" and shift_ms == 20').iloc[0]
        self.assertFalse(invalid.valid_scenario)
        self.assertEqual(invalid.reason, 'qrs_off_t_on_order')
        self.assertEqual(invalid.qrs_off_index, b['qrs_off'] + 20)
        for name in ('qrs_on', 't_on', 't_peak', 't_off'):
            self.assertEqual(invalid[name + '_index'], b[name])
        rejected_features = result['long'].query('boundary == "qrs_off" and shift_ms == 20')
        self.assertEqual(len(rejected_features), 12)
        self.assertTrue(rejected_features.perturbed.isna().all())
        self.assertTrue(rejected_features.absolute_delta.isna().all())
        rank = result['rank'].query('scope == "all" and grouping == "all" and '
                                   'feature == "QRS_Arc"').iloc[0]
        self.assertEqual(rank.n_scenarios_total, 8)
        self.assertEqual(rank.n_eligible, 8)
        self.assertEqual(rank.n_valid, 7)
        self.assertEqual(rank.n_eligible_but_invalid, 1)

    def test_missing_median_and_missing_t_keep_every_attempt(self):
        a, *_ = fixture(self.artifacts, 'patient001/no_median', no_median=True)
        b, *_ = fixture(self.artifacts, 'patient002/no_t', missing_t=True)
        b.update(label_name='MI', label=1, split='valid')
        result = sensitivity.run_feature_sensitivity(pd.DataFrame([a, b]), self.artifacts,
                                                     self.output)
        self.assertEqual(len(result['scenarios']), 32)
        self.assertEqual(len(result['long']), 384)
        none = result['scenarios'].query('record_id == "patient001/no_median"')
        self.assertFalse(none.valid_scenario.any())
        self.assertTrue(none.reason.str.startswith('median_beat_unavailable').all())
        partial = result['scenarios'].query('record_id == "patient002/no_t"')
        self.assertEqual(int(partial.valid_scenario.sum()), 8)
        self.assertTrue(partial.loc[partial.valid_scenario, 'validation_scope'].eq(
            'qrs_only_missing_t').all())
        qrs_rank = result['rank'].query('scope == "all" and grouping == "all" and '
                                       'feature == "QRS_Arc"').iloc[0]
        self.assertEqual(qrs_rank.n_records_total, 2)
        self.assertEqual(qrs_rank.n_records_eligible, 1)
        self.assertEqual(qrs_rank.n_scenarios_total, 16)
        self.assertEqual(qrs_rank.n_valid, 8)
        self.assertEqual(qrs_rank.n_unavailable, 8)
        t_rank = result['rank'].query('scope == "all" and grouping == "all" and '
                                     'feature == "T_Area"').iloc[0]
        self.assertEqual(t_rank.n_scenarios_total, 16)
        self.assertEqual(t_rank.n_valid, 0)

    def test_bad_baseline_is_not_silently_used(self):
        row, *_ = fixture(self.artifacts)
        row['QRS_Arc'] += .1
        with self.assertRaisesRegex(ValueError, 'baseline differs'):
            sensitivity.run_feature_sensitivity(pd.DataFrame([row]), self.artifacts,
                                                 self.output)
        self.assertTrue((self.output / 'sensitivity_baseline_check.csv').exists())

    def test_resume_is_equal_and_config_and_row_changes_invalidate_it(self):
        row, *_ = fixture(self.artifacts)
        frame = pd.DataFrame([row])
        cold = sensitivity.run_feature_sensitivity(frame, self.artifacts, self.output)
        warm = sensitivity.run_feature_sensitivity(frame, self.artifacts, self.output)
        self.assertEqual(warm['metadata']['cache_hits'], 1)
        for key in ('long', 'scenarios', 'baseline_check', 'summary', 'rank'):
            pd.testing.assert_frame_equal(cold[key], warm[key])
        changed = frame.copy()
        changed['split'] = 'test'
        new = sensitivity.run_feature_sensitivity(changed, self.artifacts, self.output)
        self.assertEqual(new['metadata']['cache_hits'], 0)
        # A configuration field unrelated to feature math must still invalidate cache.
        newer = sensitivity.run_feature_sensitivity(changed, self.artifacts, self.output,
                                                    cfg=replace(Config(), quiet_t_ms=21))
        self.assertEqual(newer['metadata']['cache_hits'], 0)
        self.assertNotEqual(new['metadata']['experiment_hash'],
                            newer['metadata']['experiment_hash'])
        json_path = self.artifacts / 'patient001_synthetic.json'
        artifact = json.loads(json_path.read_text())
        artifact['test_note'] = 'saved diagnostic changed'
        json_path.write_text(json.dumps(artifact))
        artifact_changed = sensitivity.run_feature_sensitivity(
            changed, self.artifacts, self.output, cfg=replace(Config(), quiet_t_ms=21))
        self.assertEqual(artifact_changed['metadata']['cache_hits'], 0)


if __name__ == '__main__':
    unittest.main()
