"""Isolated synthetic Mode A integration; never touches real annotations.

The fixtures exercise the evaluation/report/plot contract, not clinical accuracy.
Synthetic references are declared fixture coordinates, not reviewer labels for
any PTB recording. Every file is written under TemporaryDirectory.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / 'scripts'):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
import vcg_pipeline as v1
from vcg_preprocessing import Config
from scripts import vcg_reference_metrics as engine
from scripts.vcg_reference_plots import create_reference_plots
from scripts.vcg_boundary_reference_eval import save_results, write_report


class ReferenceEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vcg_reference08_integration_')
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.output = self.root / 'output'
        self.output.mkdir()
        self.cfg = Config()
        self.time = np.arange(220, dtype=float) - 50
        # Numerically distinct full-median derivatives expose accidental recuts.
        u = np.arange(220) / 20
        self.xyz = np.column_stack((np.sin(u), .3*np.cos(1.3*u), .2*np.sin(.7*u)))
        self.rows, self.annotations = [], []
        definitions = [
            ('synthetic001/full', 'synthetic001', 'HC', 'train', 'full'),
            ('synthetic001/partial', 'synthetic001', 'HC', 'train', 'partial'),
            ('synthetic002/full', 'synthetic002', 'MI', 'valid', 'full'),
            ('synthetic003/no_prediction', 'synthetic003', 'MI', 'valid', 'missing_prediction'),
            ('synthetic004/test_closed', 'synthetic004', 'HC', 'test', 'full'),
        ]
        for rid, patient, label, split, kind in definitions:
            self._add_record(rid, patient, label, split, kind)
        self.frame = pd.DataFrame(self.rows)
        self.priority = self.frame[['record_id', 'patient_id', 'label_name', 'split']].copy()
        self.priority['priority_reasons'] = 'synthetic_fixture'
        self.priority['priority_score'] = 1
        self.priority['selected_notebook07_review'] = True
        self.priority.to_csv(self.output / 'priority_review_cases.csv', index=False)
        pd.DataFrame([{'source': 'synthetic fixtures only', 'source_type': 'test fixtures'}]).to_csv(
            self.output / 'reference_availability_report.csv', index=False)

    def tearDown(self):
        self.temp.cleanup()

    def _add_record(self, rid, patient, label, split, kind):
        indices = dict(qrs_on=30, qrs_off=70, t_on=110, t_off=170)
        if kind == 'missing_prediction':
            indices = dict.fromkeys(engine.BOUNDARIES)
        safe = rid.replace('/', '_')
        npz = self.artifacts / (safe + '.npz')
        np.savez_compressed(npz, median_xyz=self.xyz, time_ms=self.time)
        artifact = {'record_id': rid, 'fs': 1000., 'time_reference': 'median_relative_ms',
                    'median_anchor_sample': 50, 'boundaries': indices}
        (self.artifacts / (safe + '.json')).write_text(json.dumps(artifact))
        row = dict(record_id=rid, patient_id=patient, label_name=label, split=split,
                   qc_status='synthetic_fixture', qc_flags='synthetic_fixture')
        row.update({b + '_ms': self.time[index] if index is not None else np.nan for b, index in indices.items()})
        speed = v1.spatial_speed(self.xyz, 1000., self.cfg)
        if kind == 'missing_prediction':
            row.update(dict.fromkeys(v1.FEATURE_NAMES, np.nan))
        else:
            row.update(v1.extract_features(self.xyz[30:71], self.xyz[110:171], 1000., self.cfg,
                       speeds={'QRS': speed[30:71], 'T': speed[110:171]}))
        self.rows.append(row)
        # Fixed reference coordinates are fixture definitions; all pairs are
        # shifted five samples from the declared synthetic prediction intervals.
        annotation = dict(record_id=rid, patient_id=patient, split=split,
                          reviewer='synthetic_reviewer_A', reference_source='synthetic independent fixture',
                          annotation_scope='median_beat', time_reference='median_relative_ms', fs=1000.,
                          signal_sha256=engine.sha256_file(npz), independent_reference=True,
                          reference_quality='good', review_status='confirmed', reference_selection='primary',
                          qrs_on_ref_ms=-25., qrs_off_ref_ms=15., t_on_ref_ms=55., t_off_ref_ms=115.)
        annotation.update({b + '_status': 'visible' for b in engine.BOUNDARIES})
        if kind == 'partial':
            annotation.update(t_on_status='uncertain', t_off_status='not_visible',
                              t_on_ref_ms=np.nan, t_off_ref_ms=np.nan)
        self.annotations.append(annotation)
        if rid == 'synthetic001/full':
            second = dict(annotation, reviewer='synthetic_reviewer_B', reference_selection='',
                          qrs_on_ref_ms=-27., qrs_off_ref_ms=13., t_on_ref_ms=53., t_off_ref_ms=113.)
            self.annotations.append(second)

    def test_mode_a_exports_reports_and_plots_preserve_failures_and_test_holdout(self):
        results = engine.evaluate_references(self.frame, pd.DataFrame(self.annotations), self.artifacts, self.cfg)
        ledger, status = save_results(results, self.frame, self.output, allow_test=False)
        report_path, questions = write_report(results, self.frame, status, self.priority, self.output)
        plotted = create_reference_plots(results, self.artifacts, self.output, self.cfg)

        self.assertEqual(status['mode'], 'A_VERIFIED_REFERENCE_EVALUATION')
        self.assertEqual(status['n_valid_boundary_references'], 14)
        self.assertEqual(status['n_records_with_all_four_references'], 3)
        self.assertEqual(status['n_records_without_reference'], 1)
        self.assertEqual(len(results['boundary_error_per_record']), 20)
        self.assertEqual(len(results['loop_overlap_per_record']), 10)
        self.assertEqual(len(results['feature_reference_comparison']), 60)
        self.assertEqual(plotted['n_evaluated_pairs'], 10)
        self.assertTrue(plotted['manifest'].status.eq('created').all())

        accepted = results['accepted_references']
        first = accepted[accepted.record_id.eq('synthetic001/full')]
        self.assertEqual(len(first), 4)
        self.assertTrue(first.reviewer.eq('synthetic_reviewer_A').all())
        self.assertEqual(len(results['interreviewer_disagreement']), 4)
        self.assertEqual(len(accepted[accepted.record_id.eq('synthetic001/partial')]), 2)
        self.assertFalse(accepted.split.eq('test').any())
        audit = results['annotation_acceptance_audit']
        self.assertTrue(audit.loc[audit.record_id.eq('synthetic004/test_closed'), 'rejection_reason']
                        .str.contains('test_evaluation_disabled').all())

        boundary_summary = results['boundary_error_summary'].query("group_by=='all'").set_index('boundary')
        self.assertTrue(boundary_summary.n_total.eq(5).all())
        self.assertEqual(boundary_summary.loc['qrs_on', 'n_reference'], 4)
        self.assertEqual(boundary_summary.loc['qrs_on', 'n_evaluated'], 3)
        self.assertEqual(boundary_summary.loc['qrs_on', 'n_reference_missing_prediction'], 1)
        self.assertEqual(boundary_summary.loc['qrs_on', 'prediction_failure_rate_on_references'], .25)
        self.assertEqual(boundary_summary.loc['qrs_on', 'mae_ms'], 5.)
        self.assertEqual(boundary_summary.loc['t_on', 'n_reference'], 3)
        self.assertEqual(boundary_summary.loc['t_on', 'n_evaluated'], 2)
        self.assertAlmostEqual(boundary_summary.loc['t_on', 'prediction_failure_rate_on_references'], 1/3)
        missing = results['boundary_error_per_record'].query("record_id=='synthetic003/no_prediction'")
        self.assertTrue(missing.error_ms.isna().all())
        self.assertTrue(missing.reference_available.all())

        # Repeated records of one patient must aggregate into one patient row.
        patients = results['patient_boundary_summary'].query("patient_id=='synthetic001'").set_index('boundary')
        self.assertEqual(len(patients), 4)
        self.assertEqual(patients.loc['qrs_on', 'n_evaluated'], 2)
        self.assertEqual(patients.loc['qrs_on', 'patient_mae_ms'], 5.)
        self.assertEqual(len(questions), 11)
        self.assertIn('MAE=5.000 ms', questions[1]['answer'])
        self.assertIn('Mean=', questions[5]['answer'])
        self.assertIn('Không phải ML feature importance', questions[8]['answer'])
        self.assertFalse(any('nan' in q['answer'].lower() for q in questions))
        self.assertFalse(any('Chưa tìm thấy tham chiếu độc lập tương thích' in q['answer'] for q in questions))
        self.assertTrue(report_path.is_file())
        loaded_summary = pd.read_csv(self.output / 'boundary_error_summary.csv')
        self.assertEqual(len(loaded_summary), len(results['boundary_error_summary']))
        self.assertTrue(plotted['gallery'].is_file())

    def test_no_prediction_and_partial_reference_cannot_become_zero_error_in_report(self):
        frame = self.frame[self.frame.record_id.eq('synthetic003/no_prediction')].copy()
        annotations = pd.DataFrame(self.annotations)
        annotations = annotations[annotations.record_id.isin(frame.record_id)]
        results = engine.evaluate_references(frame, annotations, self.artifacts, self.cfg)
        ledger, status = save_results(results, frame, self.output)
        priority = self.priority[self.priority.record_id.isin(frame.record_id)]
        _, questions = write_report(results, frame, status, priority, self.output)
        plotted = create_reference_plots(results, self.artifacts, self.output, self.cfg)
        self.assertEqual(status['n_valid_boundary_references'], 4)
        self.assertEqual(status['mode'], 'A_VERIFIED_REFERENCE_EVALUATION')
        self.assertTrue(plotted['manifest'].status.eq('unavailable').all())
        self.assertFalse(list((self.output / 'plots').rglob('*.png')))
        for answer in questions[1:5]:
            self.assertIn('Chưa thể xác định', answer['answer'])
            self.assertNotIn('MAE=0', answer['answer'])


if __name__ == '__main__':
    unittest.main()
