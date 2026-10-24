"""Scientific checks for reference coordinates, coverage and exact V1 slicing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import vcg_reference_metrics as engine
import vcg_pipeline as v1
from vcg_preprocessing import Config


class ReferenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rid = 'patient001/example'
        self.fs = 1000.0
        self.time = np.arange(220, dtype=float)-50
        u = np.arange(220)/20
        self.xyz = np.column_stack([np.sin(u),np.cos(1.3*u)*.3,np.sin(.7*u)*.2])
        self.indices = dict(qrs_on=30,qrs_off=70,t_on=110,t_off=170)
        self.jp = self.root / 'patient001_example.json'
        self.npz = self.root / 'patient001_example.npz'
        np.savez_compressed(self.npz, median_xyz=self.xyz,time_ms=self.time)
        self.artifact = dict(record_id=self.rid, fs=self.fs,time_reference='median_relative_ms',
                             median_anchor_sample=50,boundaries=self.indices)
        self.jp.write_text(json.dumps(self.artifact))
        row = dict(record_id=self.rid,patient_id='patient001',split='train',label_name='HC',
                   qc_status='needs_review',qc_flags='example_flag')
        row.update({b+'_ms': self.time[ix] for b,ix in self.indices.items()})
        speed = v1.spatial_speed(self.xyz,self.fs,Config())
        row.update(v1.extract_features(self.xyz[30:71],self.xyz[110:171],self.fs,
                                       speeds={'QRS':speed[30:71],'T':speed[110:171]}))
        self.frame = pd.DataFrame([row])
        self.annotation = dict(record_id=self.rid,patient_id='patient001',split='train',reviewer='R1',
                               reference_source='independent expert median annotation',
                               annotation_scope='median_beat',time_reference='median_relative_ms',
                               signal_sha256=engine.sha256_file(self.npz),fs=self.fs,
                               independent_reference=True,reference_quality='good',review_status='confirmed',
                               reference_selection='primary')
        for b,ix in self.indices.items():
            self.annotation[b+'_ref_ms'] = self.time[ix-5]
            self.annotation[b+'_status'] = 'visible'

    def tearDown(self):
        self.temp.cleanup()

    def test_nearest_sample_tie_and_domain(self):
        self.assertEqual(engine.time_ms_to_index(-19.5,self.time,self.fs),(30,-20.0,-.5))
        self.assertEqual(engine.time_ms_to_index(-50,self.time,self.fs),(0,-50.,0.))
        self.assertEqual(engine.time_ms_to_index(169,self.time,self.fs),(219,169.,0.))
        for bad in (-50.1,169.1,np.nan):
            with self.assertRaises(ValueError):
                engine.time_ms_to_index(bad,self.time,self.fs)
        with self.assertRaises(ValueError):
            engine.time_ms_to_index(0,self.time,500)

    def test_median_identity_and_time_alignment(self):
        median = engine.load_median(self.root,self.rid)
        self.assertEqual(median['anchor'],50)
        self.assertEqual(median['signal_sha256'],engine.sha256_file(self.npz))
        self.artifact['median_anchor_sample'] = 49
        self.jp.write_text(json.dumps(self.artifact))
        with self.assertRaises(ValueError):
            engine.load_median(self.root,self.rid)

    def test_utf8_artifact_with_vietnamese_metadata(self):
        # These UTF-8 bytes cannot be decoded using the Windows cp1258 default.
        self.artifact['note'] = 'Đánh giá tín hiệu, biên sóng T'
        self.jp.write_bytes(json.dumps(self.artifact, ensure_ascii=False).encode('utf-8'))
        median = engine.load_median(self.root, self.rid)
        self.assertEqual(median['artifact']['note'], self.artifact['note'])
        np.testing.assert_array_equal(median['xyz'], self.xyz)

    def test_no_reference_has_no_zero_errors_and_retains_denominators(self):
        results = engine.evaluate_references(self.frame,pd.DataFrame(),self.root)
        self.assertEqual(len(results['boundary_error_per_record']),4)
        self.assertEqual(len(results['loop_overlap_per_record']),2)
        self.assertEqual(len(results['feature_reference_comparison']),12)
        self.assertTrue(results['boundary_error_per_record'].error_ms.isna().all())
        self.assertTrue(results['loop_overlap_per_record'].temporal_iou.isna().all())
        self.assertTrue(results['feature_reference_comparison'].feature_error.isna().all())
        summary = results['boundary_error_summary'].query("group_by == 'all'")
        self.assertTrue(summary.n_missing_reference.eq(1).all())
        self.assertTrue(summary.mae_ms.isna().all())

    def test_signed_error_fullspeed_features_and_continuous_intervals(self):
        results = engine.evaluate_references(self.frame,pd.DataFrame([self.annotation]),self.root)
        table = results['boundary_error_per_record']
        self.assertTrue(table.error_ms.eq(5).all())
        self.assertTrue(table.evaluation_eligible.all())
        loops = results['loop_overlap_per_record'].set_index('loop')
        self.assertEqual(loops.loc['QRS','predicted_sample_count'],41)
        self.assertEqual(loops.loc['QRS','predicted_duration_ms'],40)
        self.assertEqual(loops.loc['QRS','reference_duration_ms'],40)
        self.assertAlmostEqual(loops.loc['QRS','temporal_iou'],35/45)
        self.assertEqual(loops.loc['QRS','missed_ms'],5)
        self.assertEqual(loops.loc['QRS','excess_ms'],5)
        self.assertEqual(loops.loc['QRS','reviewer'],'R1')
        self.assertEqual(loops.loc['QRS','reference_interval_kind'],'single_annotation')
        self.assertEqual(loops.loc['QRS','reference_annotation_rows'],'0')
        features = results['feature_reference_comparison'].set_index('feature')
        speed = v1.spatial_speed(self.xyz,self.fs)
        expected = v1.extract_features(self.xyz[25:66],self.xyz[105:166],self.fs,
                                        speeds={'QRS':speed[25:66],'T':speed[105:166]})
        for name in v1.FEATURE_NAMES:
            self.assertAlmostEqual(features.loc[name,'reference'],expected[name],places=12)
        self.assertTrue(results['baseline_reproduction_check'].matches.all())

    def test_missing_prediction_not_dropped(self):
        self.artifact['boundaries'] = {b:None for b in engine.BOUNDARIES}
        self.jp.write_text(json.dumps(self.artifact))
        for b in engine.BOUNDARIES:
            self.frame[b+'_ms'] = np.nan
        for feature in v1.FEATURE_NAMES:
            self.frame[feature] = np.nan
        results = engine.evaluate_references(self.frame,pd.DataFrame([self.annotation]),self.root)
        summary = results['boundary_error_summary'].query("group_by == 'all'")
        self.assertTrue(summary.n_reference.eq(1).all())
        self.assertTrue(summary.n_reference_missing_prediction.eq(1).all())
        self.assertTrue(summary.prediction_failure_rate_on_references.eq(1).all())
        self.assertTrue(summary.prediction_failure_rate_of_cohort.eq(1).all())
        self.assertTrue(summary.within_10_ms_rate.isna().all())
        self.assertTrue(summary.within_10_ms_rate_of_references.eq(0).all())
        self.assertTrue(summary.mae_ms.isna().all())
        self.assertTrue(results['loop_overlap_per_record'].status.eq('PREDICTION_UNAVAILABLE').all())
        self.assertTrue(results['feature_reference_comparison'].status.eq('PREDICTION_UNAVAILABLE').all())

    def test_metadata_hash_and_unaligned_raw_rejected(self):
        for key,value in [('signal_sha256','bad'),('fs',500),('patient_id','another'),
                          ('split','valid'),('time_reference','raw_ms'),('annotation_scope','raw_beat'),
                          ('independent_reference',False),('review_status','pending'),
                          ('reference_quality','uncertain'),('reviewer',''),('post_reveal_edit',True)]:
            bad = dict(self.annotation,**{key:value})
            accepted,audit = engine.validate_annotations(pd.DataFrame([bad]),self.frame,self.root)
            self.assertTrue(accepted.empty,msg=key)
            self.assertTrue(audit.rejection_reason.str.len().gt(0).all(),msg=key)

    def test_partial_boundaries_uncertainty_and_order(self):
        row = dict(self.annotation,t_on_status='uncertain',t_off_status='not_visible',t_off_ref_ms=np.nan)
        accepted,_ = engine.validate_annotations(pd.DataFrame([row]),self.frame,self.root)
        self.assertEqual(set(accepted.boundary),{'qrs_on','qrs_off'})
        row = dict(self.annotation,qrs_on_ref_ms=30,qrs_off_ref_ms=20)
        accepted,audit = engine.validate_annotations(pd.DataFrame([row]),self.frame,self.root)
        self.assertEqual(set(accepted.boundary),{'t_on','t_off'})
        self.assertTrue(audit[audit.boundary.str.startswith('qrs')].rejection_reason.str.contains('invalid_boundary_order').all())

    def test_two_reviewers_need_explicit_selection_and_no_repeated_samples(self):
        first = dict(self.annotation,reference_selection='')
        second = dict(self.annotation,reviewer='R2',reference_selection='',qrs_on_ref_ms=-27)
        annotations = pd.DataFrame([first,second])
        accepted,audit = engine.validate_annotations(annotations,self.frame,self.root)
        self.assertTrue(accepted.empty)
        disagreement = engine.interreviewer_disagreement(audit)
        self.assertEqual(len(disagreement),4)
        self.assertEqual(disagreement.set_index('boundary').loc['qrs_on','absolute_difference_ms'],2)
        first['reference_selection'] = 'primary'
        accepted,audit = engine.validate_annotations(pd.DataFrame([first,second]),self.frame,self.root)
        self.assertEqual(len(accepted),4)
        self.assertTrue(accepted.reviewer.eq('R1').all())
        # Duplicate primary revisions cannot become repeated independent samples.
        accepted,_ = engine.validate_annotations(pd.DataFrame([first,first]),self.frame,self.root)
        self.assertTrue(accepted.empty)

    def test_test_references_require_explicit_final_evaluation(self):
        self.frame['split'] = 'test'
        row = dict(self.annotation,split='test')
        accepted,_ = engine.validate_annotations(pd.DataFrame([row]),self.frame,self.root)
        self.assertTrue(accepted.empty)
        accepted,_ = engine.validate_annotations(pd.DataFrame([row]),self.frame,self.root,allow_test=True)
        self.assertEqual(len(accepted),4)

    def test_partial_reviewer_rows_cannot_form_reversed_selected_interval(self):
        first = dict(self.annotation,reviewer='R1',qrs_on_ref_ms=30,qrs_off_status='uncertain',
                     t_on_status='uncertain',t_off_status='uncertain')
        second = dict(self.annotation,reviewer='R2',qrs_on_status='uncertain',qrs_off_ref_ms=20,
                      t_on_status='uncertain',t_off_status='uncertain')
        accepted,audit = engine.validate_annotations(pd.DataFrame([first,second]),self.frame,self.root)
        self.assertTrue(accepted.empty)
        self.assertEqual(audit.acceptance_status.eq('selected_reference_geometry_rejected').sum(),2)

    def test_explicit_composite_reference_edges_retain_provenance(self):
        first = dict(self.annotation,reviewer='R1',qrs_off_status='uncertain')
        second = dict(self.annotation,reviewer='R2',qrs_on_status='uncertain',
                      t_on_status='uncertain',t_off_status='uncertain')
        result = engine.evaluate_references(self.frame,pd.DataFrame([first,second]),self.root)
        qrs = result['loop_overlap_per_record'].set_index('loop').loc['QRS']
        self.assertTrue(qrs.evaluation_eligible)
        self.assertEqual(qrs.reference_interval_kind,'composite_selected_edges')
        self.assertEqual(qrs.reference_annotation_rows,'0; 1')
        self.assertEqual(qrs.reviewer,'R1; R2')

    def test_overlap_disjoint_nested_and_degenerate(self):
        nested = engine.temporal_overlap(0,20,5,15)
        self.assertEqual(nested['reference_coverage'],1)
        self.assertEqual(nested['excess_ms'],10)
        self.assertEqual(nested['missed_ms'],0)
        self.assertEqual(nested['temporal_iou'],.5)
        disjoint = engine.temporal_overlap(0,10,20,30)
        self.assertEqual(disjoint['temporal_iou'],0)
        self.assertEqual(disjoint['missed_ms'],10)
        self.assertEqual(disjoint['excess_ms'],10)
        with self.assertRaises(ValueError):
            engine.temporal_overlap(0,0,0,1)

    def test_near_zero_reference_ratios_are_missing_not_infinite(self):
        np.savez_compressed(self.npz, median_xyz=np.zeros_like(self.xyz),time_ms=self.time)
        row = dict(self.annotation,signal_sha256=engine.sha256_file(self.npz))
        for feature in v1.FEATURE_NAMES:
            self.frame[feature] = 0.0
        # Degenerate synthetic PCA has zero variance; all its feature values
        # remain finite zero and its warning is irrelevant to this ratio check.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            results = engine.evaluate_references(self.frame,pd.DataFrame([row]),self.root)
        features = results['feature_reference_comparison']
        self.assertTrue(features.near_zero_reference.all())
        self.assertTrue(features.relative_feature_error_pct.isna().all())
        self.assertTrue(features.absolute_feature_error.eq(0).all())


if __name__ == '__main__':
    unittest.main()
