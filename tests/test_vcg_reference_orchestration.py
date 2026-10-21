"""Notebook08 source, preservation and diagnostic/evaluation boundary checks."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import vcg_boundary_reference_eval as orchestration
from vcg_preprocessing import Config


class ReferenceOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root/'outputs/new_reference_eval'
        self.out.mkdir(parents=True)
        self.frame = pd.DataFrame([
            dict(record_id='patient001/r1',patient_id='patient001',label_name='HC',split='train',
                 qc_status='auto_candidate',qc_flags='',feature_completeness=12,
                 t_duration_ms=120,processing_success=True),
            dict(record_id='patient002/r2',patient_id='patient002',label_name='MI',split='valid',
                 qc_status='needs_review',qc_flags='st_level_requires_review',feature_completeness=7,
                 t_duration_ms=17,processing_success=True),
            dict(record_id='patient003/r3',patient_id='patient003',label_name='MI',split='test',
                 qc_status='failed',qc_flags='features_incomplete',feature_completeness=0,
                 t_duration_ms=np.nan,processing_success=False)])

    def tearDown(self):
        self.temp.cleanup()

    def test_hash_freeze_detects_changes_to_previous_artifacts_and_notebooks(self):
        input_dir = self.root/'outputs/split06'
        artifacts = self.root/'outputs/evaluation/artifacts'
        qc = self.root/'outputs/vcg_qc_audit_v1'
        for folder in (input_dir,artifacts,qc):
            folder.mkdir(parents=True)
        for name in ('vcg_pipeline.py','vcg_preprocessing.py',
                     '05_QRS_T_Loops_Step_by_Step.ipynb',
                     '06_PTB_VCG_12_Features_Train_Valid_Test.ipynb',
                     '07_VCG_QC_vs_Delineation_Audit.ipynb'):
            content = json.dumps({'cells': [{'cell_type': 'code', 'source': ['x = 1'],
                'outputs': [], 'execution_count': None}]}) if name.endswith('.ipynb') else 'original'
            (self.root/name).write_text(content)
        (input_dir/'train_features.csv').write_text('original CSV')
        (artifacts/'record.npz').write_bytes(b'original cached signal')
        (qc/'manual_review_blinded.csv').write_text('original blank reviews')
        loaded = (self.frame,pd.DataFrame(),{'config':asdict(Config())},input_dir,artifacts.parent)
        with mock.patch.object(orchestration,'ROOT',self.root),mock.patch.object(orchestration,'load_inputs',return_value=loaded):
            _,_,_,returned_artifacts,hashes = orchestration.initialize(self.out)
            self.assertEqual(returned_artifacts,artifacts)
            self.assertTrue(orchestration.verify_protected(hashes,self.root))
            self.assertIn('outputs/vcg_qc_audit_v1/manual_review_blinded.csv',hashes)
            notebook_path = self.root/'05_QRS_T_Loops_Step_by_Step.ipynb'
            notebook = json.loads(notebook_path.read_text())
            notebook['cells'][0]['outputs'] = [{'output_type': 'stream', 'text': 'saved output'}]
            notebook_path.write_text(json.dumps(notebook))
            self.assertTrue(orchestration.verify_protected(hashes,self.root))
            notebook['cells'][0]['source'] = ['x = 2']
            notebook_path.write_text(json.dumps(notebook))
            with self.assertRaisesRegex(ValueError,'Protected V1 files changed'):
                orchestration.verify_protected(hashes,self.root)
            notebook['cells'][0]['source'] = ['x = 1']
            notebook_path.write_text(json.dumps(notebook))
            signal = artifacts/'record.npz'
            signal.write_bytes(b'changed signal')
            with self.assertRaisesRegex(ValueError,'Protected V1 files changed'):
                orchestration.verify_protected(hashes,self.root)
            with self.assertRaisesRegex(ValueError,'Previously frozen source changed'):
                orchestration.initialize(self.out)
        frozen = json.loads((self.out/'protected_input_hashes.json').read_text())
        self.assertEqual(frozen,hashes)

    def test_priority_is_diagnostic_and_test_is_not_development_eligible(self):
        qc = self.root/'qc'
        qc.mkdir()
        pd.DataFrame({'record_id':self.frame.record_id,'t_on_method_gap_ms':[80,15,200],
                      't_off_method_gap_ms':[10,150,20],'n_significant_lobes':[1,2,np.nan]}).to_csv(qc/'t_boundary_audit.csv',index=False)
        pd.DataFrame({'record_id':self.frame.record_id,'secondary_activity_count':[0,1,np.nan]}).to_csv(qc/'qrs_boundary_audit.csv',index=False)
        pd.DataFrame({'record_id':['patient001/r1','patient002/r2']}).to_csv(qc/'manual_review_blinded.csv',index=False)
        priority = orchestration.priority_cases(self.frame,self.out,qc)
        test = priority[priority.split.eq('test')]
        self.assertFalse(test.eligible_for_development_review.any())
        self.assertTrue(priority.ground_truth_status.eq('GROUND_TRUTH_UNAVAILABLE').all())
        self.assertTrue(priority.interpretation.str.contains('neither confirmed error nor disease probability').all())
        self.assertIn('t_duration_outside_v1_review_range',priority.set_index('record_id').loc['patient002/r2','priority_reasons'])
        self.assertIn('no_qc_control',priority.set_index('record_id').loc['patient001/r1','priority_reasons'])
        self.assertFalse(priority.iloc[-1].eligible_for_development_review)

    def test_source_inventory_distinguishes_empty_template_and_unverified_values(self):
        data = self.root/'content/ptbdb/physionet.org/files/ptbdb/1.0.0'
        for rid in self.frame.record_id:
            path = data/(rid+'.hea')
            path.parent.mkdir(parents=True,exist_ok=True)
            name = path.stem
            path.write_text(f'{name} 3 1000 100\n'+''.join(
                f'{name}.xyz 16 1000/mV 16 0 0 0 0 {channel}\n' for channel in ('vx','vy','vz')))
            path.with_suffix('.xyz').write_bytes(b'waveform, not annotation')
        pd.DataFrame({'record_id':['patient001/r1'],'qrs_on_review_ms':[np.nan]}).to_csv(self.root/'empty_review.csv',index=False)
        pd.DataFrame({'record_id':['patient002/r2'],'qrs_on_ref_ms':[-20]}).to_csv(self.root/'unverified_reference.csv',index=False)
        (self.root/'unverified.atr').write_bytes(b'annotation-like candidate')
        # Current output is intentionally excluded from discovery on reruns.
        pd.DataFrame({'qrs_on_ref_ms':[100]}).to_csv(self.out/'should_not_be_reimported.csv',index=False)
        report,candidates,headers = orchestration.scan_reference_sources(self.frame,self.out,self.root)
        self.assertEqual(len(headers),3)
        self.assertTrue(headers.frank_channels.eq('vx;vy;vz').all())
        self.assertTrue(headers.fs.eq(1000).all())
        states = candidates.set_index('path').status
        self.assertEqual(states['empty_review.csv'],'PENDING_EMPTY_TEMPLATE')
        self.assertEqual(states['unverified_reference.csv'],'REQUIRES_INDEPENDENCE_VALIDATION')
        self.assertEqual(states['unverified.atr'],'REQUIRES_SIGNAL_SCOPE_VALIDATION')
        self.assertFalse(candidates.path.str.contains('should_not_be_reimported').any())
        self.assertTrue(report.compatible_verified_records.eq(0).all())
        self.assertFalse(report.four_boundaries_available.any())

    def test_annotation_reading_preserves_values_and_hashes_without_filling_missing(self):
        main = self.out/'reference_annotations.csv'
        pd.DataFrame({'record_id':['patient001/r1'],'reviewer':[''],
                      'qrs_on_ref_ms':[np.nan],'review_status':['pending']}).to_csv(main,index=False)
        external = self.root/'reviewer.csv'
        pd.DataFrame({'record_id':['patient002/r2'],'reviewer':['independent_A'],
                      'qrs_on_ref_ms':[-25.25],'review_status':['confirmed']}).to_csv(external,index=False)
        original_main, original_external = main.read_bytes(),external.read_bytes()
        rows = orchestration.read_annotations(self.out,[external])
        self.assertEqual(len(rows),2)
        self.assertTrue(pd.isna(rows.loc[0,'qrs_on_ref_ms']))
        self.assertEqual(rows.loc[1,'qrs_on_ref_ms'],-25.25)
        self.assertEqual(main.read_bytes(),original_main)
        self.assertEqual(external.read_bytes(),original_external)
        hashes = json.loads((self.out/'annotation_input_hashes.json').read_text())
        self.assertEqual([r['n_rows'] for r in hashes],[1,1])
        with self.assertRaises(FileNotFoundError):
            orchestration.read_annotations(self.out,[self.root/'missing.csv'])

    def test_saving_no_references_keeps_diagnostic_mode_and_missing_errors(self):
        pd.DataFrame({'record_id':self.frame.record_id,'ground_truth_status':['pending']*3}).to_csv(self.out/'priority_review_cases.csv',index=False)
        boundaries = pd.DataFrame([dict(record_id=rid,boundary=b,reference_available=False,
                                       prediction_available=False,error_ms=np.nan)
                                  for rid in self.frame.record_id for b in orchestration.BOUNDARIES])
        ledger,status = orchestration.save_results({'boundary_error_per_record':boundaries},self.frame,self.out)
        self.assertEqual(status['mode'],'B_DIAGNOSTIC_ONLY')
        self.assertEqual(status['n_valid_boundary_references'],0)
        self.assertEqual(status['n_records_without_reference'],3)
        self.assertTrue(ledger.reference_status.eq('GROUND_TRUTH_UNAVAILABLE').all())
        reread = pd.read_csv(self.out/'boundary_error_per_record.csv')
        self.assertEqual(len(reread),12)
        self.assertTrue(reread.error_ms.isna().all())


if __name__ == '__main__':
    unittest.main()
