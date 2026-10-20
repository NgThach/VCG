"""Meaningful mathematical, failure-path and frozen-PTB integration checks."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import freqz
import vcg_t_almeida as v2
from vcg_reference_metrics import load_median
from run_vcg_t_almeida_experiment import (load_v1_inputs,process_one,verify_v1,initialize,
                                         feature_values,ARTIFACT_INPUT,safe_id,run_experiment)
from vcg_preprocessing import Config as V1Config


def synthetic(kind='positive',fs=1000):
    t=np.arange(int(1.2*fs)+1)/fs-.280
    anchor=int(.280*fs)
    qrs=np.exp(-((t-.012)/.015)**2)
    if kind=='positive':wave=.3*np.exp(-((t-.32)/.055)**2)
    elif kind=='negative':wave=-.3*np.exp(-((t-.32)/.055)**2)
    elif kind=='biphasic_positive_negative':wave=.3*np.exp(-((t-.27)/.04)**2)-.26*np.exp(-((t-.39)/.04)**2)
    elif kind=='biphasic_negative_positive':wave=-.3*np.exp(-((t-.27)/.04)**2)+.26*np.exp(-((t-.39)/.04)**2)
    else:wave=np.zeros_like(t)
    u=np.array([.8,.4,-.3]);u/=np.linalg.norm(u)
    xyz=qrs[:,None]*np.array([1.,-.4,.2])+wave[:,None]*u
    return xyz,fs,anchor,int(anchor-.04*fs),int(anchor+.05*fs),1.0


class TestQuadraticSplineWavelet(unittest.TestCase):
    def test_independent_known_filters(self):
        assert_array_equal(v2.reference_equivalent_filter(1),[2.,-2.])
        assert_array_equal(v2.reference_equivalent_filter(2),[.25,.75,.5,-.5,-.75,-.25])
        q3=np.array([1,3,6,10,11,9,4,-4,-9,-11,-10,-6,-3,-1])/32
        assert_array_equal(v2.reference_equivalent_filter(3),q3)

    def test_transfer_function_equation4(self):
        omega=np.linspace(.001,2.8,100)
        for m in (3,4,5):
            q=v2.reference_equivalent_filter(m)
            _,response=freqz(q,worN=omega)
            theory=4j*np.sin(2**(m-1)*omega/2)
            for l in range(m-1):theory*=np.cos(2**l*omega/2)**3
            assert_allclose(abs(response),abs(theory),atol=1e-12)

    def test_zero_dc_and_antisymmetry(self):
        for m in range(1,6):
            q=v2.reference_equivalent_filter(m)
            self.assertAlmostEqual(q.sum(),0.)
            assert_array_equal(q,-q[::-1])

    def test_constant_signal_and_length(self):
        xyz=np.ones((1401,3))*[1.,2.,-4.]
        for fs in (250,1000):
            w,meta=v2.wavelet_transform(xyz,fs)
            for m in (3,4,5):
                self.assertEqual(w[m].shape,xyz.shape)
                assert_allclose(w[m],0.,atol=1e-12)
                self.assertTrue(np.isfinite(w[m]).all())
            if fs==1000:self.assertTrue(all(v['alignment']=='exact_integer_delay' for v in meta.values()))

    def test_impulse_time_alignment(self):
        a=np.zeros((2001,3));a[1000]=[1,2,3]
        w,_=v2.wavelet_transform(a,1000)
        for m in (3,4,5):
            assert_allclose(w[m][1000],0.,atol=1e-12)
            assert_allclose(w[m][900:1000],-w[m][1001:1101][::-1],atol=2e-5)

    def test_analogue_band_gain_preserved_by_kernel_resampling(self):
        fs_values=(250,1000);amplitudes={}
        for fs in fs_values:
            t=np.arange(0,5,1/fs);x=np.sin(2*np.pi*5*t)
            w,_=v2.wavelet_transform(np.column_stack([x,x,x]),fs)
            amplitudes[fs]=[np.sqrt(np.mean(w[m][fs:-fs,0]**2)) for m in (3,4,5)]
        assert_allclose(amplitudes[250],amplitudes[1000],rtol=.006,atol=.001)

    def test_invalid_xyz_rejected(self):
        for a in (np.zeros((5,2)),np.full((5,3),np.nan),np.zeros((2,3))):
            with self.assertRaises(ValueError):v2.wavelet_transform(a,1000)

    def test_sampling_frequency_not_silently_capped(self):
        with self.assertRaises(ValueError):v2.wavelet_transform(np.zeros((100,3)),50)


class TestSingleLead(unittest.TestCase):
    def test_extrema_retain_sign(self):
        w=np.array([0.,1.,3.,1.,0.,-1.,-4.,-1.,0.])
        assert_array_equal(v2.modulus_extrema(w,0,8),[2,6])

    def test_zero_crossing_nearer_sample(self):
        self.assertEqual(v2.first_zero_crossing([3.,.2,-.7,-2.],0,3),1)
        self.assertEqual(v2.first_zero_crossing([-3.,-.9,.1,2.],0,3),2)
        self.assertIsNone(v2.first_zero_crossing([1.,2.,3.],0,2))

    def test_threshold_and_local_minimum_are_distinct(self):
        # Local |WT| minimum (at3) wins despite being above 25% of extremum.
        w=np.array([0.,1.2,2.,1.,1.5,4.,1.,.1])
        k,reason=v2.find_wave_boundary(w,5,0,7,.25,'t_on')
        self.assertEqual(k,3);self.assertEqual(reason,'local_modulus_minimum')

    def test_no_window_edge_fabrication(self):
        k,reason=v2.find_wave_boundary(np.array([4.,4.2,4.5,5.]),3,0,3,.25,'t_on')
        self.assertIsNone(k);self.assertEqual(reason,'no_threshold_or_local_minimum')

    def test_four_morphologies_and_inversion_invariance(self):
        for kind in ('positive','negative','biphasic_positive_negative','biphasic_negative_positive'):
            xyz,fs,anchor,*_=synthetic(kind)
            bank,_=v2.wavelet_transform(xyz,fs)
            a=v2.delineate_single_wavelet(bank[4][:,0],bank[3][:,0],(anchor+100,anchor+600),fs,(anchor,len(xyz)-1))
            self.assertTrue(a['wave_detected'],(kind,a))
            self.assertEqual(a['morphology_type'],kind)
            inverted=v2.delineate_single_wavelet(-bank[4][:,0],-bank[3][:,0],(anchor+100,anchor+600),fs,(anchor,len(xyz)-1))
            self.assertEqual(a['t_on_candidate_sample'],inverted['t_on_candidate_sample'])
            self.assertEqual(a['t_off_candidate_sample'],inverted['t_off_candidate_sample'])

    def test_rms_detection_significance_and_boundary_factors(self):
        xyz,fs,anchor,*_=synthetic();w,_=v2.wavelet_transform(xyz,fs)
        s=v2.delineate_single_wavelet(w[4][:,0],w[3][:,0],(anchor+100,anchor+600),fs,(anchor,len(xyz)-1))
        self.assertAlmostEqual(s['detection_threshold'],.25*np.sqrt(np.mean(w[4][anchor:,0]**2)))
        self.assertAlmostEqual(s['significant_threshold'],.125*np.max(abs(w[4][anchor+100:anchor+601,0])))
        self.assertAlmostEqual(s['onset_threshold'],.25*abs(w[4][s['first_significant_extremum'],0]))
        self.assertAlmostEqual(s['offset_threshold'],.4*abs(w[4][s['last_significant_extremum'],0]))
        projected=v2.delineate_single_wavelet(w[4][:,0],w[3][:,0],(anchor+100,anchor+600),fs,(anchor,len(xyz)-1),projected=True)
        self.assertAlmostEqual(projected['offset_threshold'],.25*abs(w[4][projected['last_significant_extremum'],0]))

    def test_multilead_scale_vote_requires_two(self):
        def row(scale,found=True):return dict(wavelet_scale=scale,wave_detected=found,detection_status='detected' if found else 'unresolved')
        self.assertEqual(v2.select_multilead_scale([row(5),row(4),row(5,False)]),4)
        self.assertEqual(v2.select_multilead_scale([row(5),row(4),row(5)]),5)

    def test_flat_wave_unresolved(self):
        r=v2.delineate_t(np.zeros((1201,3)),1000,280,240,330,1.)
        self.assertEqual(r['status'],'unresolved');self.assertIn('INITIALIZATION_FAILED',r['failure_reason'])
        self.assertIsNone(r['t_on_sample'])

    def test_single_lead_uses_scale4_when_peak_exists_but_boundary_missing(self):
        r=dict(detection_status='needs_review',wave_detected=True,t_peak_candidate_sample=300)
        with patch.object(v2,'delineate_single_wavelet',return_value=r) as detector:
            rows=v2.single_lead_initialization({3:np.zeros((700,3)),4:np.zeros((700,3)),5:np.zeros((700,3))},(100,600),1000,(100,600))
        self.assertEqual(detector.call_count,3)
        self.assertTrue(all(s['wavelet_scale']==4 for s in rows))


class TestTLS(unittest.TestCase):
    def test_known_3d_direction(self):
        rng=np.random.default_rng(147);u=np.array([.2,-.7,.4]);u/=np.linalg.norm(u)
        points=np.linspace(-2,3,500)[:,None]*u+np.array([8.,-2.,5.])+rng.normal(0,.0002,(500,3))
        fit=v2.tls_direction(points)
        self.assertGreater(abs(np.dot(fit['direction'],u)),.999999)
        self.assertAlmostEqual(np.linalg.norm(fit['direction']),1.)

    def test_affine_tls_translation_invariant_projection_not_centered(self):
        p=np.column_stack([np.arange(20.),np.arange(20.)*2,np.arange(20.)*3])
        a=v2.tls_direction(p);b=v2.tls_direction(p+[10.,20.,-30.])
        assert_allclose(a['direction'],b['direction'],atol=1e-13)
        assert_allclose(v2.project_wavelets(p,a['direction']),p@a['direction'])

    def test_sign_ambiguity_previous_direction(self):
        p=np.column_stack([np.arange(20.),-np.arange(20.),np.arange(20.)*2])
        a=v2.tls_direction(p);b=v2.tls_direction(p,-np.asarray(a['direction']))
        assert_allclose(a['direction'],-np.asarray(b['direction']))

    def test_degenerate_constant_and_isotropic(self):
        self.assertEqual(v2.tls_direction(np.ones((10,3)))['status'],'unresolved')
        self.assertEqual(v2.tls_direction(np.vstack([np.eye(3),-np.eye(3)]))['reason'],'tls_direction_not_identifiable')

    def _iteration(self,candidates,directions,max_iterations=4):
        bank={m:np.tile([1.,2.,3.],(800,1)) for m in (3,4,5)}
        fits=[dict(status='ok',direction=d,center=[0,0,0],singular_values=[3,1,.1],fit_quality=.9) for d in directions]
        waves=[dict(first_significant_extremum=300,last_significant_extremum=400,t_on_candidate_sample=c,
              t_off_candidate_sample=500,morphology_type='positive',t_peak_candidates=[350],failure_reason='',onset_threshold=.25) for c in candidates]
        with patch.object(v2,'tls_direction',side_effect=fits),patch.object(v2,'delineate_single_wavelet',side_effect=waves):
            return v2.iterate_tls_boundary(bank,4,(100,600),(100,600),(100,700),1000,'t_on',replace(v2.AlmeidaConfig(),max_iterations=max_iterations))

    def test_convergence_strictly_less_than_two_native_samples(self):
        r=self._iteration([200,201],[[1,0,0],[0,1,0]])
        self.assertTrue(r['converged']);self.assertEqual(r['sample'],201)

    def test_two_samples_is_not_convergence(self):
        r=self._iteration([200,202],[[1,0,0],[0,1,0]],max_iterations=2)
        self.assertFalse(r['converged']);self.assertEqual(r['stop_reason'],'max_iterations')

    def test_no_improvement_keeps_previous_not_rejected_candidate(self):
        r=self._iteration([200,225],[[0,1,0],[1,0,0]])
        self.assertEqual(r['sample'],200);self.assertEqual(r['chosen_iteration'],1)
        self.assertFalse(r['history'][-1]['accepted'])

    def test_max_iteration_reported_not_converged(self):
        r=self._iteration([200,205,210],[[1,0,0],[0,1,0],[0,0,1]],max_iterations=3)
        self.assertEqual(r['sample'],210);self.assertFalse(r['converged'])
        self.assertIn('max_iterations_reached_not_converged',r['review_flags'])

    def test_update_window_four_sigma(self):
        r=self._iteration([200,201],[[1,0,0],[0,1,0]])
        self.assertEqual(r['history'][1]['window_start'],200-61)
        self.assertEqual(r['history'][1]['window_end'],300)

    def test_invalid_order_and_peaks(self):
        self.assertFalse(v2.validate_t_bounds(200,190,100,500)[0])
        self.assertFalse(v2.validate_t_bounds(200,300,100,500,[320])[0])
        self.assertTrue(v2.validate_t_bounds(200,300,100,500,[250])[0])

    def test_end_to_end_deterministic_no_input_mutation(self):
        args=synthetic();before=args[0].copy()
        a=v2.delineate_t(*args);b=v2.delineate_t(*args)
        self.assertNotEqual(a['status'],'unresolved',a['failure_reason'])
        self.assertEqual(a['t_on_sample'],b['t_on_sample']);self.assertEqual(a['t_off_sample'],b['t_off_sample'])
        assert_array_equal(args[0],before)
        self.assertEqual(a['t_on_ms'],(a['t_on_sample']-args[2])*1000/args[1])

    def test_entrypoint_has_no_v1_t_or_label_inputs(self):
        parameters=set(inspect.signature(v2.delineate_t).parameters)
        self.assertEqual(parameters,{'xyz','fs','anchor','qrs_on','qrs_off','rr_seconds','cfg'})

    def test_unusual_duration_is_flagged_not_clipped(self):
        args=synthetic()
        normal=v2.delineate_t(*args)
        flagged=v2.delineate_t(*args,cfg=replace(v2.AlmeidaConfig(),duration_review_ms=(1.,2.)))
        self.assertEqual(normal['t_on_sample'],flagged['t_on_sample'])
        self.assertEqual(normal['t_off_sample'],flagged['t_off_sample'])
        self.assertEqual(flagged['status'],'needs_review')
        self.assertIn('t_duration_outside_review_range_not_clipped',flagged['review_flags'])

    def test_noise_does_not_create_nonfinite_or_invalid_loop_bounds(self):
        rng=np.random.default_rng(192);args=list(synthetic())
        args[0]+=rng.normal(0,.02,args[0].shape)
        result=v2.delineate_t(*args)
        self.assertIn(result['status'],('detected','needs_review','unresolved'))
        for array in result['wavelets'].values():self.assertTrue(np.isfinite(array).all())
        if result['status']!='unresolved':
            self.assertTrue(v2.validate_t_bounds(result['t_on_sample'],result['t_off_sample'],args[4],len(args[0]))[0])

    def test_no_significant_extremum_keeps_previous_mark(self):
        bank={m:np.tile([1.,2.,3.],(800,1)) for m in (3,4,5)}
        fit=dict(status='ok',direction=[1.,0.,0.],center=[0,0,0],singular_values=[3,1,.1],fit_quality=.9)
        first=dict(first_significant_extremum=300,last_significant_extremum=400,t_on_candidate_sample=200,
            t_off_candidate_sample=500,morphology_type='positive',t_peak_candidates=[350],failure_reason='',onset_threshold=.25)
        missing=dict(first_significant_extremum=None,t_on_candidate_sample=None,morphology_type='unresolved',failure_reason='no slopes')
        with patch.object(v2,'tls_direction',return_value=fit),patch.object(v2,'delineate_single_wavelet',side_effect=[first,missing]):
            r=v2.iterate_tls_boundary(bank,4,(100,600),(100,600),(100,700),1000,'t_on')
        self.assertEqual(r['sample'],200);self.assertEqual(r['chosen_iteration'],1)
        self.assertEqual(r['stop_reason'],'no_significant_extremum_or_boundary')


@unittest.skipUnless(ARTIFACT_INPUT.is_dir(),'PTB V1 artifacts unavailable')
class TestFrozenPTBIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.frame,cls.manifest,cls.v1cfg=load_v1_inputs()

    def test_manifest_no_leakage_and_all_ids(self):
        self.assertEqual(set(self.frame.record_id),set(self.manifest.record_id))
        self.assertTrue(self.frame.groupby('patient_id').split.nunique().eq(1).all())

    def test_saved_median_units_and_grid(self):
        m=load_median(ARTIFACT_INPUT,'patient104/s0306lre')
        assert_array_equal(m['time_ms'],(np.arange(len(m['xyz']))-m['anchor'])*1000/m['fs'])
        self.assertEqual(m['time_reference'],'median_relative_ms')
        self.assertEqual(m['fs'],1000.)

    def test_qrs_and_features_preserved(self):
        row=self.frame.set_index('record_id').loc['patient104/s0306lre'].to_dict();row['record_id']='patient104/s0306lre'
        m=load_median(ARTIFACT_INPUT,row['record_id']);b=m['boundaries']
        result=v2.delineate_t(m['xyz'],m['fs'],m['anchor'],b['qrs_on'],b['qrs_off'],float(np.median(m['artifact']['rr_seconds'])))
        f,parity,loops=feature_values(m,row,result,self.v1cfg)
        self.assertEqual(parity['status'],'verified')
        for k in list(f)[:7]:self.assertEqual(f[k],row[k])
        assert_array_equal(loops['qrs_loop_v1_unchanged'],m['xyz'][b['qrs_on']:b['qrs_off']+1])
        if 't_loop_v2' in loops:assert_array_equal(loops['t_loop_v2'],m['xyz'][result['t_on_sample']:result['t_off_sample']+1])

    def test_v1_t_changes_do_not_change_v2(self):
        m=load_median(ARTIFACT_INPUT,'patient051/s0179lre');b=m['boundaries']
        inputs=(m['xyz'],m['fs'],m['anchor'],b['qrs_on'],b['qrs_off'],float(np.median(m['artifact']['rr_seconds'])))
        r1=v2.delineate_t(*inputs);b['t_on']=1;b['t_off']=len(m['xyz'])-1
        r2=v2.delineate_t(*inputs)
        for key in ['status','failure_reason','t_on_sample','t_off_sample']:self.assertEqual(r1[key],r2[key])

    def test_missing_median_has_no_fake_loop_and_checkpoint_reuses(self):
        rid='patient033/s0157lre';row=self.frame.set_index('record_id').loc[rid].to_dict();row['record_id']=rid
        src=ARTIFACT_INPUT/(safe_id(rid)+'.npz');before=hashlib.sha256(src.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name in ('checkpoints','diagnostics','artifacts'):(out/name).mkdir()
            r,diag,cached=process_one(row,out,v2.AlmeidaConfig(),self.v1cfg,'test-fixture')
            self.assertFalse(r['median_present']);self.assertEqual(r['status'],'unresolved')
            with np.load(out/'artifacts'/(safe_id(rid)+'.npz')) as z:self.assertNotIn('t_loop_v2',z.files)
            again,_,cached=process_one(row,out,v2.AlmeidaConfig(),self.v1cfg,'test-fixture')
            self.assertTrue(cached);self.assertEqual(r,again)
            with self.assertRaises(ValueError):process_one(row,out,v2.AlmeidaConfig(),self.v1cfg,'changed-fingerprint')
        self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(),before)

    def test_protected_destination_rejected(self):
        with self.assertRaises(ValueError):initialize(ARTIFACT_INPUT)

    def test_full_reporting_and_gallery_for_only_missing_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            result=run_experiment(out,record_ids=['patient033/s0157lre'],plots=False,gallery=True)
            self.assertEqual(result['attempted'],1)
            self.assertEqual(result['median_available'],0)
            self.assertEqual(result['t_loops_available'],0)
            self.assertEqual(result['experiment_exceptions'],0)
            self.assertTrue((out/'gallery/index.html').is_file())

    def test_hash_guard_detects_change(self):
        self.assertTrue(verify_v1({'vcg_pipeline.py':hashlib.sha256((ROOT/'vcg_pipeline.py').read_bytes()).hexdigest()})['v1_unchanged'])
        with self.assertRaises(RuntimeError):verify_v1({'vcg_pipeline.py':'wrong-hash'})


if __name__=='__main__':unittest.main(verbosity=2)
