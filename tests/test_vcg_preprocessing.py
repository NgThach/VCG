"""Controlled signals, filter transfer functions, API and PTB smoke checks."""
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.signal import freqz
from vcg_preprocessing import (Config, analyze_powerline_noise, decide_notch,
    design_morphology_fir, filter_frequency_response, assess_input_quality, legacy_config)
from vcg_pipeline import (preprocess_xyz, detect_qrs_anchors, process_record,
                          representative_beat, FEATURE_NAMES)


def controlled_xyz(seconds=24.0, cardiac_50=False, continuous_50=False, intermittent_50=False):
    fs=1000.0;t=np.arange(int(seconds*fs))/fs;xyz=np.zeros((len(t),3))
    anchors=[]
    for center in np.arange(1,seconds-1,1.0):
        anchors.append(int(round(center*fs)))
        for mu,sigma,amplitude in [(-.025,.008,[-.15,-.05,.10]),(0,.010,[1.,.6,.15]),
                                    (.028,.009,[-.3,-.15,.10]),(-.18,.025,[.06,.03,-.02]),
                                    (.25,.05,[.25,.12,-.08])]:
            xyz+=np.exp(-.5*((t-center-mu)/sigma)**2)[:,None]*np.array(amplitude)
        if cardiac_50:
            xyz+=(.18*np.exp(-.5*((t-center)/.065)**2)*np.cos(2*np.pi*50*(t-center)))[:,None]*np.array([1.,.6,.3])
    xyz+=.001*np.random.default_rng(20261010).normal(size=xyz.shape)
    if continuous_50:xyz+=(.08*np.sin(2*np.pi*50*t))[:,None]*np.array([1.,.8,.6])
    if intermittent_50:xyz+=(.08*np.sin(2*np.pi*50*t)*(t<seconds*.30))[:,None]*np.array([1.,.8,.6])
    return xyz,fs,np.array(anchors)


class TestPowerline(unittest.TestCase):
    def test_A_clean_no_unnecessary_notch(self):
        x,fs,a=controlled_xyz()
        d=analyze_powerline_noise(x,fs,qrs_anchors=a)
        self.assertEqual(d['status'],'noise_not_detected')
        self.assertFalse(decide_notch(d)['notch_applied'])

    def test_B_continuous_line_and_noise_reduction(self):
        x,fs,a=controlled_xyz(continuous_50=True)
        d=analyze_powerline_noise(x,fs,qrs_anchors=a)
        self.assertEqual(d['status'],'noise_detected')
        self.assertTrue(decide_notch(d)['notch_applied'])
        original=x.copy()
        m,det,diag=preprocess_xyz(x,fs,return_diagnostics=True)
        self.assertTrue(diag['notch_applied'])
        np.testing.assert_array_equal(x,original)
        clean,_,_=controlled_xyz()
        interior=slice(2000,-2000)
        self.assertLess(np.sqrt(np.mean((m[interior]-clean[interior])**2)),
                        .4*np.sqrt(np.mean((x[interior]-clean[interior])**2)))

    def test_C_cardiac_50_packet_not_continuous_line(self):
        x,fs,a=controlled_xyz(cardiac_50=True)
        d=analyze_powerline_noise(x,fs,qrs_anchors=a)
        self.assertGreater(d['score_db'],Config().powerline_score_threshold_db)
        self.assertNotEqual(d['status'],'noise_detected')
        self.assertFalse(decide_notch(d)['notch_applied'])

    def test_D_unstable_and_short_are_inconclusive(self):
        x,fs,a=controlled_xyz(intermittent_50=True)
        self.assertEqual(analyze_powerline_noise(x,fs,qrs_anchors=a)['status'],'inconclusive')
        self.assertEqual(analyze_powerline_noise(x[:4000],fs,qrs_anchors=a[a<4000])['status'],'inconclusive')
        line,_,_=controlled_xyz(continuous_50=True)
        d=analyze_powerline_noise(line,fs,qrs_anchors=None)
        self.assertEqual(d['status'],'inconclusive')
        self.assertFalse(decide_notch(d)['notch_applied'])

    def test_modes_and_per_channel_decision(self):
        x,fs,a=controlled_xyz()
        t=np.arange(len(x))/fs;x[:,0]+=.10*np.sin(2*np.pi*50*t)
        d=analyze_powerline_noise(x,fs,qrs_anchors=a)
        decision=decide_notch(d)
        self.assertEqual(decision['notch_channels'],['X'])
        self.assertFalse(decide_notch(d,replace(Config(),notch_mode='off'))['notch_applied'])
        self.assertEqual(decide_notch(d,replace(Config(),notch_mode='on'))['notch_channels'],list('XYZ'))


class TestFilterAndErrors(unittest.TestCase):
    def test_E_fir_and_notch_effective_response(self):
        cfg=Config();b=design_morphology_fir(1000,cfg)
        self.assertEqual(len(b),401);np.testing.assert_allclose(b,b[::-1],atol=1e-15)
        _,h=freqz(b,worN=np.linspace(0,150,3001),fs=1000)
        self.assertLess(float(np.max(np.abs(20*np.log10(np.abs(h)**2)))),.05)
        table=filter_frequency_response(1000,cfg).set_index('frequency_hz')
        self.assertLess(table.loc[190,'fir_two_pass_db'],-60)
        self.assertLess(table.loc[200,'fir_two_pass_db'],-60)
        self.assertLess(table.loc[50,'notch_two_pass_db'],-100)
        near=table.loc[(table.index>=40)&(table.index<=60)]
        width=near.index[near['notch_two_pass_db']<=-3]
        self.assertLess(float(width.max()-width.min()),2.0)

    def test_F_shape_finite_and_invalid_data(self):
        x,fs,_=controlled_xyz(seconds=12)
        m,d=preprocess_xyz(x,fs);self.assertEqual(m.shape,x.shape);self.assertEqual(d.shape,x.shape)
        self.assertTrue(np.isfinite(m).all() and np.isfinite(d).all())
        with self.assertRaises(ValueError):preprocess_xyz(x[:1000],fs)
        with self.assertRaises(ValueError):preprocess_xyz(x,500)
        bad=x.copy();bad[100,0]=np.nan
        self.assertEqual(assess_input_quality(bad,fs)['invalid_samples'],1)
        with self.assertRaises(ValueError):preprocess_xyz(bad,fs)
        with self.assertRaises(ValueError):preprocess_xyz(x[:,0],fs)
        flat=np.zeros_like(x)
        with self.assertRaises(ValueError):detect_qrs_anchors(flat,flat,fs)
        with self.assertRaises(ValueError):representative_beat(x,np.array([0,100]),fs)

    def test_fault_detection_not_all_sharp_qrs_are_noise(self):
        x,fs,_=controlled_xyz(seconds=12)
        x[4500,0]+=2.0;x[6000:6500]=0;x[8000:8040,1]=16.0
        q=assess_input_quality(x,fs)
        self.assertTrue(q['spike_mask'][4500]);self.assertTrue(q['dropout_mask'][6200])
        self.assertTrue(q['saturation_mask'][8010])

    def test_legacy_contract_for_window_experiments(self):
        x,_,_=controlled_xyz(seconds=12)
        small=x[::2]
        m,d=preprocess_xyz(small,500,legacy_config())
        self.assertEqual(m.shape,small.shape);self.assertEqual(d.shape,small.shape)

    def test_different_clean_morphology_is_retained(self):
        x,fs,a=controlled_xyz(seconds=16)
        changed=int(a[5])
        x[changed-250:changed+500]*=-1
        result=representative_beat(x,a,fs,quality=assess_input_quality(x,fs))
        row=result['beat_qc'].set_index('sample').loc[changed]
        self.assertTrue(row['included'])
        self.assertEqual(row['morphology_status'],'different_morphology_retained')
        self.assertIn('different_morphology_retained_review',result['quality_flags'])

    def test_real_HC_MI_smoke(self):
        root=Path(__file__).resolve().parents[1]/'content/ptbdb/physionet.org/files/ptbdb/1.0.0'
        if not root.exists():self.skipTest('PTB not present')
        for record in ['patient104/s0306lre','patient001/s0010_re']:
            result=process_record(root/record)
            self.assertEqual(result['raw']['channels'],['vx','vy','vz'])
            self.assertEqual(list(result['features']),FEATURE_NAMES)
            self.assertIn(result['preprocessing']['powerline_status'],
                          ['noise_detected','noise_not_detected','inconclusive'])
            self.assertIsNotNone(result['qrs_loop'])


if __name__=='__main__':unittest.main(verbosity=2)
