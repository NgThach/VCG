"""Actual XYZ overview coverage, exact V1 shading and read-only cache checks."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vcg_visual_review_overview import create_v1_contact_sheets


class VisualOverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name); self.art=self.root/'art'; self.art.mkdir()
        self.out=self.root/'all_cases'
        self.time=(np.arange(100)-25)*2.
        self.xyz=np.column_stack([np.sin(np.arange(100)/10+j) for j in range(3)])
        self.b={'qrs_on':20,'qrs_off':30,'t_on':50,'t_off':70}
        self.row={'record_id':'p/r','patient_id':'p','label_name':'HC','split':'train',
            **{k+'_ms':self.time[v] for k,v in self.b.items()}}
        (self.art/'p_r.json').write_text(json.dumps({'record_id':'p/r','fs':500.,'median_anchor_sample':25,'boundaries':self.b}))
        np.savez_compressed(self.art/'p_r.npz',median_xyz=self.xyz,time_ms=self.time)

    def tearDown(self): self.temp.cleanup()

    def test_full_xyz_and_separate_qrs_t_shading_exactly_match_saved_samples(self):
        spans=[]; series=[]; original_span=Axes.axvspan; original_plot=Axes.plot
        def span(ax,a,z,*args,**kwargs):
            spans.append((a,z)); return original_span(ax,a,z,*args,**kwargs)
        def plot(ax,x,y,*args,**kwargs):
            series.append((np.asarray(x),np.asarray(y))); return original_plot(ax,x,y,*args,**kwargs)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.art.iterdir()}
        with mock.patch.object(Axes,'axvspan',span),mock.patch.object(Axes,'plot',plot):
            m=create_v1_contact_sheets(pd.DataFrame([self.row]),self.art,self.out)
        self.assertEqual(spans,[(-10.,10.),(50.,90.)])
        self.assertEqual(len(series),3)
        for j,(time,xyz) in enumerate(series):
            np.testing.assert_array_equal(time,self.time)
            np.testing.assert_array_equal(xyz,self.xyz[:,j])
        self.assertEqual(m.display_status.iloc[0],'median_displayed')
        self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.art.iterdir()})

    def test_cached_rerun_keeps_valid_png_and_coverage(self):
        frame=pd.DataFrame([self.row]); a=create_v1_contact_sheets(frame,self.art,self.out)
        with mock.patch.object(Figure,'savefig',side_effect=AssertionError('Should use cache')):
            b=create_v1_contact_sheets(frame,self.art,self.out)
        pd.testing.assert_frame_equal(a,b)

    def test_missing_median_and_corrupt_record_remain_visible(self):
        np.savez_compressed(self.art/'p_r.npz',detection_envelope=np.zeros(10))
        bad=dict(self.row,record_id='q/missing')
        m=create_v1_contact_sheets(pd.DataFrame([self.row,bad]),self.art,self.out)
        self.assertEqual(set(m.record_id),{'p/r','q/missing'})
        self.assertEqual(m.set_index('record_id').loc['p/r','display_status'],'median_unavailable')
        self.assertTrue(m.set_index('record_id').loc['q/missing','display_status'].startswith('render_error:'))
        self.assertTrue(all((self.out/path).is_file() for path in m.plot_path))

    def test_wrong_csv_time_is_reported_without_invented_overlay(self):
        wrong=dict(self.row,qrs_on_ms=-9)
        m=create_v1_contact_sheets(pd.DataFrame([wrong]),self.art,self.out)
        self.assertIn('CSV/artifact boundary mismatch',m.display_status.iloc[0])


if __name__=='__main__': unittest.main()
