"""Coverage, denominators, and executable filtering of the exhaustive gallery."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import vcg_all_case_gallery as gallery


class AllCaseGalleryTest(unittest.TestCase):
    def setUp(self):
        self.workspace = TemporaryDirectory(prefix='vcg-all-gallery-')
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.out = self.root / 'all_cases'
        self.figures = self.root / 'figures with spaces'
        self.figures.mkdir()
        self.records = pd.DataFrame([
            dict(record_id='p001/r1', patient_id='p001', label_name='HC', split='train',
                 qc_status='needs_review', qc_flags='flag_one;flag_two', has_median=True,
                 feature_completeness=12, t_on_gap_ms=0., t_off_gap_ms=60.,
                 t_on_signed_gap_ms=0., t_off_signed_gap_ms=-60.,
                 t_tangent_on_ms=200., t_threshold_on_ms=200.,
                 t_on_pair_available=True, t_off_pair_available=True,
                 priority_categories='method_disagreement'),
            dict(record_id='p002/r2', patient_id='p002', label_name='MI', split='test',
                 qc_status='auto_candidate', qc_flags='', has_median=True,
                 feature_completeness=12, t_on_gap_ms=101., t_off_gap_ms=np.nan,
                 t_on_pair_available=True, t_off_pair_available=False),
            dict(record_id='p003/r3', patient_id='p003', label_name='MI', split='valid',
                 qc_status='failed', qc_flags='technical_failure', has_median=False,
                 feature_completeness=0, t_on_gap_ms=np.nan, t_off_gap_ms=np.nan,
                 t_on_pair_available=False, t_off_pair_available=False),
            dict(record_id='p004/r4', patient_id='p004', label_name='HC', split='train',
                 qc_status='auto_candidate', qc_flags='', has_median=True,
                 feature_completeness=12, t_on_gap_ms=np.nan, t_off_gap_ms=10.,
                 t_on_pair_available=False, t_off_pair_available=True),
        ])
        for i in range(len(self.records)):
            (self.figures / f'case {i}.png').write_bytes(b'\x89PNG\r\n\x1a\n')
        self.manifest = pd.DataFrame(dict(record_id=self.records.record_id,
                                         plot_path=[str(self.figures/f'case {i}.png') for i in range(4)]))
        self.summary = pd.DataFrame([dict(scope='all', metric='method_gap',
                                         tolerance_ms=60, n_cases=1, n_denominator=3)])

    def build(self, records=None, manifest=None):
        return gallery.build_all_case_gallery(self.records if records is None else records,
                                              self.summary,
                                              self.manifest if manifest is None else manifest,
                                              self.out)

    def payload(self, result):
        text = Path(result['gallery_path']).read_text()
        raw = re.search(r'<script id="all-cases-payload" type="application/json">(.*?)</script>', text, re.S).group(1)
        return text, json.loads(raw)

    def run_core(self, payload, assertions):
        if shutil.which('node') is None:
            self.skipTest('Node not available')
        program = gallery.CORE_JS + '\nconst records=' + json.dumps(payload['records']) + ';\n' + assertions
        result = subprocess.run(['node', '-e', program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_every_record_has_visible_card_original_figure_and_download_row(self):
        result = self.build()
        text, payload = self.payload(result)
        self.assertEqual(result['n_records'], 4)
        self.assertEqual(result['n_median'], 3)
        self.assertEqual(result['n_median_unavailable'], 1)
        self.assertEqual(text.count('<article class="case"'), 4)
        self.assertEqual(text.count('loading="lazy"'), 4)
        self.assertEqual(set(pd.read_csv(result['all_cases_index']).record_id), set(self.records.record_id))
        self.assertEqual(set(r['record_id'] for r in payload['records']), set(self.records.record_id))
        for row in payload['records']:
            self.assertIn('figures%20with%20spaces/case%20', row['plot_url'])
        self.assertIn('chưa phải số ca tách sai', text)
        self.assertIn('Tập test: chỉ xem mô tả, không tinh chỉnh', text)
        self.assertIn('Không có median beat: xem tín hiệu raw', text)
        self.assertNotIn('https://', text)

    def test_membership_denominators_keep_missing_separate_from_zero(self):
        _, payload = self.payload(self.build())
        actual = self.run_core(payload, '''console.log(JSON.stringify({
          counts:membershipCounts(records), on:thresholdCounts(records,'on',[0,60,100]),
          any:thresholdCounts(records,'any',[0,10,60,100]),
          all:selectRecords(records,{tolerance:'all',sort:'gap_desc'}).map(r=>r.record_id),
          above60:selectRecords(records,{tolerance:'60',sort:'gap_desc'}).map(r=>r.record_id)
        }));''')
        self.assertEqual(actual['counts'], dict(total=4, both=1, any=3, missing=3, none=1))
        self.assertEqual(actual['on'][0], dict(tolerance=0, count=1, denominator=2, missing=2))
        self.assertEqual(actual['any'][1], dict(tolerance=10, count=2, denominator=3, missing=1))
        self.assertEqual(actual['above60'], ['p002/r2'])
        self.assertEqual(actual['all'], ['p002/r2', 'p001/r1', 'p004/r4', 'p003/r3'])

    def test_filters_use_exact_qc_flag_and_strict_threshold_and_both_pair(self):
        _, payload = self.payload(self.build())
        actual = self.run_core(payload, '''const ids=f=>selectRecords(records,f).map(r=>r.record_id);
          console.log(JSON.stringify({
          exact:ids({qc_flag:'flag_one',tolerance:'all'}), partial:ids({qc_flag:'flag',tolerance:'all'}),
          label:ids({label_name:'MI',split:'test',tolerance:'all'}),
          onZero:ids({boundary:'on',tolerance:'0'}),
          off60:ids({boundary:'off',tolerance:'60'}),
          both:ids({boundary:'both',tolerance:'all'}),
          search:ids({search:'P003',tolerance:'all'}),
          missing:ids({pair_status:'neither',tolerance:'all'}),
          category:ids({category:'method_disagreement',tolerance:'all'})
          }));''')
        self.assertEqual(actual['exact'], ['p001/r1'])
        self.assertEqual(actual['partial'], [])
        self.assertEqual(actual['label'], ['p002/r2'])
        self.assertEqual(actual['onZero'], ['p002/r2'])
        self.assertEqual(actual['off60'], [])
        self.assertEqual(actual['both'], ['p001/r1'])
        self.assertEqual(actual['search'], ['p003/r3'])
        self.assertEqual(actual['missing'], ['p003/r3'])
        self.assertEqual(actual['category'], ['p001/r1'])

    def test_missing_pair_is_not_invented_from_numeric_gap(self):
        records = self.records.copy()
        records.loc[1, 't_on_pair_available'] = False
        _, payload = self.payload(self.build(records))
        row = payload['records'][1]
        self.assertFalse(row['t_on_pair_available'])
        self.assertEqual(row['method_pair_status'], 'neither')
        self.assertIsNone(row['max_method_gap_ms'])
        actual = self.run_core(payload, "console.log(JSON.stringify(thresholdCounts(records,'on',[0])));")
        self.assertEqual(actual[0], dict(tolerance=0, count=0, denominator=1, missing=3))

    def test_source_tables_and_unrelated_review_files_are_unchanged(self):
        before = self.records.copy(deep=True)
        review = self.root / 'review'
        review.mkdir()
        annotation = review / 'independent_annotation.html'
        annotation.write_text('existing independent annotation')
        result = self.build()
        pd.testing.assert_frame_equal(before, self.records)
        self.assertEqual(annotation.read_text(), 'existing independent annotation')
        text, _ = self.payload(result)
        self.assertIn('../review/independent_annotation.html', text)
        self.assertIn('Đã xem dự đoán có thể làm lệch', text)

    def test_duplicates_membership_mismatch_and_missing_figures_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.build(pd.concat([self.records, self.records.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.build(manifest=self.manifest.iloc[:-1])
        missing = self.manifest.copy()
        missing.loc[1, 'plot_path'] = str(self.root/'missing.png')
        with self.assertRaisesRegex(FileNotFoundError, 'Figure does not exist'):
            self.build(manifest=missing)

    def test_metadata_escaping_and_json_never_break_out_of_script(self):
        records = self.records.copy()
        records.loc[0, 'qc_flags'] = '</script><script>alert(1)</script>'
        text, payload = self.payload(self.build(records))
        self.assertEqual(payload['records'][0]['qc_flags'], '</script><script>alert(1)</script>')
        self.assertNotIn('<script>alert(1)</script>', text)
        self.assertIn('&lt;/script&gt;', text)
        self.assertNotIn('NaN', re.search(r'<script id="all-cases-payload".*?>(.*?)</script>', text, re.S).group(1))

    def test_browser_app_refresh_filters_cards_and_exports_real_csv(self):
        if shutil.which('node') is None:
            self.skipTest('Node not available')
        _, payload = self.payload(self.build())
        program = r'''
const assert=require('assert');let exported=null;
class Element {
 constructor(){this.children=[];this.dataset={};this.value='';this.textContent='';this.style={};this.listeners={};this.hidden=false;this.tagName='SELECT';}
 append(...items){for(const i of items){this.children=this.children.filter(v=>v!==i);this.children.push(i);}}
 replaceChildren(){this.children=[];}
 addEventListener(name,fn){this.listeners[name]=fn;}
 click(){if(this.listeners.click)this.listeners.click();}
}
const payload=PAYLOAD;
const elems=Object.fromEntries(['all-cases-payload','counts-chart','case-list','visible-count','pair-count','empty-state','reset','export-selected'].map(k=>[k,new Element()]));
elems['all-cases-payload'].textContent=JSON.stringify(payload);
const cases=payload.records.map(r=>{const e=new Element();e.dataset.recordId=r.record_id;return e;});
const defs={search:'',label_name:'',split:'',qc_status:'',qc_flag:'',category:'',boundary:'any',tolerance:'all',sort:'gap_desc',pair_status:''};
const inputs=Object.entries(defs).map(([key,value])=>{const e=new Element();e.dataset={filter:key,default:value};e.value=value;return e;});
global.document={getElementById:k=>elems[k],querySelectorAll:s=>s==='.case'?cases:inputs,createElement:()=>new Element()};
global.Blob=class {constructor(parts){exported=parts.join('');}};
global.URL={createObjectURL:()=>'',revokeObjectURL:()=>{}};global.setTimeout=fn=>fn();
'''.replace('PAYLOAD', json.dumps(payload)) + gallery.CORE_JS + gallery.APP_JS + r'''
assert.equal(elems['visible-count'].textContent,'4 / 4 ca đang hiển thị');
const tolerance=inputs.find(e=>e.dataset.filter==='tolerance');tolerance.value='60';tolerance.listeners.change();
assert.equal(elems['visible-count'].textContent,'1 / 4 ca đang hiển thị');
assert.deepEqual(cases.filter(e=>!e.hidden).map(e=>e.dataset.recordId),['p002/r2']);
elems['export-selected'].click();assert.equal(exported.charCodeAt(0),0xfeff);
assert(exported.includes(String.fromCharCode(13,10)));
assert(exported.includes('"p002/r2"'));assert(!exported.includes('"p003/r3"'));
elems['reset'].click();assert.equal(cases.filter(e=>!e.hidden).length,4);
console.log(JSON.stringify({ok:true}));
'''
        result = subprocess.run(['node', '-e', program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
