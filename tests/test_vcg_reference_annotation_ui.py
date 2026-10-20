"""Independent annotation UI contracts, including executable browser logic.

The DOM/canvas smoke test runs the actual generated JavaScript in Node with
minimal browser stubs. It tests workflow behavior, not pixel-perfect rendering.
"""
from __future__ import annotations

import base64
import hashlib
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
import vcg_reference_annotation_ui as ui


def extract_payload(html):
    return json.loads(re.search(r'<script id="annotation-payload" type="application/json">(.*?)</script>',
                                html, re.S).group(1))


def executable_scripts(html):
    return '\n'.join(re.findall(r'<script id="annotation-(?:core|app)">(.*?)</script>', html, re.S))


class AnnotationToolTest(unittest.TestCase):
    def setUp(self):
        self.workspace = TemporaryDirectory(prefix='vcg08-ui-tests-')
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.out = self.root / 'out'
        self.time = np.arange(901, dtype=float) - 200
        self.xyz = np.column_stack([np.sin(self.time/50), np.cos(self.time/80), self.time/3000])
        self.frame = pd.DataFrame([
            dict(record_id='patient001/s0001', patient_id='patient001', split='train',
                 label_name='HC', diagnosis='Healthy control', fs=1000,
                 qrs_on_ms=-30., qrs_off_ms=50., t_on_ms=200., t_off_ms=450.),
            dict(record_id='patient002/s0002', patient_id='patient002', split='valid',
                 label_name='MI', diagnosis='Myocardial infarction', fs=1000,
                 qrs_on_ms=np.nan, qrs_off_ms=np.nan, t_on_ms=np.nan, t_off_ms=np.nan),
        ])
        self.write_artifact('patient001/s0001', True)
        self.write_artifact('patient002/s0002', False)

    def write_artifact(self, rid, median):
        path = self.artifacts / rid.replace('/', '_')
        artifact = dict(record_id=rid, fs=1000., time_reference='median_relative_ms',
                        median_anchor_sample=200 if median else None,
                        boundaries=dict(qrs_on=170, qrs_off=250, t_on=400, t_off=650) if median else {})
        path.with_suffix('.json').write_text(json.dumps(artifact))
        if median:
            np.savez_compressed(path.with_suffix('.npz'), median_xyz=self.xyz,
                                time_ms=self.time, speed=np.linspace(1, 40, len(self.time)))
        else:
            np.savez_compressed(path.with_suffix('.npz'), anchors=np.array([1000, 2000]))

    def build(self, records=None, artifact_root=None):
        return ui.build_annotation_tool(self.frame, records, artifact_root or self.artifacts, self.out)

    def test_blank_reference_times_and_unavailable_median_are_explicit(self):
        result = self.build()
        self.assertEqual(result['records'], 2)
        self.assertEqual(result['median_available'], 1)
        self.assertEqual(result['median_unavailable'], 1)
        table = pd.read_csv(result['template_path'])
        for boundary in ui.BOUNDARIES:
            self.assertTrue(table[boundary + '_ref_ms'].isna().all())
            self.assertTrue(table[boundary + '_status'].eq('not_visible').all())
        self.assertTrue(table.reviewer.isna().all())
        self.assertTrue(table.review_status.eq('pending').all())
        self.assertFalse(table.independent_reference.any())
        self.assertTrue(table.annotation_protocol.eq('frank_xyz_median_manual_v1').all())
        self.assertTrue((self.out/'review/annotation_protocol.md').is_file())
        self.assertIn('annotation_protocol.md', Path(result['html_path']).read_text())
        payload = extract_payload(Path(result['html_path']).read_text())
        first, missing = payload['records']
        self.assertEqual(first['time_ms'], self.time.tolist())
        np.testing.assert_allclose(first['xyz'], self.xyz, atol=6e-9)
        expected_sha = hashlib.sha256((self.artifacts/'patient001_s0001.npz').read_bytes()).hexdigest()
        self.assertEqual(first['signal_sha256'], expected_sha)
        self.assertEqual(json.loads(base64.b64decode(first['encoded_prediction'])),
                         {'qrs_on': -30., 'qrs_off': 50., 't_on': 200., 't_off': 450.})
        self.assertFalse(missing['signal_available'])
        self.assertEqual(missing['xyz'], [])
        self.assertEqual(missing['encoded_prediction'], '')
        self.assertIn('Không có median', missing['unavailable_reason'])
        self.assertIn('MEDIAN_UNAVAILABLE', table.iloc[1].review_notes)

    def test_existing_reference_csv_is_never_overwritten_and_inputs_are_unchanged(self):
        self.out.mkdir()
        reference = self.out/'reference_annotations.csv'
        original = b'record_id,reviewer,review_notes\npatient001/s0001,expert,do not overwrite\n'
        reference.write_bytes(original)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.artifacts.iterdir()}
        self.build()
        self.assertEqual(reference.read_bytes(), original)
        self.build()
        self.assertEqual(reference.read_bytes(), original)
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.artifacts.iterdir()})

    def test_selected_records_can_extend_and_parent_artifact_path_is_supported(self):
        result = self.build(pd.DataFrame({'record_id':['patient002/s0002','patient001/s0001','patient002/s0002']}), self.root)
        self.assertEqual(result['selected_record_ids'], ['patient002/s0002','patient001/s0001'])
        with self.assertRaisesRegex(ValueError, 'Unknown annotation'):
            self.build(['unknown/record'])
        with self.assertRaisesRegex(ValueError, 'No records'):
            self.build([])

    def test_invalid_time_coordinate_and_prediction_mismatch_are_rejected(self):
        source = self.artifacts/'patient001_s0001.json'
        original = source.read_text()
        artifact = json.loads(original)
        artifact['time_reference'] = 'raw_recording_ms'
        source.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'median coordinates'):
            self.build()
        source.write_text(original)
        bad = self.frame.copy()
        bad.loc[0, 't_off_ms'] = 451
        with self.assertRaisesRegex(ValueError, 'CSV/artifact mismatch'):
            ui.build_annotation_tool(bad, None, self.artifacts, self.out)

    def test_payload_escapes_script_in_record_metadata_and_has_no_network_dependencies(self):
        self.frame.loc[0, 'diagnosis'] = '</script><script>alert(1)</script>'
        result = self.build()
        html = Path(result['html_path']).read_text()
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertEqual(extract_payload(html)['records'][0]['diagnosis'], self.frame.iloc[0].diagnosis)
        self.assertNotRegex(html, r'<(?:script|link)[^>]+(?:src|href)=')
        self.assertNotIn('fetch(', html)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to execute standalone browser logic')
    def test_generated_javascript_blinding_reviewer_isolation_csv_and_edit_gates(self):
        result = self.build()
        html = Path(result['html_path']).read_text()
        payload = extract_payload(html)
        script = NODE_STUB.replace('__PAYLOAD__', json.dumps(json.dumps(payload)))
        script += '\n' + executable_scripts(html) + '\n' + NODE_ASSERTIONS
        target = self.root/'ui_smoke.cjs'
        target.write_text(script)
        checked = subprocess.run(['node', '--check', str(target)], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        executed = subprocess.run(['node', str(target)], capture_output=True, text=True)
        self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
        self.assertIn('annotation UI smoke passed', executed.stdout)


NODE_STUB = r'''
const assert=require('node:assert/strict');
const payloadText=__PAYLOAD__;
const elements=new Map(),localMemory=new Map();
class Element{
 constructor(id=''){this.id=id;this.value=id==='record'?'0':id==='active-boundary'?'qrs_on':id==='quality'?'unrated':id==='review-status'?'pending':id==='source'?'independent_visual_annotation':id.endsWith('-status')?'not_visible':'';this.checked=false;this.disabled=false;this.textContent=id==='annotation-payload'?payloadText:'';this.innerHTML='';this.children=[];this.listeners={};const set=new Set(['compare','diagnosis'].includes(id)?['hidden']:[]);this.classList={add:x=>set.add(x),remove:x=>set.delete(x),contains:x=>set.has(x),toggle:(x,force)=>{if(force===undefined)force=!set.has(x);force?set.add(x):set.delete(x);return force;}};}
 addEventListener(name,fn){this.listeners[name]=fn;}
 appendChild(child){this.children.push(child);}
 querySelector(name){return get(this.id+'-'+name);}
 getBoundingClientRect(){return {width:1050,height:610,left:0,top:0};}
 getContext(){return new Proxy({}, {get:(target,key)=>target[key]||(target[key]=()=>{}),set:(target,key,value)=>{target[key]=value;return true;}});}
 setPointerCapture(){}click(){}
}
function get(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);}
global.document={getElementById:get,createElement:name=>new Element(name)};
global.window={devicePixelRatio:1,addEventListener:()=>{}};
global.localStorage={getItem:key=>localMemory.get(key)||null,setItem:(key,value)=>localMemory.set(key,value)};
let allowConfirmation=true;
global.confirm=()=>allowConfirmation;
if(!global.crypto)global.crypto=require('node:crypto').webcrypto;
'''

NODE_ASSERTIONS = r'''
(async()=>{
 const r=records[0];
 assert.equal(shownPrediction,false);
 assert.equal(prediction,null);
 assert.equal($('reveal').disabled,true);
 assert.equal($('compare').classList.contains('hidden'),true);
 for(const b of C.boundaries)assert.equal($(b+'-value').value,'');
 assert.equal($('reviewer').value,'');
 $('reviewer').value='expert_A';switchReviewer();
 reveal();assert.equal(prediction,null,'unsaved reference cannot reveal V1');
 for(const [b,v] of Object.entries({qrs_on:-30,qrs_off:50,t_on:200,t_off:450})){$(b+'-value').value=String(v);$(b+'-status').value='visible';}
 $('quality').value='good';$('review-status').value='confirmed';$('independent').checked=true;$('notes').value='comma, quote " and\nnew line';change();
 assert.equal(draft.independent_reference,true);
 saveRow();assert.equal(dirty,false);assert.equal(session.rows[r.record_id].review_status,'confirmed');
 assert.equal(C.bool(session.rows[r.record_id].independent_reference),true);
 assert.ok(session.rows[r.record_id].annotation_id);assert.ok(session.rows[r.record_id].created_at);
 const savedBeforeReveal={...session.rows[r.record_id]};
 reveal();assert.equal(shownPrediction,true);assert.equal(prediction.t_on,200);
 assert.equal(session.rows[r.record_id].independent_reference,true,'reveal preserves earlier independent save');
 assert.equal(session.rows[r.record_id].prediction_revealed,true);
 let exported=null;download=(content,name)=>{exported={content,name};};exportRows();
 assert.equal(C.parseCSV(exported.content)[0].reviewer,'expert_A');
 $('reviewer').value='expert_B';switchReviewer();
 assert.equal(shownPrediction,false);assert.equal(prediction,null);
 assert.equal(Object.keys(session.rows).length,0);
 for(const b of C.boundaries)assert.equal($(b+'-value').value,'','second reviewer starts blind');
 for(const [b,v] of Object.entries({qrs_on:-29,qrs_off:51,t_on:201,t_off:451})){$(b+'-value').value=String(v);$(b+'-status').value='visible';}
 $('quality').value='good';$('review-status').value='confirmed';$('independent').checked=true;change();saveRow();
 const bRow={...session.rows[r.record_id]};exportRows();const parsed=C.parseCSV(exported.content);
 assert.equal(parsed.length,1);assert.equal(parsed[0].reviewer,'expert_B');
 assert.equal(Number(parsed[0].t_on_ref_ms),201);
 const combined=C.toCSV([savedBeforeReveal,bRow],DATA.columns);
 await importRows({target:{files:[{text:async()=>combined}],value:'x'}});
 assert.equal(Object.keys(session.rows).length,1);assert.equal(session.rows[r.record_id].reviewer,'expert_B');
 assert.equal(hasExposure(r.record_id),false);
 $('reviewer').value='expert_A';switchReviewer();
 assert.equal(shownPrediction,false,'reload does not automatically reveal prediction');
 assert.equal(hasExposure(r.record_id),true);
 assert.equal(draft.qrs_on_ref_ms,-30);
 $('t_off-value').value='452';change();
 assert.equal(draft.independent_reference,false);assert.equal(draft.post_reveal_edit,true);
 assert.equal(draft.review_status,'pending');saveRow();
 assert.equal(session.rows[r.record_id].independent_reference,false);
 assert.equal(session.rows[r.record_id].post_reveal_edit,true);
 $('reviewer').value='expert_B';switchReviewer();
 $('qrs_on-value').value='-28';change();allowConfirmation=false;
 $('record').value='1';$('record').listeners.change();
 assert.equal($('record').value,'0','cancelled record change restores original selection');
 allowConfirmation=true;saveRow();navigate(1);
 assert.equal(rec().signal_available,false);assert.equal($('qrs_on-value').disabled,true);assert.equal($('reveal').disabled,true);
 const missing=C.blankRow(records[1],'expert_B');missing.review_status='confirmed';missing.reference_quality='good';missing.independent_reference=true;
 assert.ok(C.validateRow(missing,records[1]).some(s=>s.includes('Không có median')));
 const valid={...savedBeforeReveal};
 assert.deepEqual(C.validateRow(valid,r),[]);
 assert.ok(C.validateRow({...valid,fs:500},r).length);
 assert.ok(C.validateRow({...valid,signal_sha256:'bad'},r).length);
 assert.ok(C.validateRow({...valid,qrs_on_ref_ms:-30.5},r).some(s=>s.includes('sample')));
 assert.ok(C.validateRow({...valid,qrs_on_ref_ms:100},r).some(s=>s.includes('sớm hơn')));
 assert.ok(C.validateRow({...valid,t_off_ref_ms:10000},r).some(s=>s.includes('ngoài median')));
 const values=[{a:'x,y',b:'a"b',c:'one\ntwo',d:'',e:true}];
 assert.deepEqual(C.parseCSV(C.toCSV(values,['a','b','c','d','e'])),[{a:'x,y',b:'a"b',c:'one\ntwo',d:'',e:'true'}]);
 assert.throws(()=>C.parseCSV('a,b\n"unfinished,b'));
 assert.throws(()=>C.parseCSV('a,a\n1,2'));
 assert.equal(C.nearestTime(r,-30.2),-30);
 assert.equal(C.nearestTime(r,-30.5),-31,'midpoint chooses earlier sample');
 assert.equal(C.nearestTime(r,200.5),200,'positive midpoint chooses earlier sample');
 assert.equal(C.nearestTime(r,-200.01),'','outside times are rejected, not clipped');
 assert.equal(C.nearestTime(r,700.01),'','outside times are rejected, not clipped');
 assert.equal(C.canReveal(C.blankRow(r,'expert_C'),r),false,'blank save cannot reveal V1');
 assert.equal(C.canReveal({...savedBeforeReveal,independent_reference:false},r),false,'nonindependent initial save cannot reveal');
 console.log('annotation UI smoke passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''


if __name__ == '__main__':
    unittest.main()
