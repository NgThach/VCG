"""Exercise the generated offline UI and manual-review persistence in Node."""
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
import vcg_visual_review_ui as ui


def payload_from(markup):
    return json.loads(re.search(r'<script id="visual-review-payload" type="application/json">(.*?)</script>', markup, re.S).group(1))


class VisualReviewUITest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix='vcg08-visual-ui-')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.frame = pd.DataFrame([
            dict(record_id='patient001/s1', patient_id='patient001', label_name='HC', split='train', fs=1000, qc_status='auto_candidate', qc_flags='', qrs_on_ms=-30., qrs_off_ms=50., t_on_ms=200., t_off_ms=450.),
            dict(record_id='patient002/s2', patient_id='patient002', label_name='MI', split='valid', fs=500, qc_status='needs_review', qc_flags='flag1|flag2', qrs_on_ms=-30., qrs_off_ms=50., t_on_ms=np.nan, t_off_ms=450.),
            dict(record_id='patient003/s3', patient_id='patient003', label_name='MI', split='test', fs=1000, qc_status='failed', qc_flags='median_unavailable', qrs_on_ms=np.nan, qrs_off_ms=np.nan, t_on_ms=np.nan, t_off_ms=np.nan),
        ])
        self.signals = self.frame.copy()
        self.signals['has_median'] = [True, True, False]
        self.signals['signal_sha256'] = ['a'*64, 'b'*64, 'c'*64]
        self.signals['payload_path'] = ['signal_data/'+r.replace('/', '_')+'.js' for r in self.frame.record_id]
        self.panels = pd.DataFrame({'record_id':self.frame.record_id, 'plot_path':['plots/a.png','plots/b.png','plots/c.png']})
        self.signal_payloads = {}
        for _, r in self.frame.iterrows():
            has_median = r.record_id != 'patient003/s3'
            time = np.arange(-200, 701, 1000/r.fs) if has_median else np.array([])
            xyz = np.column_stack([np.sin(time/70), np.cos(time/60), time/2000]) if has_median else np.empty((0,3))
            boundary_times = {b:float(r[b+'_ms']) if pd.notna(r[b+'_ms']) else None for b in ui.BOUNDARIES}
            boundaries = {b: int(np.argmin(abs(time-v))) if v is not None and len(time) else None for b,v in boundary_times.items()}
            self.signal_payloads[r.record_id] = dict(record_id=r.record_id, available=has_median, reason='' if has_median else 'No saved V1 median', meta=dict(fs=r.fs), time_ms=time.tolist(), xyz=xyz.tolist(), speed=np.ones(len(time)).tolist(), smooth_speed=np.ones(len(time)).tolist(), magnitude=np.linalg.norm(xyz,axis=1).tolist(), t_magnitude=np.linalg.norm(xyz,axis=1).tolist(), t_slope=np.ones(len(time)).tolist(), boundaries={**boundaries,'q_threshold':.2,'st_level':.1,'t_end_baseline':0.,'tp':[1,10]}, boundary_times_ms={**boundary_times, 'qrs_speed_peak':0., 't_peak':300.}, qrs_search=[max(0,int(.08*r.fs)),max(0,int(.35*r.fs))], qrs_loop_indices=[boundaries['qrs_on'],boundaries['qrs_off']] if has_median else None, t_loop_indices=[boundaries['t_on'],boundaries['t_off']] if has_median and boundaries['t_on'] is not None else None)

    def build(self):
        return ui.build_visual_review_gallery(self.frame, self.signals, self.panels, self.out)

    def test_exact_cohort_coverage_and_no_candidate_membership_filter(self):
        result = self.build()
        self.assertEqual((result['records'], result['median_available'], result['median_unavailable']), (3,2,1))
        markup = Path(result['html_path']).read_text()
        payload = payload_from(markup)
        self.assertEqual([r['record_id'] for r in payload['records']], self.frame.record_id.tolist())
        self.assertEqual([r['segmentation_status'] for r in payload['records']], ['full','partial','missing_median'])
        self.assertNotIn('fetch(', markup)
        self.assertNotRegex(markup, r'(?:src|href)="https?://')
        self.assertIn('window.VCGVisualReview={receiveRecord}', markup)
        self.assertEqual(payload['records'][2]['qrs_on_ms'], None)
        template = pd.read_csv(result['review_template_path'])
        self.assertFalse(template.independent_reference.any())
        self.assertTrue(template.source.eq('manual_visual_review').all())
        self.assertTrue(template.review_status.eq('visual_review_only').all())
        self.assertTrue(template.qrs_on_manual_ms.isna().all())
        self.assertTrue(template.qrs_on_v1_minus_manual_ms.isna().all())

    def test_regeneration_preserves_csv_and_stable_session_identity(self):
        result = self.build()
        original_sha = result['dataset_sha256']
        path = Path(result['review_template_path'])
        path.write_text('reviewer,record_id\nexpert,do not overwrite\n')
        reference = self.out.parent/'reference_annotations.csv'
        reference.write_text('reference stays unchanged\n')
        self.panels['plot_path'] = 'new_render.png'
        self.signals['payload_sha256'] = ['d'*64]*3
        rebuilt = self.build()
        self.assertEqual(rebuilt['dataset_sha256'], original_sha)
        self.assertEqual(path.read_text(), 'reviewer,record_id\nexpert,do not overwrite\n')
        self.assertEqual(reference.read_text(), 'reference stays unchanged\n')
        self.frame.loc[0,'t_on_ms'] = 201
        self.signals.loc[0,'t_on_ms'] = 201
        self.assertNotEqual(self.build()['dataset_sha256'], original_sha)

    def test_manifest_coverage_duplicates_and_html_injection(self):
        self.signals = self.signals.iloc[:2]
        with self.assertRaisesRegex(ValueError, 'exact cohort'):
            self.build()
        self.signals = pd.concat([self.signals, self.signals.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'unique record_id'):
            self.build()
        self.signals = self.frame.copy()
        self.signals['has_median'] = True
        self.frame.loc[0, 'diagnosis'] = '</script><script>alert(1)</script>'
        markup = Path(self.build()['html_path']).read_text()
        self.assertNotIn('</script><script>alert(1)', markup)
        self.assertEqual(payload_from(markup)['records'][0]['diagnosis'], self.frame.iloc[0].diagnosis)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for generated JavaScript workflow tests')
    def test_actual_dom_workflow_navigation_filter_zoom_layers_manual_csv_and_reload(self):
        markup = Path(self.build()['html_path']).read_text()
        payload = payload_from(markup)
        ids = re.findall(r'id="([^\"]+)"', markup)
        checked = re.findall(r'id="([^\"]+)"[^>]*type="checkbox"[^>]*checked', markup)
        heights = dict(re.findall(r'id="(plot-[^\"]+)" data-height="([^\"]+)"', markup))
        source = (NODE_STUB.replace('__PAYLOAD_TEXT__', json.dumps(json.dumps(payload)))
                  .replace('__SIGNALS__', json.dumps(self.signal_payloads))
                  .replace('__IDS__', json.dumps(ids))
                  .replace('__CHECKED__', json.dumps(checked))
                  .replace('__HEIGHTS__', json.dumps(heights)))
        source += '\n'.join(re.findall(r'<script id="visual-review-(?:core|app)">(.*?)</script>', markup, re.S))
        source += NODE_ASSERTIONS
        target = self.out/'visual_ui.cjs'
        target.write_text(source)
        checked_script = subprocess.run(['node', '--check', str(target)], capture_output=True, text=True)
        self.assertEqual(checked_script.returncode, 0, checked_script.stderr)
        run = subprocess.run(['node', str(target)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('visual UI workflow passed', run.stdout)


NODE_STUB = r'''
const assert=require('node:assert/strict'),payloadText=__PAYLOAD_TEXT__,signalPayloads=__SIGNALS__;
const ids=__IDS__,checkedIds=__CHECKED__,heights=__HEIGHTS__,elements=new Map(),localMemory=new Map();
let denyStorage=false;
class Element{
 constructor(id=''){this.id=id;this.value=id==='active-boundary'?'qrs_on':id.endsWith('-status')?'khong_danh_gia':'';this.checked=checkedIds.includes(id);this.disabled=false;this.hidden=false;this.open=false;this.textContent=id==='visual-review-payload'?payloadText:'';this.children=[];this.listeners={};this.style={};this.dataset={height:heights[id]};this.classList={toggle:()=>{}};}
 addEventListener(name,fn){this.listeners[name]=fn;}
 appendChild(child){this.children.push(child);return child;}
 replaceChildren(){this.children=[];}
 setPointerCapture(){}remove(){}removeAttribute(name){delete this[name];}click(){}
 getBoundingClientRect(){return{width:1100,height:Number(this.dataset.height)||330,left:0,top:0};}
 getContext(){if(!this.context)this.context=new Proxy({}, {get:(o,k)=>o[k]||(o[k]=()=>{}),set:(o,k,v)=>{o[k]=v;return true;}});return this.context;}
}
function get(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);}
ids.forEach(get);
global.document={getElementById:get,createElement:name=>new Element(name),head:{appendChild(script){const safe=decodeURI(script.src).split('/').at(-1).replace(/\.js$/,'');const found=Object.values(signalPayloads).find(p=>p.record_id.replaceAll('/','_')===safe);if(found){window.VCGVisualReview.receiveRecord(found.record_id,found);if(script.onload)script.onload();}else if(script.onerror)script.onerror();}}};
global.window={devicePixelRatio:1,addEventListener:()=>{}};
global.localStorage={getItem:key=>{if(denyStorage)throw Error('storage blocked');return localMemory.get(key)||null;},setItem:(key,value)=>{if(denyStorage)throw Error('storage blocked');localMemory.set(key,value);}};
'''

NODE_ASSERTIONS = r'''
(async()=>{
 assert.equal(records.length,3);assert.equal(selected.length,3);
 assert.equal(payload.record_id,records[0].record_id);
 assert.equal($('record').children.length,3,'every record is in initial selector');
 assert.equal(session.exposures[currentId]!==undefined,true,'default markers disclose prediction');
 $('reviewer').value='reviewer_A';switchReviewer();
 $('t_on-manual').value='205';$('t_on-status').value='qua_som';$('t_on-note').value='too early, by eye';$('review-notes').value='quote " and comma,\nnew line';manualChange();
 assert.equal(session.rows[currentId].t_on_v1_minus_manual_ms,-5);
 assert.equal(session.rows[currentId].independent_reference,false);
 assert.equal(session.rows[currentId].source,'manual_visual_review');
 assert.equal(session.rows[currentId].review_status,'visual_review_only');
 assert.ok(session.rows[currentId].created_at);assert.ok(session.rows[currentId].updated_at);
 const savedId=currentId,key=STORE+'reviewer_A';assert.ok(localMemory.has(key));
 const fullWidth=views.full[1]-views.full[0];zoom('full',.5);assert.ok(views.full[1]-views.full[0]<fullWidth);
 const beforePan=[...views.full];panView('full',.1);assert.ok(views.full[0]>beforePan[0]);
 const originalQrs=[...views.qrs];$('plot-qrs').listeners.wheel({preventDefault(){},shiftKey:false,deltaY:-1,clientX:550});assert.ok(views.qrs[1]-views.qrs[0]<originalQrs[1]-originalQrs[0]);
 $('hide-all').onclick();assert.equal(predictionsShown(),false);assert.equal(draft.has_viewed_prediction,true,'hiding never erases exposure');
 $('show-qrs_on').checked=true;layerChange();assert.equal(predictionsShown(),true);
 $('click-mode').checked=true;$('active-boundary').value='qrs_on';
 $('plot-full').listeners.pointerdown({clientX:400,pointerId:1});$('plot-full').listeners.pointerup({clientX:400,clientY:100});assert.equal(typeof draft.qrs_on_manual_ms,'number','click adds a sample');
 const clickValue=draft.qrs_on_manual_ms;
 $('plot-full').listeners.pointerdown({clientX:400,pointerId:1});$('plot-full').listeners.pointermove({clientX:440});$('plot-full').listeners.pointerup({clientX:440,clientY:100});assert.equal(draft.qrs_on_manual_ms,clickValue,'drag pans without placing manual boundary');
 navigate(1);assert.equal(currentId,records[1].record_id);assert.equal(payload.available,true);assert.equal(payload.t_loop_indices,null);assert.equal(draft.t_on_v1_ms,'');assert.equal(draft.t_on_v1_minus_manual_ms,'');
 navigate(1);assert.equal(currentId,records[2].record_id);assert.equal(payload.available,false);assert.equal($('qrs_on-manual').disabled,true,'missing median disables manual time');assert.ok($('message').textContent.includes('No saved V1 median'));
 navigate(1);assert.equal(currentId,savedId);assert.equal(draft.t_on_manual_ms,205);assert.equal(draft.review_notes,'quote " and comma,\nnew line');
 $('filter-label_name').value='MI';filterRecords();assert.equal(selected.length,2);assert.equal(currentId,records[1].record_id);
 $('filter-segmentation_status').value='missing_median';filterRecords();assert.equal(selected.length,1);assert.equal(currentId,records[2].record_id);
 $('filter-label_name').value='';$('filter-segmentation_status').value='';$('filter-search').value='patient001';filterRecords();assert.equal(selected.length,1);assert.equal(currentId,savedId);
 $('filter-search').value='';filterRecords();setMode('overview');assert.equal($('overview').children.length,3);assert.equal($('detail').hidden,true);assert.equal(Object.keys(session.exposures).length,3);setMode('detail');
 $('filter-search').value='no_matching_recording_ever';filterRecords();assert.equal(selected.length,0);assert.equal(payload,null);assert.equal($('detail').hidden,true);assert.equal($('qrs_on-manual').disabled,true,'empty filters cannot annotate a stale record');
 $('filter-search').value='';filterRecords();assert.equal(selected.length,3);assert.equal($('detail').hidden,false);assert.ok(payload.available,'clearing filters restores the first real signal');
 $('reviewer').value='reviewer_B';switchReviewer();assert.equal(draft.t_on_manual_ms,'','reviewers have isolated drafts');$('reviewer').value='reviewer_A';switchReviewer();assert.equal(draft.t_on_manual_ms,205);
 let downloaded=null;download=(content,name)=>{downloaded={content,name};};exportRows();let imported=C.parseCSV(downloaded.content);assert.equal(imported.find(r=>r.record_id===savedId).t_on_v1_minus_manual_ms,'-5');assert.ok(imported.every(r=>r.independent_reference==='false'));assert.ok(imported.every(r=>r.review_status==='visual_review_only'));
 const originalCSV=downloaded.content;session={rows:{},exposures:{}};await importRows({target:{files:[{text:async()=>originalCSV}],value:'x'}});assert.equal(session.rows[savedId].t_on_manual_ms,205);assert.equal(session.rows[savedId].independent_reference,false);
 const forged=imported.map(r=>({...r,independent_reference:'true',source:'independent_reference',review_status:'confirmed'}));await importRows({target:{files:[{text:async()=>C.toCSV(forged,DATA.review_columns)}],value:'x'}});assert.equal(session.rows[savedId].independent_reference,false);assert.equal(session.rows[savedId].source,'manual_visual_review');assert.equal(session.rows[savedId].review_status,'visual_review_only');
 assert.equal(C.nearestTime(signalPayloads[savedId],-30.5),-31);assert.equal(C.nearestTime(signalPayloads[savedId],700.1),'');assert.equal(C.nearestTime(signalPayloads[records[1].record_id],201),200,'fs500 uses actual samples');
 assert.equal(C.finalize({...C.blankRow(records[2],'X'),qrs_on_manual_ms:''},records[2]).qrs_on_v1_minus_manual_ms,'','missing is not zero');
 assert.deepEqual(C.parseCSV(C.toCSV([{a:'a,b',b:'a"b',c:'line\nnext',d:''}],['a','b','c','d'])),[{a:'a,b',b:'a"b',c:'line\nnext',d:''}]);assert.throws(()=>C.parseCSV('a,a\n1,2'));assert.throws(()=>C.parseCSV('a,b\n"unfinished,b'));
 const appendOriginal=document.head.appendChild;cache.delete(records[1].record_id);document.head.appendChild=node=>node.onerror();openRecord(records[1].record_id);assert.equal(selected.length,3,'payload load failure must never drop cohort membership');assert.equal(currentId,records[1].record_id);assert.ok($('message').textContent.includes('Không tải được'));assert.equal($('qrs_on-manual').disabled,true);document.head.appendChild=appendOriginal;openRecord(savedId);
 denyStorage=true;$('review-notes').value='export survives unavailable storage';saveRow();assert.equal(storageOK,false);assert.ok($('storage-status').textContent.includes('Không lưu được'));exportRows();assert.ok(downloaded.content.includes('export survives unavailable storage'));
 console.log('visual UI workflow passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''


if __name__ == '__main__':
    unittest.main()
