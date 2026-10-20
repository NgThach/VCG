#!/usr/bin/env python3
"""Offline, exhaustive visual review of frozen V1 Frank XYZ segmentation.

This UI never creates an independent reference. Signals are lazy-loaded as
local script callbacks so opening the HTML with file:// needs no web server.
Browser drafts are namespaced by signal/prediction identity and reviewer, not
by UI version, so regenerating figures does not erase a review session.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

BOUNDARIES = ('qrs_on', 'qrs_off', 't_on', 't_off')
REVIEW_STATUSES = ('dat', 'qua_som', 'qua_muon', 'khong_ro', 'khong_danh_gia')
REVIEW_COLUMNS = ['record_id', 'patient_id', 'label_name', 'split', 'fs',
                  'signal_sha256', 'time_reference', 'reviewer', 'source',
                  'independent_reference', 'review_status', 'created_at',
                  'updated_at', 'has_viewed_prediction', 'prediction_first_viewed_at',
                  'blind_unblind_status', 'review_notes'] + [
    f'{boundary}_{suffix}' for boundary in BOUNDARIES
    for suffix in ('v1_ms', 'manual_ms', 'v1_minus_manual_ms', 'status', 'note')]


def _clean(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _json(value):
    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).replace('<', '\\u003c')


def _as_frame(manifest):
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    if isinstance(manifest, dict):
        for key in ('manifest', 'frame', 'records'):
            if key in manifest:
                return _as_frame(manifest[key])
        for key in ('manifest_path', 'manifest_csv'):
            if key in manifest:
                return pd.read_csv(manifest[key])
    if isinstance(manifest, (str, Path)):
        return pd.read_csv(manifest)
    return pd.DataFrame(manifest)


def _relative_url(value, output_dir):
    if not value:
        return ''
    path = Path(str(value))
    if path.is_absolute():
        path = Path(os.path.relpath(path, output_dir))
    elif not (output_dir / path).is_file() and (Path.cwd() / path).is_file():
        path = Path(os.path.relpath(path.resolve(), output_dir))
    return quote(path.as_posix(), safe='/')


def _items(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x]
    return [x.strip() for x in str(value).replace(';', '|').split('|') if x.strip()]


def build_visual_review_gallery(frame, signal_manifest, panel_manifest, output_dir):
    """Replace the existing primary gallery; keep all cohort rows visible.

    The signal and panel manifests must cover the exact cohort once. Existing
    review/reference CSVs are never read as truth or overwritten. All browser
    sessions persist independently of regenerated HTML/PNG content.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, signals, panels = frame.copy(), _as_frame(signal_manifest), _as_frame(panel_manifest)
    if not frame.record_id.is_unique:
        raise ValueError('Duplicate cohort record_id')
    ids = set(frame.record_id.astype(str))
    for label, manifest in [('signal', signals), ('panel', panels)]:
        if 'record_id' not in manifest or not manifest.record_id.is_unique:
            raise ValueError(f'{label} manifest must have unique record_id')
        if set(manifest.record_id.astype(str)) != ids:
            raise ValueError(f'{label} manifest does not cover the exact cohort')
    signal_lookup = signals.set_index('record_id').to_dict('index')
    panel_lookup = panels.set_index('record_id').to_dict('index')
    records = []
    for _, row in frame.iterrows():
        record = _clean(row.to_dict())
        rid = str(record['record_id'])
        signal, panel = _clean(signal_lookup[rid]), _clean(panel_lookup[rid])
        record.update(signal)
        record['record_id'] = rid
        payload_path = next((signal.get(k) for k in ('payload_path', 'signal_path', 'script_path')
                             if signal.get(k)), f'signal_data/{rid.replace("/", "_")}.js')
        record['payload_url'] = _relative_url(payload_path, output_dir)
        plot_path = next((panel.get(k) for k in ('plot_path', 'png_path', 'figure_path')
                          if panel.get(k)), '')
        record['plot_url'] = _relative_url(plot_path, output_dir)
        record['qc_flags'] = str(record.get('qc_flags') or '')
        record['qc_flag_items'] = _items(record.get('qc_flags'))
        record['warnings'] = _items(record.get('warnings'))
        for source, keys in ((signal, ('reason', 'data_integrity_issue', 'data_integrity_issues', 'integrity_issues', 'load_error')), (panel, ('render_error', 'plot_error', 'reason'))):
            for key in keys:
                for warning in _items(source.get(key)):
                    if warning not in record['warnings']:
                        record['warnings'].append(warning)
        if not record['warnings']:
            record['warnings'] = record['qc_flag_items']
        available = record.get('has_median', record.get('available', record.get('signal_available')))
        if isinstance(available, str):
            available = available.lower() in ('true', '1', 'yes')
        if available is None:
            available = record.get('median_available', False)
        record['has_median'] = bool(available)
        count = sum(record.get(b + '_ms') is not None for b in BOUNDARIES)
        record['segmentation_status'] = ('missing_median' if not available else
                                         'full' if count == 4 else 'partial')
        record['signal_sha256'] = signal.get('signal_sha256') or signal.get('artifact_npz_sha256') or ''
        if not record['signal_sha256']:
            record['signal_sha256'] = signal.get('payload_sha256') or ''
        records.append(record)
    identity = [(r['record_id'], r['signal_sha256'],
                 *[r.get(b + '_ms') for b in BOUNDARIES]) for r in records]
    dataset_sha = hashlib.sha256(_json(sorted(identity)).encode()).hexdigest()
    payload = dict(schema_version=1, dataset_sha256=dataset_sha, records=records,
                   review_columns=REVIEW_COLUMNS, review_statuses=list(REVIEW_STATUSES),
                   boundaries=list(BOUNDARIES))
    rows = []
    for r in records:
        row = dict.fromkeys(REVIEW_COLUMNS, '')
        row.update({k: r.get(k, '') for k in ('record_id', 'patient_id', 'label_name', 'split',
                                             'fs', 'signal_sha256')})
        row.update(time_reference='median_relative_ms', source='manual_visual_review',
                   independent_reference=False, review_status='visual_review_only',
                   has_viewed_prediction=False, blind_unblind_status='unknown')
        for b in BOUNDARIES:
            row[b + '_v1_ms'] = r.get(b + '_ms')
            row[b + '_status'] = 'khong_danh_gia'
        rows.append(row)
    template_path = output_dir / 'manual_visual_review_template.csv'
    if not template_path.exists():
        pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_csv(template_path, index=False)
    target = output_dir / 'all_cases_gallery.html'
    markup = HTML.replace('__VISUAL_REVIEW_PAYLOAD__', _json(payload))
    target.write_text(markup, encoding='utf-8')
    result = dict(html_path=str(target), gallery_path=str(target), records=len(records),
                  median_available=sum(r['has_median'] for r in records),
                  median_unavailable=sum(not r['has_median'] for r in records),
                  dataset_sha256=dataset_sha, review_template_path=str(template_path),
                  default_mode='single_record_detail', lazy_local_scripts=True,
                  independent_reference=False, source='manual_visual_review',
                  review_csv_preserved=True)
    (output_dir / 'visual_review_ui_manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


CORE_JS = r'''
const VisualReviewCore=(()=>{
 const boundaries=['qrs_on','qrs_off','t_on','t_off'];
 const finite=x=>x!==''&&x!==null&&x!==undefined&&Number.isFinite(Number(x));
 const bool=x=>x===true||String(x).toLowerCase()==='true'||String(x)==='1';
 function matches(r,f){const term=String(f.search||'').trim().toLowerCase();if(term&&![r.record_id,r.patient_id].some(v=>String(v||'').toLowerCase().includes(term)))return false;for(const k of ['label_name','split','qc_status','segmentation_status'])if(f[k]&&r[k]!==f[k])return false;if(f.qc_flag&&!(r.qc_flag_items||[]).includes(f.qc_flag))return false;if(f.warnings==='yes'&&!(r.warnings||[]).length)return false;if(f.warnings==='no'&&(r.warnings||[]).length)return false;return true;}
 function blankRow(r,reviewer){const x={};for(const k of ['record_id','patient_id','label_name','split','fs','signal_sha256'])x[k]=r[k]??'';Object.assign(x,{time_reference:'median_relative_ms',reviewer,source:'manual_visual_review',independent_reference:false,review_status:'visual_review_only',created_at:'',updated_at:'',has_viewed_prediction:false,prediction_first_viewed_at:'',blind_unblind_status:'prediction_hidden_no_exposure_recorded',review_notes:''});for(const b of boundaries){x[b+'_v1_ms']=finite(r[b+'_ms'])?Number(r[b+'_ms']):'';x[b+'_manual_ms']='';x[b+'_v1_minus_manual_ms']='';x[b+'_status']='khong_danh_gia';x[b+'_note']='';}return x;}
 function finalize(row,r,exposed=false){const x={...row,source:'manual_visual_review',independent_reference:false,review_status:'visual_review_only'};x.has_viewed_prediction=bool(x.has_viewed_prediction)||exposed;x.blind_unblind_status=x.has_viewed_prediction?'unblinded_visual_review':'prediction_hidden_no_exposure_recorded';for(const b of boundaries){x[b+'_v1_ms']=finite(r[b+'_ms'])?Number(r[b+'_ms']):'';x[b+'_v1_minus_manual_ms']=finite(x[b+'_v1_ms'])&&finite(x[b+'_manual_ms'])?Number((x[b+'_v1_ms']-Number(x[b+'_manual_ms'])).toFixed(8)):'';}return x;}
 function nearestTime(p,v){if(!p||!p.available||!finite(v)||!p.time_ms.length)return '';const t=p.time_ms,n=Number(v);if(n<t[0]-1e-8||n>t[t.length-1]+1e-8)return '';let a=0,z=t.length-1;while(a<z){const m=Math.floor((a+z)/2);if(t[m]<n)a=m+1;else z=m;}if(a>0&&n-t[a-1]<=t[a]-n+1e-9)a--;return t[a];}
 function clampView(view,full,minWidth){let width=Math.max(minWidth,Math.min(full[1]-full[0],view[1]-view[0]));width=Math.min(width,full[1]-full[0]);let a=Math.max(full[0],Math.min(view[0],full[1]-width));return[a,a+width];}
 function csvCell(v){const s=v===null||v===undefined?'':String(v);return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
 function toCSV(rows,columns){return '\ufeff'+[columns.map(csvCell).join(','),...rows.map(r=>columns.map(c=>csvCell(r[c])).join(','))].join('\r\n')+'\r\n';}
 function parseCSV(input){const text=String(input).replace(/^\ufeff/,''),all=[];let row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else if(!quoted&&cell.length)throw Error('CSV quote không hợp lệ');else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(v=>v!==''))all.push(row);row=[];cell='';}else cell+=c;}if(quoted)throw Error('CSV chưa đóng dấu quote');if(cell!==''||row.length){row.push(cell);if(row.some(v=>v!==''))all.push(row);}if(!all.length)return[];const columns=all.shift();if(new Set(columns).size!==columns.length)throw Error('CSV có cột trùng');return all.map(values=>{if(values.length!==columns.length)throw Error('Số cột CSV không khớp');return Object.fromEntries(columns.map((c,i)=>[c,values[i]]));});}
 return{boundaries,finite,bool,matches,blankRow,finalize,nearestTime,clampView,toCSV,parseCSV};
})();
'''

APP_JS = r'''
'use strict';
const DATA=JSON.parse(document.getElementById('visual-review-payload').textContent),C=VisualReviewCore,$=id=>document.getElementById(id),records=DATA.records,recordMap=new Map(records.map(r=>[r.record_id,r]));
const COLORS={qrs_on:'#087db6',qrs_off:'#158367',t_on:'#bb5d00',t_off:'#8053b8'},LABELS={qrs_on:'QRS-on',qrs_off:'QRS-off',t_on:'T-on',t_off:'T-off'},XYZCOLORS=['#176aa0','#c96b14','#7850b1'];
const STORE='vcg-visual-review-v1:'+DATA.dataset_sha256+':';
let currentId=records.length?records[0].record_id:'',selected=[...records],reviewer='',session={rows:{},exposures:{}},draft=null,payload=null,cache=new Map(),views={},pan=null,hover=null,storageOK=true,mode='detail',rotation=-.72,tilt=.46,loadingId='',scriptNode=null;
function rec(){return recordMap.get(currentId);}
function say(message,kind=''){const el=$('message');el.textContent=message;el.className='message '+kind;}
function fields(){return Object.fromEntries(['search','label_name','split','qc_status','qc_flag','segmentation_status','warnings'].map(k=>[k,$('filter-'+k).value]));}
function persist(){try{localStorage.setItem(STORE+(reviewer||'__anonymous__'),JSON.stringify(session));$('storage-status').textContent='Đã lưu trong trình duyệt; xuất CSV để sao lưu.';}catch(error){storageOK=false;$('storage-status').textContent='Không lưu được LocalStorage. Dữ liệu còn trong phiên này: hãy xuất CSV trước khi đóng trang.';}}
function readSession(){try{const x=JSON.parse(localStorage.getItem(STORE+(reviewer||'__anonymous__')))||{};session={rows:x.rows||{},exposures:x.exposures||{}};}catch(error){storageOK=false;session={rows:{},exposures:{}};$('storage-status').textContent='Không đọc được LocalStorage; hãy nhập CSV đã sao lưu.';}}
function predictionsShown(){return C.boundaries.some(b=>$('show-'+b).checked)||$('show-candidates').checked||$('show-tangents').checked;}
function expose(rid){if(!rid)return;const now=new Date().toISOString();session.exposures[rid]=session.exposures[rid]||now;const row=session.rows[rid];if(row){row.has_viewed_prediction=true;row.prediction_first_viewed_at=row.prediction_first_viewed_at||session.exposures[rid];row.blind_unblind_status='unblinded_visual_review';}if(rid===currentId&&draft){draft.has_viewed_prediction=true;draft.prediction_first_viewed_at=draft.prediction_first_viewed_at||session.exposures[rid];draft.blind_unblind_status='unblinded_visual_review';}persist();}
function readForm(){if(!draft)return null;const row={...draft};for(const b of C.boundaries){const raw=$(b+'-manual').value;row[b+'_manual_ms']=raw===''?'':Number(raw);row[b+'_status']=$(b+'-status').value;row[b+'_note']=$(b+'-note').value;}row.review_notes=$('review-notes').value;return row;}
function fillForm(){if(!draft)return;for(const b of C.boundaries){$(b+'-manual').value=draft[b+'_manual_ms']??'';$(b+'-status').value=draft[b+'_status']||'khong_danh_gia';$(b+'-note').value=draft[b+'_note']||'';$(b+'-prediction').textContent=C.finite(draft[b+'_v1_ms'])?draft[b+'_v1_ms']+' ms':'Thiếu';$(b+'-delta').textContent=C.finite(draft[b+'_v1_minus_manual_ms'])?(Number(draft[b+'_v1_minus_manual_ms'])>=0?'+':'')+draft[b+'_v1_minus_manual_ms']+' ms':'—';} $('review-notes').value=draft.review_notes||'';$('exposure-state').textContent=draft.has_viewed_prediction?'Đã xem prediction; ẩn lại không xóa lịch sử.':'Chưa ghi nhận hiện prediction trong phiên reviewer này. Đây vẫn là visual review, không phải reference độc lập.';}
function saveRow(options={}){if(!draft||!rec())return false;let row=readForm();const errors=[];for(const b of C.boundaries){const v=row[b+'_manual_ms'];if(v!==''&&!C.finite(v))errors.push(LABELS[b]+': cần số hữu hạn hoặc để trống');if(C.finite(v)&&payload&&payload.available&&(Number(v)<payload.time_ms[0]||Number(v)>payload.time_ms.at(-1)))errors.push(LABELS[b]+': ngoài miền median');}if(errors.length){say(errors.join('; '),'error');return false;}const now=new Date().toISOString();row=C.finalize(row,rec(),!!session.exposures[currentId]);row.reviewer=reviewer;row.created_at=row.created_at||now;row.updated_at=now;row.prediction_first_viewed_at=row.prediction_first_viewed_at||session.exposures[currentId]||'';session.rows[currentId]={...row};draft={...row};persist();fillForm();$('saved-state').textContent='Đã lưu '+now+' · '+Object.keys(session.rows).length+'/'+records.length+' ca trong phiên';if(!options.quiet)say('Đã lưu review trực quan. Δ = thời gian V1 − thời gian thủ công; giá trị dương nghĩa V1 muộn hơn.');return true;}
function manualChange(){if(saveRow({quiet:true}))drawAll();}
function switchReviewer(){saveRow({quiet:true});reviewer=$('reviewer').value.trim();readSession();openRecord(currentId,{save:false});$('session-name').textContent=reviewer||'Phiên chưa đặt tên (vẫn lưu cục bộ)';}
function enableManual(enabled){for(const b of C.boundaries){$(b+'-manual').disabled=!enabled;$(b+'-snap').disabled=!enabled;} $('active-boundary').disabled=!enabled;}
function updateMetadata(){const r=rec();if(!r)return;$('record-title').textContent=r.record_id;$('metadata').textContent=[r.patient_id,r.label_name,'split '+r.split,'fs '+(r.fs??'?')+' Hz','QC '+(r.qc_status||'không rõ'),'phân đoạn '+r.segmentation_status].join(' · ');$('qc-details').textContent='Cảnh báo / QC: '+((r.qc_flags||'Không có cờ QC')+(r.warnings?.length?' · '+r.warnings.join(' | '):''));$('test-notice').hidden=r.split!=='test';$('data-provenance').textContent='Median V1 đã lưu; anchor = 0 ms · signal_sha256: '+(r.signal_sha256||'xem manifest');$('static-figure').href=r.plot_url||'#';$('static-preview').removeAttribute('src');$('static-caption').textContent=r.has_median?'Hình V1 gốc độ phân giải cao (A–D và loop 3D)':'Không có median: hình chẩn đoán raw; không dùng thời gian raw thay median';$('static-figure').hidden=!r.plot_url;$('record').value=currentId;$('position').textContent=(selected.findIndex(x=>x.record_id===currentId)+1)+' / '+selected.length+' trong bộ lọc · '+records.length+' toàn cohort';}
function emptyCanvases(message){for(const name of ['full','qrs','t','loops']){const c=$('plot-'+name),p=layout(c),ctx=c.getContext('2d');ctx.clearRect(0,0,p.w,p.h);ctx.fillStyle='#fff';ctx.fillRect(0,0,p.w,p.h);ctx.fillStyle='#8b4c22';ctx.font='14px system-ui';ctx.fillText(message,24,48);}}
function openRecord(rid,options={}){if(options.save!==false&&!saveRow({quiet:true}))return false;if(!recordMap.has(rid))return false;currentId=rid;payload=null;views={};hover=null;loadingId=rid;draft=C.finalize(session.rows[rid]||C.blankRow(rec(),reviewer),rec(),!!session.exposures[rid]);fillForm();updateMetadata();enableManual(false);$('static-details').open=false;emptyCanvases('Đang tải tín hiệu đã lưu…');expose(rid);if(cache.has(rid)){receiveRecord(rid,cache.get(rid));return true;}const script=document.createElement('script');script.src=rec().payload_url;script.async=true;script.onerror=()=>{if(currentId!==rid)return;say('Không tải được '+rec().payload_url+'. Ca vẫn được giữ trong danh sách; xem hình diagnostic và nguyên nhân từ manifest.','error');emptyCanvases('Lỗi tải tín hiệu — recording vẫn có trong cohort.');enableManual(false);};script.onload=()=>{if(currentId===rid&&!cache.has(rid)){say('Payload không trả callback cho '+rid,'error');emptyCanvases('Payload không hợp lệ; không tự tạo dữ liệu thay thế.');}script.remove();};document.head.appendChild(script);scriptNode=script;return true;}
function receiveRecord(rid,data){cache.set(rid,data);if(cache.size>12){for(const key of cache.keys()){if(key!==currentId){cache.delete(key);break;}}}if(rid!==currentId)return;payload=data;if(!data||data.record_id!==rid){say('record_id trong payload không khớp; không hiển thị dữ liệu.','error');payload=null;return;}if(!data.available){enableManual(false);say(data.reason||'Không có median V1; không tạo biên hoặc loop thay thế.','error');emptyCanvases('Không có median V1; mở hình raw diagnostic bên dưới.');return;}if(!Array.isArray(data.time_ms)||data.time_ms.length!==data.xyz.length||data.time_ms.length<2){say('Payload median không hợp lệ; không tạo dữ liệu thay thế.','error');payload=null;return;}const diffs=[];for(const b of C.boundaries){const actual=data.boundary_times_ms?.[b];const expected=rec()[b+'_ms'];if((C.finite(actual)!==C.finite(expected))||(C.finite(actual)&&Math.abs(Number(actual)-Number(expected))>1e-6))diffs.push(b);}if(diffs.length){say('Biên payload khác manifest V1: '+diffs.join(', ')+'. Dừng overlay để không thay prediction.','error');payload=null;emptyCanvases('Lỗi nhất quán mốc V1; xem manifest kiểm tra.');return;}const first=data.time_ms[0],last=data.time_ms.at(-1),q=data.qrs_search||[],b=data.boundaries||{};views.full=[first,last];views.qrs=C.clampView([q.length?data.time_ms[q[0]]-20:-120,q.length?data.time_ms[Math.min(q[1],data.time_ms.length-1)]+20:150],[first,last],20);const qoff=data.boundary_times_ms?.qrs_off;const ts=b.t_search||[];views.t=C.clampView([C.finite(qoff)?Number(qoff)-30:(ts.length?data.time_ms[ts[0]]-30:80),last],[first,last],20);enableManual(true);say('Tín hiệu median V1 gốc. Cuộn để zoom; kéo ngang để pan. Bật “Đặt mốc bằng click” để đánh dấu.');drawAll();}
window.VCGVisualReview={receiveRecord};
function filterRecords(){
 if(draft&&!saveRow({quiet:true}))return;
 selected=records.filter(r=>C.matches(r,fields()));$('record').replaceChildren();
 for(const r of selected){const el=document.createElement('option');el.value=r.record_id;el.textContent=r.record_id+' · '+r.label_name+' · '+r.split+' · '+r.segmentation_status;$('record').appendChild(el);}
 if(!selected.length){
  $('position').textContent='0 / '+records.length+' ca theo bộ lọc';$('prev').disabled=true;$('next').disabled=true;
  currentId='';payload=null;draft=null;enableManual(false);emptyCanvases('Không có recording phù hợp với bộ lọc.');
  $('record-title').textContent='Không có recording phù hợp';
  for(const id of ['metadata','qc-details','data-provenance'])$(id).textContent='';
  for(const b of C.boundaries){$(b+'-prediction').textContent='';$(b+'-manual').value='';$(b+'-delta').textContent='';$(b+'-note').value='';}
  say('Không có recording phù hợp. Xóa bộ lọc để xem lại toàn bộ cohort.');$('overview').replaceChildren();$('detail').hidden=true;return;
 }
 $('detail').hidden=mode!=='detail';$('prev').disabled=false;$('next').disabled=false;
 const target=selected.some(r=>r.record_id===currentId)?currentId:selected[0].record_id;openRecord(target,{save:false});if(mode==='overview')buildOverview();
}
function navigate(delta){if(!selected.length)return;const i=selected.findIndex(r=>r.record_id===currentId);openRecord(selected[(i+delta+selected.length)%selected.length].record_id);}
function buildOverview(){const box=$('overview');box.replaceChildren();for(const r of selected){const article=document.createElement('article');article.className='overview-case';article.dataset.recordId=r.record_id;const button=document.createElement('button');button.className='case-link';button.textContent=r.record_id+' · '+r.label_name+' · '+r.split;button.onclick=()=>{setMode('detail');openRecord(r.record_id);};article.appendChild(button);const status=document.createElement('p');status.textContent=r.segmentation_status+' · '+(r.qc_status||'')+(r.warnings?.length?' · có cảnh báo':'');article.appendChild(status);if(r.plot_url){const img=document.createElement('img');img.src=r.plot_url;img.loading='lazy';img.decoding='async';img.alt='V1 '+r.record_id;img.onclick=button.onclick;article.appendChild(img);}box.appendChild(article);expose(r.record_id);} }
function setMode(value){saveRow({quiet:true});mode=value;$('detail').hidden=value!=='detail';$('overview').hidden=value!=='overview';$('mode-detail').classList.toggle('selected',value==='detail');$('mode-overview').classList.toggle('selected',value==='overview');if(value==='overview')buildOverview();else drawAll();}
function layerChange(){if(predictionsShown())expose(currentId);if(draft){draft=C.finalize(draft,rec(),!!session.exposures[currentId]);fillForm();}drawAll();}
function layout(canvas){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1,w=Math.max(360,rect.width||1100),h=Number(canvas.dataset.height)||340;if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return{w,h,left:76,right:w-25,top:29,bottom:h-37};}
function niceStep(n){const power=10**Math.floor(Math.log10(Math.max(n,1e-10))),part=n/power;return(part<=1?1:part<=2?2:part<=5?5:10)*power;}
function numeric(v){if(Math.abs(v)<1e-10)return'0';return Math.abs(v)>=100?String(Math.round(v)):Number(v.toPrecision(3)).toString();}
function timeAtIndex(index){return payload&&Number.isInteger(index)&&index>=0&&index<payload.time_ms.length?payload.time_ms[index]:null;}
function boundaryTime(name){return payload?.boundary_times_ms?.[name]??null;}
function drawTimeline(name,defs){const canvas=$('plot-'+name),ctx=canvas.getContext('2d'),p=layout(canvas),t=payload.time_ms,view=views[name]||views.full;ctx.clearRect(0,0,p.w,p.h);ctx.fillStyle='#fff';ctx.fillRect(0,0,p.w,p.h);ctx.font='12px system-ui';const gap=39,height=(p.bottom-p.top-gap*(defs.length-1))/defs.length,xx=v=>p.left+(v-view[0])/(view[1]-view[0])*(p.right-p.left);let lo=0,hi=t.length-1;while(lo<t.length-1&&t[lo]<view[0])lo++;while(hi>0&&t[hi]>view[1])hi--;lo=Math.max(0,lo-1);hi=Math.min(t.length-1,hi+1);
 for(let panel=0;panel<defs.length;panel++){const d=defs[panel],top=p.top+panel*(height+gap),bottom=top+height;let ymin=Infinity,ymax=-Infinity;for(const s of d.series)for(let i=lo;i<=hi;i++)if(C.finite(s.values?.[i])){ymin=Math.min(ymin,Number(s.values[i]));ymax=Math.max(ymax,Number(s.values[i]));}for(const line of d.horizontal||[])if(C.finite(line.value)){ymin=Math.min(ymin,Number(line.value));ymax=Math.max(ymax,Number(line.value));}if(!Number.isFinite(ymin)){ymin=0;ymax=1;}const pad=Math.max((ymax-ymin)*.11,d.unit==='mV'?0.003:0.1);ymin-=pad;ymax+=pad;const yy=v=>bottom-(v-ymin)/(ymax-ymin)*height;ctx.save();ctx.beginPath();ctx.rect(p.left,top,p.right-p.left,height);ctx.clip();for(const [a,z,color]of[['qrs_on','qrs_off','#e0f2fc'],['t_on','t_off','#fff0dc']])if($('show-'+a).checked&&$('show-'+z).checked&&C.finite(boundaryTime(a))&&C.finite(boundaryTime(z))){ctx.fillStyle=color;ctx.fillRect(xx(boundaryTime(a)),top,xx(boundaryTime(z))-xx(boundaryTime(a)),height);}if(name==='qrs'&&Array.isArray(payload.qrs_search)){const a=timeAtIndex(payload.qrs_search[0]),z=timeAtIndex(Math.min(payload.qrs_search[1],t.length-1));if(C.finite(a)&&C.finite(z)){ctx.fillStyle='#5e829411';ctx.fillRect(xx(a),top,xx(z)-xx(a),height);}}ctx.restore();ctx.strokeStyle='#e3e9ef';ctx.lineWidth=1;ctx.textAlign='right';for(let j=0;j<=4;j++){const y=top+height*j/4;ctx.beginPath();ctx.moveTo(p.left,y);ctx.lineTo(p.right,y);ctx.stroke();ctx.fillStyle='#536979';ctx.fillText(numeric(ymax-(ymax-ymin)*j/4),p.left-9,y+4);}ctx.textAlign='left';ctx.fillStyle='#173d52';ctx.font='bold 12px system-ui';ctx.fillText(d.title+' ('+d.unit+')',p.left,top-10);ctx.font='12px system-ui';let lx=p.left+240;for(const s of d.series){ctx.fillStyle=s.color;ctx.fillText(s.name,lx,top-10);lx+=s.name.length*7+23;}
 ctx.save();ctx.beginPath();ctx.rect(p.left,top,p.right-p.left,height);ctx.clip();const step=niceStep((view[1]-view[0])/10);ctx.strokeStyle='#e7edf2';ctx.lineWidth=1;for(let x=Math.ceil(view[0]/step)*step;x<=view[1];x+=step){ctx.beginPath();ctx.moveTo(xx(x),top);ctx.lineTo(xx(x),bottom);ctx.stroke();}for(const s of d.series){ctx.strokeStyle=s.color;ctx.lineWidth=s.width||1.4;ctx.setLineDash(s.dash||[]);ctx.beginPath();let started=false;for(let i=lo;i<=hi;i++){if(!C.finite(s.values?.[i])){started=false;continue;}const x=xx(t[i]),y=yy(s.values[i]);if(started)ctx.lineTo(x,y);else{ctx.moveTo(x,y);started=true;}}ctx.stroke();ctx.setLineDash([]);}for(const line of d.horizontal||[]){if(!C.finite(line.value))continue;ctx.strokeStyle=line.color;ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(p.left,yy(line.value));ctx.lineTo(p.right,yy(line.value));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=line.color;ctx.fillText(line.name+' '+numeric(Number(line.value)),p.left+8,yy(line.value)-4);}
 for(const b of C.boundaries){if($('show-'+b).checked&&C.finite(boundaryTime(b))){ctx.strokeStyle=COLORS[b];ctx.lineWidth=1.9;ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(xx(boundaryTime(b)),top);ctx.lineTo(xx(boundaryTime(b)),bottom);ctx.stroke();if(panel===0){ctx.fillStyle=COLORS[b];const index=C.boundaries.indexOf(b);ctx.fillText(LABELS[b]+' '+numeric(Number(boundaryTime(b))),xx(boundaryTime(b))+4,top+14+(index%2)*14);}}if($('show-manual').checked&&draft&&C.finite(draft[b+'_manual_ms'])){ctx.strokeStyle='#dc2765';ctx.lineWidth=2;ctx.setLineDash([7,4]);ctx.beginPath();ctx.moveTo(xx(Number(draft[b+'_manual_ms'])),top);ctx.lineTo(xx(Number(draft[b+'_manual_ms'])),bottom);ctx.stroke();ctx.setLineDash([]);}}
 for(const marker of d.markers||[]){if(!C.finite(marker.time))continue;ctx.strokeStyle=marker.color;ctx.lineWidth=1.1;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(xx(marker.time),top);ctx.lineTo(xx(marker.time),bottom);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=marker.color;ctx.fillText(marker.name,xx(marker.time)+4,bottom-8);}
 if(name==='t'&&d.unit==='mV'&&d.kind==='magnitude'&&$('show-tangents').checked){const b=payload.boundaries||{},m=payload.t_magnitude||[],slope=payload.t_slope||[];for(const[contact,intercept,color,label]of[['t_rise_index','t_tangent_on','#9b6519','tiếp tuyến lên'],['t_fall_index','t_tangent_off','#7450a2','tiếp tuyến xuống']]){const i=b[contact],a=timeAtIndex(i),z=boundaryTime(intercept);if(!C.finite(a)||!C.finite(m[i])||!C.finite(slope[i]))continue;const hasCrossing=C.finite(z),start=Math.max(view[0],hasCrossing?Math.min(a,z)-12:a-18),end=Math.min(view[1],hasCrossing?Math.max(a,z)+12:a+18);ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.setLineDash([8,3]);ctx.beginPath();ctx.moveTo(xx(start),yy(m[i]+slope[i]*(start-a)/1000));ctx.lineTo(xx(end),yy(m[i]+slope[i]*(end-a)/1000));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=color;ctx.beginPath();ctx.arc(xx(a),yy(m[i]),3.5,0,2*Math.PI);ctx.fill();ctx.fillText(label+(hasCrossing?'':' · chưa có giao điểm'),xx(a)+5,yy(m[i])-7);}}
 if(hover&&hover.name===name){ctx.strokeStyle='#536779';ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(xx(hover.time),top);ctx.lineTo(xx(hover.time),bottom);ctx.stroke();ctx.setLineDash([]);}ctx.restore();ctx.strokeStyle='#a8bac6';ctx.strokeRect(p.left,top,p.right-p.left,height);if(panel===defs.length-1){ctx.fillStyle='#435e70';ctx.textAlign='center';for(let x=Math.ceil(view[0]/step)*step;x<=view[1]+1e-9;x+=step)ctx.fillText(numeric(x),xx(x),bottom+18);ctx.fillText('Thời gian so với QRS anchor (ms)',(p.left+p.right)/2,p.h-3);ctx.textAlign='left';}}
 $('range-'+name).textContent=view.map(n=>n.toFixed(1)).join(' → ')+' ms';}
function drawAll(){if(!payload?.available)return;const xyz=[0,1,2].map((i)=>({name:['X','Y','Z'][i],values:payload.xyz.map(v=>v[i]),color:XYZCOLORS[i]}));drawTimeline('full',[{title:'A · Median Frank XYZ',unit:'mV',series:xyz}]);const b=payload.boundaries||{};drawTimeline('qrs',[{title:'B · QRS Frank XYZ',unit:'mV',series:xyz},{title:'Spatial velocity',unit:'mV/s',series:[{name:'velocity gốc',values:payload.speed,color:'#a0aab5',width:1},{name:'V1 dùng phân đoạn',values:payload.smooth_speed,color:'#153d66',width:1.7}],horizontal:[{value:b.q_threshold,name:'Ngưỡng QRS',color:'#b54431'}],markers:[{time:boundaryTime('qrs_speed_peak'),name:'speed peak',color:'#8c2e4e'},...(payload.qrs_search||[]).map(i=>({time:timeAtIndex(Math.min(i,payload.time_ms.length-1)),name:'cửa sổ tìm QRS',color:'#687b88'}))]}]);const horizontal=[{value:b.st_level,name:'ST',color:'#c36b18'},{value:b.t_end_baseline,name:b.tp?'TP':'PQ fallback',color:'#7351a2'}];if($('show-candidates').checked)horizontal.push({value:b.t_threshold,name:'ngưỡng T',color:'#4b8578'},{value:b.t_end_threshold,name:'ngưỡng cuối T',color:'#527b92'});const markers=[{time:boundaryTime('t_peak'),name:'T peak',color:'#ae2756'}];if($('show-candidates').checked)for(const k of ['t_tangent_on','t_threshold_on','t_tangent_off','t_threshold_off'])markers.push({time:boundaryTime(k),name:k.replace('t_',''),color:k.includes('threshold')?'#368474':'#9a6c99'});drawTimeline('t',[{title:'C · T Frank XYZ',unit:'mV',series:xyz},{title:'Magnitude |XYZ|',unit:'mV',kind:'magnitude',series:[{name:'magnitude gốc',values:payload.magnitude,color:'#a5adb5',width:1},{name:'magnitude V1',values:payload.t_magnitude,color:'#234a68',width:1.8}],horizontal,markers}]);drawLoops();}
function drawLoops(){const canvas=$('plot-loops'),p=layout(canvas),ctx=canvas.getContext('2d');ctx.clearRect(0,0,p.w,p.h);ctx.fillStyle='#fff';ctx.fillRect(0,0,p.w,p.h);let extent=0;for(const indices of [payload.qrs_loop_indices,payload.t_loop_indices])if(indices)for(let i=indices[0];i<=indices[1];i++)for(const v of payload.xyz[i]||[])extent=Math.max(extent,Math.abs(v));extent=Math.max(extent*1.12,.001);const half=p.w/2,scale=Math.min(half-80,p.h-95)/(2*extent);const project=(v,cx,cy)=>{const ca=Math.cos(rotation),sa=Math.sin(rotation),ct=Math.cos(tilt),st=Math.sin(tilt),x=ca*v[0]-sa*v[1],y=sa*v[0]+ca*v[1],z=v[2];return[cx+x*scale,cy-(ct*z-st*y)*scale];};for(const[slot,label,indices,color]of[[0,'QRS loop',payload.qrs_loop_indices,'#187fa5'],[1,'T loop',payload.t_loop_indices,'#c87815']]){const cx=half*(slot+.5),cy=p.h*.52;ctx.font='bold 14px system-ui';ctx.fillStyle='#234558';ctx.fillText(label,half*slot+22,25);ctx.font='12px system-ui';if(!indices){ctx.fillStyle='#8c5128';ctx.fillText('Thiếu biên hợp lệ — không dựng loop.',half*slot+22,58);continue;}for(let a=0;a<3;a++){const base=project([0,0,0],cx,cy),v=[0,0,0];v[a]=extent;const z=project(v,cx,cy);ctx.strokeStyle=['#6fa4b9','#c69e68','#9a83bb'][a];ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(...base);ctx.lineTo(...z);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(['X','Y','Z'][a]+' (mV)',z[0]+4,z[1]);}ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.beginPath();for(let i=indices[0];i<=indices[1];i++){const point=project(payload.xyz[i],cx,cy);i===indices[0]?ctx.moveTo(...point):ctx.lineTo(...point);}ctx.stroke();for(const [index,fill,name]of[[indices[0],'#14844f','bắt đầu'],[indices[1],'#d0444c','kết thúc']]){const v=project(payload.xyz[index],cx,cy);ctx.fillStyle=fill;ctx.beginPath();ctx.arc(v[0],v[1],4,0,2*Math.PI);ctx.fill();ctx.fillText(name,v[0]+6,v[1]-6);}const stride=Math.max(1,Math.floor((indices[1]-indices[0])/5));for(let i=indices[0]+stride;i<indices[1];i+=stride){const a=project(payload.xyz[i-1],cx,cy),z=project(payload.xyz[i],cx,cy),ang=Math.atan2(z[1]-a[1],z[0]-a[0]);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(z[0]-6*Math.cos(ang-.5),z[1]-6*Math.sin(ang-.5));ctx.lineTo(...z);ctx.lineTo(z[0]-6*Math.cos(ang+.5),z[1]-6*Math.sin(ang+.5));ctx.stroke();}ctx.fillStyle='#4c6576';ctx.fillText('sample '+indices[0]+' → '+indices[1]+' · '+timeAtIndex(indices[0])+' → '+timeAtIndex(indices[1])+' ms',half*slot+22,p.h-30);}ctx.fillStyle='#526a7a';ctx.fillText('Cùng gốc XYZ, cùng tỉ lệ '+numeric(extent)+' mV; điểm xanh bắt đầu, đỏ kết thúc; mũi tên theo thời gian.',22,p.h-7);}
function changeView(name,next){if(!payload?.available)return;views[name]=C.clampView(next,[payload.time_ms[0],payload.time_ms.at(-1)],Math.max(8,10*1000/payload.meta.fs));drawAll();}
function zoom(name,factor,at=null){if(!payload?.available||!views[name])return;const view=views[name],mid=at===null?(view[0]+view[1])/2:at;changeView(name,[mid-(mid-view[0])*factor,mid+(view[1]-mid)*factor]);}
function panView(name,fraction){const view=views[name];if(view){const d=(view[1]-view[0])*fraction;changeView(name,[view[0]+d,view[1]+d]);}}
function pointerTime(name,event){const canvas=$('plot-'+name),p=layout(canvas),x=event.clientX-canvas.getBoundingClientRect().left;return views[name][0]+(x-p.left)/(p.right-p.left)*(views[name][1]-views[name][0]);}
function bindCanvas(name){const canvas=$('plot-'+name);canvas.addEventListener('wheel',event=>{if(!payload?.available)return;event.preventDefault();if(event.shiftKey)panView(name,.1*Math.sign(event.deltaY||event.deltaX));else zoom(name,event.deltaY>0?1.2:1/1.2,pointerTime(name,event));},{passive:false});canvas.addEventListener('pointerdown',event=>{if(!payload?.available)return;pan={name,x:event.clientX,start:[...views[name]],moved:false};canvas.setPointerCapture(event.pointerId);});canvas.addEventListener('pointermove',event=>{if(!payload?.available)return;if(pan&&pan.name===name){const dx=event.clientX-pan.x;if(Math.abs(dx)>3)pan.moved=true;if(pan.moved){const p=layout(canvas),dt=dx/(p.right-p.left)*(pan.start[1]-pan.start[0]);changeView(name,[pan.start[0]-dt,pan.start[1]-dt]);return;}}hover={name,time:C.nearestTime(payload,pointerTime(name,event))};$('hover').textContent='Con trỏ '+hover.time+' ms · '+LABELS[$('active-boundary').value];drawAll();});canvas.addEventListener('pointerup',event=>{const moved=pan?.moved;pan=null;if(moved||!$('click-mode').checked||!payload?.available)return;const p=layout(canvas),box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;if(x<p.left||x>p.right||y<p.top||y>p.bottom)return;const b=$('active-boundary').value,value=C.nearestTime(payload,pointerTime(name,event));if(value==='')return;$(b+'-manual').value=value;manualChange();});canvas.addEventListener('pointerleave',()=>{if(!pan){hover=null;drawAll();}});}
function download(content,name){const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),500);}
function exportRows(){if(!saveRow({quiet:true}))return;const rows=Object.values(session.rows).map(row=>C.finalize(row,recordMap.get(row.record_id),!!session.exposures[row.record_id]));download(C.toCSV(rows,DATA.review_columns),'manual_visual_review_'+(reviewer||'anonymous').replace(/[^a-zA-Z0-9_-]/g,'_')+'.csv');say('Đã xuất '+rows.length+' dòng review. independent_reference luôn false; CSV này không tự trở thành ground truth.');}
async function importRows(event){const file=event.target.files?.[0];if(!file)return;try{const parsed=C.parseCSV(await file.text()),mine=parsed.filter(x=>String(x.reviewer||'').trim()===reviewer),seen=new Set(),ready=[];if(!mine.length)throw Error('CSV không có dòng của reviewer hiện tại. Đặt đúng tên reviewer trước khi nhập.');for(const input of mine){if(seen.has(input.record_id))throw Error('Trùng recording của reviewer trong CSV: '+input.record_id);seen.add(input.record_id);const r=recordMap.get(input.record_id);if(!r)throw Error('record_id ngoài cohort: '+input.record_id);if(input.signal_sha256!==r.signal_sha256||Number(input.fs)!==Number(r.fs)||input.time_reference!=='median_relative_ms')throw Error('Hash tín hiệu, fs hoặc hệ thời gian không khớp: '+input.record_id);const row={...C.blankRow(r,reviewer),...input};for(const b of C.boundaries){const key=b+'_manual_ms';if(row[key]!==''&&!C.finite(row[key]))throw Error('Thời gian không hợp lệ: '+key);if(C.finite(row[key])&&!r.has_median)throw Error('Không có median để đặt biên: '+r.record_id);if(C.finite(row[key])&&((C.finite(r.time_start_ms)&&Number(row[key])<Number(r.time_start_ms))||(C.finite(r.time_end_ms)&&Number(row[key])>Number(r.time_end_ms))))throw Error('Biên ngoài miền median: '+r.record_id+' '+b);row[key]=row[key]===''?'':Number(row[key]);if(!DATA.review_statuses.includes(row[b+'_status']))throw Error('Trạng thái biên không hợp lệ: '+b);}row.has_viewed_prediction=C.bool(input.has_viewed_prediction)||!!session.exposures[r.record_id];ready.push(C.finalize(row,r,row.has_viewed_prediction));}for(const row of ready){const old=session.rows[row.record_id];if(old?.updated_at&&row.updated_at&&old.updated_at>row.updated_at)continue;session.rows[row.record_id]=row;if(row.has_viewed_prediction)session.exposures[row.record_id]=session.exposures[row.record_id]||row.prediction_first_viewed_at||new Date().toISOString();}persist();openRecord(currentId,{save:false});say('Đã nhập '+ready.length+' dòng. Giữ review mới hơn theo updated_at; lịch sử đã xem prediction được giữ.');}catch(error){say('Không nhập CSV: '+error.message,'error');}finally{event.target.value='';}}
function setup(){for(const key of ['label_name','split','qc_status']){const values=[...new Set(records.map(r=>r[key]).filter(Boolean))].sort();for(const value of values){const o=document.createElement('option');o.value=value;o.textContent=value;$('filter-'+key).appendChild(o);}}for(const value of [...new Set(records.flatMap(r=>r.qc_flag_items||[]))].sort()){const o=document.createElement('option');o.value=value;o.textContent=value;$('filter-qc_flag').appendChild(o);}for(const key of ['search','label_name','split','qc_status','qc_flag','segmentation_status','warnings'])$('filter-'+key).addEventListener(key==='search'?'input':'change',filterRecords);$('record').onchange=()=>openRecord($('record').value);$('prev').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);$('mode-detail').onclick=()=>setMode('detail');$('mode-overview').onclick=()=>setMode('overview');$('use-reviewer').onclick=switchReviewer;$('reviewer').addEventListener('keydown',e=>{if(e.key==='Enter')switchReviewer();});$('save').onclick=()=>saveRow();$('export').onclick=exportRows;$('import').onchange=importRows;$('review-notes').addEventListener('input',manualChange);$('hide-all').onclick=()=>{for(const b of C.boundaries)$('show-'+b).checked=false;$('show-candidates').checked=false;$('show-tangents').checked=false;layerChange();};$('show-all').onclick=()=>{for(const b of C.boundaries)$('show-'+b).checked=true;layerChange();};for(const b of C.boundaries){$('show-'+b).addEventListener('change',layerChange);for(const suffix of ['manual','status','note'])$(b+'-'+suffix).addEventListener('input',manualChange);$(b+'-snap').onclick=()=>{if(!payload?.available)return;const value=C.nearestTime(payload,$(b+'-manual').value);if(value===''){say('Thời gian ngoài miền median hoặc không hợp lệ; không gán mép cửa sổ.','error');return;}$(b+'-manual').value=value;manualChange();};}for(const key of ['manual','candidates','tangents'])$('show-'+key).addEventListener('change',layerChange);for(const name of ['full','qrs','t']){bindCanvas(name);$('zoom-'+name+'-in').onclick=()=>zoom(name,1/1.5);$('zoom-'+name+'-out').onclick=()=>zoom(name,1.5);$('pan-'+name+'-left').onclick=()=>panView(name,-.2);$('pan-'+name+'-right').onclick=()=>panView(name,.2);$('fit-'+name).onclick=()=>{if(!payload?.available)return;if(name==='full')changeView(name,[payload.time_ms[0],payload.time_ms.at(-1)]);else{const saveCurrent=draft;receiveRecord(currentId,payload);draft=saveCurrent;fillForm();}};}$('rotation').oninput=()=>{rotation=Number($('rotation').value);if(payload?.available)drawLoops();};$('tilt').oninput=()=>{tilt=Number($('tilt').value);if(payload?.available)drawLoops();};$('static-details').addEventListener('toggle',()=>{if($('static-details').open){expose(currentId);$('static-preview').src=rec().plot_url;}});$('static-figure').onclick=()=>expose(currentId);window.addEventListener('resize',drawAll);window.addEventListener('beforeunload',()=>saveRow({quiet:true}));readSession();filterRecords();}
setup();
'''


def _controls(name):
    return f'''<div class="plot-tools"><button id="zoom-{name}-in">＋ zoom</button><button id="zoom-{name}-out">− zoom</button><button id="pan-{name}-left">←</button><button id="pan-{name}-right">→</button><button id="fit-{name}">Về cửa sổ ban đầu</button><span id="range-{name}" class="muted"></span></div>'''


def _boundary_rows():
    labels = dict(zip(BOUNDARIES, ['QRS-on', 'QRS-off', 'T-on', 'T-off']))
    status_labels = {'dat': 'đạt', 'qua_som': 'V1 quá sớm', 'qua_muon': 'V1 quá muộn',
                     'khong_ro': 'không rõ biên', 'khong_danh_gia': 'không đánh giá'}
    return ''.join(f'''<tr><th>{labels[b]}</th><td id="{b}-prediction"></td><td><input id="{b}-manual" type="number" step="any" aria-label="{labels[b]} thủ công ms"><button id="{b}-snap" class="small">Về sample</button></td><td id="{b}-delta"></td><td><select id="{b}-status">{''.join(f'<option value="{s}"{ " selected" if s == "khong_danh_gia" else ""}>{status_labels[s]}</option>' for s in REVIEW_STATUSES)}</select></td><td><input id="{b}-note" aria-label="Ghi chú {labels[b]}"></td></tr>''' for b in BOUNDARIES)


HTML = r'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Notebook 08 · Quan sát QRS/T V1 trên toàn bộ recordings</title>
<style>
:root{font:15px/1.5 system-ui,-apple-system,sans-serif;color:#193747;background:#edf2f6}*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:20px}.card{background:white;border:1px solid #d4e0e8;border-radius:12px;padding:18px;margin-bottom:15px;box-shadow:0 2px 7px #17394b07}h1{font-size:25px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}h3{font-size:16px;margin:0 0 7px}p{margin:7px 0}.muted{color:#657c8c;font-size:13px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.grow{flex:1;min-width:240px}button,input,select,textarea{font:inherit}button{border:1px solid #b9ccd8;background:#f6fafc;border-radius:6px;padding:7px 12px;color:#23495e;cursor:pointer}button:hover{background:#e4f1f9}button.selected,.primary{background:#176b92;color:white;border-color:#176b92}button:disabled,input:disabled{opacity:.5;cursor:default}input,select,textarea{border:1px solid #b7cbd8;border-radius:5px;padding:7px;max-width:100%;background:white}input[type=checkbox]{accent-color:#146b93}label{display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap}.filters{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:10px}.filters select,.filters input{width:100%;min-width:0}.record-nav{display:grid;grid-template-columns:auto minmax(200px,1fr) auto;gap:10px;margin-top:12px}.message{border-left:4px solid #3188ad;padding:9px 12px;background:#edf7fc;margin-top:10px;white-space:pre-wrap}.message.error{border-color:#bf6843;background:#fff3e9}.test-note{background:#fff7e4;border-left:4px solid #c59126;padding:8px 12px}.layers{display:flex;gap:13px;flex-wrap:wrap;padding:8px 0}.layers label{font-size:14px}.plot-tools{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:8px 0}canvas{width:100%;display:block;touch-action:none;border:1px solid #d5e1e8;border-radius:5px;background:#fff}.review-table{width:100%;border-collapse:collapse}.review-table th,.review-table td{border-bottom:1px solid #dce7ed;padding:9px 7px;text-align:left;font-size:14px}.review-table input[type=number]{width:115px}.review-table input:not([type=number]){width:100%;min-width:100px}.small{font-size:11px;padding:3px 5px;margin-left:3px}textarea{width:100%;min-height:70px}.hint{font-size:13px;color:#5d7382}a{color:#126c98}#static-preview{width:100%;height:auto}#overview{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:14px}.overview-case{background:white;border:1px solid #d4e0e8;padding:10px;border-radius:9px;min-width:0}.overview-case img{width:100%;cursor:zoom-in}.overview-case p{font-size:12px}.case-link{width:100%;text-align:left;font-weight:600}#detail[hidden],#overview[hidden],[hidden]{display:none!important}.legend-line{display:inline-block;width:22px;border-top:3px solid #1680a6}.manual-line{border-color:#dc2765;border-top-style:dashed}.warning{color:#965725}summary{cursor:pointer;font-weight:600}details{margin-top:12px}code{overflow-wrap:anywhere}@media(max-width:850px){main{padding:10px}.filters{grid-template-columns:1fr 1fr}#overview{grid-template-columns:1fr}.table-wrap{overflow-x:auto}.card{padding:12px}.review-table{min-width:830px}}
</style></head><body><main>
<section class="card"><h1>Quan sát trực tiếp QRS/T trên median Frank XYZ V1</h1><p>Xem toàn bộ cohort, phóng to tín hiệu, kiểm tra vùng cắt và loop, rồi ghi lại nhận xét thủ công. Các biên là prediction V1 đã lưu; mức đồng thuận tangent/threshold không xác nhận biên đúng.</p>
<div class="row"><button id="mode-detail" class="selected">Chi tiết một recording</button><button id="mode-overview">Tổng quan nhiều recordings</button><span class="hint">Offline · không cần server · chỉ tải dữ liệu recording đang mở</span></div>
<div class="filters" style="margin-top:12px"><input id="filter-search" placeholder="Tìm record_id / patient_id" aria-label="Tìm recording"><select id="filter-label_name" aria-label="HC/MI"><option value="">HC + MI</option></select><select id="filter-split" aria-label="Split"><option value="">train + valid + test</option></select><select id="filter-qc_status" aria-label="Trạng thái QC"><option value="">Mọi trạng thái QC</option></select><select id="filter-segmentation_status" aria-label="Phân đoạn"><option value="">Mọi trạng thái phân đoạn</option><option value="full">Đủ bốn biên</option><option value="partial">Thiếu ít nhất một biên</option><option value="missing_median">Thiếu median</option></select><select id="filter-qc_flag" aria-label="Cờ QC"><option value="">Mọi cờ QC</option></select><select id="filter-warnings" aria-label="Cảnh báo"><option value="">Có / không có cảnh báo</option><option value="yes">Có cảnh báo</option><option value="no">Không có cảnh báo</option></select></div>
<div class="record-nav"><button id="prev">← Previous</button><select id="record" aria-label="Danh sách tất cả recordings"></select><button id="next">Next →</button></div><p id="position" class="muted"></p>
</section>
<div id="overview" hidden></div>
<div id="detail">
<section class="card"><h2 id="record-title"></h2><p id="metadata"></p><p id="qc-details" class="hint"></p><p id="test-notice" class="test-note" hidden>Tập test: chỉ quan sát và ghi review; không dùng để chỉnh ngưỡng hoặc thuật toán.</p><p id="data-provenance" class="muted"></p><div id="message" class="message"></div>
<div class="layers"><strong>Lớp prediction V1:</strong><label><input id="show-qrs_on" type="checkbox" checked>QRS-on</label><label><input id="show-qrs_off" type="checkbox" checked>QRS-off</label><label><input id="show-t_on" type="checkbox" checked>T-on</label><label><input id="show-t_off" type="checkbox" checked>T-off</label><button id="hide-all" class="small">Ẩn prediction + candidates</button><button id="show-all" class="small">Hiện bốn biên</button></div>
<div class="layers"><label><input id="show-manual" type="checkbox" checked><span class="legend-line manual-line"></span>Mốc thủ công</label><label><input id="show-candidates" type="checkbox">Candidates (nét chấm)</label><label><input id="show-tangents" type="checkbox">Tiếp tuyến và điểm tiếp xúc</label></div><p class="hint">V1: nét liền và vùng QRS xanh / T cam. Thủ công: nét đứt hồng. Candidates không thay các biên cuối cùng. Kéo ngang để pan; cuộn chuột để zoom, Shift + cuộn để pan.</p>
</section>
<section class="card"><h2>A · Toàn bộ median beat XYZ và hai khoảng cắt V1</h2><p class="hint">Tín hiệu ngoài QRS/T được giữ để quan sát sườn đầu, đoạn ST và đuôi T. Anchor của median ở 0 ms.</p>__FULL_CONTROLS__<canvas id="plot-full" data-height="330" style="height:330px" aria-label="Toàn median XYZ"></canvas></section>
<section class="card"><h2>B · Phóng to QRS: XYZ và spatial velocity</h2>__QRS_CONTROLS__<canvas id="plot-qrs" data-height="485" style="height:485px" aria-label="QRS XYZ và velocity"></canvas></section>
<section class="card"><h2>C · Phóng to T: XYZ, magnitude, ST và TP</h2>__T_CONTROLS__<canvas id="plot-t" data-height="500" style="height:500px" aria-label="T XYZ và magnitude"></canvas></section>
<section class="card"><h2>D · QRS loop và T loop 3D bằng đúng biên V1</h2><div class="row"><label>Xoay <input id="rotation" type="range" min="-3.14" max="3.14" step=".01" value="-.72"></label><label>Góc nghiêng <input id="tilt" type="range" min="-1.2" max="1.2" step=".01" value=".46"></label></div><canvas id="plot-loops" data-height="395" style="height:395px" aria-label="QRS và T loops 3D"></canvas><p class="hint">Hai loop cùng gốc và tỉ lệ; không nối QRS-off sang T-on. Không dựng loop khi thiếu biên hợp lệ. Hình gốc bên dưới có các phép chiếu bổ sung.</p>
<details id="static-details"><summary id="static-caption">Hình độ phân giải cao</summary><a id="static-figure" target="_blank" rel="noopener"><img id="static-preview" alt="Hình QRS/T V1 độ phân giải cao" loading="lazy"></a></details></section>
<section class="card"><h2>Manual Visual Review</h2><p>Ghi lại mốc tự quan sát và đánh giá V1. <strong>Δ = V1 − thủ công:</strong> dương = V1 muộn hơn, âm = V1 sớm hơn. Đây là review trực quan: bảng số V1 và loop được hiển thị khi mở recording, nên lượt này được ghi nhận đã xem prediction; <code>independent_reference=false</code> luôn được giữ.</p><div class="row"><label>Reviewer <input id="reviewer" placeholder="Tên / mã reviewer"></label><button id="use-reviewer">Mở phiên reviewer</button><span id="session-name" class="hint">Phiên chưa đặt tên (vẫn lưu cục bộ)</span></div><div class="row" style="margin-top:12px"><label><input id="click-mode" type="checkbox">Đặt mốc bằng click trên tín hiệu</label><label>Mốc đang đặt <select id="active-boundary"><option value="qrs_on">QRS-on</option><option value="qrs_off">QRS-off</option><option value="t_on">T-on</option><option value="t_off">T-off</option></select></label><span id="hover" class="muted"></span></div><p class="hint">Click chọn sample gần nhất; kéo không đặt mốc. Nhập thời gian ms để chỉnh chính xác; “Về sample” chọn sample gần nhất, không gán mép cửa sổ khi nhập ngoài miền median.</p><div class="table-wrap"><table class="review-table"><thead><tr><th>Biên</th><th>V1</th><th>Thủ công (ms)</th><th>V1 − thủ công</th><th>Đánh giá V1</th><th>Ghi chú biên</th></tr></thead><tbody>__BOUNDARY_ROWS__</tbody></table></div><label for="review-notes">Ghi chú recording</label><textarea id="review-notes" placeholder="Hình thái, biên không rõ, cắt sớm/muộn, lý do không đánh giá…"></textarea><p id="exposure-state" class="hint"></p><div class="row"><button id="save" class="primary">Lưu review</button><button id="export">Xuất CSV</button><label>Nhập lại CSV <input id="import" type="file" accept=".csv,text/csv"></label></div><p id="saved-state" class="muted"></p><p id="storage-status" class="hint"></p><p class="hint">Tự lưu khi chỉnh và trước Previous / Next. Xuất CSV thường xuyên: dữ liệu trình duyệt có thể bị xóa khi dọn cache hoặc đổi ứng dụng. Tạo lại notebook/HTML giữ nguyên khóa phiên theo dữ liệu V1 và reviewer. CSV này không ghi đè <code>reference_annotations.csv</code>.</p></section>
</div><footer class="muted">Frank XYZ và prediction V1 được giữ nguyên. Ca thiếu median / biên / T peak / loop vẫn xuất hiện; dữ liệu thiếu không được đổi thành 0. Để đánh giá sai số với ground truth, dùng quy trình reference riêng trong phần bổ sung notebook.</footer>
<script id="visual-review-payload" type="application/json">__VISUAL_REVIEW_PAYLOAD__</script><script id="visual-review-core">__CORE_JS__</script><script id="visual-review-app">__APP_JS__</script></main></body></html>'''
HTML = (HTML.replace('__FULL_CONTROLS__', _controls('full'))
        .replace('__QRS_CONTROLS__', _controls('qrs'))
        .replace('__T_CONTROLS__', _controls('t'))
        .replace('__BOUNDARY_ROWS__', _boundary_rows())
        .replace('__CORE_JS__', CORE_JS).replace('__APP_JS__', APP_JS))
