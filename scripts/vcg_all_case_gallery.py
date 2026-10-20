#!/usr/bin/env python3
"""Offline, exhaustive visual inspection of saved V1 delineation diagnostics.

Method disagreement is a descriptive comparison, never a substitute for an
independent expert reference. Every input recording gets one visible card and
one original-resolution figure link, including unavailable median beats.
"""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

TOLERANCES = (0, 10, 20, 40, 60, 100)


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
    except (ValueError, TypeError):
        pass
    return str(value)


def _json(value):
    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).replace('<', '\\u003c')


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _boolean(value):
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return value.lower().strip() in ('true', '1', 'yes')
    return bool(value) if not pd.isna(value) else False


def _escape(value):
    return html.escape('' if value is None else str(value), quote=True)


def _ms(value):
    return 'Thiếu' if _number(value) is None else f'{float(value):.1f} ms'


def _figure_path(value, output_dir):
    path = Path(str(value)).expanduser()
    candidates = [path] if path.is_absolute() else [output_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Figure does not exist: {value}')


def _url(path, output_dir):
    return quote(os.path.relpath(path, output_dir), safe='/')


def _options(values, all_label):
    return '<option value="">' + _escape(all_label) + '</option>' + ''.join(
        f'<option value="{_escape(v)}">{_escape(v)}</option>' for v in sorted({v for v in values if v}))


def _card(record):
    rid = _escape(record['record_id'])
    unavailable = not record['has_median']
    notice = ('Không có median beat: xem tín hiệu raw và lý do thất bại trong hình. '
              'Không quy đổi biên raw thành biên median.' if unavailable else
              'Biên màu trong hình là kết quả V1 và hai cách đặt biên nội bộ; '
              'chưa có biên chuyên gia độc lập để kết luận đúng/sai.')
    pair_rows = []
    for boundary, name in [('on', 'T-on'), ('off', 'T-off')]:
        available = record['t_' + boundary + '_pair_available']
        gap = record.get('t_' + boundary + '_gap_ms')
        signed = record.get('t_' + boundary + '_signed_gap_ms')
        delta = _ms(gap) if available else 'Thiếu cặp so sánh'
        pair_rows.append(
            f'<tr><th>{name}</th><td>{_ms(record.get("t_tangent_" + boundary + "_ms"))}</td>'
            f'<td>{_ms(record.get("t_threshold_" + boundary + "_ms"))}</td>'
            f'<td class="gap">{delta}</td><td>{_ms(signed) if available else "Thiếu"}</td></tr>')
    test_notice = ('<p class="test-note">Tập test: chỉ xem mô tả, không tinh chỉnh thuật toán hoặc ngưỡng.</p>'
                   if record.get('split') == 'test' else '')
    ref = _escape(record.get('reference_status') or 'Chưa có tham chiếu độc lập hợp lệ')
    flags = _escape(record.get('qc_flags') or 'Không có cờ QC')
    link = _escape(record['plot_url'])
    return f'''<article class="case" data-record-id="{rid}">
<div class="case-title"><h2>{rid}</h2><span>{_escape(record.get('label_name'))} · {_escape(record.get('split'))} · {_escape(record.get('qc_status'))}</span></div>
{test_notice}<p class="signal-note">{notice}</p>
<p class="metadata">QC: {flags}<br>Features: {_escape(record.get('feature_completeness'))}/12 · Tham chiếu: {ref}</p>
<table class="pair-table"><thead><tr><th>Biên</th><th>Tiếp tuyến</th><th>Ngưỡng</th><th>|Tiếp tuyến − ngưỡng|</th><th>Tiếp tuyến − ngưỡng</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table>
<p class="missing-note">“Thiếu cặp” không phải chênh lệch 0 ms. Số dương ở cột cuối: tiếp tuyến đặt biên muộn hơn ngưỡng; số âm: sớm hơn.</p>
<a class="figure-link" href="{link}" target="_blank" rel="noopener"><img src="{link}" alt="Biểu đồ QRS/T của {rid}; mở hình gốc để phóng to" loading="lazy" decoding="async"></a>
<p class="figure-caption"><a href="{link}" target="_blank" rel="noopener">Mở hình gốc để phóng to từng biên và từng loop ↗</a></p>
</article>'''


# This pure core is also executed by the Node tests; the browser uses the same
# functions for visible membership and denominator counts.
CORE_JS = r'''
function available(record, boundary) {
  const value = record['t_' + boundary + '_gap_ms'];
  return record['t_' + boundary + '_pair_available'] === true && typeof value === 'number' && Number.isFinite(value);
}
function gapFor(record, boundary) {
  if (boundary === 'on' || boundary === 'off') return available(record, boundary) ? Math.abs(record['t_' + boundary + '_gap_ms']) : null;
  const vals = ['on', 'off'].filter(b => available(record, b)).map(b => Math.abs(record['t_' + b + '_gap_ms']));
  if (boundary === 'both' && vals.length !== 2) return null;
  return vals.length ? Math.max(...vals) : null;
}
function matches(record, filters) {
  const term = (filters.search || '').toLowerCase().trim();
  if (term && ![record.record_id,record.patient_id].some(v => String(v || '').toLowerCase().includes(term))) return false;
  for (const key of ['label_name','split','qc_status']) if (filters[key] && record[key] !== filters[key]) return false;
  if (filters.qc_flag && !(record.qc_flag_items || []).includes(filters.qc_flag)) return false;
  if (filters.category && !(record.category_items || []).includes(filters.category)) return false;
  if (filters.pair_status && record.method_pair_status !== filters.pair_status) return false;
  const gap = gapFor(record, filters.boundary || 'any');
  if (filters.tolerance !== '' && filters.tolerance !== 'all' && filters.tolerance != null) {
    if (gap === null || !(gap > Number(filters.tolerance))) return false;
  } else if (filters.boundary === 'both' && gap === null) return false;
  return true;
}
function selectRecords(records, filters) {
  const selected = records.filter(r => matches(r, filters));
  const boundary = filters.boundary || 'any';
  if (filters.sort === 'id') selected.sort((a,b) => a.record_id.localeCompare(b.record_id));
  else selected.sort((a,b) => {
    const ag = gapFor(a,boundary), bg = gapFor(b,boundary);
    if (ag === null && bg === null) return a.record_id.localeCompare(b.record_id);
    if (ag === null) return 1; if (bg === null) return -1;
    return (filters.sort === 'gap_asc' ? ag-bg : bg-ag) || a.record_id.localeCompare(b.record_id);
  });
  return selected;
}
function membershipCounts(records) {
  return {total:records.length, both:records.filter(r=>available(r,'on')&&available(r,'off')).length,
    any:records.filter(r=>available(r,'on')||available(r,'off')).length,
    missing:records.filter(r=>!available(r,'on')||!available(r,'off')).length,
    none:records.filter(r=>!available(r,'on')&&!available(r,'off')).length};
}
function thresholdCounts(records, boundary, tolerances) {
  const present = records.filter(r=>gapFor(r,boundary)!==null);
  return tolerances.map(t=>({tolerance:t,count:present.filter(r=>gapFor(r,boundary)>t).length,
    denominator:present.length,missing:records.length-present.length}));
}
'''


APP_JS = r'''
const DATA = JSON.parse(document.getElementById('all-cases-payload').textContent);
const records = DATA.records, cards = new Map([...document.querySelectorAll('.case')].map(e=>[e.dataset.recordId,e]));
const controls = [...document.querySelectorAll('[data-filter]')];
const fields = () => Object.fromEntries(controls.map(e=>[e.dataset.filter,e.value]));
const chart = document.getElementById('counts-chart');
function drawChart(selected) {
  chart.replaceChildren();
  for (const [boundary,label] of [['on','T-on'],['off','T-off'],['any','T-on hoặc T-off']]) {
    const block = document.createElement('div'); block.className='chart-group';
    const title=document.createElement('h3');title.textContent=label;block.append(title);
    for (const result of thresholdCounts(selected,boundary,DATA.tolerances)) {
      const row=document.createElement('div');row.className='chart-row';
      const labelEl=document.createElement('span');labelEl.textContent='> '+result.tolerance+' ms';
      const track=document.createElement('span');track.className='bar-track';
      const bar=document.createElement('span');bar.className='bar';bar.style.width=(result.denominator?100*result.count/result.denominator:0)+'%';track.append(bar);
      const count=document.createElement('strong');count.textContent=result.count+' / '+result.denominator;
      row.append(labelEl,track,count);row.title=result.missing+' ca thiếu cặp, không nằm trong mẫu số';block.append(row);
    }
    const missing=document.createElement('p'); const first=thresholdCounts(selected,boundary,[0])[0];
    missing.className='missing-note';missing.textContent=first.missing+' ca thiếu cặp so sánh';block.append(missing);chart.append(block);
  }
}
let selected=[];
function refresh() {
  selected=selectRecords(records,fields());const keep=new Set(selected.map(r=>r.record_id));
  for(const [id,card] of cards) card.hidden=!keep.has(id);
  const container=document.getElementById('case-list');for(const row of selected)container.append(cards.get(row.record_id));
  const c=membershipCounts(selected);
  document.getElementById('visible-count').textContent=c.total+' / '+records.length+' ca đang hiển thị';
  document.getElementById('pair-count').textContent=c.both+' ca đủ cả hai cặp · '+c.any+' ca có ít nhất một cặp · '+c.missing+' ca thiếu ít nhất một cặp ('+c.none+' ca thiếu cả hai)';
  document.getElementById('empty-state').hidden=c.total!==0;drawChart(selected);
}
controls.forEach(e=>e.addEventListener(e.tagName==='INPUT'?'input':'change',refresh));
document.getElementById('reset').addEventListener('click',()=>{controls.forEach(e=>e.value=e.dataset.default||'');refresh();});
document.getElementById('export-selected').addEventListener('click',()=>{
  const keys=['record_id','patient_id','label_name','split','qc_status','qc_flags','has_median','feature_completeness','t_on_gap_ms','t_off_gap_ms','t_on_pair_available','t_off_pair_available','method_pair_status','reference_status','plot_url'];
  const cell=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
  const csv=[keys.join(','),...selected.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8;'}));
  const a=document.createElement('a');a.href=url;a.download='selected_visual_cases.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
refresh();
'''


def build_all_case_gallery(records, summary, manifest, output_dir):
    """Build a self-contained gallery, validating one real figure per record.

    ``output_dir`` is the ``all_cases`` directory. Manifest plot paths can be
    absolute, relative to it, or workspace-relative. Input tables are untouched.
    The output index contains every record and relative links to its figure.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(records).copy()
    manifest = pd.DataFrame(manifest).copy()
    summary = pd.DataFrame(summary).copy()
    required = {'record_id', 'patient_id', 'label_name', 'split', 'qc_status', 'has_median'}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f'Missing record columns: {sorted(missing)}')
    for name, frame in [('records', records), ('manifest', manifest)]:
        if 'record_id' not in frame or frame.record_id.isna().any() or frame.record_id.duplicated().any():
            raise ValueError(f'{name} must contain unique, nonempty record_id values')
    if records.empty:
        raise ValueError('No records to visualize')
    if set(records.record_id) != set(manifest.record_id):
        raise ValueError('Manifest membership must exactly match all records')
    if 'plot_path' not in manifest:
        raise ValueError('Manifest must contain plot_path')
    plots = manifest.set_index('record_id').plot_path.to_dict()
    payload = []
    for row in records.to_dict('records'):
        row = _clean(row)
        row['has_median'] = _boolean(row['has_median'])
        row['qc_flag_items'] = [v.strip() for v in str(row.get('qc_flags') or '').split(';') if v.strip()]
        row['category_items'] = [v.strip() for v in str(row.get('priority_categories') or '').split(';') if v.strip()]
        for boundary in ('on', 'off'):
            key = 't_' + boundary + '_gap_ms'
            row[key] = _number(row.get(key))
            available_key = 't_' + boundary + '_pair_available'
            row[available_key] = _boolean(row.get(available_key)) and row[key] is not None
        on, off = row['t_on_pair_available'], row['t_off_pair_available']
        row['method_pair_status'] = 'both' if on and off else 'on_only' if on else 'off_only' if off else 'neither'
        row['max_method_gap_ms'] = max([abs(row[f't_{b}_gap_ms']) for b in ('on', 'off') if row[f't_{b}_pair_available']], default=None)
        row['plot_url'] = _url(_figure_path(plots[row['record_id']], output_dir), output_dir)
        payload.append(row)
    index_path = output_dir / 'all_cases_index.csv'
    index = pd.DataFrame(payload).drop(columns=['qc_flag_items', 'category_items'])
    index.to_csv(index_path, index=False)
    summary_path = output_dir / 'method_gap_summary.csv'
    summary.to_csv(summary_path, index=False)
    flags = [v for r in payload for v in r['qc_flag_items']]
    categories = [v for r in payload for v in r['category_items']]
    patient_count = records.patient_id.nunique()
    median_count = sum(r['has_median'] for r in payload)
    both_count = sum(r['method_pair_status'] == 'both' for r in payload)
    figure_count = len(payload)
    annotation = output_dir.parent / 'review' / 'independent_annotation.html'
    annotation_link = (f'<a href="{_escape(_url(annotation, output_dir))}">Công cụ đánh dấu tham chiếu độc lập</a>' if annotation.is_file() else '')
    parts = dict(
        total=figure_count, patients=patient_count, median=median_count, unavailable=figure_count-median_count,
        both=both_count, labels=_options([r.get('label_name') for r in payload], 'HC và MI'),
        splits=_options([r.get('split') for r in payload], 'Mọi tập'),
        statuses=_options([r.get('qc_status') for r in payload], 'Mọi trạng thái QC'),
        flags=_options(flags, 'Mọi cờ QC'), categories=_options(categories, 'Mọi nhóm cần xem'),
        annotation_link=annotation_link, cards='\n'.join(_card(r) for r in payload),
        payload=_json(dict(records=payload, tolerances=list(TOLERANCES))), core=CORE_JS, app=APP_JS,
    )
    page = _HTML
    for key, value in parts.items():
        page = page.replace('@@' + key.upper() + '@@', str(value))
    gallery_path = output_dir / 'all_cases_gallery.html'
    gallery_path.write_text(page, encoding='utf-8')
    return dict(gallery_path=gallery_path, all_cases_index=index_path, summary_path=summary_path,
                n_records=figure_count, n_patients=int(patient_count), n_figures=figure_count,
                n_median=median_count, n_median_unavailable=figure_count-median_count)


_HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Notebook 08 · Toàn bộ @@TOTAL@@ biểu đồ QRS/T</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#172b42;background:#eef2f6;font-size:15px}*{box-sizing:border-box}body{margin:0}main{max-width:1480px;margin:auto;padding:26px 24px}h1{font-size:28px;line-height:1.25;margin:0 0 12px}h2{font-size:18px;margin:0;overflow-wrap:anywhere}h3{font-size:16px;margin:0 0 12px}p{line-height:1.5}a{color:#075fa5}button,select,input{font:inherit;border:1px solid #aebccc;border-radius:7px;background:#fff;padding:9px;color:#172b42}button{cursor:pointer;font-weight:600}.notice{background:#fff6da;border-left:5px solid #b87a06;padding:13px 17px}.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0}.tile{background:white;border:1px solid #d5dde7;border-radius:10px;padding:16px}.tile strong{display:block;font-size:28px}.tile span{font-size:13px}.downloads{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}.controls{position:sticky;top:0;z-index:3;background:#f8fafdf5;border:1px solid #c7d3e1;border-radius:10px;backdrop-filter:blur(8px);padding:14px;margin:22px 0}.filter-grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px}label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}input,select{width:100%;min-width:0;font-size:13px}.search{grid-column:span 2}.control-footer{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}#visible-count{font-size:18px;font-weight:750}#pair-count{font-size:13px;color:#40556d;margin:9px 0 0}#counts-chart{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;background:#fff;padding:20px;border-radius:10px;border:1px solid #d5dde7}.chart-row{display:grid;grid-template-columns:66px 1fr 85px;gap:8px;align-items:center;margin:9px 0;font-size:12px}.chart-row strong{text-align:right}.bar-track{height:14px;background:#e6edf5;border-radius:3px;overflow:hidden}.bar{display:block;height:100%;background:#336fa0}.case{background:#fff;border:1px solid #c6d2e0;border-radius:12px;margin:22px 0;overflow:hidden;box-shadow:0 2px 5px #142a3e08}.case-title{padding:17px 20px;background:#eaf0f7;display:flex;align-items:center;justify-content:space-between;gap:10px}.case-title span{font-size:13px;white-space:nowrap}.case>p{margin:12px 20px}.signal-note,.metadata{font-size:13px;color:#344b62}.test-note{color:#80560c;background:#fff5d5;padding:9px 12px;border-radius:5px;font-size:13px;font-weight:600}.pair-table{border-collapse:collapse;width:calc(100% - 40px);margin:14px 20px;font-size:13px}.pair-table th,.pair-table td{border:1px solid #d6dfe8;padding:9px;text-align:left}.pair-table thead{background:#f0f4f9}.gap{font-weight:700}.missing-note{font-size:12px;color:#51667b}.figure-link{display:block;margin:16px 10px 0}.figure-link img{display:block;width:100%;height:auto;min-height:180px;object-fit:contain;background:#f5f7fa}.figure-caption{font-size:13px;text-align:right}.instructions{font-size:14px}.instructions li{margin:6px 0;line-height:1.5}[hidden]{display:none!important}#empty-state{text-align:center;background:white;padding:40px;border-radius:10px}footer{font-size:13px;color:#53657a;margin:20px 0}@media(max-width:900px){main{padding:16px 10px}.tiles{grid-template-columns:repeat(3,1fr)}.filter-grid{grid-template-columns:repeat(3,1fr)}#counts-chart{grid-template-columns:1fr}.case-title{align-items:flex-start;flex-direction:column}.controls{position:static}}@media(max-width:550px){.tiles{grid-template-columns:repeat(2,1fr)}.filter-grid{grid-template-columns:repeat(2,1fr)}.pair-table{font-size:11px;width:calc(100% - 16px);margin:12px 8px}.pair-table th,.pair-table td{padding:5px}.case>p{margin-left:12px;margin-right:12px}}
</style></head><body><main>
<h1>Toàn bộ @@TOTAL@@ trường hợp: QRS/T trên tín hiệu và loop</h1>
<p>Mỗi ca có biểu đồ thực tế và số chênh lệch giữa tiếp tuyến với ngưỡng. Tất cả ca đều có mặt, gồm cả ca không có cờ QC và ca không tạo được median.</p>
<p class="notice"><strong>Đọc đúng các con số:</strong> “Chênh lệch” ở trang này là sự bất đồng giữa hai cách đặt biên nội bộ V1. Chưa có tham chiếu chuyên gia độc lập hợp lệ, nên số ca chênh lệch chưa phải số ca tách sai. Các mức 10–100 ms là bộ lọc mô tả, không phải ngưỡng xác nhận chính xác lâm sàng.</p>
<div class="tiles"><div class="tile"><strong>@@TOTAL@@</strong><span>Ca có biểu đồ</span></div><div class="tile"><strong>@@PATIENTS@@</strong><span>Bệnh nhân</span></div><div class="tile"><strong>@@MEDIAN@@</strong><span>Có median beat</span></div><div class="tile"><strong>@@UNAVAILABLE@@</strong><span>Không tạo được median</span></div><div class="tile"><strong>@@BOTH@@</strong><span>Đủ cặp cho cả T-on/T-off</span></div></div>
<div class="downloads"><a href="all_cases_index.csv" download>Danh sách toàn bộ ca (CSV)</a><a href="method_gap_summary.csv" download>Bảng đếm theo nhóm và ngưỡng (CSV)</a>@@ANNOTATION_LINK@@</div>
<details class="instructions"><summary>Cách xem hình và giữ tính độc lập khi đánh dấu tham chiếu</summary><ul><li>Hình cho thấy QRS/T V1, các đường tiếp tuyến/ngưỡng và loop tương ứng. Mở hình gốc để đọc rõ từng mốc.</li><li>Chọn một biên và mức chênh lệch để xem <em>toàn bộ</em> ca thỏa điều kiện; xuất danh sách đang lọc để đối chiếu.</li><li>Ca thiếu cặp vẫn hiển thị khi chọn “Mọi ca”. Chúng không nằm trong mẫu số tính chênh lệch và không được gán 0 ms.</li><li>Đã xem dự đoán có thể làm lệch việc đánh dấu chuyên gia. Người cung cấp tham chiếu độc lập nên dùng công cụ annotation trước khi xem trang này.</li><li>Ca test chỉ dùng để mô tả, không dùng để chọn hoặc tinh chỉnh thuật toán V1.</li></ul></details>
<section class="controls" aria-label="Bộ lọc tất cả ca"><div class="filter-grid">
<label class="search">Record hoặc bệnh nhân<input data-filter="search" placeholder="Ví dụ patient051 hoặc s0179"></label>
<label>Nhãn<select data-filter="label_name">@@LABELS@@</select></label><label>Tập dữ liệu<select data-filter="split">@@SPLITS@@</select></label><label>Trạng thái QC<select data-filter="qc_status">@@STATUSES@@</select></label>
<label>Cờ QC cụ thể<select data-filter="qc_flag">@@FLAGS@@</select></label><label>Nhóm cần xem<select data-filter="category">@@CATEGORIES@@</select></label>
<label>Biên dùng để lọc<select data-filter="boundary" data-default="any"><option value="any">T-on hoặc T-off</option><option value="on">T-on</option><option value="off">T-off</option><option value="both">Đủ cặp ở cả hai biên</option></select></label>
<label>Chênh lệch tuyệt đối<select data-filter="tolerance" data-default="all"><option value="all">Mọi ca, kể cả thiếu cặp</option><option value="0">&gt; 0 ms</option><option value="10">&gt; 10 ms</option><option value="20">&gt; 20 ms</option><option value="40">&gt; 40 ms</option><option value="60">&gt; 60 ms</option><option value="100">&gt; 100 ms</option></select></label>
<label>Sắp xếp<select data-filter="sort" data-default="gap_desc"><option value="gap_desc">Chênh lệch lớn trước</option><option value="gap_asc">Chênh lệch nhỏ trước</option><option value="id">Record ID</option></select></label>
<label>Cặp tiếp tuyến/ngưỡng<select data-filter="pair_status"><option value="">Mọi tình trạng</option><option value="both">Đủ cả hai biên</option><option value="on_only">Chỉ đủ T-on</option><option value="off_only">Chỉ đủ T-off</option><option value="neither">Thiếu cả hai biên</option></select></label>
</div><div class="control-footer"><span id="visible-count">@@TOTAL@@ / @@TOTAL@@ ca đang hiển thị</span><button id="reset">Hiện lại tất cả</button><button id="export-selected">Tải các ca đang lọc (CSV)</button></div><p id="pair-count">Đang đếm cặp biên…</p></section>
<h2>Số ca chênh lệch trong nhóm đang hiển thị</h2><p class="missing-note">Mỗi thanh dùng mẫu số riêng là số ca có đủ cặp ở biên đó. “T-on hoặc T-off” dùng ca có ít nhất một cặp. Ca thiếu được ghi riêng; các ngưỡng dùng dấu &gt;.</p>
<section id="counts-chart" aria-label="Số trường hợp theo mức chênh lệch"></section>
<p id="empty-state" hidden>Không có ca thỏa bộ lọc. Chọn “Hiện lại tất cả” để xem toàn bộ dữ liệu.</p>
<section id="case-list" aria-label="Biểu đồ thực tế từng ca">@@CARDS@@</section>
<footer>Trang hoạt động offline; không gửi tín hiệu hoặc thông tin bản ghi lên máy chủ. @@TOTAL@@ ca, @@TOTAL@@ liên kết hình gốc, không lấy mẫu bỏ bớt trường hợp.</footer>
<script id="all-cases-payload" type="application/json">@@PAYLOAD@@</script>
<script id="all-cases-core">@@CORE@@</script><script id="all-cases-app">@@APP@@</script>
</main></body></html>'''
