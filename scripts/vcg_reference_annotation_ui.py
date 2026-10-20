#!/usr/bin/env python3
"""Offline, initially blinded annotation of saved V1 Frank XYZ median beats.

The reviewer supplies all reference times. Predictions are encoded separately
and may only be drawn after saving and then explicitly revealing that record.
This is a workflow safeguard, not a security boundary against source inspection.
No network, server, annotation package, or browser extension is required.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOUNDARIES = ('qrs_on', 'qrs_off', 't_on', 't_off')
ANNOTATION_PROTOCOL_ID = 'frank_xyz_median_manual_v1'
REFERENCE_COLUMNS = [
    'record_id', 'patient_id', 'split', 'reviewer', 'reference_source',
    'annotation_scope', 'qrs_on_ref_ms', 'qrs_off_ref_ms', 't_on_ref_ms',
    't_off_ref_ms', 'reference_quality', 'review_status', 'time_reference',
    'fs', 'signal_sha256', 'independent_reference', 'reference_selection',
    'annotation_id', 'created_at', 'updated_at', 'prediction_revealed',
    'post_reveal_edit', 'review_notes', 'annotation_protocol',
] + [f'{b}_{suffix}' for b in BOUNDARIES for suffix in ('status', 'note')]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    # Prevent record metadata from breaking out of an application/json script.
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).replace('<', '\\u003c')


def _metadata_text(value):
    return '' if value is None or pd.isna(value) else str(value)


def _record_payload(row, artifact_root):
    rid = str(row['record_id'])
    safe = rid.replace('/', '_')
    array_path = artifact_root / (safe + '.npz')
    meta_path = artifact_root / (safe + '.json')
    if not array_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f'Missing saved V1 artifacts for {rid}')
    artifact = json.loads(meta_path.read_text())
    if artifact.get('record_id') != rid:
        raise ValueError(f'Artifact record_id mismatch: {rid}')
    if artifact.get('time_reference') != 'median_relative_ms':
        raise ValueError(f'Artifact is not in median coordinates: {rid}')
    fs = float(artifact.get('fs') or row.get('fs'))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'Invalid fs: {rid}')
    rec = dict(record_id=rid, patient_id=str(row['patient_id']),
               split=str(row['split']), label_name=_metadata_text(row.get('label_name')),
               diagnosis=_metadata_text(row.get('diagnosis')), fs=fs,
               signal_sha256=_sha(array_path), annotation_scope='median_beat',
               time_reference='median_relative_ms',
               reference_anchor_sample=artifact.get('median_anchor_sample'),
               signal_available=False, unavailable_reason='', time_ms=[],
               xyz=[], magnitude=[], speed=[], encoded_prediction='')
    with np.load(array_path, allow_pickle=False) as arrays:
        if 'median_xyz' not in arrays:
            rec['unavailable_reason'] = 'Không có median beat V1; không thể đặt biên trong hệ median. Không so thời gian raw với median.'
            return rec
        xyz = np.asarray(arrays['median_xyz'], dtype=float)
        time = np.asarray(arrays['time_ms'], dtype=float)
        anchor = artifact.get('median_anchor_sample')
        if anchor is None:
            raise ValueError(f'Missing median anchor: {rid}')
        expected_time = (np.arange(len(xyz)) - int(anchor)) * 1000.0 / fs
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(time) != len(xyz)
                or len(time) < 2 or not np.isfinite(xyz).all()
                or not np.isfinite(time).all()
                or not np.allclose(time, expected_time, atol=1e-7, rtol=0)):
            raise ValueError(f'Invalid median signal / time coordinates: {rid}')
        magnitude = np.linalg.norm(xyz, axis=1)
        if 'speed' not in arrays:
            raise ValueError(f'Saved V1 spatial speed missing: {rid}')
        speed = np.asarray(arrays['speed'], dtype=float)
        if speed.shape != time.shape or not np.isfinite(speed).all():
            raise ValueError(f'Invalid saved spatial speed: {rid}')
        rec.update(signal_available=True, time_ms=np.round(time, 8).tolist(),
                   xyz=np.round(xyz, 8).tolist(),
                   magnitude=np.round(magnitude, 8).tolist(),
                   speed=np.round(speed, 8).tolist())
        prediction = {}
        for boundary in BOUNDARIES:
            ix = artifact.get('boundaries', {}).get(boundary)
            value = None
            if ix is not None:
                ix = int(ix)
                if not 0 <= ix < len(time):
                    raise ValueError(f'Prediction outside median: {rid} {boundary}')
                value = float(time[ix])
                csv_value = row.get(boundary + '_ms')
                if pd.notna(csv_value) and not np.isclose(value, float(csv_value), atol=1e-7, rtol=0):
                    raise ValueError(f'Prediction CSV/artifact mismatch: {rid} {boundary}')
            prediction[boundary] = value
        rec['encoded_prediction'] = base64.b64encode(_json(prediction).encode()).decode()
    return rec


def build_annotation_tool(frame, review_records, artifact_root, outdir, cfg=None):
    """Build a portable HTML and blank CSV without altering existing references.

    ``review_records`` may be a DataFrame with record_id, an iterable of IDs,
    or None (all frame records). Existing reference_annotations.csv is preserved
    byte for byte. HTML localStorage is separate for each named reviewer.
    ``artifact_root`` accepts either the artifacts folder or its parent.
    """
    del cfg  # Only frozen cached signals are displayed; no V1 recomputation.
    outdir, artifact_root = Path(outdir), Path(artifact_root)
    if (artifact_root / 'artifacts').is_dir():
        artifact_root = artifact_root / 'artifacts'
    if not frame.record_id.is_unique:
        raise ValueError('Duplicate records in annotation population')
    ids = (frame.record_id.tolist() if review_records is None else
           review_records.record_id.tolist() if isinstance(review_records, pd.DataFrame)
           else list(review_records))
    ids = list(dict.fromkeys(str(x) for x in ids))
    if not ids:
        raise ValueError('No records selected for independent annotation')
    lookup = frame.set_index('record_id', drop=False)
    unknown = set(ids) - set(lookup.index)
    if unknown:
        raise ValueError(f'Unknown annotation records: {sorted(unknown)}')
    records = [_record_payload(lookup.loc[rid], artifact_root) for rid in ids]
    dataset_sha = hashlib.sha256(_json([
        (r['record_id'], r['signal_sha256']) for r in records]).encode()).hexdigest()
    templates = []
    for record in records:
        row = dict.fromkeys(REFERENCE_COLUMNS, '')
        row.update({k: record[k] for k in ('record_id', 'patient_id', 'split', 'fs',
                                          'signal_sha256', 'annotation_scope', 'time_reference')})
        row.update(reference_source='independent_visual_annotation',
                   reference_quality='unrated', review_status='pending',
                   independent_reference=False, reference_selection='primary',
                   prediction_revealed=False, post_reveal_edit=False,
                   annotation_protocol=ANNOTATION_PROTOCOL_ID)
        for boundary in BOUNDARIES:
            row[boundary + '_status'] = 'not_visible'
        if not record['signal_available']:
            row['review_notes'] = 'MEDIAN_UNAVAILABLE: cannot annotate median beat; no raw-to-median mapping invented.'
        templates.append(row)
    review_dir = outdir / 'review'
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / 'annotation_protocol.md').write_text(ANNOTATION_PROTOCOL, encoding='utf-8')
    template = pd.DataFrame(templates, columns=REFERENCE_COLUMNS)
    template_path = review_dir / 'annotation_template.csv'
    template.to_csv(template_path, index=False)
    reference_path = outdir / 'reference_annotations.csv'
    if not reference_path.exists():
        template.to_csv(reference_path, index=False)
    payload = dict(schema_version=1, dataset_sha256=dataset_sha,
                   boundaries=list(BOUNDARIES), columns=REFERENCE_COLUMNS,
                   records=records)
    html = HTML.replace('__ANNOTATION_PAYLOAD__', _json(payload))
    html_path = review_dir / 'independent_annotation.html'
    html_path.write_text(html, encoding='utf-8')
    result = dict(html_path=str(html_path), template_path=str(reference_path),
                  review_template_path=str(template_path), records=len(records),
                  median_available=sum(r['signal_available'] for r in records),
                  median_unavailable=sum(not r['signal_available'] for r in records),
                  selected_record_ids=ids, dataset_sha256=dataset_sha,
                  annotation_protocol=ANNOTATION_PROTOCOL_ID,
                  protocol_path=str(review_dir / 'annotation_protocol.md'),
                  reference_csv_preserved=True,
                  independence_rule='save blind before reveal; any edit after reveal invalidates independence')
    (review_dir / 'annotation_tool_manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


ANNOTATION_PROTOCOL = '''# Quy ước annotation thủ công Frank XYZ trên median beat

Mã protocol: `frank_xyz_median_manual_v1`.

Đây là định nghĩa thao tác của nghiên cứu này để hai reviewer sử dụng cùng một
quy ước. Đây không phải chuẩn lâm sàng đã được xác nhận hay một detector mới.
Mốc do reviewer đặt là tham chiếu thủ công có độ bất định; không tự trở thành
chân lý tuyệt đối chỉ vì đã lưu CSV.

## Trước khi đặt mốc

Hai người đánh giá có kinh nghiệm đọc dạng sóng cần thống nhất các định nghĩa
dưới đây trước khi xem biên V1. Mỗi người thực hiện lượt đầu độc lập, với mã
reviewer riêng, trước khi xem gallery QC, hình diagnostic của notebook 07/08,
biên V1 hoặc biên của reviewer còn lại trên các recording đang đánh giá.
Nếu đã xem dự đoán của một recording, không tự nhận lượt đặt mới trên recording
đó là độc lập; sử dụng reviewer chưa xem hoặc quy trình phân xử được ghi rõ.

Chỉ đặt mốc trực tiếp trên median Frank X, Y, Z đã lưu. Trục thời gian là
`median_relative_ms`, anchor của median ở 0 ms. Kiểm tra record, fs và hash tín
hiệu; không chuyển thời gian raw recording sang median theo phỏng đoán.

## Bốn mốc cần đánh dấu

- **QRS-on:** điểm bắt đầu sớm nhất có thể nhận biết của hoạt động khử cực QRS
  trong ba kênh X, Y, Z, so với nền ngay trước QRS. Quan sát cả ba kênh để chọn
  mốc chung của vector, không chỉ lấy onset của một kênh thuận mắt.
- **QRS-off:** điểm kết thúc muộn nhất có thể nhận biết của thành phần QRS trong
  ba kênh, tại chuyển tiếp sang ST. Không kéo biên qua hoạt động ST chỉ để buộc
  vector trở về gốc; nếu QRS và ST không phân biệt rõ, ghi uncertain.
- **T-on:** chuyển tiếp từ ST sang khởi đầu hoạt động T có thể nhận biết trên
  XYZ. ST chênh, T thấp, nhiều thùy hoặc biến thiên liên tục có thể khiến mốc
  này khó xác định; giữ uncertain/not_visible khi không có cơ sở rõ để chọn.
- **T-off:** kết thúc thành phần T cuối cùng còn nhận biết trên XYZ, tại chỗ
  chuyển về nền sau T theo phán đoán của reviewer. Cân nhắc hình thái nhiều
  thùy và tín hiệu chồng lấp; ghi rõ nếu không phân biệt được T tận cùng với
  nhiễu hoặc hoạt động sóng khác.

Magnitude và spatial speed của toàn median chỉ hỗ trợ quan sát. Không tự biến
điểm cực đại đạo hàm, tiếp tuyến, giao threshold, giới hạn độ rộng hay một
baseline số học thành quy tắc đặt mốc bắt buộc. Không có marker tự động trong
lượt đặt đầu. Không điều chỉnh mốc để giống V1 hoặc đủ 12 features.

## Độ chắc chắn và lưu dữ liệu

Chọn visible khi thấy rõ; uncertain khi có một vùng thời gian hợp lý nhưng
không chắc điểm chính xác; not_visible khi không thể nhận biết; rejected khi
reference không sử dụng được. Ghi chú riêng từng biên, định nghĩa đã dùng và
nguồn bất định. Có thể để thời gian trống; không dùng số 0 thay dữ liệu thiếu.
Các điểm nhập được đưa về sample gần nhất khi chủ động chọn chức năng đó;
nếu nằm đúng giữa hai samples, chọn sample sớm hơn. Thời gian ngoài median bị
từ chối. Thứ tự hợp lệ không chứng minh rằng các mốc đã đúng về sinh lý.

Lưu lượt độc lập trước khi hiện V1. Nếu chỉnh sửa sau khi hiện dự đoán, giao
diện đánh dấu post_reveal_edit và bỏ tính độc lập. Hai reviewer xuất CSV riêng;
primary/secondary phải được chọn theo kế hoạch đã định trước, không dựa vào ai
gần V1 hơn. Ghi rõ nguồn và quyết định phân xử nếu có adjudication, giữ nguyên
các lượt đầu để tính bất đồng. confirmed/good là xác nhận của reviewer; notebook
vẫn kiểm tra provenance và từng biên trước khi chấp nhận.

Không có median: báo MEDIAN_UNAVAILABLE, không đặt mốc thay bằng raw và không
tạo loop reference. Bộ 63 ca được ưu tiên kiểm tra, không phải mẫu ngẫu nhiên
đại diện toàn cohort. Không dùng test để sửa ngưỡng hoặc tối ưu thuật toán.
'''


HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Notebook 08 · Independent Frank VCG annotation</title>
<style>
:root{font:15px/1.5 system-ui,-apple-system,sans-serif;color:#142735;background:#f2f5f8;color-scheme:light}
*{box-sizing:border-box}body{margin:0}header{background:#13354a;color:#fff;padding:22px 28px}header h1{margin:0;font-size:24px}header p{margin:6px 0;max-width:1100px}main{max-width:1500px;margin:18px auto;padding:0 20px}.card{background:white;border:1px solid #d4dee6;border-radius:10px;padding:16px;margin:14px 0}.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:block;font-size:13px;font-weight:600}input,select,button,textarea{font:inherit;padding:8px;border:1px solid #adbdc9;border-radius:5px;background:white;color:inherit}input[type=checkbox]{width:17px;height:17px;vertical-align:middle}button{cursor:pointer;background:#ecf2f6}button.primary{background:#146284;color:white;border-color:#146284}button.warn{background:#fff0d5;color:#623f00}button:disabled{opacity:.45;cursor:default}label small{font-weight:400}input[type=number]{width:125px}textarea{width:100%;min-height:55px}select{max-width:100%}#record{min-width:320px;max-width:570px}.grow{flex:1}.pill{padding:3px 8px;border-radius:4px;background:#e6eef3;display:inline-block;font-size:12px}.hint{font-size:13px;color:#465e70}.status{padding:10px 13px;border-left:4px solid #298490;background:#edf8f9;margin-top:10px;white-space:pre-wrap}.status.error{border-color:#b35227;background:#fff2e9}.status.ok{border-color:#27835e;background:#edf8f0}#metadata{margin:10px 0}#plot{width:100%;height:610px;display:block;touch-action:none;cursor:crosshair;border:1px solid #d4dee6;border-radius:6px;background:#fff}#plotlabel{margin:8px 0;color:#465e70;font-size:13px}#boundary-table{border-collapse:collapse;width:100%;margin-top:12px}#boundary-table th,#boundary-table td{border-bottom:1px solid #dfe6ed;text-align:left;padding:8px}#boundary-table th{background:#eef3f7;font-size:13px}#boundary-table input[type=text]{width:100%;min-width:130px}.marker{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px}.active-row{background:#f1f8ff}.alert{font-size:13px;background:#fff7e7;padding:10px;border-radius:5px}.spacer{margin-top:12px}.mono{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}details{margin-top:10px}summary{cursor:pointer;font-weight:600}a{color:#135b83}footer{margin:22px 0;color:#465e70;font-size:13px}#compare{white-space:pre-wrap;font-size:13px}#source{min-width:280px}.hidden{display:none!important}@media(max-width:850px){main{padding:0 8px}header{padding:18px}#boundary-table th,#boundary-table td{padding:5px}#boundary-table input[type=number]{width:90px}#plot{height:560px}.tablewrap{overflow-x:auto}}
</style></head><body>
<header><h1>Đặt mốc tham chiếu độc lập trên median Frank VCG</h1>
<p>Đặt QRS-on, QRS-off, T-on, T-off trên cùng median XYZ đã lưu của V1. Biên dự đoán chỉ được hiển thị khi bạn đã lưu rồi chủ động bấm “Hiện dự đoán V1”.</p>
<p class="hint" style="color:#d2e5ee">Ứng dụng hoạt động ngoại tuyến. Mỗi reviewer có vùng lưu riêng trong trình duyệt. Xuất CSV để sao lưu và đưa vào notebook 08; trang này không tự ghi vào file project.</p></header>
<main>
<section class="card"><div class="row"><label>Reviewer độc lập <input id="reviewer" type="text" autocomplete="off" placeholder="Nhập mã/tên người đánh giá" list="reviewer-examples"><datalist id="reviewer-examples"><option value="reviewer_A"><option value="reviewer_B"></datalist></label><button id="use-reviewer" class="primary">Mở phiên reviewer</button><span id="session" class="pill">Chưa mở phiên</span><span id="progress" class="hint"></span></div>
<p class="hint">Hai reviewer nên dùng mã riêng và không xem annotation của nhau trước khi hoàn thành độc lập. Danh tính reviewer là quy trình làm việc; trang HTML không thực hiện xác thực người dùng.</p>
<p class="hint">Quy ước chung: <code>frank_xyz_median_manual_v1</code>. Đọc <a href="annotation_protocol.md" target="_blank" rel="noopener">hướng dẫn định nghĩa bốn biên</a> và thống nhất giữa hai reviewer trước khi xem V1 hoặc diagnostic của notebook 07/08. Khi T-on/T-off không rõ, đánh dấu uncertain/not_visible.</p>
<div class="row"><label class="grow">Recording <select id="record"></select></label><button id="prev">← Trước</button><button id="next">Tiếp →</button><label><input type="checkbox" id="show-diagnosis"> Hiện bệnh lý (tùy chọn)</label></div><div id="metadata"></div><div id="diagnosis" class="hint hidden"></div><div id="signal-hash" class="mono"></div>
</section>
<section class="card"><div class="row"><label>Biên đặt khi bấm đồ thị <select id="active-boundary"><option value="qrs_on">QRS-on</option><option value="qrs_off">QRS-off</option><option value="t_on">T-on</option><option value="t_off">T-off</option></select></label><button id="fit">Toàn median</button><button id="zoom-in">Phóng to +</button><button id="zoom-out">Thu nhỏ −</button><button id="pan-left">← Dịch</button><button id="pan-right">Dịch →</button><span id="view-range" class="hint"></span></div>
<p id="plotlabel">Bấm để đặt biên; phím 1–4 chọn biên. Cuộn để zoom; giữ Shift và kéo để dịch. Mốc được đặt tại sample gần nhất. Trục X: ms tương đối với median anchor.</p><canvas id="plot" aria-label="Frank XYZ, magnitude và spatial speed của median beat"></canvas><div id="hover" class="hint">Chưa chọn điểm.</div>
<div class="tablewrap"><table id="boundary-table"><thead><tr><th>Biên</th><th>Thời gian (ms)</th><th>Khả năng xác định</th><th>Ghi chú riêng</th><th></th></tr></thead><tbody></tbody></table></div>
<p class="hint">Không thấy rõ biên: chọn “uncertain” hoặc “not_visible”, không đoán để đủ bốn số. QRS/T được thể hiện bằng dải màu từ các biên bạn đặt; độ rộng không phải tiêu chuẩn chẩn đoán.</p>
</section>
<section class="card"><div class="row"><label>Nguồn tham chiếu <input id="source" type="text" value="independent_visual_annotation"></label><label>Chất lượng reference <select id="quality"><option value="unrated">Chưa đánh giá</option><option value="good">good · đủ tin cậy</option><option value="uncertain">uncertain · chưa chắc</option><option value="low">low · thấp</option></select></label><label>Trạng thái review <select id="review-status"><option value="pending">pending · chưa xác nhận</option><option value="confirmed">confirmed · reviewer xác nhận</option><option value="rejected">rejected · loại reference</option></select></label><label>Vai trò reference <select id="reference-selection"><option value="primary">primary · reference chính</option><option value="secondary">secondary · reviewer đối chiếu</option><option value="adjudicated">adjudicated · đã phân xử</option></select></label></div>
<p class="hint">Khi hợp nhất hai reviewer, chọn rõ primary/secondary theo kế hoạch đã định trước; chỉ dùng adjudicated sau quy trình phân xử độc lập. Nhiều reference cùng vai trò chính sẽ được notebook báo xung đột, không tự chọn theo tên file hoặc thứ tự dòng.</p>
<label class="spacer"><input id="independent" type="checkbox"> Tôi tự đặt các mốc này trước khi xem biên V1 hoặc biên từ reviewer khác; không sao chép dự đoán thành reference.</label>
<label class="spacer">Ghi chú recording <textarea id="notes" placeholder="Định nghĩa biên đã dùng, độ chắc chắn, tín hiệu bất thường, lý do từ chối..."></textarea></label>
<div class="row spacer"><button id="save" class="primary">Lưu annotation độc lập</button><button id="reveal" class="warn" disabled>Hiện dự đoán V1 sau khi lưu</button><button id="clear" type="button">Xóa các mốc của bản nháp</button><span id="saved-state" class="hint"></span></div>
<div id="message" class="status">Mở phiên reviewer để bắt đầu. Không có mốc dự đoán nào được vẽ ở lần đặt đầu.</div>
<div id="compare" class="spacer hidden"></div>
<details><summary>Quy tắc độc lập và chấp nhận reference</summary><p class="hint">Lưu annotation không có nghĩa là ground truth tự động. Notebook kiểm tra reviewer, nguồn, median/time base, sampling rate, hash tín hiệu, trạng thái, chất lượng và từng biên. Chỉ những mốc “visible” có giá trị hợp lệ trong annotation được xác nhận, chất lượng “good” và độc lập mới có thể được đánh giá. Bất kỳ chỉnh sửa nào sau khi hiện dự đoán đều làm annotation của phiên đó mất tính độc lập, chuyển về pending. Cần người đánh giá khác chưa xem dự đoán hoặc quy trình adjudication độc lập để tạo reference mới.</p><p class="hint">Lớp ẩn dự đoán chỉ hạn chế thiên lệch khi dùng giao diện. Không thể ngăn người dùng chủ động đọc mã nguồn HTML. Tránh xem source, notebook QC và gallery dự đoán trước khi hoàn thành review độc lập.</p></details>
</section>
<section class="card"><div class="row"><button id="export" class="primary">Xuất CSV của reviewer hiện tại</button><label>Nhập lại CSV đã lưu <input id="import" type="file" accept=".csv,text/csv"></label><span id="storage-status" class="hint"></span></div>
<p class="hint">Xuất thường xuyên. LocalStorage có thể bị xóa khi đổi trình duyệt hoặc dọn dữ liệu. CSV chỉ chứa annotation của reviewer hiện tại; nhập CSV không mở annotation của reviewer khác. Sau khi xuất, hợp nhất các reviewer trong <code>reference_annotations.csv</code> bằng notebook 08; không tự ghi đè dữ liệu review cũ.</p>
</section><footer id="footer">Frank vx/vy/vz có sẵn của PTB; không chuyển đổi ECG → VCG. Tất cả mốc là median_relative_ms. Không thay median thiếu bằng raw hoặc số 0.</footer>
</main>
<script id="annotation-payload" type="application/json">__ANNOTATION_PAYLOAD__</script>
<script id="annotation-core">
'use strict';
const AnnotationCore = (() => {
 const boundaries=['qrs_on','qrs_off','t_on','t_off'];
 const bool=x=>x===true||String(x).toLowerCase()==='true'||String(x)==='1';
 const finite=x=>x!==''&&x!==null&&x!==undefined&&Number.isFinite(Number(x));
 function blankRow(r,reviewer){const x={record_id:r.record_id,patient_id:r.patient_id,split:r.split,reviewer,reference_source:'independent_visual_annotation',annotation_scope:'median_beat',reference_quality:'unrated',review_status:'pending',time_reference:'median_relative_ms',fs:r.fs,signal_sha256:r.signal_sha256,independent_reference:false,reference_selection:'primary',annotation_id:'',created_at:'',updated_at:'',prediction_revealed:false,post_reveal_edit:false,review_notes:'',annotation_protocol:'frank_xyz_median_manual_v1'};for(const b of boundaries){x[b+'_ref_ms']='';x[b+'_status']='not_visible';x[b+'_note']='';}return x;}
 function protectAfterExposure(row,exposed){const copy={...row};if(exposed){copy.independent_reference=false;copy.post_reveal_edit=true;copy.prediction_revealed=true;copy.review_status='pending';}return copy;}
 function validateRow(row,r){const errors=[];if(!String(row.reviewer||'').trim())errors.push('Cần mã reviewer.');if(!String(row.reference_source||'').trim())errors.push('Cần nguồn reference.');if(!['primary','secondary','adjudicated'].includes(row.reference_selection))errors.push('Cần vai trò reference hợp lệ.');if(row.annotation_scope!=='median_beat'||row.time_reference!=='median_relative_ms')errors.push('Chỉ hỗ trợ annotation trực tiếp trên median.');if(Number(row.fs)!==Number(r.fs)||row.signal_sha256!==r.signal_sha256)errors.push('Sampling rate hoặc hash tín hiệu không khớp.');let visible=0;for(const b of boundaries){const value=row[b+'_ref_ms'],status=row[b+'_status'];if(!['visible','uncertain','not_visible','rejected'].includes(status))errors.push(b+': trạng thái không hợp lệ.');if(value!==''&&value!==null&&value!==undefined&&!finite(value))errors.push(b+': thời gian phải là số hoặc để trống.');if(status==='visible'){visible++;if(!finite(value))errors.push(b+': visible cần thời gian số.');}if(finite(value)&&(!r.signal_available||Number(value)<r.time_ms[0]-1e-7||Number(value)>r.time_ms[r.time_ms.length-1]+1e-7))errors.push(b+': thời gian ngoài median.');if(finite(value)&&r.signal_available){const index=(Number(value)-r.time_ms[0])*r.fs/1000;if(Math.abs(index-Math.round(index))>1e-5)errors.push(b+': thời gian chưa nằm trên sample; chọn sample gần nhất.');}}
 for(const [a,b] of [['qrs_on','qrs_off'],['qrs_off','t_on'],['t_on','t_off']])if(finite(row[a+'_ref_ms'])&&finite(row[b+'_ref_ms'])&&Number(row[a+'_ref_ms'])>=Number(row[b+'_ref_ms']))errors.push(a+' phải sớm hơn '+b+'.');
 if(row.review_status==='confirmed'){if(!r.signal_available)errors.push('Không có median: không thể xác nhận reference.');if(!bool(row.independent_reference)||bool(row.post_reveal_edit))errors.push('Reference confirmed phải độc lập, chưa chỉnh sau khi xem dự đoán.');if(row.reference_quality!=='good')errors.push('Reference confirmed cần quality=good; giữ pending nếu chưa chắc.');if(!visible)errors.push('Reference confirmed cần ít nhất một biên visible.');}return errors;}
 function csvCell(value){const s=value===null||value===undefined?'':String(value);return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
 function toCSV(rows,columns){return '\ufeff'+[columns.map(csvCell).join(','),...rows.map(r=>columns.map(c=>csvCell(r[c])).join(','))].join('\r\n')+'\r\n';}
 function parseCSV(input){const text=String(input).replace(/^\ufeff/,''),all=[];let row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else if(!quoted&&cell.length)throw Error('CSV quote không hợp lệ.');else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(x=>x!==''))all.push(row);row=[];cell='';}else cell+=c;}if(quoted)throw Error('CSV chưa đóng dấu quote.');if(cell!==''||row.length){row.push(cell);if(row.some(x=>x!==''))all.push(row);}if(!all.length)return [];const columns=all.shift();if(new Set(columns).size!==columns.length)throw Error('CSV có cột trùng.');return all.map(values=>{if(values.length!==columns.length)throw Error('Số cột CSV không khớp.');return Object.fromEntries(columns.map((c,i)=>[c,values[i]]));});}
 function nearestTime(r,value){if(!r.signal_available||!finite(value))return '';const v=Number(value);if(v<r.time_ms[0]||v>r.time_ms[r.time_ms.length-1])return '';const index=(v-r.time_ms[0])*r.fs/1000,lo=Math.floor(index),fraction=index-lo,n=fraction<=.5+1e-9?lo:lo+1;return r.time_ms[n];}
 function canReveal(row,r,exposed=false){return !!row&&r.signal_available&&(exposed||(bool(row.independent_reference)&&!bool(row.post_reveal_edit)&&(boundaries.some(b=>finite(row[b+'_ref_ms'])||String(row[b+'_note']||'').trim())||String(row.review_notes||'').trim())));}
 return {boundaries,bool,finite,blankRow,protectAfterExposure,validateRow,toCSV,parseCSV,nearestTime,canReveal};
})();
</script>
<script id="annotation-app">
'use strict';
const DATA=JSON.parse(document.getElementById('annotation-payload').textContent),C=AnnotationCore,$=id=>document.getElementById(id);
const COLORS={qrs_on:'#1c79b8',qrs_off:'#1b8b77',t_on:'#b65b00',t_off:'#7b4fbd'},LABELS={qrs_on:'QRS-on',qrs_off:'QRS-off',t_on:'T-on',t_off:'T-off'};
const STORE='vcg-reference08:'+DATA.dataset_sha256+':',records=DATA.records,recordMap=new Map(records.map(r=>[r.record_id,r]));
let reviewer='',session={rows:{},exposures:{}},draft=null,dirty=false,shownPrediction=false,prediction=null,view=null,hoverTime=null,pan=null,storageOK=true,currentRecordIndex=0;
const canvas=$('plot'),ctx=canvas.getContext('2d');
function rec(){return records[Number($('record').value)||0];}
function say(text,kind=''){const el=$('message');el.textContent=text;el.className='status '+kind;}
function storageRead(key,fallback){try{return JSON.parse(localStorage.getItem(key))||fallback;}catch(e){storageOK=false;return fallback;}}
function persist(){if(!reviewer)return;try{localStorage.setItem(STORE+reviewer,JSON.stringify(session));$('storage-status').textContent='Đã lưu cục bộ; hãy xuất CSV để sao lưu.';}catch(e){storageOK=false;$('storage-status').textContent='Trình duyệt không cho lưu cục bộ; xuất CSV trước khi đóng trang.';}}
function hasExposure(rid){return !!session.exposures[rid];}
function readForm(){if(!draft)return null;const row={...draft};for(const b of C.boundaries){const raw=$(b+'-value').value;row[b+'_ref_ms']=raw===''?'':Number(raw);row[b+'_status']=$(b+'-status').value;row[b+'_note']=$(b+'-note').value;}row.reference_source=$('source').value.trim();row.reference_quality=$('quality').value;row.review_status=$('review-status').value;row.reference_selection=$('reference-selection').value;row.independent_reference=$('independent').checked;row.review_notes=$('notes').value;return row;}
function fillForm(){for(const b of C.boundaries){$(b+'-value').value=draft[b+'_ref_ms'];$(b+'-status').value=draft[b+'_status'];$(b+'-note').value=draft[b+'_note'];}for(const [id,key] of [['source','reference_source'],['quality','reference_quality'],['review-status','review_status'],['reference-selection','reference_selection'],['notes','review_notes']])$(id).value=draft[key]||'';$('independent').checked=C.bool(draft.independent_reference);$('independent').disabled=hasExposure(rec().record_id);}
function change(){if(!reviewer||!draft)return;draft=readForm();if(hasExposure(rec().record_id)){draft=C.protectAfterExposure(draft,true);fillForm();say('Đã sửa sau khi xem dự đoán: annotation này không còn độc lập và đã chuyển pending. Cần reviewer/adjudication độc lập để tạo reference được chấp nhận.','error');}dirty=true;$('reveal').disabled=true;$('saved-state').textContent='Có thay đổi chưa lưu';draw();}
function formEnabled(){const enabled=!!reviewer,available=rec().signal_available;for(const b of C.boundaries){$(b+'-value').disabled=!enabled||!available;$(b+'-status').disabled=!enabled||!available;$(b+'-note').disabled=!enabled;}for(const id of ['source','quality','review-status','reference-selection','notes','save','clear','export'])$(id).disabled=!enabled;$('active-boundary').disabled=!enabled||!available;$('independent').disabled=!enabled||!available||hasExposure(rec().record_id);}
function changeRecord(){const r=rec();currentRecordIndex=Number($('record').value)||0;draft=session.rows[r.record_id]?{...session.rows[r.record_id]}:C.blankRow(r,reviewer);dirty=false;shownPrediction=false;prediction=null;hoverTime=null;view=r.signal_available?[r.time_ms[0],r.time_ms[r.time_ms.length-1]]:[0,1];fillForm();formEnabled();$('metadata').textContent=r.record_id+' · '+r.patient_id+' · split '+r.split+' · fs '+r.fs+' Hz · '+(r.signal_available?r.time_ms.length+' samples, median anchor = 0 ms':'MEDIAN_UNAVAILABLE');$('diagnosis').textContent=(r.label_name||'')+' · '+(r.diagnosis||'');$('signal-hash').textContent='signal_sha256 (saved NPZ): '+r.signal_sha256;$('compare').classList.add('hidden');$('reveal').disabled=!reviewer||!C.canReveal(session.rows[r.record_id],r,hasExposure(r.record_id));$('saved-state').textContent=session.rows[r.record_id]?'Đã có annotation của reviewer hiện tại':'Chưa có annotation của reviewer hiện tại';if(!r.signal_available)say(r.unavailable_reason,'error');else if(hasExposure(r.record_id))say('Reviewer này đã từng xem dự đoán ở recording này. Annotation đã lưu trước đó được giữ; chỉnh sửa tiếp sẽ mất tính độc lập.');else say('Chế độ đặt mốc độc lập: biên V1 và biên của reviewer khác đang ẩn.');updateProgress();draw();}
function switchReviewer(){const name=$('reviewer').value.trim();if(!name){say('Nhập mã hoặc tên reviewer trước.','error');return;}if(dirty&&!confirm('Bản nháp chưa lưu sẽ bị bỏ khi đổi reviewer. Tiếp tục?'))return;reviewer=name;session=storageRead(STORE+reviewer,{rows:{},exposures:{}});session.rows=session.rows||{};session.exposures=session.exposures||{};for(const [rid,row] of Object.entries(session.rows))if(C.bool(row.prediction_revealed))session.exposures[rid]=true;$('session').textContent='Phiên: '+reviewer;$('import').disabled=false;changeRecord();if(!storageOK)$('storage-status').textContent='Lưu cục bộ không khả dụng; cần xuất CSV.';}
function updateProgress(){const rows=Object.values(session.rows);$('progress').textContent=rows.length+'/'+records.length+' đã lưu · '+rows.filter(x=>x.review_status==='confirmed'&&C.bool(x.independent_reference)).length+' confirmed độc lập';}
function saveRow(){if(!reviewer)return;let row=readForm();if(hasExposure(row.record_id)&&dirty)row=C.protectAfterExposure(row,true);row.reviewer=reviewer;const errors=C.validateRow(row,rec());if(errors.length){say(errors.join('\n'),'error');return;}const now=new Date().toISOString();row.annotation_id=row.annotation_id||(globalThis.crypto&&crypto.randomUUID?crypto.randomUUID():'ref-'+Date.now()+'-'+Math.random().toString(16).slice(2));row.created_at=row.created_at||now;row.updated_at=now;row.prediction_revealed=hasExposure(row.record_id);session.rows[row.record_id]={...row};draft={...row};dirty=false;persist();fillForm();$('reveal').disabled=!C.canReveal(row,rec(),hasExposure(row.record_id));$('saved-state').textContent='Đã lưu '+now;say('Đã lưu annotation '+row.review_status+'. Có thể xuất CSV; chỉ bấm hiện dự đoán sau khi hoàn thành review độc lập.','ok');updateProgress();draw();}
function reveal(){const r=rec(),saved=session.rows[r.record_id];if(dirty||!C.canReveal(saved,r,hasExposure(r.record_id))){say('Cần lưu annotation độc lập có mốc hoặc ghi chú đánh giá trước khi xem dự đoán.','error');return;}if(!confirm('Hiện dự đoán V1? Annotation vừa lưu được giữ. Mọi chỉnh sửa sau đó sẽ bị đánh dấu không còn độc lập.'))return;prediction=JSON.parse(atob(r.encoded_prediction));session.exposures[r.record_id]=true;saved.prediction_revealed=true;draft.prediction_revealed=true;shownPrediction=true;persist();$('independent').disabled=true;$('compare').classList.remove('hidden');$('compare').textContent='Chỉ đối chiếu sau khi lưu — không phải xác nhận ground truth:\n'+C.boundaries.map(b=>LABELS[b]+': reference '+(C.finite(saved[b+'_ref_ms'])?saved[b+'_ref_ms']+' ms':'thiếu')+'; V1 '+(prediction[b]===null?'không có prediction':prediction[b]+' ms')).join('\n');say('Đã hiện V1 bằng nét đứt. Mốc reference là nét liền. Mọi chỉnh sửa tiếp theo sẽ mất tính độc lập.');draw();}
function exportRows(){if(!reviewer)return;if(dirty&&!confirm('Bản nháp chưa lưu không nằm trong CSV. Vẫn xuất các bản đã lưu?'))return;const rows=Object.values(session.rows).filter(x=>x.reviewer===reviewer);if(!rows.length){say('Chưa có annotation đã lưu để xuất.','error');return;}download(C.toCSV(rows,DATA.columns),'reference_annotations_'+reviewer.replace(/[^a-zA-Z0-9_-]/g,'_')+'.csv');}
function download(content,name){const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),500);}
async function importRows(event){const file=event.target.files[0];if(!file)return;if(!reviewer){say('Mở phiên reviewer trước khi nhập CSV.','error');return;}try{const parsed=C.parseCSV(await file.text()),mine=parsed.filter(x=>String(x.reviewer||'').trim()===reviewer);if(!mine.length)throw Error('CSV không có annotation cho reviewer hiện tại. Không mở biên của reviewer khác.');const validated=[],seen=new Set();for(const item of mine){if(seen.has(item.record_id))throw Error('CSV có nhiều annotation của cùng reviewer cho '+item.record_id+'; cần adjudication riêng.');seen.add(item.record_id);const r=recordMap.get(item.record_id);if(!r)throw Error('Recording không có trong trang này: '+item.record_id);const row={...C.blankRow(r,reviewer),...item};if(!String(item.annotation_protocol||'').trim())row.annotation_protocol='';for(const b of C.boundaries)row[b+'_ref_ms']=row[b+'_ref_ms']===''?'':Number(row[b+'_ref_ms']);for(const key of ['independent_reference','prediction_revealed','post_reveal_edit'])row[key]=C.bool(row[key]);if(row.post_reveal_edit){row.independent_reference=false;row.review_status='pending';}if(hasExposure(row.record_id)&&!C.bool(row.prediction_revealed))throw Error('Không thể nhập annotation tự nhận chưa xem V1 sau khi reviewer này đã hiện V1: '+row.record_id);const errors=C.validateRow(row,r);if(errors.length)throw Error(row.record_id+': '+errors.join(' '));validated.push(row);}const overwrite=validated.filter(r=>session.rows[r.record_id]);if(overwrite.length&&!confirm('Thay '+overwrite.length+' annotation hiện có của reviewer này bằng dữ liệu CSV?'))return;for(const row of validated){session.rows[row.record_id]=row;if(C.bool(row.prediction_revealed))session.exposures[row.record_id]=true;}persist();changeRecord();say('Đã nhập '+validated.length+' annotation cho '+reviewer+'. '+(parsed.length-mine.length)+' dòng reviewer khác không được mở.','ok');}catch(error){say('Không nhập CSV: '+error.message,'error');}finally{event.target.value='';}}
function setupTable(){const body=$('boundary-table').querySelector('tbody');for(const b of C.boundaries){const tr=document.createElement('tr');tr.id=b+'-row';tr.innerHTML='<td><span class="marker" style="background:'+COLORS[b]+'"></span>'+LABELS[b]+'</td><td><input id="'+b+'-value" type="number" step="any" aria-label="'+LABELS[b]+' milliseconds"></td><td><select id="'+b+'-status"><option value="not_visible">not_visible · không thấy</option><option value="visible">visible · thấy rõ</option><option value="uncertain">uncertain · chưa chắc</option><option value="rejected">rejected · loại</option></select></td><td><input id="'+b+'-note" type="text" aria-label="Ghi chú '+LABELS[b]+'"></td><td><button id="'+b+'-snap" title="Đưa thời gian nhập về sample gần nhất">Sample gần nhất</button></td>';body.appendChild(tr);for(const suffix of ['value','status','note'])$(b+'-'+suffix).addEventListener('input',change);$(b+'-snap').onclick=()=>{if(!reviewer||!rec().signal_available)return;const value=C.nearestTime(rec(),$(b+'-value').value);if(value!==''){$(b+'-value').value=value;change();}};}highlightActive();}
function highlightActive(){for(const b of C.boundaries)$(b+'-row').classList.toggle('active-row',b===$('active-boundary').value);}
function layout(){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1,w=rect.width,h=rect.height;if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}ctx.setTransform(ratio,0,0,ratio,0,0);return {w,h,left:73,right:w-23,top:32,bottom:h-38,gap:25};}
function draw(){const p=layout(),r=rec();ctx.clearRect(0,0,p.w,p.h);ctx.fillStyle='#fff';ctx.fillRect(0,0,p.w,p.h);ctx.font='12px system-ui';if(!r.signal_available){ctx.fillStyle='#824522';ctx.fillText('MEDIAN_UNAVAILABLE — không có tín hiệu median để đặt biên.',28,55);ctx.fillStyle='#455e70';ctx.fillText('Không dùng raw recording thay median khi chưa có phép căn chỉnh được kiểm chứng.',28,83);return;}if(!view)view=[r.time_ms[0],r.time_ms[r.time_ms.length-1]];const ph=(p.bottom-p.top-2*p.gap)/3,xx=t=>p.left+(t-view[0])/(view[1]-view[0])*(p.right-p.left);const visible=[];for(let i=0;i<r.time_ms.length;i++)if(r.time_ms[i]>=view[0]&&r.time_ms[i]<=view[1])visible.push(i);const panels=[{title:'Frank XYZ (mV)',series:[r.xyz.map(v=>v[0]),r.xyz.map(v=>v[1]),r.xyz.map(v=>v[2])],colors:['#1476a5','#d1771e','#7b50ad'],names:['X','Y','Z']},{title:'|XYZ| (mV)',series:[r.magnitude],colors:['#2e7e63'],names:['magnitude']},{title:'Spatial speed V1 (mV/s)',series:[r.speed],colors:['#374c66'],names:['speed toàn median']}];
 for(let panel=0;panel<3;panel++){const obj=panels[panel],top=p.top+panel*(ph+p.gap),bottom=top+ph;let ymin=Infinity,ymax=-Infinity;for(const s of obj.series)for(const i of visible){ymin=Math.min(ymin,s[i]);ymax=Math.max(ymax,s[i]);}if(!Number.isFinite(ymin)){ymin=0;ymax=1;}const pad=Math.max((ymax-ymin)*.09,1e-6);ymin-=pad;ymax+=pad;const yy=v=>bottom-(v-ymin)/(ymax-ymin)*ph;ctx.save();ctx.beginPath();ctx.rect(p.left,top,p.right-p.left,ph);ctx.clip();ctx.fillStyle='#eaf4fa';if(draft&&C.finite(draft.qrs_on_ref_ms)&&C.finite(draft.qrs_off_ref_ms))ctx.fillRect(xx(draft.qrs_on_ref_ms),top,xx(draft.qrs_off_ref_ms)-xx(draft.qrs_on_ref_ms),ph);ctx.fillStyle='#fff1e3';if(draft&&C.finite(draft.t_on_ref_ms)&&C.finite(draft.t_off_ref_ms))ctx.fillRect(xx(draft.t_on_ref_ms),top,xx(draft.t_off_ref_ms)-xx(draft.t_on_ref_ms),ph);ctx.restore();
 ctx.strokeStyle='#dbe4eb';ctx.lineWidth=1;for(let j=0;j<=4;j++){const y=top+ph*j/4;ctx.beginPath();ctx.moveTo(p.left,y);ctx.lineTo(p.right,y);ctx.stroke();ctx.fillStyle='#526879';ctx.textAlign='right';ctx.fillText((ymax-(ymax-ymin)*j/4).toPrecision(3),p.left-9,y+4);}ctx.textAlign='left';ctx.fillStyle='#183b51';ctx.font='bold 12px system-ui';ctx.fillText(obj.title,p.left,top-9);ctx.font='12px system-ui';let legendX=p.left+230;obj.names.forEach((name,i)=>{ctx.fillStyle=obj.colors[i];ctx.fillText(name,legendX,top-9);legendX+=name.length*7+25;});
 ctx.save();ctx.beginPath();ctx.rect(p.left,top,p.right-p.left,ph);ctx.clip();for(let k=0;k<obj.series.length;k++){ctx.strokeStyle=obj.colors[k];ctx.lineWidth=1.4;ctx.beginPath();let begun=false;for(let i=Math.max(0,(visible[0]||0)-1);i<=Math.min(r.time_ms.length-1,(visible[visible.length-1]||r.time_ms.length-1)+1);i++){const x=xx(r.time_ms[i]),y=yy(obj.series[k][i]);if(!begun){ctx.moveTo(x,y);begun=true;}else ctx.lineTo(x,y);}ctx.stroke();}for(const b of C.boundaries){if(draft&&C.finite(draft[b+'_ref_ms'])){ctx.setLineDash([]);ctx.strokeStyle=COLORS[b];ctx.lineWidth=1.8;ctx.beginPath();ctx.moveTo(xx(Number(draft[b+'_ref_ms'])),top);ctx.lineTo(xx(Number(draft[b+'_ref_ms'])),bottom);ctx.stroke();}if(shownPrediction&&prediction&&prediction[b]!==null){ctx.setLineDash([6,4]);ctx.strokeStyle=COLORS[b];ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(xx(prediction[b]),top);ctx.lineTo(xx(prediction[b]),bottom);ctx.stroke();ctx.setLineDash([]);}}if(hoverTime!==null){ctx.strokeStyle='#6b7f8d';ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(xx(hoverTime),top);ctx.lineTo(xx(hoverTime),bottom);ctx.stroke();ctx.setLineDash([]);}ctx.restore();ctx.strokeStyle='#93a9b9';ctx.strokeRect(p.left,top,p.right-p.left,ph);}
 const step=niceStep((view[1]-view[0])/10);ctx.textAlign='center';ctx.fillStyle='#4a6273';for(let t=Math.ceil(view[0]/step)*step;t<=view[1];t+=step){const x=xx(t);ctx.fillText(String(Math.round(t*100)/100),x,p.h-19);}ctx.fillText('Thời gian median (ms); anchor = 0 ms',(p.left+p.right)/2,p.h-2);ctx.textAlign='left';$('view-range').textContent=view.map(v=>v.toFixed(1)).join(' → ')+' ms';}
function niceStep(n){const power=10**Math.floor(Math.log10(n)),part=n/power;return(part<=1?1:part<=2?2:part<=5?5:10)*power;}
function clampView(a,b){const r=rec();if(!r.signal_available)return;const first=r.time_ms[0],last=r.time_ms[r.time_ms.length-1],width=Math.max(15*1000/r.fs,Math.min(last-first,b-a));a=Math.min(Math.max(a,first),last-width);view=[a,a+width];draw();}
function zoom(scale,anchor=null){if(!rec().signal_available)return;const mid=anchor===null?(view[0]+view[1])/2:anchor;clampView(mid-(mid-view[0])*scale,mid+(view[1]-mid)*scale);}
function pointerTime(event){const p=layout(),x=event.clientX-canvas.getBoundingClientRect().left;return view[0]+(x-p.left)/(p.right-p.left)*(view[1]-view[0]);}
canvas.addEventListener('pointerdown',event=>{if(event.shiftKey&&rec().signal_available){pan={x:event.clientX,view:[...view]};canvas.setPointerCapture(event.pointerId);}});
canvas.addEventListener('pointermove',event=>{if(!rec().signal_available)return;if(pan){const p=layout(),dt=(event.clientX-pan.x)/(p.right-p.left)*(pan.view[1]-pan.view[0]);clampView(pan.view[0]-dt,pan.view[1]-dt);return;}hoverTime=C.nearestTime(rec(),pointerTime(event));$('hover').textContent='Con trỏ: '+hoverTime+' ms · đặt '+LABELS[$('active-boundary').value];draw();});
canvas.addEventListener('pointerup',event=>{if(pan){pan=null;return;}if(!reviewer||!rec().signal_available||event.shiftKey)return;const p=layout(),box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;if(x<p.left||x>p.right||y<p.top||y>p.bottom)return;const b=$('active-boundary').value;$(b+'-value').value=C.nearestTime(rec(),pointerTime(event));$(b+'-status').value='visible';change();});
canvas.addEventListener('pointerleave',()=>{if(!pan){hoverTime=null;draw();}});canvas.addEventListener('wheel',event=>{if(!rec().signal_available)return;event.preventDefault();if(event.shiftKey){const d=(view[1]-view[0])*.1*Math.sign(event.deltaY||event.deltaX);clampView(view[0]+d,view[1]+d);}else zoom(event.deltaY>0?1.2:1/1.2,pointerTime(event));},{passive:false});
function navigate(index){if(dirty&&!confirm('Bản nháp chưa lưu sẽ bị bỏ. Chuyển recording?'))return;$('record').value=String((index+records.length)%records.length);changeRecord();}
$('footer').textContent='Frank vx/vy/vz có sẵn của PTB; không chuyển đổi ECG → VCG. Tất cả mốc là median_relative_ms. '+records.filter(r=>!r.signal_available).length+' / '+records.length+' ca đã chọn không có median; không thay bằng raw hoặc số 0. Khi nằm đúng giữa hai samples, chọn sample sớm hơn; thời gian ngoài median bị từ chối, không cắt về biên.';
$('reference-selection').addEventListener('input',change);
for(let i=0;i<records.length;i++){const option=document.createElement('option');option.value=String(i);option.textContent=(i+1)+'. '+records[i].record_id+' · '+records[i].split+(records[i].signal_available?'':' · không có median');$('record').appendChild(option);}
setupTable();$('use-reviewer').onclick=switchReviewer;$('reviewer').addEventListener('keydown',e=>{if(e.key==='Enter')switchReviewer();});$('record').addEventListener('change',()=>{if(dirty&&!confirm('Bỏ bản nháp chưa lưu và đổi recording?')){$('record').value=String(currentRecordIndex);return;}changeRecord();});$('prev').onclick=()=>navigate(Number($('record').value)-1);$('next').onclick=()=>navigate(Number($('record').value)+1);$('active-boundary').onchange=highlightActive;$('show-diagnosis').onchange=()=>{$('diagnosis').classList.toggle('hidden',!$('show-diagnosis').checked);};$('save').onclick=saveRow;$('reveal').onclick=reveal;$('clear').onclick=()=>{if(!reviewer)return;for(const b of C.boundaries){$(b+'-value').value='';$(b+'-status').value='not_visible';}change();};$('export').onclick=exportRows;$('import').onchange=importRows;$('import').disabled=true;$('fit').onclick=()=>{if(rec().signal_available){view=[rec().time_ms[0],rec().time_ms[rec().time_ms.length-1]];draw();}};$('zoom-in').onclick=()=>zoom(1/1.5);$('zoom-out').onclick=()=>zoom(1.5);$('pan-left').onclick=()=>{const d=(view[1]-view[0])*.2;clampView(view[0]-d,view[1]-d);};$('pan-right').onclick=()=>{const d=(view[1]-view[0])*.2;clampView(view[0]+d,view[1]+d);};for(const id of ['source','quality','review-status','independent','notes'])$(id).addEventListener('input',change);window.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;if(['1','2','3','4'].includes(event.key)){$('active-boundary').value=C.boundaries[Number(event.key)-1];highlightActive();}});window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});window.addEventListener('resize',draw);changeRecord();
</script></body></html>'''
