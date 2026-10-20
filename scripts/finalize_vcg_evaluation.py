#!/usr/bin/env python3
"""Add descriptive failure diagnostics to an already completed frozen run.

This never modifies candidates, split, QC, or expert annotation columns.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd


def table(frame):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| '+' | '.join(map(cell, frame.columns))+' |',
             '| '+' | '.join('---' for _ in frame.columns)+' |']
    lines += ['| '+' | '.join(map(cell, row))+' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def finalize(output):
    output = Path(output)
    frame = pd.read_csv(output/'features/all_features.csv')
    beat = pd.read_csv(output/'evaluation/beat_quality_results.csv')
    count = frame.groupby('label_name').size()
    methods = frame.groupby(['label_name', 't_method']).size().rename('n_recordings').reset_index()
    methods['all_label_recordings'] = methods.label_name.map(count)
    methods['pct_of_all_label_recordings'] = methods.n_recordings/methods.all_label_recordings*100
    methods.to_csv(output/'evaluation/t_method_summary.csv', index=False)
    beat_summary = beat.groupby(['label_name', 'reason']).agg(
        n_beats=('sample', 'size'), n_records=('record_id', 'nunique'),
        n_patients=('patient_id', 'nunique')).reset_index()
    beat_summary['all_detected_beats'] = beat_summary.label_name.map(beat.groupby('label_name').size())
    beat_summary['pct_of_detected_beats'] = beat_summary.n_beats/beat_summary.all_detected_beats*100
    beat_summary.to_csv(output/'evaluation/beat_exclusion_summary.csv', index=False)
    failed = frame[~frame.processing_success].copy()
    reasons = []
    for rid in failed.record_id:
        selected = beat[beat.record_id.eq(rid)]
        reasons.append('; '.join(f'{name}: {n}' for name, n in selected.reason.value_counts().items()))
    failed['beat_exclusion_reasons'] = reasons
    fields = ['record_id', 'label_name', 'split', 'n_qrs_detected', 'n_beats_used',
              'n_beats_excluded', 'failure_stage', 'beat_exclusion_reasons']
    failed[fields].to_csv(output/'evaluation/failure_mode_summary.csv', index=False)
    separation = []
    for label, group in frame.groupby('label_name'):
        separation.append(dict(label_name=label, n_recordings=len(group),
            input_quality_flagged=int(group.input_qc_flags.notna().sum()),
            additional_evaluation_flagged=int(group.evaluation_flags.notna().sum()),
            v1_auto_with_input_warning=int((group.qc_status.eq('auto_candidate') & group.input_qc_flags.notna()).sum())))
    pd.DataFrame(separation).to_csv(output/'evaluation/additional_qc_summary.csv', index=False)
    missing_t = frame[frame.processing_success & ~frame.t_boundary_valid]
    mi = frame[frame.label_name.eq('MI')]
    mi_t = int(mi.t_boundary_valid.sum())
    mi_t_range = int(mi.t_algorithm_range_valid.sum())
    audit = pd.read_csv(output/'evaluation/boundary_reference_audit.csv')
    accepted = int(audit.accepted.eq(True).sum())
    log = (output/'evaluation/run.log').read_text()
    resumes = re.findall(r'Eligible=(\d+); requested=(\d+); resumed=(\d+); pending=(\d+)', log)
    resumed = bool(resumes and int(resumes[-1][2]) == len(frame) and int(resumes[-1][3]) == 0)
    resume_text = (f'Resume đã dùng lại đủ {len(frame)} checkpoints, không chạy lại record. '
                   if resumed else 'Lần gọi gần nhất chưa xác nhận resume toàn bộ checkpoints. ')
    missing_fields = ['record_id', 'split', 'feature_completeness', 't_method', 't_qc_flags']
    marker = '\n<!-- detailed-batch-diagnostics -->\n'
    report_path = output/'full_evaluation_report.md'
    base = report_path.read_text().split(marker)[0]
    config = json.loads((output/'config.json').read_text())
    split_text = json.dumps(config.get('split_details', {}), ensure_ascii=False, indent=2, default=str)
    if split_text in base and '```json\n'+split_text not in base:
        base = base.replace(split_text, '```json\n'+split_text+'\n```', 1)
    lines = [marker, '## Chẩn đoán chi tiết các ca không hoàn tất', '',
        'Các số sau được lấy từ đúng các hàm V1. Khi median beat không tạo được, '
        'batch giữ các dòng quyết định nhịp từ biến cục bộ của hàm trước lúc exception '
        'thoát ra; không đổi điều kiện giữ/loại hoặc tính lại nhịp theo quy tắc khác.', '',
        table(failed[fields]), '',
        f'Cả {len(failed)} ca đều có anchors do detector trả về. Điểm dừng là không còn đủ ba '
        'nhịp theo các quy tắc RR/cửa sổ/lỗi dữ liệu để tạo median beat; đây không '
        'phải bằng chứng detector bỏ sót toàn bộ QRS. `rr_outlier` có thể phản ánh '
        'bất thường nhịp thật hoặc detector; `adjacent_qrs_in_window` phản ánh khoảng '
        'cửa sổ không đáp ứng điều kiện của V1. Cần nhãn nhịp độc lập để phân biệt.', '',
        f'{len(missing_t)} recording chạy tới cuối nhưng thiếu phân đoạn T chỉ còn bảy đặc trưng '
        'QRS; các đặc trưng T được giữ NaN:', '', table(missing_t[missing_fields]), '',
        'Các record có T duration ngoài 60–400 ms vẫn giữ candidate features theo '
        f'đúng V1; trong {mi_t} ca MI đủ T-on/off, có {mi_t_range} ca qua giới hạn duration và '
        'thứ tự thuật toán. Đây là giới hạn kiểm tra kỹ thuật, không là nhãn bệnh '
        'hay kết luận biên sai.', '',
        '## Phân bố phương pháp T và quyết định nhịp', '', table(methods.round(2)), '',
        'Tên `t_method` mô tả các candidate đã giải được. Khi cặp biên cuối không '
        'hợp lệ hoặc thiếu một biên, tên này vẫn có thể chứa một phương pháp đã '
        'giải được; phải đọc cùng `t_boundary_valid` và `t_qc_flags`.', '',
        table(beat_summary.round(2)), '',
        'Tỷ lệ nhịp có mẫu số là tổng anchors trong từng nhãn, không phải số '
        'recordings. Nhiều nhịp cùng recording/bệnh nhân phụ thuộc nhau; các tỷ lệ '
        'này chỉ mô tả quy tắc chọn median. Hình thái khác được giữ và gắn cờ theo '
        'V1, không tự loại vì là MI.', '',
        '## Phân biệt bộ QC và chuẩn bị chú giải độc lập', '',
        'Trạng thái `auto_candidate`/`needs_review` giữ nguyên bộ cờ của V1. '
        'Các cờ đầu vào và `eval_*` được lưu riêng để không âm thầm đổi định nghĩa '
        'QC của baseline. Xem [additional_qc_summary.csv](evaluation/additional_qc_summary.csv). '
        'Trong lần chạy này, không có auto_candidate nào mang cờ input-quality.', '',
        '[CSV review không chứa dự đoán/QC/nhãn bệnh](annotations/manual_review_blinded.csv) '
        'đi kèm [tín hiệu median không vẽ biên dự đoán](annotations/blinded_signals/). '
        'Dùng bộ này cho lượt chú giải độc lập đầu; template gốc có `pred_*` dành '
        'cho đối chiếu sau đó. Những ca không tạo được median được ghi '
        '`no_median_beat`: không có median để chuyên gia chú giải; xem tín hiệu '
        'recording và đánh giá QRS/tiêu chí giữ nhịp riêng. Không tạo mốc tham '
        'chiếu trên một median không tồn tại.', '',
        f'Đã chạy script so sánh với template đang pending: **{accepted}/{len(audit)} dòng được nhận '
        'làm chuẩn**. Khi không có dòng được nhận, bảng boundary comparison chỉ có header. Đây là kiểm tra '
        'không dùng nhầm dự đoán làm ground truth; chưa phải đánh giá accuracy.', '',
        '## Kiểm chứng triển khai', '',
        '- [Log kiểm tra tự động](evaluation/test_results.log).',
        '- [Đối chiếu trực tiếp V1 trên toàn cohort](evaluation/native_v1_consistency_summary.json).',
        '- [Trạng thái hoàn tất và hash baseline/mã đánh giá](evaluation/completion.json).',
        '- [Hướng dẫn chạy, resume và review](baseline_v1/README_vcg_full_evaluation.md).', '',
        resume_text+'Baseline, '
        'mã đánh giá, thư viện, hash header/tín hiệu và split phải khớp trước khi '
        'dùng lại kết quả. Không đổi thuật toán hoặc tối ưu tham số theo test.', '']
    report_path.write_text(base+'\n'.join(lines), encoding='utf-8')
    repo = Path(__file__).resolve().parents[1]
    sources = ['README_vcg_full_evaluation.md', 'prepare_vcg_blind_review.py',
               'finalize_vcg_evaluation.py']
    hashes = {}
    for name in sources:
        source = repo/'scripts'/name
        shutil.copy2(source, output/'baseline_v1'/name)
        hashes['scripts/'+name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ['test_vcg_annotation_comparison.py', 'test_vcg_full_evaluation.py']:
        source = repo/'tests'/name
        shutil.copy2(source, output/'baseline_v1/tests'/name)
        hashes['tests/'+name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output/'evaluation/postprocessing_manifest.json').write_text(
        json.dumps(hashes, indent=2)+'\n')
    print('Saved failure-mode, beat-exclusion, T-method summaries and report appendix.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/vcg_full_evaluation'))
    finalize(parser.parse_args().output_dir)
