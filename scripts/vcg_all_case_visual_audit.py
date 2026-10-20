"""Exhaustive visual audit of saved candidate disagreements, never ground truth."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = (0, 10, 20, 40, 60, 100)
COLORS = {'t_on': '#146bb3', 't_off': '#dc7415'}


def build_disagreement_tables(frame, output_dir, qc_dir=None):
    """Use exact saved tangent/threshold timestamps; retain absent pairs."""
    out = Path(output_dir) / 'all_cases'
    out.mkdir(parents=True, exist_ok=True)
    if not frame.record_id.is_unique:
        raise ValueError('Duplicate record_id')
    records = frame.copy().reset_index(drop=True)
    records['has_median'] = records.processing_success.astype(bool)
    for boundary, suffix in [('t_on', 'on'), ('t_off', 'off')]:
        first = pd.to_numeric(records[f't_tangent_{suffix}_ms'], errors='coerce')
        second = pd.to_numeric(records[f't_threshold_{suffix}_ms'], errors='coerce')
        pair = np.isfinite(first) & np.isfinite(second)
        records[boundary + '_pair_available'] = pair
        records[boundary + '_signed_gap_ms'] = (first - second).where(pair)
        records[boundary + '_gap_ms'] = (first - second).abs().where(pair)
        records[boundary + '_pair_status'] = np.select(
            [pair, np.isfinite(first), np.isfinite(second)],
            ['both_candidates', 'threshold_missing', 'tangent_missing'], default='both_missing')
    on, off = records.t_on_pair_available, records.t_off_pair_available
    records['method_pair_status'] = np.select([on & off, on, off], ['both', 'on_only', 'off_only'], default='neither')
    records['max_method_gap_ms'] = records[['t_on_gap_ms', 't_off_gap_ms']].max(axis=1, skipna=True)
    records['any_pair_available'] = on | off
    for threshold in TOLERANCES:
        records[f'any_gap_gt{threshold}'] = records[['t_on_gap_ms', 't_off_gap_ms']].gt(threshold).any(axis=1)
    records['observed_equal_only_on_available_pairs'] = records.any_pair_available & records.max_method_gap_ms.eq(0)
    records['reference_status'] = 'GROUND_TRUTH_UNAVAILABLE'
    ledger_path = Path(output_dir) / 'reference_availability_per_record.csv'
    if ledger_path.is_file():
        ledger = pd.read_csv(ledger_path).set_index('record_id')
        records['reference_status'] = records.record_id.map(ledger.reference_status).fillna('GROUND_TRUTH_UNAVAILABLE')
    qc_dir = Path(qc_dir or ROOT / 'outputs/vcg_qc_audit_v1')
    t = pd.read_csv(qc_dir / 't_boundary_audit.csv').set_index('record_id')
    q = pd.read_csv(qc_dir / 'qrs_boundary_audit.csv').set_index('record_id')
    records['n_significant_lobes'] = records.record_id.map(t.n_significant_lobes)
    records['qrs_secondary_activity_count'] = records.record_id.map(q.secondary_activity_count)
    # An outside-search component is not necessarily ventricular depolarization.
    records['qrs_secondary_screen_positive'] = records.record_id.map(
        q.screening_signals.fillna('').str.contains('secondary', regex=False))
    flags = records.qc_flags.fillna('').astype(str)
    categories = {
        't_on_gap_gt60ms': records.t_on_gap_ms.gt(60),
        't_off_gap_gt100ms': records.t_off_gap_ms.gt(100),
        'any_method_gap_gt0': records.any_gap_gt0,
        'any_method_gap_gt20': records.any_gap_gt20,
        'any_method_gap_gt60': records.any_gap_gt60,
        'st_flag': flags.str.split(';').map(lambda xs: 'st_level_requires_review' in [x.strip() for x in xs]),
        't_duration_review': records.t_duration_ms.lt(60) | records.t_duration_ms.gt(400),
        't_multiple_lobes': records.n_significant_lobes.gt(1),
        'qrs_outside_velocity_activity': records.qrs_secondary_activity_count.gt(0),
        'qrs_secondary_screen': records.qrs_secondary_screen_positive,
        'features_incomplete': records.feature_completeness.lt(12),
        'median_missing': ~records.has_median,
        'no_qc_flags': flags.str.strip().eq('') & records.has_median,
        'candidate_pair_missing': ~(on & off),
    }
    records['priority_categories'] = [
        ';'.join(key for key, mask in categories.items() if bool(mask.loc[ix])) for ix in records.index]
    rows = []
    for scope in ['all', 'train_valid', 'train', 'valid', 'test']:
        base = records if scope == 'all' else records[records.split.isin(['train', 'valid'])] if scope == 'train_valid' else records[records.split.eq(scope)]
        for label in ['all', 'HC', 'MI']:
            group = base if label == 'all' else base[base.label_name.eq(label)]
            for boundary in ['t_on', 't_off', 'any_boundary']:
                pair = group.any_pair_available if boundary == 'any_boundary' else group[boundary + '_pair_available']
                for threshold in TOLERANCES:
                    changed = group[f'any_gap_gt{threshold}'] if boundary == 'any_boundary' else group[boundary + '_gap_ms'].gt(threshold)
                    rows.append({'scope': scope, 'label_name': label, 'boundary': boundary,
                        'metric': 'absolute_tangent_threshold_gap_strictly_gt', 'tolerance_ms': threshold,
                        'n_total': len(group), 'n_cases': int((changed & pair).sum()),
                        'n_denominator': int(pair.sum()), 'n_missing_pair': int((~pair).sum()),
                        'n_patients': group.loc[changed & pair, 'patient_id'].nunique(),
                        'pct_of_available_pairs': 100 * (changed & pair).sum() / pair.sum() if pair.sum() else np.nan,
                        'interpretation': 'Method disagreement, not independently verified boundary error',
                        'test_scope': 'Descriptive only; no tuning'})
    summary = pd.DataFrame(rows)
    category_rows = []
    for key, mask in categories.items():
        for label in ['all', 'HC', 'MI']:
            eligible = pd.Series(True, index=records.index) if label == 'all' else records.label_name.eq(label)
            category_rows.append({'category': key, 'label_name': label,
                'n_cases': int((mask & eligible).sum()), 'n_denominator': int(eligible.sum()),
                'n_patients': records.loc[mask & eligible, 'patient_id'].nunique(),
                'interpretation': 'Overlapping diagnostic groups; no confirmed error label'})
    groups = pd.DataFrame(category_rows)
    records = records.sort_values(['max_method_gap_ms', 'record_id'], ascending=[False, True], na_position='last').reset_index(drop=True)
    records['display_rank'] = np.arange(1, len(records) + 1)
    records.to_csv(out / 'all_case_disagreements.csv', index=False)
    summary.to_csv(out / 'method_gap_summary.csv', index=False)
    groups.to_csv(out / 'diagnostic_group_counts.csv', index=False)
    for threshold in TOLERANCES:
        records[records[f'any_gap_gt{threshold}']].to_csv(out / f'cases_gap_gt{threshold}ms.csv', index=False)
    for key in categories:
        mask = records.priority_categories.str.split(';').map(lambda xs: key in xs)
        records[mask].to_csv(out / f'cases_{key}.csv', index=False)
    config = {'source': 'Saved V1 tangent and threshold candidates in Notebook06 CSVs',
        'signed_gap': 'tangent_ms - threshold_ms', 'absolute_gap': 'abs(tangent_ms - threshold_ms)',
        'threshold_comparison': 'strictly greater than; descriptive levels only',
        'tolerances_ms': list(TOLERANCES), 'unavailable_pairs_not_zero': True,
        'group_overlap': 'A record may occur in more than one boundary/category; union counts each record once',
        'qrs_reference_or_candidate_pair_available': False,
        'qrs_note': 'No independent QRS reference/second saved QRS boundary set; show actual V1 cuts and diagnostics only',
        'actual_error_status': 'Consult accepted references; method gaps do not establish correctness',
        'test_use': 'All448 displayed as explicitly requested; no threshold fitting or algorithm tuning'}
    (out / 'visual_audit_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    return {'records': records, 'summary': summary, 'categories': groups}


def plot_disagreement_summary(tables, output_dir):
    records, summary, categories = (tables[k] for k in ['records', 'summary', 'categories'])
    out = Path(output_dir) / 'all_cases' / 'summary_plots'
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(fig, name, note='Chênh lệch phương pháp ≠ sai số so với tham chiếu. Test chỉ xem mô tả; không tinh chỉnh.'):
        fig.text(.02, .012, note, fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, .94))
        path = out / name
        fig.savefig(path, dpi=130, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        paths.append({'plot': name, 'plot_path': str(path.relative_to(Path(output_dir))), 'n_records': len(records)})
    fig, ax = plt.subplots(figsize=(11, 4.6))
    available = np.array([int(records[b + '_pair_available'].sum()) for b in ['t_on', 't_off']])
    unavailable = len(records) - available
    ax.barh(['T-on', 'T-off'], available, color='#2c8a78', label='Đủ tangent + threshold')
    ax.barh(['T-on', 'T-off'], unavailable, left=available, color='#c8cbd0', label='Thiếu ít nhất một candidate')
    for j, (a, z) in enumerate(zip(available, unavailable)):
        ax.text(a/2, j, f'{a}/{len(records)} đủ cặp', ha='center', va='center', fontsize=11, color='white')
        ax.text(a+z/2, j, f'{z} thiếu', ha='center', va='center', fontsize=10)
    ax.set(xlabel='Số recordings', xlim=(0, len(records)), title='Có bao nhiêu ca thực sự đối chiếu được hai phương pháp?')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2)
    save(fig, '01_candidate_coverage.png')

    all_counts = summary[summary.scope.eq('all') & summary.label_name.eq('all')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(TOLERANCES))
    for j, boundary in enumerate(['t_on', 't_off']):
        values = all_counts[all_counts.boundary.eq(boundary)].set_index('tolerance_ms').loc[list(TOLERANCES)]
        bars = axes[0].bar(x+(j-.5)*.36, values.n_cases, .36, color=COLORS[boundary],
            label=f"{'T-on' if boundary=='t_on' else 'T-off'} (đủ cặp n={int(values.n_denominator.iloc[0])})")
        axes[0].bar_label(bars, padding=3, fontsize=9)
    unique = all_counts[all_counts.boundary.eq('any_boundary')].set_index('tolerance_ms').loc[list(TOLERANCES)]
    bars = axes[1].bar(x, unique.n_cases, color='#7751a8', width=.65)
    axes[1].bar_label(bars, padding=3)
    for ax in axes:
        ax.set(xticks=x, xticklabels=[f'>{t}' for t in TOLERANCES], xlabel='Ngưỡng chênh tuyệt đối (ms)', ylabel='Số recordings')
        ax.grid(axis='y', alpha=.18)
        ax.set_ylim(0, max(unique.n_cases.max()*1.18, 1))
    axes[0].set_title('Đếm riêng từng biên (các mức là tích lũy)'); axes[0].legend(fontsize=9)
    axes[1].set_title('Số ca duy nhất: T-on hoặc T-off chênh ngưỡng')
    save(fig, '02_counts_by_disagreement_ms.png', 'Không cộng các cột tích lũy hoặc cộng T-on với T-off. Mẫu thiếu candidate không được coi là khớp/sai.')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.1))
    for ax, boundary, suffix in zip(axes, ['t_on', 't_off'], ['on', 'off']):
        pair = records[records[boundary + '_pair_available']]
        for label, color in [('HC', '#2176aa'), ('MI', '#c46636')]:
            group = pair[pair.label_name.eq(label)]
            ax.scatter(group[f't_threshold_{suffix}_ms'], group[f't_tangent_{suffix}_ms'], s=23, alpha=.75, color=color, label=f'{label}: n={len(group)}')
        if len(pair):
            vals = pair[[f't_threshold_{suffix}_ms', f't_tangent_{suffix}_ms']].to_numpy()
            lo, hi = np.nanmin(vals)-12, np.nanmax(vals)+12
            ax.plot([lo, hi], [lo, hi], '--', color='.35', label='Hai thời điểm trùng nhau')
        ax.set(title=f"{'T-on' if suffix=='on' else 'T-off'}: đủ cặp {len(pair)}/{len(records)}", xlabel='Threshold (ms so với median anchor)', ylabel='Tangent (ms so với median anchor)')
        ax.grid(alpha=.18); ax.legend(fontsize=9)
    save(fig, '03_candidate_times_scatter.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, boundary in zip(axes, ['t_on', 't_off']):
        vals = records[boundary + '_gap_ms'].dropna()
        if len(vals):
            ax.hist(vals, bins=np.arange(0, max(210, vals.max()+10), 10), color=COLORS[boundary], edgecolor='white')
        ax.set(title=f'{boundary}: n={len(vals)}, thiếu={len(records)-len(vals)}', xlabel='|Tangent − threshold| (ms)', ylabel='Số recordings')
        ax.grid(axis='y', alpha=.2)
    save(fig, '04_method_gap_distribution.png')

    by_label = summary[summary.scope.eq('all') & summary.label_name.isin(['HC','MI']) & summary.boundary.isin(['t_on','t_off']) & summary.tolerance_ms.eq(60)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, boundary in zip(axes, ['t_on', 't_off']):
        sub = by_label[by_label.boundary.eq(boundary)].set_index('label_name').reindex(['HC','MI'])
        ax.bar(['HC','MI'], sub.pct_of_available_pairs, color=['#2176aa','#c46636'])
        for j, label in enumerate(['HC','MI']):
            row = sub.loc[label]
            ax.text(j, row.pct_of_available_pairs+1, f'{int(row.n_cases)}/{int(row.n_denominator)} đủ cặp\nthiếu {int(row.n_missing_pair)}/{int(row.n_total)}', ha='center', fontsize=10)
        ax.set(title=boundary+' — chênh >60 ms', ylabel='% trong số có đủ cặp', ylim=(0,100))
    save(fig, '05_disagreements_by_label.png', 'Tỷ lệ chỉ trên các cặp quan sát được, không ước lượng tỷ lệ lỗi HC/MI trên toàn cohort.')

    fig, ax = plt.subplots(figsize=(11, 8.8))
    ordered = records.sort_values(['max_method_gap_ms','record_id'], ascending=[False,True], na_position='last')
    matrix = ordered[['t_on_gap_ms','t_off_gap_ms']].to_numpy(float)
    cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#babec5')
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap, vmin=0, vmax=max(200,np.nanmax(matrix)))
    ax.set(xticks=[0,1], xticklabels=['T-on','T-off'], xlabel='Candidate tangent so với threshold', ylabel='Mỗi hàng = một recording, xếp theo chênh lớn nhất giảm dần', title=f'Toàn bộ {len(records)} ca — ô xám là thiếu cặp, không phải 0 ms')
    ticks = np.linspace(0,len(records)-1,10,dtype=int)
    ax.set_yticks(ticks, [f'{i+1}: {ordered.iloc[i].record_id}' for i in ticks], fontsize=8)
    fig.colorbar(im, ax=ax, label='Chênh tuyệt đối (ms)')
    save(fig, '06_all_records_gap_map.png')

    labels = {'st_flag':'Cờ ST', 't_duration_review':'T duration cần review', 't_multiple_lobes':'T nhiều thùy',
        'qrs_outside_velocity_activity':'Có hoạt động vận tốc ngoài khoảng QRS',
        'qrs_secondary_screen':'Đạt screen phụ QRS', 'features_incomplete':'Thiếu features', 'median_missing':'Không tạo được median', 'no_qc_flags':'Không có cờ QC'}
    vals = categories[categories.label_name.eq('all') & categories.category.isin(labels)].set_index('category').reindex(list(labels))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    bars = ax.barh([labels[k] for k in vals.index], vals.n_cases, color='#4b829e')
    ax.bar_label(bars, padding=3); ax.invert_yaxis(); ax.set(xlabel='Số recordings (nhóm có thể chồng nhau)', title='Các nhóm cần xem hình thực tế — không tự gán là tách sai', xlim=(0,max(1,vals.n_cases.max()*1.2)))
    save(fig, '07_diagnostic_groups.png', 'Hoạt động ngoài QRS có thể là P/ST; ST và MI không tự chứng minh biên sai. Toàn bộ nhóm có thể lọc trong gallery.')
    result = pd.DataFrame(paths)
    result.to_csv(Path(output_dir)/'all_cases/summary_plot_manifest.csv', index=False)
    return result


def create_all_case_contact_sheets(records, artifact_root, output_dir, page_size=12):
    """Every record gets a real magnitude/time panel, including missing medians.

    Full-resolution six-panel figures are separate; sheets are an exhaustive
    overview for embedding in the notebook, not a sampled selection.
    """
    if not isinstance(page_size, int) or not 1 <= page_size <= 12:
        raise ValueError('page_size must be an integer from 1 through 12')
    out = Path(output_dir)/'all_cases/contact_sheets'; out.mkdir(parents=True,exist_ok=True)
    root = Path(artifact_root)
    if (root/'artifacts').is_dir(): root = root/'artifacts'
    manifest = []
    for start in range(0,len(records),page_size):
        page = records.iloc[start:start+page_size]
        number=start//page_size+1
        path=out/f'page_{number:03d}.png'
        checkpoint=path.with_suffix('.json')
        digest=hashlib.sha256(Path(__file__).read_bytes()+page.to_json(orient='records',double_precision=12).encode())
        for rid in page.record_id:
            safe=rid.replace('/','_')
            for suffix in ['.npz','.json']:
                digest.update((root/(safe+suffix)).read_bytes())
        fingerprint=digest.hexdigest()
        saved=json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
        if (path.is_file() and saved.get('fingerprint')==fingerprint and
                saved.get('png_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()):
            manifest.extend({'page':number,'record_id':row.record_id,'display_rank':int(row.display_rank),
                'plot_path':str(path.relative_to(Path(output_dir)))} for row in page.itertuples(index=False))
            continue
        fig, axes = plt.subplots(4,3,figsize=(17,12.5))
        for ax, (_, row) in zip(axes.flat,page.iterrows()):
            safe=row.record_id.replace('/','_')
            jp=root/(safe+'.json'); npz=root/(safe+'.npz')
            artifact=json.loads(jp.read_text())
            with np.load(npz,allow_pickle=False) as arrays:
                if 'median_xyz' not in arrays:
                    ax.text(.5,.5,'KHÔNG CÓ MEDIAN V1\nKhông có biên/loop để so sánh',ha='center',va='center',transform=ax.transAxes,color='#92472d',fontsize=11)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    time=arrays['time_ms']; xyz=arrays['median_xyz']
                    magnitude=arrays['t_magnitude'] if 't_magnitude' in arrays else np.linalg.norm(xyz,axis=1)
                    b=artifact.get('boundaries',{})
                    ix=b.get('qrs_off'); left=float(time[ix])+5 if ix is not None else 70.
                    cand=[float(row[c]) for c in ['t_on_ms','t_off_ms','t_tangent_on_ms','t_tangent_off_ms','t_threshold_on_ms','t_threshold_off_ms'] if pd.notna(row.get(c))]
                    right=min(float(time[-1]),max([450.]+[v+35 for v in cand]))
                    mask=(time>=left)&(time<=right)
                    ax.plot(time[mask],magnitude[mask],color='#263c46',lw=1)
                    for boundary,suffix in [('t_on','on'),('t_off','off')]:
                        pred=row.get(boundary+'_ms')
                        if pd.notna(pred): ax.axvline(pred,color=COLORS[boundary],lw=1.1)
                        tangent=row.get('t_tangent_'+suffix+'_ms'); threshold=row.get('t_threshold_'+suffix+'_ms')
                        if pd.notna(tangent): ax.axvline(tangent,color=COLORS[boundary],ls=':',lw=.9)
                        if pd.notna(threshold): ax.axvline(threshold,color='#8c4fb0',ls='--',lw=.9)
                        if pd.notna(tangent) and pd.notna(threshold): ax.axvspan(min(tangent,threshold),max(tangent,threshold),color=COLORS[boundary],alpha=.12)
                    ax.set_xlim(left,right); ax.set_xlabel('ms so với median anchor',fontsize=8); ax.set_ylabel('|XYZ| (mV)',fontsize=8)
                    ax.tick_params(labelsize=8); ax.grid(alpha=.15)
            def gap(boundary):
                value=row[boundary+'_gap_ms']; return f'{value:g}' if pd.notna(value) else 'thiếu cặp'
            ax.set_title(f"#{int(row.display_rank)} {row.record_id} · {row.label_name} · {row.split}\nΔon={gap('t_on')} ms; Δoff={gap('t_off')} ms",fontsize=8.5)
        for ax in axes.flat[len(page):]: ax.axis('off')
        fig.suptitle(f'Tất cả ca — trang {number}/{int(np.ceil(len(records)/page_size))} | recordings {start+1}–{start+len(page)}/{len(records)}',fontsize=14)
        fig.text(.02,.012,'V1: nét liền; tangent: nét chấm; threshold: nét đứt tím; vùng tô = khoảng giữa hai candidate. Δ chỉ là bất đồng phương pháp. Mỗi ca có hình chi tiết trong gallery.',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95))
        fig.savefig(path,dpi=115,facecolor='white',bbox_inches='tight'); plt.close(fig)
        checkpoint.write_text(json.dumps({'fingerprint':fingerprint,'png_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})+'\n')
        for _, row in page.iterrows():
            manifest.append({'page':number,'record_id':row.record_id,'display_rank':int(row.display_rank),
                'plot_path':str(path.relative_to(Path(output_dir)))})
    result=pd.DataFrame(manifest)
    result.to_csv(Path(output_dir)/'all_cases/contact_sheet_manifest.csv',index=False)
    if not result.record_id.is_unique or set(result.record_id)!=set(records.record_id):
        raise ValueError('Contact sheets failed exhaustive record coverage')
    return result
