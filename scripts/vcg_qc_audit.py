#!/usr/bin/env python3
"""Frozen V1 QC/delineation audit, used by notebook 07.

All additional screens are technical review hints, never reference error labels.
Neither original feature CSVs nor V1 algorithms are modified.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_qc_audit_matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcg_pipeline import FEATURE_NAMES, robust_sigma, ms_samples
from vcg_preprocessing import Config
from run_full_vcg_evaluation import sha256, write_json, jsonable

SCREENS = {
    'method_gap_ms': 60.0,
    'qrs_secondary_component_min_ms': 10.0,
    'qrs_secondary_component_peak_fraction': .15,
    't_on_peak_fraction': .50,
    't_off_peak_fraction': .50,
    't_tail_after_ms': 20.0,
    't_tail_peak_fraction': .20,
}
COMMON = ['record_id', 'patient_id', 'label', 'label_name', 'split', 'qc_status', 'qc_flags', 'feature_completeness']
BASELINE_FILES = ['vcg_pipeline.py', 'vcg_preprocessing.py', 'scripts/run_full_vcg_evaluation.py',
    '05_QRS_T_Loops_Step_by_Step.ipynb', '06_PTB_VCG_12_Features_Train_Valid_Test.ipynb']


def split_flags(text):
    return sorted({item.strip() for item in str(text or '').split(';') if item.strip() and item.strip() != 'nan'})


def bool_column(series):
    return series.map(lambda v: str(v).lower() in {'true', '1', '1.0'}).astype(bool)


def load_inputs(input_dir=None, evaluation_dir=None):
    input_dir = Path(input_dir or ROOT/'outputs/vcg_12_features_train_valid_test')
    evaluation_dir = Path(evaluation_dir or ROOT/'outputs/vcg_full_evaluation')
    tables = []
    for name in ['train', 'valid', 'test']:
        part = pd.read_csv(input_dir/f'{name}_features.csv')
        if not part['split'].eq(name).all():
            raise ValueError(f'Wrong split in {name}_features.csv')
        tables.append(part)
    frame = pd.concat(tables, ignore_index=True).sort_values('record_id').reset_index(drop=True)
    manifest = pd.read_csv(input_dir/'split_manifest.csv')
    config = json.loads((input_dir/'config.json').read_text())
    if len(frame) != 448 or not frame.record_id.is_unique:
        raise ValueError('This PTB audit requires the verified 448 unique HC/MI recordings')
    if set(frame.record_id) != set(manifest.record_id) or not manifest.record_id.is_unique:
        raise ValueError('Manifest and feature population differ')
    if frame.groupby('patient_id').split.nunique().gt(1).any():
        raise ValueError('Patient leakage')
    left = frame.set_index('record_id')
    right = manifest.set_index('record_id').loc[left.index]
    for key in ['patient_id', 'split', 'label_name']:
        if not left[key].eq(right[key]).all():
            raise ValueError(f'Manifest mismatch: {key}')
    if not frame.label.eq(frame.label_name.map({'HC': 0, 'MI': 1})).all():
        raise ValueError('Header label mapping differs')
    for key in ['processing_success', 'extraction_success', 'eligible_for_features',
                'qrs_boundary_valid', 't_boundary_valid', 'qrs_algorithm_range_valid', 't_algorithm_range_valid']:
        frame[key] = bool_column(frame[key])
    frame['qc_flags'] = frame.qc_flags.fillna('')
    frame['n_flags'] = frame.qc_flags.map(lambda s: len(split_flags(s)))
    if not np.isfinite(frame[FEATURE_NAMES]).sum(axis=1).eq(frame.feature_completeness).all():
        raise ValueError('Feature completeness inconsistent')
    if frame.boundary_time_reference.ne('median_relative_ms').any():
        raise ValueError('Unexpected boundary time reference')
    for filename, expected in config['source_sha256'].items():
        if sha256(ROOT/filename) != expected:
            raise ValueError(f'V1 source changed: {filename}')
    return frame, manifest, config, input_dir, evaluation_dir


def initialize_audit(frame, input_dir, evaluation_dir, output_dir, source_config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['plots/summary', 'plots/HC', 'plots/MI', 'plots/failure_cases', 'blinded_review', 'source_snapshot']:
        (output_dir/name).mkdir(parents=True, exist_ok=True)
    input_names = ['train_features.csv', 'valid_features.csv', 'test_features.csv', 'split_manifest.csv', 'config.json']
    artifact_root = Path(evaluation_dir)/'artifacts'
    artifact_hashes = {}
    for rid in frame.record_id:
        safe = rid.replace('/', '_')
        for suffix in ['.json', '.npz']:
            p = artifact_root/(safe+suffix)
            if not p.is_file():
                raise FileNotFoundError(p)
            artifact_hashes[p.name] = sha256(p)
    frozen = dict(
        input_sha256={name: sha256(Path(input_dir)/name) for name in input_names},
        baseline_sha256={name: sha256(ROOT/name) for name in BASELINE_FILES},
        artifact_sha256=artifact_hashes,
        config_v1=source_config['config'],
        screening_rules=SCREENS,
        screen_interpretation='Technical triage only; no independent reference error labels',
        reference_time_base='median_relative_ms',
        inference_scope='Recording/patient descriptive; proposals based on train+valid only',
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']},
    )
    path = output_dir/'audit_config.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get('frozen_inputs') != jsonable(frozen):
            raise ValueError('Audit source/data/config changed; choose a new output directory')
    else:
        write_json(path, {'frozen_inputs': frozen, 'baseline_unchanged': True,
            'no_qc_correction': True, 'independent_references_used': 0})
    return artifact_root


def qc_tables(frame, output_dir):
    output_dir = Path(output_dir)
    all_flags = sorted({flag for text in frame.qc_flags for flag in split_flags(text)})
    matrix = pd.DataFrame({flag: frame.qc_flags.map(lambda s: flag in split_flags(s)).astype(int) for flag in all_flags}, index=frame.index)
    rows = []
    for flag in all_flags:
        row = {'qc_flag': flag}
        for label in ['HC', 'MI']:
            subset = frame.label_name.eq(label)
            n = int(matrix.loc[subset, flag].sum())
            row.update({f'{label}_count': n, f'{label}_denominator': int(subset.sum()),
                        f'{label}_pct': n/subset.sum()*100})
        row['total'] = int(matrix[flag].sum())
        row['only_flag_count'] = int((matrix[flag].eq(1) & frame.n_flags.eq(1)).sum())
        row['full12_with_flag'] = int((matrix[flag].eq(1) & frame.extraction_success).sum())
        rows.append(row)
    frequency = pd.DataFrame(rows).sort_values(['total', 'qc_flag'], ascending=[False, True])
    frequency.to_csv(output_dir/'qc_flag_frequency.csv', index=False)
    cooccurrence = matrix.T.dot(matrix)
    cooccurrence.to_csv(output_dir/'qc_flag_cooccurrence.csv', index_label='qc_flag')
    label_rows = []
    for split in ['all', 'train', 'valid', 'test']:
        for label in ['HC', 'MI']:
            group = frame[frame.label_name.eq(label) & (True if split == 'all' else frame.split.eq(split))]
            row = dict(split=split, label_name=label, n_records=len(group), n_patients=group.patient_id.nunique())
            for status in ['auto_candidate', 'needs_review', 'failed']:
                n = int(group.qc_status.eq(status).sum())
                row.update({status: n, status+'_pct': n/len(group)*100 if len(group) else np.nan})
            row.update(single_flag=int(group.n_flags.eq(1).sum()), multiple_flags=int(group.n_flags.gt(1).sum()),
                full12=int(group.extraction_success.sum()), missing_features=int((~group.extraction_success).sum()),
                full12_and_review=int((group.extraction_success & group.qc_status.eq('needs_review')).sum()))
            label_rows.append(row)
    label_summary = pd.DataFrame(label_rows)
    label_summary.to_csv(output_dir/'qc_by_label.csv', index=False)
    nflag_summary = frame.groupby(['label_name', 'n_flags']).agg(n_records=('record_id', 'size'),
        full12=('extraction_success', 'sum')).reset_index()
    nflag_summary['full12_pct'] = nflag_summary.full12/nflag_summary.n_records*100
    nflag_summary.to_csv(output_dir/'feature_completeness_by_flag_count.csv', index=False)
    total = dict(n_records=len(frame), n_patients=frame.patient_id.nunique(),
        no_flags=int(frame.n_flags.eq(0).sum()), needs_review=int(frame.qc_status.eq('needs_review').sum()),
        processing_failed=int((~frame.processing_success).sum()),
        single_flag=int(frame.n_flags.eq(1).sum()), multiple_flags=int(frame.n_flags.gt(1).sum()),
        full12=int(frame.extraction_success.sum()),
        full12_and_review=int((frame.extraction_success & frame.qc_status.eq('needs_review')).sum()),
        missing_features=int((~frame.extraction_success).sum()))
    write_json(output_dir/'qc_totals.json', total)
    return {'frequency': frequency, 'cooccurrence': cooccurrence, 'by_label': label_summary,
            'by_nflags': nflag_summary, 'totals': total, 'matrix': matrix}


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _fraction(value, denominator):
    return float(value/max(abs(denominator), 1e-9)) if np.isfinite(value) and np.isfinite(denominator) else np.nan


def audit_record(row, artifact_root, cfg=None):
    cfg = cfg or Config()
    safe = row['record_id'].replace('/', '_')
    art = json.loads((Path(artifact_root)/(safe+'.json')).read_text())
    common = {key: row[key] for key in COMMON}
    common.update(evidence_level='unverified', audit_group='D_unverified',
                  boundary_time_reference='median_relative_ms')
    q, t = dict(common), dict(common)
    if 'median_anchor_sample' not in art:
        q.update(available=False, reason='no_representative_median', screening_signals='')
        t.update(available=False, reason='no_representative_median', screening_signals='')
        return q, t
    with np.load(Path(artifact_root)/(safe+'.npz')) as saved:
        arrays = {key: saved[key] for key in saved.files}
    b, fs = art['boundaries'], float(art['fs'])
    xyz, time = arrays['median_xyz'], arrays['time_ms']
    speed, smooth, mag = arrays['speed'], arrays['smooth_speed'], arrays['t_magnitude']
    anchor = int(art['median_anchor_sample'])
    # Fail fast if a cached waveform has a different time base/boundary than notebook 06.
    for name in ['qrs_on', 'qrs_off', 't_on', 't_peak', 't_off']:
        observed = np.nan if b[name] is None else float(time[b[name]])
        if not np.allclose(observed, row[name+'_ms'], equal_nan=True, atol=1e-8, rtol=0):
            raise ValueError(f'Artifact boundary mismatch {row["record_id"]}: {name}')
    qlo = max(0, anchor+int(cfg.qrs_search_ms[0]*fs/1000))
    qhi = min(len(time)-1, anchor+int(cfg.qrs_search_ms[1]*fs/1000))
    qp = b['qrs_speed_peak']
    qon, qoff = b['qrs_on'], b['qrs_off']
    qok = qon is not None and qoff is not None and qon < anchor < qoff
    pq0, pq1 = art['pq_indices']
    qbase = float(np.median(smooth[pq0:pq1]))
    qnoise = robust_sigma(smooth[pq0:pq1])
    q.update(available=qok, reason='' if qok else 'missing_or_unordered_boundaries',
        qrs_anchor_ms=float(time[anchor]), median_anchor_index=anchor,
        qrs_on_ms=row['qrs_on_ms'], qrs_off_ms=row['qrs_off_ms'], qrs_duration_ms=row['qrs_duration_ms'],
        qrs_search_start_ms=float(time[qlo]), qrs_search_end_ms=float(time[qhi]),
        velocity_peak_ms=float(time[qp]), velocity_peak_mV_s=float(smooth[qp]),
        qrs_threshold_mV_s=b['q_threshold'], pq_velocity_median=qbase, pq_velocity_noise=qnoise,
        threshold_recomputed=max(qbase+4*qnoise, cfg.qrs_fraction*float(smooth[qp])))
    qsignals = []
    if qok:
        q['left_search_margin_ms'] = (qon-qlo)*1000/fs
        q['right_search_margin_ms'] = (qhi-qoff)*1000/fs
        q['duration_within_v1_range'] = 40 <= (qoff-qon)*1000/fs <= 240
        outside = smooth[qlo:qhi+1] > b['q_threshold']
        guard = ms_samples(cfg.quiet_qrs_ms, fs)
        a, z = max(0, qon-qlo-guard), min(len(outside), qoff-qlo+guard+1)
        outside[a:z] = False
        components = []
        for lo, hi in _runs(outside):
            peak_fraction = float(smooth[qlo+lo:qlo+hi].max()/max(smooth[qp], 1e-9))
            if (hi-lo)*1000/fs >= SCREENS['qrs_secondary_component_min_ms']:
                components.append(dict(on_ms=float(time[qlo+lo]), off_ms=float(time[qlo+hi-1]),
                    duration_ms=(hi-lo)*1000/fs, peak_fraction=peak_fraction))
        q['outside_components_json'] = json.dumps(components)
        q['secondary_activity_count'] = sum(c['peak_fraction'] >= SCREENS['qrs_secondary_component_peak_fraction'] for c in components)
        q['secondary_max_peak_fraction'] = max([c['peak_fraction'] for c in components], default=0)
        # This may be P/ST/other activity: it is not a missed-QRS annotation.
        if q['secondary_activity_count']:
            qsignals.append('secondary_velocity_activity_in_qrs_search_window')
        if min(q['left_search_margin_ms'], q['right_search_margin_ms']) < 8:
            qsignals.append('qrs_near_v1_search_edge')
        if not q['duration_within_v1_range']:
            qsignals.append('qrs_duration_outside_v1_range')
        q['loop_endpoint_distance_mV'] = float(np.linalg.norm(xyz[qoff]-xyz[qon]))
        q['loop_endpoint_distance_fraction'] = _fraction(q['loop_endpoint_distance_mV'], np.linalg.norm(xyz[qon:qoff+1], axis=1).max())
    peaks, _ = find_peaks(smooth[qlo:qhi+1], prominence=.15*max(float(smooth[qp]), 1e-9), distance=ms_samples(10, fs))
    q['velocity_peak_count_in_search'] = len(peaks)
    q['screening_signals'] = '; '.join(qsignals)
    t.update(available=bool(row['t_boundary_valid']), reason='' if row['t_boundary_valid'] else 'missing_t_boundaries',
        t_on_ms=row['t_on_ms'], t_peak_ms=row['t_peak_ms'], t_off_ms=row['t_off_ms'],
        t_duration_ms=row['t_duration_ms'], t_method=b['t_method'],
        st_level=b['st_level'], tp_level=b['t_end_baseline'],
        t_on_method_gap_ms=row['t_on_method_gap_ms'], t_off_method_gap_ms=row['t_off_method_gap_ms'],
        t_tangent_on_ms=row.get('t_tangent_on_ms'), t_tangent_off_ms=row.get('t_tangent_off_ms'),
        t_threshold_on_ms=row.get('t_threshold_on_ms'), t_threshold_off_ms=row.get('t_threshold_off_ms'))
    tsignals = []
    peak, ton, toff = b['t_peak'], b['t_on'], b['t_off']
    mbase = float(np.median(mag[pq0:pq1]))
    mnoise = max(robust_sigma(mag[pq0:pq1]), 1e-6)
    t.update(pq_magnitude_baseline=mbase, pq_magnitude_noise=mnoise)
    if peak is not None:
        amp = float(mag[peak]-mbase)
        delta = float(b['st_level']-mbase)
        threshold = max(5*mnoise, .15*amp)
        t.update(t_peak_magnitude_mV=float(mag[peak]), t_peak_amplitude_above_pq_mV=amp,
            st_minus_pq_mV=delta, st_flag_threshold_mV=threshold,
            st_trigger_margin_mV=delta-threshold, st_delta_peak_fraction=_fraction(delta, amp),
            st_flag_recomputed=bool(delta > threshold),
            st_flag_present='st_level_requires_review' in split_flags(row['qc_flags']))
        lo, hi = b['t_search']
        t.update(t_search_start_ms=float(time[lo]), t_search_end_ms=float(time[hi]))
        prominence = max(4*mnoise, .015*float(np.max(mag[qon:qoff+1])))
        ps, props = find_peaks(mag[lo:hi+1], prominence=prominence, distance=ms_samples(40, fs))
        ps += lo
        valid = (ps >= qoff+ms_samples(50, fs)) & (ps <= hi-ms_samples(30, fs))
        selected_peak = np.flatnonzero(ps == peak)
        if len(selected_peak):
            significant = (np.abs(ps-peak) <= ms_samples(100, fs)) & (props['prominences'] >= .35*props['prominences'][selected_peak[0]])
            t.update(n_candidate_peaks=len(ps), n_valid_peaks=int(valid.sum()), n_significant_lobes=int(significant.sum()),
                significant_lobes_outside_valid_window=int((significant & ~valid).sum()),
                candidate_peaks_json=json.dumps([{'time_ms': float(time[p]), 'peak_valid': bool(v), 'significant': bool(s)} for p,v,s in zip(ps,valid,significant)]))
            if (significant & ~valid).any():
                tsignals.append('significant_lobe_outside_peak_valid_window')
        rise, fall = b['t_rise_index'], b['t_fall_index']
        slope = arrays.get('t_slope')
        for name, index in [('rise', rise), ('fall', fall)]:
            if index is not None:
                t[name+'_contact_ms'] = float(time[index])
                t[name+'_magnitude_mV'] = float(mag[index])
                t[name+'_slope_mV_s'] = float(slope[index])
        if rise is not None and slope is not None and slope[rise] > 1e-9:
            # Algebraic counterfactual only: do not replace actual V1 boundaries.
            pq_intercept = float(time[rise] + (mbase-mag[rise])/slope[rise]*1000)
            st_intercept = float(time[rise] + (b['st_level']-mag[rise])/slope[rise]*1000)
            t.update(counterfactual_pq_intercept_ms=pq_intercept,
                unrounded_v1_st_intercept_ms=st_intercept,
                st_to_pq_intercept_shift_ms=st_intercept-pq_intercept)
        if ton is not None:
            fraction = _fraction(mag[ton]-b['st_level'], mag[peak]-b['st_level'])
            t['t_on_actual_magnitude_peak_fraction'] = fraction
            if fraction > SCREENS['t_on_peak_fraction']:
                tsignals.append('t_on_magnitude_above_half_of_peak_relative_to_st')
        if toff is not None:
            fraction = _fraction(mag[toff]-b['t_end_baseline'], mag[peak]-b['t_end_baseline'])
            t['t_off_actual_magnitude_peak_fraction'] = fraction
            if fraction > SCREENS['t_off_peak_fraction']:
                tsignals.append('t_off_magnitude_above_half_of_peak_relative_to_tp')
            post = toff+ms_samples(SCREENS['t_tail_after_ms'], fs)
            if post <= hi:
                tail = _fraction(mag[post]-b['t_end_baseline'], mag[peak]-b['t_end_baseline'])
                t['t_tail_20ms_residual_peak_fraction'] = tail
                if tail > SCREENS['t_tail_peak_fraction']:
                    tsignals.append('residual_magnitude_20ms_after_t_off')
            t['margin_to_typical_next_anchor_ms'] = row['median_rr_ms']-float(time[toff])
        for side in ['on', 'off']:
            gap = row.get(f't_{side}_method_gap_ms', np.nan)
            if np.isfinite(gap) and gap > SCREENS['method_gap_ms']:
                tsignals.append(f't_{side}_methods_gap_gt60ms')
    if not bool(row['t_boundary_valid']):
        tsignals.append('t_boundary_unresolved')
    elif not bool(row['t_algorithm_range_valid']):
        tsignals.append('t_duration_outside_v1_range')
    t['screening_signals'] = '; '.join(tsignals)
    return q, t


def boundary_audits(frame, artifact_root, output_dir, cfg=None):
    qrows, trows = [], []
    for row in tqdm(frame.to_dict('records'), desc='Boundary audit', unit='record'):
        q, t = audit_record(row, artifact_root, cfg)
        qrows.append(q); trows.append(t)
    qrs, twave = pd.DataFrame(qrows), pd.DataFrame(trows)
    output_dir = Path(output_dir)
    qrs.to_csv(output_dir/'qrs_boundary_audit.csv', index=False)
    twave.to_csv(output_dir/'t_boundary_audit.csv', index=False)
    joined = frame.merge(qrs[['record_id', 'screening_signals']], on='record_id', validate='one_to_one').rename(columns={'screening_signals': 'qrs_screening_signals'})
    joined = joined.merge(twave[['record_id', 'screening_signals']], on='record_id', validate='one_to_one').rename(columns={'screening_signals': 't_screening_signals'})
    joined['suspicion_indicators'] = joined[['qrs_screening_signals', 't_screening_signals']].fillna('').agg(lambda s: '; '.join(v for v in s if v), axis=1)
    joined['screen_positive'] = joined.suspicion_indicators.ne('')
    joined['screening_route'] = np.select([
        ~joined.processing_success,
        joined.n_flags.gt(0) & joined.screen_positive,
        joined.n_flags.eq(0) & joined.screen_positive,
        joined.n_flags.gt(0),
    ], ['technical_failure', 'flagged_screen_candidate_B', 'unflagged_screen_candidate_C', 'flagged_unverified'], default='unflagged_unverified')
    joined['audit_group'] = np.where(joined.processing_success, 'D_unverified', 'technical_failure')
    joined['reference_verdict'] = 'not_assessed'
    joined.to_csv(output_dir/'audit_record_index.csv', index=False)
    joined[joined.screen_positive | ~joined.processing_success].to_csv(output_dir/'suspicious_records.csv', index=False)
    # Ranking is triage only. It is not an error probability and does not modify QC.
    joined['review_priority_score'] = ((~joined.processing_success).astype(int)*100
        + (~joined.extraction_success).astype(int)*40 + joined.screen_positive.astype(int)*20
        + joined.n_flags.clip(upper=10))
    joined['development_scope'] = np.where(joined.split.eq('test'), 'descriptive_test_only', 'train_valid_review')
    joined.sort_values(['development_scope', 'review_priority_score', 'record_id'], ascending=[False, False, True]).to_csv(output_dir/'review_priority.csv', index=False)
    return qrs, twave, joined


def failure_analysis(frame, artifact_root, output_dir):
    rows = []
    for _, row in frame[~frame.extraction_success].iterrows():
        art = json.loads((Path(artifact_root)/(row.record_id.replace('/', '_')+'.json')).read_text())
        beat_rows = art.get('beat_qc', [])
        reasons = pd.Series([b['reason'] for b in beat_rows], dtype=str).value_counts().to_dict()
        steps = dict(input_quality='completed', qrs_detection='anchors_returned',
            beat_selection='completed' if row.processing_success else 'fewer_than_three_retained',
            representative_beat='available' if row.processing_success else 'not_created',
            qrs_delineation='boundaries_available' if row.qrs_boundary_valid else 'not_run_no_median',
            t_delineation='boundaries_available' if row.t_boundary_valid else ('unresolved' if row.processing_success else 'not_run_no_median'),
            feature_extraction='partial_QRS_only' if row.feature_completeness == 7 else 'not_run_no_median')
        rows.append({**{k: row[k] for k in COMMON}, **steps,
            'detected_anchors': row.n_qrs_detected, 'retained_beats': row.n_beats_used,
            'excluded_beats': row.n_beats_excluded, 'beat_reasons_json': json.dumps(reasons),
            'input_qc_flags': row.input_qc_flags, 'failure_stage': row.failure_stage,
            'error': row.error, 't_qc_flags': row.t_qc_flags,
            'independent_error_confirmation': False})
    result = pd.DataFrame(rows)
    result.to_csv(Path(output_dir)/'failed_records_analysis.csv', index=False)
    return result


def create_review_template(joined, output_dir, per_label=30):
    output_dir = Path(output_dir)
    path = output_dir/'manual_review_template.csv'
    if path.exists():
        return pd.read_csv(path)
    development = joined[joined.split.isin(['train', 'valid'])].copy()
    selected_ids = set()
    # Include every incomplete case on train/valid, then stratify QC and patient.
    selected_ids.update(development.loc[~development.extraction_success, 'record_id'])
    for label in ['HC', 'MI']:
        group = development[development.label_name.eq(label)].sample(frac=1, random_state=42)
        for has_flags, minimum in [(False, 10), (True, 20)]:
            sub = group[group.n_flags.gt(0).eq(has_flags)]
            ids = [rid for rid in selected_ids if rid in set(sub.record_id)]
            used = set(group[group.record_id.isin(selected_ids)].patient_id)
            for _, row in sub.iterrows():
                if len(ids) >= min(minimum, len(sub)):
                    break
                if row.record_id in selected_ids or row.patient_id in used:
                    continue
                selected_ids.add(row.record_id); used.add(row.patient_id); ids.append(row.record_id)
        while int(group.record_id.isin(selected_ids).sum()) < min(per_label, len(group)):
            unselected = group[~group.record_id.isin(selected_ids)]
            used = set(group[group.record_id.isin(selected_ids)].patient_id)
            diverse = unselected[~unselected.patient_id.isin(used)]
            selected_ids.add((diverse if len(diverse) else unselected).iloc[0].record_id)
        # Cover rare observed QC flags without using test to select improvements.
        for flag in sorted({f for s in group.qc_flags for f in split_flags(s)}):
            flagged = group[group.qc_flags.map(lambda s: flag in split_flags(s))]
            if not flagged.record_id.isin(selected_ids).any():
                selected_ids.add(flagged.iloc[0].record_id)
        # Ensure unflagged screen-positive cases also receive manual attention.
        candidates = group[group.n_flags.eq(0) & group.screen_positive]
        if len(candidates) and not candidates.record_id.isin(selected_ids).any():
            selected_ids.add(candidates.iloc[0].record_id)
    selected = development[development.record_id.isin(selected_ids)].copy().sort_values(['label_name', 'record_id'])
    result = selected[['record_id', 'patient_id', 'label', 'label_name', 'split', 'qc_flags']].copy()
    for name in ['qrs_on', 'qrs_off', 't_on', 't_off']:
        result[name+'_pred_ms'] = selected[name+'_ms']
        result[name+'_review_ms'] = np.nan
    result['time_reference'] = 'median_relative_ms'
    result['reference_anchor_sample'] = np.nan
    result['reviewer'] = ''
    result['review_status'] = 'unverified'
    result['independent_reference'] = False
    result['review_notes'] = ''
    result['proposed_group_A_B_C_D'] = ''
    result['signal_availability'] = np.where(selected.processing_success, 'median_available', 'no_median')
    result.to_csv(path, index=False)
    return result


def review_references(frame, output_dir, evaluation_dir, acceptance_ms=None, review_path=None):
    """Accept only confirmed independent expert input, never screening labels."""
    from compare_vcg_annotations import compare_boundaries
    output_dir = Path(output_dir)
    review = pd.read_csv(review_path or output_dir/'manual_review_template.csv')
    aliases = {k+'_review_ms': k+'_ms' for k in ['qrs_on', 'qrs_off', 't_on', 't_off']}
    converted = review.rename(columns=aliases)
    comparison, summary, audit = compare_boundaries(frame, converted, Path(evaluation_dir))
    if acceptance_ms is not None and (not np.isfinite(acceptance_ms) or acceptance_ms <= 0):
        raise ValueError('Reference acceptance criterion must be explicitly positive')
    comparison.to_csv(output_dir/'reference_comparison.csv', index=False)
    summary.to_csv(output_dir/'reference_error_summary.csv', index=False)
    audit.to_csv(output_dir/'reference_acceptance_audit.csv', index=False)
    write_json(output_dir/'reference_status.json', dict(
        expert_rows=len(review), accepted_rows=int(audit.accepted.sum()) if len(audit) else 0,
        boundary_references=len(comparison), acceptance_ms=acceptance_ms,
        accuracy_available=bool(len(comparison)),
        qc_false_alarm_rate='not_assessed_without_confirmed_complete_references_and_declared_acceptance_criterion'))
    verdicts = frame[COMMON].copy()
    verdicts['confirmed_four_boundaries'] = False
    verdicts['reference_verdict'] = 'unverified'
    if len(comparison):
        complete_groups = comparison.groupby(['record_id', 'reference_row']).boundary.nunique()
        multiple = set(complete_groups[complete_groups.eq(4)].reset_index().groupby('record_id').size().loc[lambda s: s.gt(1)].index)
        for (rid, refrow), group in comparison.groupby(['record_id', 'reference_row']):
            if set(group.boundary) != {'qrs_on', 'qrs_off', 't_on', 't_off'}:
                continue
            if not group.comparison_scope.eq('median_estimate_vs_expert_same_median').all():
                continue
            mask = verdicts.record_id.eq(rid)
            verdicts.loc[mask, 'confirmed_four_boundaries'] = True
            if rid in multiple:
                verdicts.loc[mask, 'reference_verdict'] = 'multiple_references_require_adjudication'
                continue
            if acceptance_ms is not None:
                ok = group.matched.all() and group.absolute_error_ms.le(acceptance_ms).all()
                verdicts.loc[mask, 'reference_verdict'] = 'within_declared_criterion' if ok else 'outside_declared_criterion'
            else:
                verdicts.loc[mask, 'reference_verdict'] = 'confirmed_no_acceptance_criterion'
    verdicts.to_csv(output_dir/'reference_record_verdicts.csv', index=False)
    reference_groups = []
    for label in ['HC', 'MI']:
        group = verdicts[verdicts.label_name.eq(label)]
        for status in ['auto_candidate', 'needs_review', 'failed']:
            subset = group[group.qc_status.eq(status)]
            reference_groups.append(dict(label_name=label, qc_status=status, n_records=len(subset),
                qc_reference_verdict='unverified' if not subset.confirmed_four_boundaries.any() else 'see_reference_record_verdicts',
                annotation_confirmed=int(subset.confirmed_four_boundaries.sum()),
                reference_within_criterion=int(subset.reference_verdict.eq('within_declared_criterion').sum()),
                reference_outside_criterion=int(subset.reference_verdict.eq('outside_declared_criterion').sum())))
    pd.DataFrame(reference_groups).to_csv(output_dir/'qc_reference_status.csv', index=False)
    # Explicit denominators; unavailable without independent complete labels and criterion.
    rate_rows = []
    for label in ['HC', 'MI']:
        group = verdicts[verdicts.label_name.eq(label)]
        passing = group.reference_verdict.eq('within_declared_criterion')
        error = group.reference_verdict.eq('outside_declared_criterion')
        warning = group.qc_status.eq('needs_review')
        n_good, n_error = int(passing.sum()), int(error.sum())
        rate_rows.append(dict(label_name=label, n_verified_good=n_good, n_verified_error=n_error,
            false_alarms=int((warning & passing).sum()), true_warnings=int((warning & error).sum()),
            missed_errors=int((group.qc_status.eq('auto_candidate') & error).sum()),
            false_alarm_rate=float((warning & passing).sum()/n_good) if n_good else np.nan,
            missed_error_rate=float((group.qc_status.eq('auto_candidate') & error).sum()/n_error) if n_error else np.nan,
            acceptance_ms=acceptance_ms))
    pd.DataFrame(rate_rows).to_csv(output_dir/'qc_reference_rates.csv', index=False)
    return comparison, summary, audit


def technical_plots(frame, qc, qrs, twave, output_dir):
    output_dir = Path(output_dir)
    target = output_dir/'plots/summary'
    target.mkdir(exist_ok=True)
    paths = []
    def save(fig, name):
        fig.savefig(target/name, dpi=120, bbox_inches='tight'); plt.close(fig)
        paths.append((target/name).relative_to(output_dir).as_posix())
    frequency = qc['frequency'].sort_values('total')
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(frequency.qc_flag, frequency.total, color='#287c8e')
    ax.set(xlabel='Recordings with flag (overlap allowed)', title='Observed QC flags — all 448 recordings')
    save(fig, 'qc_flag_counts.png')
    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(frequency))
    ax.barh(y-.18, frequency.HC_pct, .36, label='HC / 80', color='#2486aa')
    ax.barh(y+.18, frequency.MI_pct, .36, label='MI / 368', color='#d67037')
    ax.set(yticks=y, yticklabels=frequency.qc_flag, xlabel='% of each diagnostic label', title='QC frequencies by HC / MI')
    ax.legend(); save(fig, 'qc_flag_rates_by_label.png')
    top = qc['frequency'].head(18).qc_flag.tolist()
    co = qc['cooccurrence'].loc[top, top]
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(co, cmap='YlOrRd')
    ax.set(xticks=np.arange(len(top)), xticklabels=top, yticks=np.arange(len(top)), yticklabels=top,
        title='Flag co-occurrence counts (diagonal = flag frequency)')
    plt.setp(ax.get_xticklabels(), rotation=75, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.colorbar(im, ax=ax, label='Recordings'); save(fig, 'qc_cooccurrence.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color in [('HC', '#2486aa'), ('MI', '#d67037')]:
        group = qc['by_nflags'][qc['by_nflags'].label_name.eq(label)]
        axes[0].plot(group.n_flags, group.n_records, marker='o', label=label, color=color)
        axes[1].plot(group.n_flags, group.full12_pct, marker='o', label=label, color=color)
    axes[0].set(xlabel='Number of flags', ylabel='Recordings', title='QC flags per recording')
    axes[1].set(xlabel='Number of flags', ylabel='% full 12 features', title='Completeness is independent of QC verdict')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    save(fig, 'flags_and_feature_completeness.png')
    if twave is None:
        return paths
    st = twave[twave.st_minus_pq_mV.notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, color in [('HC', '#2486aa'), ('MI', '#d67037')]:
        group = st[st.label_name.eq(label)]
        axes[0].scatter(group.st_minus_pq_mV, group.st_flag_threshold_mV, s=13, alpha=.55, label=label, color=color)
        axes[1].scatter(group.st_delta_peak_fraction, group.t_on_actual_magnitude_peak_fraction, s=13, alpha=.55, label=label, color=color)
        axes[2].hist(group.st_to_pq_intercept_shift_ms.dropna(), bins=25, alpha=.45, label=label, color=color)
    limits = [min(0, float(st.st_minus_pq_mV.min())), float(max(st.st_minus_pq_mV.max(), st.st_flag_threshold_mV.max()))]
    axes[0].plot(limits, limits, 'k--', lw=.8)
    axes[0].set(xlabel='ST magnitude minus PQ (mV)', ylabel='V1 threshold (mV)', title='Flag trigger: x > y')
    axes[1].set(xlabel='(ST − PQ) / T amplitude', ylabel='Actual magnitude at T-on / peak above ST', title='Descriptive shape quantities, not error')
    axes[2].set(xlabel='ST vs PQ tangent intercept shift (ms)', ylabel='Recordings', title='Algebraic reference sensitivity only')
    for ax in axes: ax.legend(); ax.grid(alpha=.15)
    save(fig, 'st_rule_and_t_on.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color in [('HC', '#2486aa'), ('MI', '#d67037')]:
        group = twave[twave.label_name.eq(label)]
        for ax, field in zip(axes, ['t_on_method_gap_ms', 't_off_method_gap_ms']):
            ax.hist(group[field].dropna(), bins=25, alpha=.45, label=label, color=color)
            ax.set(xlabel='Absolute method gap (ms)', ylabel='Recordings', title=field)
    for ax in axes: ax.legend(); ax.grid(alpha=.15)
    save(fig, 't_method_disagreement.png')
    return paths


def sensitivity_plots(sensitivity, output_dir):
    long = sensitivity['long']
    output_dir = Path(output_dir)
    # Plot only finite own-loop perturbations on train/valid for development summaries.
    use = long[long['split'].isin(['train', 'valid'])].copy()
    relative = 'relative_delta'
    if relative not in use:
        return []
    own = use.apply(lambda r: str(r['feature']).startswith('QRS') == str(r['boundary']).startswith('qrs'), axis=1)
    use = use[own]
    grouped = use.groupby(['feature', 'boundary', 'shift_ms'])[relative].median().unstack(['boundary', 'shift_ms'])
    grouped = grouped.reindex(FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(grouped.to_numpy()*100, cmap='YlOrRd', aspect='auto')
    ax.set(yticks=np.arange(len(grouped)), yticklabels=grouped.index,
        xticks=np.arange(len(grouped.columns)), xticklabels=[f'{a}\n{b:+g} ms' for a,b in grouped.columns],
        title='Median relative feature change (%) — train/valid only; no reference accuracy')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.colorbar(im, ax=ax, label='% absolute relative change')
    path = output_dir/'plots/summary/feature_sensitivity_heatmap.png'
    fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
    return [path.relative_to(output_dir).as_posix()]


def markdown_table(frame):
    def cell(value): return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(cell, frame.columns))+' |',
        '| '+' | '.join('---' for _ in frame.columns)+' |']+
        ['| '+' | '.join(map(cell, row))+' |' for row in frame.itertuples(index=False, name=None)])


def prepare_blinded_review(output_dir, artifact_root, data_root, cfg=None):
    """Unannotated signals + blank expert fields, without disease/QC/prediction."""
    from vcg_pipeline import load_xyz
    cfg = cfg or Config()
    output_dir = Path(output_dir)
    sample = pd.read_csv(output_dir/'manual_review_template.csv')
    template = output_dir/'manual_review_blinded.csv'
    if template.exists():
        return pd.read_csv(template)
    rows = []
    for _, row in sample.iterrows():
        safe = row.record_id.replace('/', '_')
        art = json.loads((Path(artifact_root)/(safe+'.json')).read_text())
        with np.load(Path(artifact_root)/(safe+'.npz')) as saved:
            if 'median_xyz' in saved:
                xyz, tt = saved['median_xyz'], saved['time_ms']
                time_reference = 'median_relative_ms'
                availability = 'median_available'
                title = 'Median XYZ; no predicted boundaries shown'
                note = ''
            else:
                raw = load_xyz(Path(data_root)/row.record_id, cfg.seconds)
                xyz = raw['xyz'][:int(8*raw['fs'])]
                tt = np.arange(len(xyz))*1000/raw['fs']
                time_reference = 'raw_recording_ms'
                availability = 'no_median_raw_first8s_only'
                title = 'Raw XYZ first 8 s; median was not created'
                note = 'Không có median để chú giải biên; xem detector/beat selection riêng trên recording gốc.'
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout='constrained')
        for j, ax in enumerate(axes[:3]):
            ax.plot(tt, xyz[:,j], color=['#2563eb', '#059669', '#d97706'][j], lw=.8)
            ax.set_ylabel('XYZ'[j]+' (mV)')
        axes[3].plot(tt, np.linalg.norm(xyz, axis=1), color='#0891b2', lw=.8)
        axes[3].set_ylabel('Magnitude (mV)')
        axes[-1].set_xlabel(time_reference)
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle(row.record_id+' | '+title, fontsize=11)
        path = output_dir/'blinded_review'/(safe+'.png')
        fig.savefig(path, dpi=110); plt.close(fig)
        ref = dict(record_id=row.record_id, patient_id=row.patient_id,
            time_reference=time_reference, reference_anchor_sample=np.nan,
            reviewer='', review_status='unverified', independent_reference=False,
            review_notes=note, signal_availability=availability,
            signal_plot=path.relative_to(output_dir).as_posix())
        for name in ['qrs_on', 'qrs_off', 't_on', 't_off']: ref[name+'_review_ms'] = np.nan
        rows.append(ref)
    blinded = pd.DataFrame(rows)
    blinded.to_csv(template, index=False)
    return blinded


def representative_examples(joined, output_dir):
    """Deterministic illustrative selection; development proposals use train/valid."""
    output_dir = Path(output_dir)
    development = joined[joined.split.isin(['train', 'valid'])].copy()
    groups = []
    def add(title, selected, count=2):
        for _, r in selected.head(count).iterrows():
            groups.append(dict(group=title, record_id=r.record_id, patient_id=r.patient_id,
                label_name=r.label_name, split=r.split, qc_status=r.qc_status,
                qc_flags=r.qc_flags, feature_completeness=r.feature_completeness,
                suspicion_indicators=r.suspicion_indicators,
                independent_confirmation='unverified'))
    for label in ['HC', 'MI']:
        group = development[development.label_name.eq(label)]
        add(label+'_no_flags', group[group.n_flags.eq(0)])
        add(label+'_many_flags', group.sort_values(['n_flags', 'record_id'], ascending=[False, True]))
        add(label+'_ST_flag', group[group.qc_flags.map(lambda s:'st_level_requires_review' in split_flags(s))])
    add('T_on_screen_candidate', development[development.t_screening_signals.fillna('').str.contains('t_on_')])
    add('T_off_methods_disagree', development[development.qc_flags.str.contains('t_end_methods_disagree')])
    add('QRS_complex_or_long_for_visual_review_only', development.sort_values('qrs_duration_ms', ascending=False))
    add('full12_with_flags', development[development.extraction_success & development.n_flags.gt(0)])
    add('no_flags_with_secondary_screen', development[development.n_flags.eq(0) & development.screen_positive], count=4)
    add('incomplete_features', development[~development.extraction_success], count=len(development))
    result = pd.DataFrame(groups)
    result.to_csv(output_dir/'representative_examples.csv', index=False)
    return result


def write_reference_source_audit(output_dir):
    audit = {
        'local_sources_checked': ['PTB waveform tree', 'Notebook06 outputs', 'outputs/vcg_full_evaluation/annotations'],
        'matching_confirmed_four_boundary_annotations_found': False,
        'local_review_status': 'Existing and Notebook07 templates pending/unverified; no fabricated references',
        'sources': [
            {'url': 'https://physionet.org/content/ptbdb/1.0.0/',
             'finding': 'Original PTB signal/header release; no matching four-boundary reference file found locally.'},
            {'url': 'https://physionet.org/content/challenge-2006/1.0.0/',
             'finding': 'Same PTB IDs, but QT is lead-II observer-selected representative beat; not four Frank median boundaries.'},
            {'url': 'https://physionet.org/content/ptb-xl-plus/1.0.0/',
             'finding': 'Different PTB-XL ECG IDs; fiducials are ECGDeli automated outputs, not matching independent original-PTB labels.'},
        ],
        'independent_boundary_accuracy': 'not_assessed',
    }
    write_json(Path(output_dir)/'reference_sources.json', audit)
    return audit


def write_audit_report(output_dir):
    """Eleven answers computed from actual audit results, with evidence levels."""
    output_dir = Path(output_dir)
    f = pd.read_csv(output_dir/'audit_record_index.csv').fillna({'qc_flags':'', 'suspicion_indicators':''})
    q = pd.read_csv(output_dir/'qrs_boundary_audit.csv')
    t = pd.read_csv(output_dir/'t_boundary_audit.csv')
    freq = pd.read_csv(output_dir/'qc_flag_frequency.csv')
    by_label = pd.read_csv(output_dir/'qc_by_label.csv')
    totals = json.loads((output_dir/'qc_totals.json').read_text())
    failures = pd.read_csv(output_dir/'failed_records_analysis.csv')
    rank = pd.read_csv(output_dir/'feature_sensitivity_full12_rank.csv')
    primary = rank[(rank.scope == 'train_valid') & (rank.grouping == 'all')].copy()
    primary['median_relative_pct'] = primary.relative_delta_median*100
    primary['p95_relative_pct'] = primary.relative_delta_p95*100
    ranking = primary[['feature','unit','absolute_delta_median','absolute_delta_p95','median_relative_pct','p95_relative_pct','n_valid','n_eligible_but_invalid']].round(5)
    scenarios = pd.read_csv(output_dir/'sensitivity_scenarios.csv')
    reference = json.loads((output_dir/'reference_status.json').read_text())
    sources = write_reference_source_audit(output_dir)
    review = pd.read_csv(output_dir/'manual_review_template.csv')
    stonly = int(f.qc_flags.eq('st_level_requires_review').sum())
    stflag = f.qc_flags.map(lambda s:'st_level_requires_review' in split_flags(s))
    noflag = f.n_flags.eq(0)
    noflag_screen = int((noflag & f.screen_positive).sum())
    st_table = []
    for label in ['HC', 'MI']:
        group = t[t.label_name.eq(label)]
        st_table.append(dict(label=label, recordings=len(group),
            st_flag=int(group.st_flag_present.fillna(False).sum()),
            st_minus_pq_median_mV=group.st_minus_pq_mV.median(),
            st_delta_peak_fraction_median=group.st_delta_peak_fraction.median(),
            counterfactual_intercept_shift_median_ms=group.st_to_pq_intercept_shift_ms.median(),
            t_on_threshold_available=int(group.t_threshold_on_ms.notna().sum())))
    st_table = pd.DataFrame(st_table).round(5)
    missing_t_after_median = int((f.processing_success & ~f.t_boundary_valid).sum())
    bad_duration = int((f.t_boundary_valid & ~f.t_algorithm_range_valid).sum())
    actual_lobe_caveat = int(t.significant_lobes_outside_valid_window.fillna(0).gt(0).sum())
    q_secondary = int(q.secondary_activity_count.fillna(0).gt(0).sum())
    t_on_gap = int(t.t_on_method_gap_ms.gt(SCREENS['method_gap_ms']).sum())
    t_off_gap = int(t.t_off_method_gap_ms.gt(SCREENS['method_gap_ms']).sum())
    st_threshold_missing = int((t.st_flag_present.fillna(False) & t.t_threshold_on_ms.isna()).sum())
    source_findings = json.loads((output_dir/'source_audit_findings.json').read_text())
    findings_table = pd.DataFrame(source_findings)
    main_label = by_label[by_label.split.eq('all')].copy()
    sample_counts = review.groupby('label_name').agg(n_records=('record_id','size'),n_patients=('patient_id','nunique')).reset_index()
    # All numerical descriptions are generated after complete computation.
    answers = [
        ('Câu 1 — Vì sao needs_review cao?', 'QC/source đã kiểm toán; chưa xác minh biên độc lập',
         f"{totals['needs_review']}/{len(f)} ({100*totals['needs_review']/len(f):.2f}%) có ít nhất một cờ mà logic V1 đưa vào status. V1 dùng any-flag, không có mức độ nặng/nhẹ. Có {totals['full12_and_review']} ca vừa đủ 12 features vừa needs_review. Đủ feature và qua QC là hai tiêu chí khác nhau."),
        ('Câu 2 — Flags nào đóng góp nhiều nhất?', 'Đếm trực tiếp toàn cohort',
         '; '.join(f'{r.qc_flag}: {r.total}' for _, r in freq.head(4).iterrows())+f'. Có {stonly} ca chỉ mang cờ ST. Không cộng các tần suất vì một record có nhiều cờ.'),
        ('Câu 3 — QC có quá nhạy không?', 'Chưa đủ annotation để kết luận',
         f'Cờ ST là ưu tiên kiểm chứng vì xuất hiện ở {int(stflag.sum())} ca và là cờ duy nhất ở {stonly} ca. Tần suất cao không xác định false alarm. Chưa có ca nào được chuyên gia xác nhận đủ bốn biên; false-alarm rate và missed-error rate để NaN.'),
        ('Câu 4 — Có bằng chứng QRS sai không?', 'Bằng chứng kỹ thuật, chưa xác minh sinh lý',
         f'{int(f.qrs_boundary_valid.sum())}/{len(f)} có cặp QRS; {int((~f.processing_success).sum())} ca không tạo được median dù detector trả anchors. {q_secondary} ca vượt screen hoạt động vận tốc phụ đã khai báo. QRS rộng/nhiều đỉnh được xem hình, không tự gọi lỗi; không có sai số QRS-on/off tham chiếu.'),
        ('Câu 5 — T sai ở đâu?', 'Ca chưa giải được và candidate cần review; chưa có lỗi biên xác nhận',
         f'{missing_t_after_median} ca có median nhưng không đủ biên T; {bad_duration} ca có biên T với duration ngoài giới hạn V1; {int(freq.loc[freq.qc_flag.eq("t_end_methods_disagree"),"total"].sum())} ca bất đồng T-off >100 ms. Screens >{SCREENS["method_gap_ms"]:g} ms có {t_on_gap} ca T-on và {t_off_gap} ca T-off. Chưa thể xác định tỷ lệ T-on/T-peak/T-off thực sự sai.'),
        ('Câu 6 — ST phản ánh bệnh lý hay lỗi?', 'Giả thuyết cần tham chiếu',
         f'V1 kiểm tra ST_mag−PQ_mag > max(5×noise_PQ,0.15×amp_T), một phía trên magnitude không dấu. Cờ không chẩn đoán ECG ST elevation/depression. {st_threshold_missing}/{int(stflag.sum())} ca có cờ ST không giải được threshold-on; vì threshold-on dựa PQ còn mức ST cao hơn, đây là thiếu một kiểm tra chéo hữu dụng, không phải xác nhận lỗi tangent-on. Không xác định được nguyên nhân baseline, hình thái hay biên nếu thiếu chú giải độc lập.'),
        ('Câu 7 — QC có bỏ sót lỗi không?', 'Chưa xác định missed-error; có ứng viên cần kiểm tra',
         f'Đã kiểm tra tất cả {int(noflag.sum())} ca không có cờ. {noflag_screen} ca vượt ít nhất một screen phụ (chủ yếu bất đồng phương pháp). Đây là candidate nhóm C, không phải {noflag_screen} lỗi xác nhận. {len(f)-totals["needs_review"]} ca ngoài needs_review gồm {int(noflag.sum())} auto_candidate và {totals["processing_failed"]} failed, không phải 70 ca đều không cảnh báo.'),
        ('Câu 8 — Feature nào nhạy với sai lệch biên?', 'Thử nghiệm perturbation kỹ thuật, không đánh giá accuracy',
         'Trên các record đủ 12 features, xếp hạng train/valid theo median |ΔF|/(|F|+1e−9): '+
         '; '.join(f'{r.feature}: {100*r.relative_delta_median:.2f}% (p95={100*r.relative_delta_p95:.2f}%)' for _,r in primary.head(5).iterrows())+
         '. Chỉ gộp các biên thuộc chính loop của feature, giữ full-median SG speed rồi cắt. Những MaxVec/MaxVel không đổi trong khoảng dịch này không được suy rộng là luôn bất biến.'),
        ('Câu 9 — HC/MI khác nhau thế nào?', 'Mô tả theo record, không suy luận các record độc lập',
         '; '.join(f'{r.label_name}: review {int(r.needs_review)}/{int(r.n_records)} ({r.needs_review_pct:.2f}%), đủ 12={int(r.full12)}' for _,r in main_label.iterrows())+
         '. Cờ ST và biến thiên hình thái có tần suất khác nhau; chưa chứng minh khác biệt là lỗi phân đoạn do bệnh. Không dùng nhãn để thay threshold.'),
        ('Câu 10 — Đủ tin cậy để ML chưa?', 'Đánh giá mức bằng chứng hiện có',
         f'{totals["full12"]} record có thể dùng làm ứng viên cho ML thăm dò, nhưng độ chính xác biên chưa được xác nhận. Features có độ nhạy khác nhau; missing/QC phụ thuộc nhóm bệnh có thể làm lệch chọn mẫu. Chưa đủ căn cứ gọi bộ 12 features đã được kiểm chứng đáng tin cậy. Điểm phân loại cao trong tương lai cũng không xác nhận các biên đúng.'),
        ('Câu 11 — Ưu tiên V2 nào?', 'Đề xuất dựa train/valid và code, chưa thay V1',
         f'Ưu tiên chú giải độc lập và kiểm chứng quy tắc ST/PQ, định nghĩa T-on và các trường hợp T-peak chọn lobe nhỏ muộn. Tách ý nghĩa cảnh báo hình thái/dữ liệu khỏi lỗi biên đã xác nhận sau review; chưa xóa flags hay đổi ngưỡng. Kiểm tra quy tắc loại beat trên các ca không đủ median. Caveat significant-lobe không áp lại valid mask xuất hiện thực tế ở {actual_lobe_caveat} ca; chưa phải lỗi cohort đã chứng minh. Mọi chỉnh sửa chỉ chọn trên train/valid, cần test độc lập mới nếu quyết định thiết kế dựa quan sát test.'),
    ]
    write_json(output_dir/'audit_conclusions.json', [{'question': a, 'evidence_level': b, 'answer': c} for a,b,c in answers])
    lines = [
        '# Notebook 07 — Kiểm toán QC và phân đoạn QRS/T của V1', '',
        'Báo cáo được tạo từ kết quả tính toán trên toàn bộ 448 recordings. Không đổi V1, features Notebook 06, nhãn hay split. Không huấn luyện classifier.', '',
        '## Tính toàn vẹn và phạm vi', '',
        f"{len(f)} recordings, {f.patient_id.nunique()} patients; {totals['no_flags']} không cờ, {totals['needs_review']} needs_review, {totals['processing_failed']} failed. Đã kiểm tra cả nhóm có và không có QC. Các thời điểm là median_relative_ms; QRS-on âm không phải lỗi.", '',
        markdown_table(main_label.round(3)), '',
        '## Cơ chế QC và tần suất', '',
        'Bảng [qc_rules_inventory.csv](qc_rules_inventory.csv) truy từng điều kiện và ngưỡng về đúng dòng source. Phân biệt flags nhập trực tiếp vào status, input-quality diagnostics, lý do loại beat và cờ batch/evaluation. Các ngưỡng nghiên cứu chưa được hiệu chuẩn độc lập.', '',
        markdown_table(freq[['qc_flag','HC_count','MI_count','HC_pct','MI_pct','total','only_flag_count']].round(3)), '',
        'Input flags không được nhập thẳng vào status V1; mask của chúng có thể ảnh hưởng việc giữ/loại beat. eligible_for_features dựa cờ QC, trong khi extract_features vẫn chạy và giữ candidate khi có cờ.', '',
        '## Quy tắc ST và kiểm tra chéo T-on', '',
        markdown_table(st_table), '',
        'Phép thay mức ST bằng PQ trong phương trình tiếp tuyến chỉ đo ảnh hưởng đại số của reference với điểm đặt và độ dốc giữ nguyên. Không dùng giao điểm thay thế làm biên mới, không cập nhật features, không coi đây là phương pháp độc lập. Tương quan giữa các lượng cùng công thức không chứng minh ST gây lỗi biên.', '',
        'Cần đọc số lượng method gaps cùng mẫu số: threshold-on chỉ có ở một phần nhỏ records. Thiếu threshold-on không được gán gap=0 hoặc xem như hai phương pháp đồng ý.', '',
        '## Bảy ca thiếu features và các giai đoạn', '',
        markdown_table(failures[['record_id','label_name','split','feature_completeness','detected_anchors','retained_beats','beat_reasons_json','t_delineation']]), '',
        'failed là không hoàn tất bước tạo median, không mặc định là detector không phát hiện QRS. Các xung hẹp/RR bị loại là kết quả heuristic; cần quan sát tín hiệu gốc để xác định nguyên nhân.', '',
        '## Độ nhạy biên — các ca đủ 12 features', '',
        'Mỗi lần chỉ dịch một biên ±10/±20 ms, giữ biên khác; không clamp và không sửa V1. Tính lại features từ median XYZ cùng công thức, giữ SG velocity trên toàn median. ε=1e−9 trong đơn vị gốc từng feature. Báo cả |ΔF| và |ΔF|/(|F|+ε), có marker gần zero. Chỉ so trong từng feature khi đơn vị khác nhau.', '',
        f'{len(scenarios)} scenarios trên toàn cohort; {int((~scenarios.valid_scenario).sum())} scenarios không hợp lệ/không khả dụng, vẫn giữ mẫu số. Các thống kê dưới đây chỉ dùng train/valid và records ban đầu đủ 12 features; invalid perturbations không có ΔF, không được điền 0.', '',
        markdown_table(ranking), '',
        'Các con số là độ nhạy với dịch biên giả định, không phải sai số biên thực hay mức quan trọng của feature trong classifier. P95 tính trên các perturbation hợp lệ; nhiều recording cùng patient không phải mẫu độc lập.', '',
        '## Phân loại A/B/C/D và tham chiếu', '',
        f"{int(reference['accepted_rows'])} dòng chú giải được nhận làm chuẩn. Khi chưa có chú giải độc lập, {int(f.processing_success.sum())} ca có median nằm nhóm D_chưa xác minh; {int((~f.processing_success).sum())} ca lỗi kỹ thuật tách riêng. A chưa được xác nhận; B/C trong screening_route chỉ là ứng viên review, không phải nhãn đúng/sai.", '',
        f'Mẫu review khởi đầu có {len(review)} recordings, chỉ thuộc train/valid:', '',
        markdown_table(sample_counts), '',
        '[manual_review_blinded.csv](manual_review_blinded.csv) và tín hiệu không overlay trong [blinded_review](blinded_review/) dành cho lượt đầu, ẩn nhãn và dự đoán. File template đầy đủ dùng đối chiếu sau chú giải. Không có median thì không tạo annotation cho median không tồn tại; raw first8s phục vụ review detector/beat selection riêng.', '',
        'Chỉ nhận review_status=confirmed, có reviewer và independent_reference=true; kiểm tra time base và phạm vi median. Các ngưỡng 10/20/40/60 ms chỉ mô tả phân phối sai số nếu có chuẩn. QC false alarm/missed error cần chú giải đủ bốn biên và tiêu chí sai số khai báo trước. Hai reviewer khác nhau chưa phân xử thì không tự lấy dòng cuối làm chuẩn.', '',
        'Nguồn tham chiếu đã kiểm tra:', '',
    ]
    for source in sources['sources']:
        lines += [f"- {source['finding']} [Nguồn]({source['url']})"]
    lines += ['', '## Trả lời 11 câu hỏi', '']
    for title, level, answer in answers:
        lines += ['### '+title, '', '**Mức bằng chứng:** '+level+'.', '', answer, '']
    lines += ['## Artifacts và tái lập', '',
        '- [Notebook](../../07_VCG_QC_vs_Delineation_Audit.ipynb), [gallery tất cả records](qc_review_gallery.html), [ví dụ đại diện](representative_examples.csv).',
        '- [QRS boundary audit](qrs_boundary_audit.csv), [T boundary audit](t_boundary_audit.csv), [review priority](review_priority.csv).',
        '- [Sensitivity raw](feature_sensitivity.csv), [scenario validity](sensitivity_scenarios.csv), [full12 summary](feature_sensitivity_full12_summary.csv).',
        '- [Reference comparison](reference_comparison.csv): bảng không có dòng khi chưa có chuẩn hợp lệ.',
        '- [Audit config](audit_config.json) lưu hash input/baseline/artifacts và các technical screens đã khai báo.', '',
        'Không dùng test để lựa chọn V2/threshold. Thống kê test chỉ mô tả; nếu quyết định sau khi xem test ảnh hưởng thiết kế, cần test độc lập khác cho kết luận xác nhận.', '']
    (output_dir/'qc_audit_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return pd.DataFrame([{'question':a, 'evidence_level':b, 'answer':c} for a,b,c in answers])


def extra_analysis_tables(output_dir):
    output_dir = Path(output_dir)
    f = pd.read_csv(output_dir/'audit_record_index.csv')
    t = pd.read_csv(output_dir/'t_boundary_audit.csv')
    f[f.n_flags.eq(0)].to_csv(output_dir/'no_qc_records_audit.csv', index=False)
    records = f[['record_id','screen_positive','qrs_screening_signals','t_screening_signals']]
    st = t.merge(records, on='record_id', validate='one_to_one')
    st['st_ratio_band'] = pd.cut(st.st_delta_peak_fraction,
        [-np.inf,0,.08,.15,.30,.50,1,np.inf],
        labels=['<=0','0..0.08','0.08..0.15','0.15..0.30','0.30..0.50','0.50..1','>1']).astype(str)
    strata = st.groupby(['label_name','split','st_ratio_band'], observed=True).agg(
        n_records=('record_id','size'), st_flags=('st_flag_present','sum'),
        screen_positive=('screen_positive','sum'),
        t_on_fraction_median=('t_on_actual_magnitude_peak_fraction','median'),
        t_on_gap_available=('t_on_method_gap_ms','count'),
        t_on_gap_median=('t_on_method_gap_ms','median')).reset_index()
    strata['confirmed_boundary_errors'] = np.nan
    strata['interpretation'] = 'Descriptive strata; no annotation-confirmed error threshold'
    strata.to_csv(output_dir/'st_stratified_review.csv', index=False)
    method_rows = []
    for label, group in t.groupby('label_name'):
        for side in ['on','off']:
            values = group[f't_{side}_method_gap_ms'].dropna()
            method_rows.append(dict(label_name=label, boundary='t_'+side,
                all_label_recordings=len(group), n_both_methods=len(values),
                n_missing_comparison=len(group)-len(values), median_gap_ms=values.median(),
                p95_gap_ms=values.quantile(.95), max_gap_ms=values.max(),
                n_gap_gt60=int(values.gt(60).sum()), n_gap_gt100=int(values.gt(100).sum()),
                reference_accuracy='not_assessed'))
    method = pd.DataFrame(method_rows)
    method.to_csv(output_dir/'method_agreement_summary.csv', index=False)
    screens = []
    for label, group in f.groupby('label_name'):
        for indicator in sorted({flag for s in group.suspicion_indicators.fillna('') for flag in split_flags(s)}):
            selected = group.suspicion_indicators.fillna('').map(lambda s: indicator in split_flags(s))
            screens.append(dict(label_name=label, screen=indicator, n_records=int(selected.sum()),
                n_label=len(group), pct=100*selected.mean(),
                n_unflagged=int((selected & group.n_flags.eq(0)).sum()),
                evidence_level='technical_screen_only'))
    pd.DataFrame(screens).to_csv(output_dir/'technical_screen_summary.csv', index=False)
    sensitivity = pd.read_csv(output_dir/'feature_sensitivity.csv')
    own = sensitivity[sensitivity.own_loop_boundary & sensitivity.valid_feature & sensitivity.original_feature_completeness.eq(12)]
    patient = own.groupby(['patient_id','label_name','split','feature','unit']).agg(
        n_records=('record_id','nunique'), n_valid_scenarios=('relative_delta','count'),
        absolute_delta_median=('absolute_delta','median'), relative_delta_median=('relative_delta','median')).reset_index()
    patient.to_csv(output_dir/'feature_sensitivity_patient_medians.csv', index=False)
    return {'st_strata': strata, 'method_agreement': method, 'screen_summary': pd.DataFrame(screens), 'patient_sensitivity': patient}
