#!/usr/bin/env python3
"""Evaluate frozen notebook-05 V1 on every HC/MI PTB recording.

No detector/delineator parameters are fitted here. Availability and automated
QC are not reference accuracy. Run with the notebook's `ipcv` environment.
Each record has an atomic checkpoint, trace JSON, NPZ, and review plots.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_full_evaluation_matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wfdb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import vcg_pipeline as v1
from vcg_preprocessing import Config

BOUNDARIES = ['qrs_on', 'qrs_off', 't_on', 't_peak', 't_off']
SOURCES = ['vcg_pipeline.py', 'vcg_preprocessing.py',
           'vcg_preprocessing_evaluation.py', 'tests/test_vcg_preprocessing.py',
           '05_QRS_T_Loops_Step_by_Step.ipynb', '02_VCG_MI_HC_pipeline.ipynb']
ENGINE_SOURCES = ['scripts/run_full_vcg_evaluation.py',
                  'scripts/vcg_evaluation_reporting.py', 'scripts/compare_vcg_annotations.py']
QRS_FLAGS = {'qrs_on_not_found', 'qrs_off_not_found',
             'qrs_duration_or_anchor_requires_review', 'pq_overlaps_qrs',
             'qrs_near_search_limit'}
ANNOTATION_COLUMNS = ['record_id', 'patient_id', 'label_name', 'split',
    'time_reference', 'reference_anchor_sample', 'reviewer', 'review_status',
    'independent_reference', 'qrs_on_ms', 'qrs_off_ms', 't_on_ms', 't_off_ms',
    'pred_qrs_on_ms', 'pred_qrs_off_ms', 'pred_t_on_ms', 'pred_t_off_ms',
    'qc_status', 'qc_flags', 'review_stratum', 'plot_path', 'loop_plot_path', 'notes']


def sha256(path):
    from project_integrity import file_digest
    return file_digest(path)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def normalize_diagnosis(value):
    return ' '.join(str(value).strip().lower().split())


def discover_records(data_root):
    """Union on-disk records and RECORDS; missing files stay in inventory."""
    data_root = Path(data_root)
    paths = {p.relative_to(data_root).with_suffix('').as_posix()
             for ext in ['*.hea', '*.dat', '*.xyz']
             for p in data_root.rglob(ext) if p.parent.name.startswith('patient')}
    manifest = data_root / 'RECORDS'
    listed = []
    if manifest.exists():
        listed = [s.strip() for s in manifest.read_text(encoding="utf-8").splitlines()
                  if s.strip() and not s.startswith('#')]
        paths.update(listed)
    duplicate_manifest = sorted({s for s in listed if listed.count(s) > 1})
    rows = []
    for record_id in sorted(paths):
        path = data_root / record_id
        row = dict(record_id=record_id, patient_id=Path(record_id).parent.name,
                   diagnosis='', label_name='OTHER', label=np.nan,
                   missing_signal=False, missing_metadata=False,
                   header_error='', listed_in_RECORDS=record_id in listed,
                   included=False)
        try:
            header = wfdb.rdheader(str(path))
            reason = next((s.split(':', 1)[1].strip() for s in header.comments
                           if s.lower().startswith('reason for admission:')), '')
            norm = normalize_diagnosis(reason)
            row.update(diagnosis=reason, fs=header.fs, total_samples=header.sig_len,
                       channels='; '.join(header.sig_name),
                       missing_metadata=norm in {'', 'n/a', 'unknown', 'nan'})
            row['header_sha256'] = sha256(path.with_suffix('.hea'))
            row['signal_sha256'] = json.dumps({name: sha256(path.parent/name)
                for name in sorted(set(header.file_name)) if (path.parent/name).is_file()}, sort_keys=True)
            if norm == 'myocardial infarction':
                row.update(label_name='MI', label=1, included=True)
            elif norm in {'healthy control', 'healthy controls'}:
                row.update(label_name='HC', label=0, included=True)
            row['missing_signal'] = any(not (path.parent / name).is_file()
                                        for name in set(header.file_name))
        except Exception as exc:
            row.update(missing_metadata=True, header_error=f'{type(exc).__name__}: {exc}',
                       missing_signal=not path.with_suffix('.xyz').exists())
        rows.append(row)
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError(f'No PTB records at {data_root}')
    if inventory.record_id.duplicated().any() or duplicate_manifest:
        raise ValueError(f'Duplicate recording IDs: {duplicate_manifest}')
    return inventory


def make_split(inventory, existing_features=None, existing_manifest=None):
    eligible = inventory[inventory.included].copy()
    if eligible.groupby('patient_id').label.nunique().gt(1).any():
        raise ValueError('Conflicting HC/MI diagnoses within a patient.')
    people = eligible[['patient_id', 'label_name']].drop_duplicates().reset_index(drop=True)
    split = {}
    if existing_manifest:
        source = pd.read_csv(existing_manifest)
        source['split'] = source['split'].replace({'validation': 'val'})
        if source.groupby('patient_id').split.nunique().gt(1).any():
            raise ValueError('Patient leakage in existing split manifest')
        if not set(source.split).issubset({'train', 'val', 'test'}):
            raise ValueError('Unknown split names in existing manifest')
        split.update(source.drop_duplicates('patient_id').set_index('patient_id').split.to_dict())
        if set(people.patient_id) - set(split):
            raise ValueError('Existing manifest does not cover every eligible patient')
        details = {'method': 'Reuse explicit existing manifest', 'source': str(existing_manifest)}
    elif existing_features and Path(existing_features).exists():
        # Notebook 02 splits the original feature CSV in its existing row order.
        source = pd.read_csv(existing_features)
        old = source[['patient_id', 'label']].drop_duplicates().reset_index(drop=True)
        if old.groupby('patient_id').label.nunique().gt(1).any():
            raise ValueError('Conflicting labels in legacy features')
        if set(old.patient_id) != set(people.patient_id):
            raise ValueError('Cannot reproduce legacy split: patient population differs')
        original_train, original_test = train_test_split(
            old, test_size=.25, random_state=42, stratify=old.label)
        new_train, validation = train_test_split(
            original_train, test_size=.20, random_state=42, stratify=original_train.label)
        for name, subset in [('train', new_train), ('val', validation), ('test', original_test)]:
            split.update(dict.fromkeys(subset.patient_id, name))
        details = {'method': 'Reproduce notebook 02 patient 75/25 split; preserve original test; reserve 20% of original train for validation',
                   'source': '02_VCG_MI_HC_pipeline.ipynb cell 36 + vcg_12_features.csv row order',
                   'source_sha256': sha256(existing_features), 'random_state': 42,
                   'patient_ratios': '60/15/25', 'test_preserved': True,
                   'legacy_failed_records_retained_by_patient': ['patient007/s0029lre', 'patient010/s0042lre']}
    else:
        train, holdout = train_test_split(people, test_size=.30, random_state=42,
                                          stratify=people.label_name)
        validation, test = train_test_split(holdout, test_size=.50, random_state=42,
                                           stratify=holdout.label_name)
        for name, subset in [('train', train), ('val', validation), ('test', test)]:
            split.update(dict.fromkeys(subset.patient_id, name))
        details = {'method': 'New stratified patient-level 70/15/15 split', 'random_state': 42}
    eligible['split'] = eligible.patient_id.map(split)
    if eligible.split.isna().any() or eligible.groupby('patient_id').split.nunique().ne(1).any():
        raise ValueError('Incomplete or leaking split')
    details['counts'] = eligible.groupby(['split', 'label_name']).agg(
        patients=('patient_id', 'nunique'), recordings=('record_id', 'size')).reset_index().to_dict('records')
    return eligible, details


def run_staged(record_path, cfg, trace):
    """Same calls/aggregation as process_record; trace survives a later failure."""
    trace['stage'] = 'load'
    raw = v1.load_xyz(record_path, cfg.seconds)
    trace['raw'] = raw
    fs = raw['fs']
    trace['stage'] = 'preprocessing'
    morph, detsig, prep = v1.preprocess_xyz(raw['xyz'], fs, cfg, return_diagnostics=True)
    trace.update(morph=morph, detection_xyz=detsig, preprocessing=prep)
    trace['stage'] = 'qrs_detection'
    detector = v1.detect_qrs_anchors(morph, detsig, fs, cfg)
    trace['detector'] = detector
    trace['stage'] = 'representative_beat'
    rep = representative_with_diagnostics(morph, detector['anchors'], fs, cfg,
                                          prep['input_quality'], trace)
    trace['rep'] = rep
    trace['stage'] = 'delineation'
    b = v1.delineate_loops(rep, fs, cfg)
    trace['boundaries'] = b
    trace['stage'] = 'features'
    qrs = v1.cut_loop(rep['xyz'], b['qrs_on'], b['qrs_off'])
    tloop = v1.cut_loop(rep['xyz'], b['t_on'], b['t_off'])
    flags = list(dict.fromkeys(detector['flags'] + b['flags'] + rep['quality_flags']))
    if prep['powerline_status'] == 'inconclusive':
        flags.append('powerline_inconclusive_review')
    baseline_residual = {}
    if b.get('tp') is not None:
        pq, tp = rep['pq'], b['tp']
        pv = np.median(rep['xyz'][pq[0]:pq[1]], axis=0)
        tv = np.median(rep['xyz'][tp[0]:tp[1]], axis=0)
        baseline_residual = {'pq_tp_vector_offset_mV': float(np.linalg.norm(tv - pv)),
            'note': 'Offset PQ–TP có thể là drift/tail T; không phải đo nhiễu chắc chắn.'}
    if rep['n_used'] < 5:
        flags.append('few_representative_beats')
    if rep['n_excluded'] / len(detector['anchors']) > .4:
        flags.append('many_excluded_beats')
    speeds = {name: None if loop is None else b['speed'][b[a]:b[z]+1]
              for name, loop, a, z in [('QRS', qrs, 'qrs_on', 'qrs_off'),
                                        ('T', tloop, 't_on', 't_off')]}
    features = v1.extract_features(qrs, tloop, fs, cfg, speeds=speeds)
    if not np.isfinite(list(features.values())).all():
        flags.append('features_incomplete')
    result = dict(raw=raw, morph=morph, detection_xyz=detsig, detector=detector,
        rep=rep, boundaries=b, qrs_loop=qrs, t_loop=tloop, features=features,
        flags=list(dict.fromkeys(flags)), status='needs_review' if flags else 'auto_candidate',
        eligible_for_features=not flags, config=asdict(cfg), preprocessing=prep,
        baseline_residual=baseline_residual)
    trace['stage'] = 'complete'
    return result


def representative_with_diagnostics(morph, anchors, fs, cfg, quality, trace):
    """Observe V1's own exclusion rows on failure, without changing its logic.

    V1 raises when fewer than three beats remain. A frame-local observer retains
    those already computed diagnostics before the exception unwinds. Successful
    outputs still come directly from the original representative_beat function.
    """
    previous = sys.gettrace()
    captured = {}
    representative_code = getattr(v1.representative_beat, '__code__', None)
    def observe(frame, event, arg):
        if frame.f_code is representative_code:
            if event == 'exception' and 'rows' in frame.f_locals:
                values = frame.f_locals
                captured.update(beat_qc=[dict(r) for r in values['rows']],
                    n_used=len(values.get('beats', [])),
                    n_excluded=len(values['rows'])-len(values.get('beats', [])),
                    used_samples=list(values.get('kept', [])))
            return observe
        return None
    sys.settrace(observe)
    try:
        return v1.representative_beat(morph, anchors, fs, cfg, quality=quality)
    except Exception:
        if captured:
            trace['representative_failure_diagnostics'] = captured
        raise
    finally:
        sys.settrace(previous)


def flags_string(flags):
    return '; '.join(dict.fromkeys(flags))


def describe_result(metadata, trace, result=None, error=''):
    row = {k: metadata[k] for k in ['patient_id', 'record_id', 'split', 'diagnosis', 'label', 'label_name']}
    row.update(dict.fromkeys(v1.FEATURE_NAMES + [k+'_ms' for k in BOUNDARIES] + [
        'qrs_duration_ms', 't_duration_ms', 'median_rr_ms', 'st_level', 'tp_level',
        't_on_method_gap_ms', 't_off_method_gap_ms', 'fs', 'n_beats_used', 'n_beats_excluded'], np.nan))
    row.update(processing_success=result is not None, extraction_success=False,
        eligible_for_features=False,
        feature_completeness=0, qc_status='failed', qc_flags='', input_qc_flags='',
        evaluation_flags='', qrs_qc_flags='', t_qc_flags='', n_qrs_detected=0,
        qrs_boundary_valid=False, qrs_algorithm_range_valid=False,
        t_boundary_valid=False, t_algorithm_range_valid=False,
        qrs_boundary_method='spatial_speed_threshold_quiet_run', t_method='unresolved',
        failure_stage='' if result is not None else trace.get('stage', 'load'), error=error,
        plot_path='', loop_plot_path='', detection_plot_path='', plot_error='')
    raw = trace.get('raw')
    if raw:
        row.update(fs=raw['fs'], samples_loaded=len(raw['xyz']),
                   seconds_loaded=len(raw['xyz'])/raw['fs'])
    prep = trace.get('preprocessing', {})
    row['input_qc_flags'] = flags_string(prep.get('input_quality', {}).get('flags', []))
    for key in ['notch_applied', 'notch_mode', 'powerline_status', 'powerline_score_db']:
        row[key] = prep.get(key)
    detector = trace.get('detector')
    if detector:
        rr = detector['rr']
        row.update(n_qrs_detected=len(detector['anchors']), median_rr_ms=float(np.median(rr))*1000,
            qrs_detection_flags=flags_string(detector['flags']),
            rr_short_count=int((rr < .30).sum()), rr_long_count=int((rr > 2).sum()),
            suspected_missed_rr_count=int((rr > 1.7*np.median(rr)).sum()),
            suspected_extra_rr_count=int((rr < .6*np.median(rr)).sum()))
    rep = trace.get('rep')
    if rep:
        row.update(n_beats_used=rep['n_used'], n_beats_excluded=rep['n_excluded'],
                   excluded_beat_fraction=rep['n_excluded']/len(detector['anchors']))
    elif trace.get('representative_failure_diagnostics'):
        partial = trace['representative_failure_diagnostics']
        row.update(n_beats_used=partial['n_used'], n_beats_excluded=partial['n_excluded'],
                   excluded_beat_fraction=partial['n_excluded']/len(detector['anchors']))
    b = trace.get('boundaries')
    if b is not None and rep:
        for key in BOUNDARIES + ['t_tangent_on', 't_tangent_off', 't_threshold_on', 't_threshold_off']:
            row[key+'_ms'] = np.nan if b[key] is None else float(rep['time_ms'][b[key]])
        row['qrs_duration_ms'] = row['qrs_off_ms'] - row['qrs_on_ms']
        row['t_duration_ms'] = row['t_off_ms'] - row['t_on_ms']
        qok = bool(np.isfinite([row['qrs_on_ms'], row['qrs_off_ms']]).all()
                   and row['qrs_on_ms'] < row['qrs_off_ms'])
        tok = bool(np.isfinite([row[k+'_ms'] for k in ['qrs_off', 't_on', 't_peak', 't_off']]).all()
                   and row['qrs_off_ms'] < row['t_on_ms'] < row['t_peak_ms'] < row['t_off_ms'])
        row.update(qrs_boundary_valid=qok, t_boundary_valid=tok,
            qrs_algorithm_range_valid=qok and row['qrs_on_ms'] < 0 < row['qrs_off_ms']
                and 40 <= row['qrs_duration_ms'] <= 240,
            t_algorithm_range_valid=tok and 60 <= row['t_duration_ms'] <= 400,
            st_level=b['st_level'], tp_level=b['t_end_baseline'], t_method=b['t_method'],
            qrs_qc_flags=flags_string(f for f in b['flags'] if f in QRS_FLAGS),
            t_qc_flags=flags_string(f for f in b['flags'] if f not in QRS_FLAGS),
            t_on_method_gap_ms=abs(row['t_tangent_on_ms']-row['t_threshold_on_ms']),
            t_off_method_gap_ms=abs(row['t_tangent_off_ms']-row['t_threshold_off_ms']))
        extra = []
        if row['t_on_method_gap_ms'] > 100:
            extra.append('eval_t_on_methods_gap_gt100ms')
        if row['suspected_extra_rr_count']:
            extra.append('eval_short_rr_lt0.6median_suspected_extra_or_irregular')
        row['evaluation_flags'] = flags_string(extra)
    if result:
        row.update(result['features'])
        count = int(np.isfinite([row[f] for f in v1.FEATURE_NAMES]).sum())
        row.update(qc_status=result['status'], qc_flags=flags_string(result['flags']),
            feature_completeness=count, extraction_success=count == len(v1.FEATURE_NAMES),
            eligible_for_features=result['eligible_for_features'])
    else:
        row['qc_flags'] = flags_string((detector['flags'] if detector else []) + ['processing_failed', 'features_incomplete'])
    missing = {}
    for name, names in [('QRS', v1.FEATURE_NAMES[:7]), ('T', v1.FEATURE_NAMES[7:])]:
        if any(not np.isfinite(row[f]) for f in names):
            missing[name] = (f'pipeline_failed_at_{row["failure_stage"]}: {error}' if result is None
                else 'missing_or_too_short_loop; ' + row[f'{"qrs" if name=="QRS" else "t"}_qc_flags'])
    row['feature_missing_reasons'] = json.dumps(missing, ensure_ascii=False)
    return row


def save_trace(out, row, trace, result):
    safe = row['record_id'].replace('/', '_')
    raw, rep, b = trace.get('raw'), trace.get('rep'), trace.get('boundaries')
    artifact = {'record_id': row['record_id'], 'fs': row.get('fs'),
        'time_reference': 'median_relative_ms', 'raw_time_origin_sample': 0,
        'failure_stage': row['failure_stage'], 'error': row['error']}
    if trace.get('traceback'):
        artifact['traceback'] = trace['traceback']
    if trace.get('representative_failure_diagnostics'):
        artifact.update(trace['representative_failure_diagnostics'])
    arrays = {}
    if trace.get('detector'):
        detector = trace['detector']
        artifact.update(anchors=detector['anchors'], qrs_anchors=detector['anchors'],
                        rr_seconds=detector['rr'], qrs_detection_flags=detector['flags'])
        arrays.update(anchors=detector['anchors'], detection_envelope=detector['envelope'],
                      detection_threshold=detector['threshold'])
    if rep:
        artifact.update(used_samples=rep['used_samples'], median_anchor_sample=rep['anchor'],
            median_time_start_ms=float(rep['time_ms'][0]),
            median_time_end_ms=float(rep['time_ms'][-1]), median_n_samples=len(rep['xyz']),
            pq_indices=rep['pq'], beat_qc=rep['beat_qc'].to_dict('records'),
            representative_quality_flags=rep['quality_flags'])
        arrays.update(median_xyz=rep['xyz'], time_ms=rep['time_ms'], used_samples=rep['used_samples'])
    if b is not None:
        artifact['boundaries'] = {k: v for k, v in b.items() if not isinstance(v, np.ndarray)}
        arrays.update({k: b[k] for k in ['speed', 'smooth_speed', 'magnitude', 't_magnitude']})
        if b['t_slope'] is not None:
            arrays['t_slope'] = b['t_slope']
    prep = trace.get('preprocessing')
    if prep:
        artifact['preprocessing'] = {k: v for k, v in prep.items()
            if k not in {'input_quality', 'powerline'} and not isinstance(v, np.ndarray)}
        artifact['input_quality'] = {k: v for k, v in prep['input_quality'].items()
            if not k.endswith('_mask') and not k.startswith('psd_')}
        artifact['input_quality_mask_counts'] = {k: int(v.sum()) for k, v in prep['input_quality'].items()
                                                if k.endswith('_mask')}
        powerline = prep.get('powerline', {})
        artifact['powerline'] = {k: v for k, v in powerline.items()
            if k not in {'psd_frequency_hz', 'psd_density', 'windows'}}
        if isinstance(powerline.get('windows'), pd.DataFrame):
            artifact['powerline']['windows'] = powerline['windows'].to_dict('records')
    artifact['result'] = row
    write_json(out / 'artifacts' / (safe+'.json'), artifact)
    if arrays:
        np.savez_compressed(out / 'artifacts' / (safe+'.npz'), **arrays)


def record_plots(out, row, trace, result):
    folder = out / 'plots' / row['label_name']
    safe = row['record_id'].replace('/', '_')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    if result:
        fig = v1.plot_delineation(result)
        ax = fig.axes[2]
        for key, color, label in [('st_level', '#ea580c', 'ST reference'),
                                   ('t_end_baseline', '#7c3aed', 'TP reference')]:
            value = result['boundaries'][key]
            if np.isfinite(value):
                ax.axhline(value, color=color, ls='--', lw=.8, label=label)
        b, rep = result['boundaries'], result['rep']
        if b['t_peak'] is not None:
            fig.axes[0].axvline(rep['time_ms'][b['t_peak']], color='#7c3aed', ls='--', lw=.7, label='T peak')
        ax.legend(fontsize=7, ncol=2)
        if len(row['qc_flags']) > 170:
            import textwrap
            fig._suptitle.set_text(f'{row["record_id"]} | {row["qc_status"]}\n' + '\n'.join(textwrap.wrap(row['qc_flags'], 150)))
            fig._suptitle.set_fontsize(8)
        path = folder / (safe+'_delineation.png')
        fig.savefig(path, dpi=95)
        plt.close(fig)
        row['plot_path'] = path.relative_to(out).as_posix()
        fig = v1.plot_loops(result)
        path = folder / (safe+'_loops.png')
        fig.savefig(path, dpi=85)
        plt.close(fig)
        row['loop_plot_path'] = path.relative_to(out).as_posix()
        # Detection strips are useful for RR/median failures and suspected count errors.
        if result['detector']['flags'] or row.get('suspected_extra_rr_count', 0):
            fig = v1.plot_detection(result, seconds=8)
            path = folder / (safe+'_detection.png')
            fig.savefig(path, dpi=85)
            plt.close(fig)
            row['detection_plot_path'] = path.relative_to(out).as_posix()
    else:
        fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout='constrained')
        raw = trace.get('raw')
        if raw:
            n = min(len(raw['xyz']), int(8*raw['fs']))
            for j, ax in enumerate(axs):
                ax.plot(np.arange(n)/raw['fs'], raw['xyz'][:n, j], lw=.7)
                for anchor in trace.get('detector', {}).get('anchors', []):
                    if anchor < n:
                        ax.axvline(anchor/raw['fs'], color='#ea580c', alpha=.5)
                ax.set_ylabel('XYZ'[j]+' (mV)')
            axs[-1].set_xlabel('Recording time (s)')
        fig.suptitle(f'{row["record_id"]} | failure at {row["failure_stage"]}\n{row["error"]}', fontsize=9)
        path = out / 'plots' / 'failure_cases' / (safe+'.png')
        fig.savefig(path, dpi=95)
        plt.close(fig)
        row['plot_path'] = path.relative_to(out).as_posix()


def process_one(metadata, data_root, output, cfg_dict, plots=True):
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_dict.items()})
    trace, result, error = {}, None, ''
    started = time.perf_counter()
    try:
        result = run_staged(Path(data_root)/metadata['record_id'], cfg, trace)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        trace['traceback'] = traceback.format_exc()
    row = describe_result(metadata, trace, result, error)
    out = Path(output)
    if plots:
        try:
            record_plots(out, row, trace, result)
        except Exception as exc:
            row['plot_error'] = f'{type(exc).__name__}: {exc}'
        finally:
            plt.close('all')
    row['runtime_seconds'] = time.perf_counter()-started
    save_trace(out, row, trace, result)
    safe = row['record_id'].replace('/', '_')
    write_json(out / 'checkpoints' / (safe+'.json'), row)
    return row


def review_sample(frame, per_label=30):
    """Patient diversity first, then cover QC strata; no correctness labels."""
    selected = []
    for label, group in frame.groupby('label_name', sort=True):
        group = group.copy()
        group['review_stratum'] = np.select([
            ~group.processing_success, group.qrs_qc_flags.fillna('').ne(''),
            group.t_qc_flags.fillna('').ne(''), group.qc_status.eq('needs_review')],
            ['processing_failure', 'qrs_review', 't_review', 'other_qc'], default='no_v1_qc_flags')
        group = group.sample(frac=1, random_state=42)
        chosen, patients = [], set()
        strata = [g for _, g in group.groupby('review_stratum', sort=True)]
        while len(chosen) < min(per_label, len(group)):
            progress = False
            for stratum in strata:
                candidates = stratum[~stratum.record_id.isin(chosen) & ~stratum.patient_id.isin(patients)]
                if candidates.empty:
                    continue
                r = candidates.iloc[0]
                chosen.append(r.record_id)
                patients.add(r.patient_id)
                progress = True
                if len(chosen) >= min(per_label, len(group)):
                    break
            if not progress:
                leftovers = group[~group.record_id.isin(chosen)]
                chosen.extend(leftovers.record_id.head(min(per_label, len(group))-len(chosen)).tolist())
                break
        # Extra representative for every observed V1 flag on train/val, if not covered.
        covered = set()
        for flags in group[group.record_id.isin(chosen)].qc_flags.fillna(''):
            covered.update(flags.split('; '))
        for _, r in group[group.split.ne('test')].iterrows():
            new_flags = set(str(r.qc_flags).split('; ')) - covered - {''}
            if new_flags:
                chosen.append(r.record_id)
                covered.update(new_flags)
        selected.append(group[group.record_id.isin(chosen)])
    sample = pd.concat(selected, ignore_index=True)
    rows = []
    for _, r in sample.iterrows():
        row = {k: r.get(k, '') for k in ANNOTATION_COLUMNS if k in sample.columns}
        row.update(time_reference='median_relative_ms', reference_anchor_sample=np.nan,
            reviewer='', review_status='pending', independent_reference=False,
            notes='', **{k+'_ms': np.nan for k in ['qrs_on', 'qrs_off', 't_on', 't_off']},
            **{'pred_'+k+'_ms': r.get(k+'_ms', np.nan) for k in ['qrs_on', 'qrs_off', 't_on', 't_off']})
        rows.append(row)
    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)


def write_tables(out, frame):
    frame = frame.sort_values('record_id').reset_index(drop=True)
    frame.to_csv(out/'features/all_features.csv', index=False)
    for split in ['train', 'val', 'test']:
        frame[frame.split.eq(split)].to_csv(out/f'features/{split}_features.csv', index=False)
    common = ['patient_id', 'record_id', 'split', 'diagnosis', 'label', 'label_name']
    for filename, fields in [
        ('qrs_detection_results', ['n_qrs_detected', 'n_beats_used', 'n_beats_excluded', 'median_rr_ms',
            'qrs_detection_flags', 'rr_short_count', 'rr_long_count', 'suspected_missed_rr_count',
            'suspected_extra_rr_count', 'processing_success', 'failure_stage', 'error']),
        ('qrs_delineation_results', ['qrs_on_ms', 'qrs_off_ms', 'qrs_duration_ms', 'qrs_boundary_method',
            'qrs_boundary_valid', 'qrs_algorithm_range_valid', 'qrs_qc_flags', 'evaluation_flags']),
        ('t_delineation_results', ['t_on_ms', 't_peak_ms', 't_off_ms', 't_duration_ms', 't_method',
            'st_level', 'tp_level', 't_boundary_valid', 't_algorithm_range_valid', 't_qc_flags',
            't_tangent_on_ms', 't_tangent_off_ms', 't_threshold_on_ms', 't_threshold_off_ms',
            't_on_method_gap_ms', 't_off_method_gap_ms', 'evaluation_flags'])]:
        frame.reindex(columns=common+fields).to_csv(out/f'evaluation/{filename}.csv', index=False)
    frame[~frame.processing_success].to_csv(out/'evaluation/failed_records.csv', index=False)
    beat_rows = []
    for _, record in frame.iterrows():
        path = out/'artifacts'/(record.record_id.replace('/', '_')+'.json')
        if path.exists():
            for beat in json.loads(path.read_text(encoding="utf-8")).get('beat_qc', []):
                beat_rows.append({**{key: record[key] for key in common}, **beat})
    pd.DataFrame(beat_rows).to_csv(out/'evaluation/beat_quality_results.csv', index=False)
    # Empty schema explicitly means no independent corresponding reference found.
    path = out/'evaluation/boundary_reference_comparison.csv'
    if not path.exists():
        pd.DataFrame(columns=['record_id', 'patient_id', 'label_name', 'boundary',
            'predicted_ms', 'reference_ms', 'signed_error_ms', 'absolute_error_ms',
            'reference_source', 'comparison_scope']).to_csv(path, index=False)
    template_path = out/'annotations/manual_review_template.csv'
    if not template_path.exists():
        review_sample(frame).to_csv(template_path, index=False)
    return frame


def initialize_run(args, inventory, manifest, split_details):
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'{out} already has results. Use --resume or a new --output-dir.')
    for directory in ['features', 'evaluation', 'annotations', 'checkpoints', 'artifacts',
                       'baseline_v1', 'plots/HC', 'plots/MI', 'plots/failure_cases', 'plots/summary']:
        (out/directory).mkdir(parents=True, exist_ok=True)
    hashes = {name: sha256(REPO/name) for name in SOURCES}
    engine_hashes = {name: sha256(REPO/name) for name in ENGINE_SOURCES}
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas',
                'matplotlib', 'scikit-learn', 'wfdb', 'tqdm']}
    current = dict(baseline='V1 unchanged notebook 05', config=asdict(Config(seconds=args.seconds)),
        baseline_hashes=hashes, engine_hashes=engine_hashes, versions=versions,
        data_root=str(args.data_root.resolve()),
        inventory_sha256=hashlib.sha256(inventory.to_csv(index=False).encode()).hexdigest(),
        split_sha256=hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest())
    config_path = out/'config.json'
    if args.resume:
        if not config_path.exists():
            raise ValueError('--resume requires config.json')
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if jsonable({k: config.get(k) for k in current}) != jsonable(current):
            raise ValueError('Resume blocked: V1 source/config/data inventory/split changed')
    else:
        config = {**current, 'created_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
            'python': sys.version, 'platform': platform.platform(), 'split_details': split_details,
            'requested_notebook': '05_QRS_T_Loops_Step_by_Step(2).ipynb',
            'actual_notebook': '05_QRS_T_Loops_Step_by_Step.ipynb',
            'notebook_note': 'Requested (2) filename absent; existing notebook imports current shared V1 modules.',
            'annotation_audit': 'No corresponding independent beat/QRS/T boundary annotations in local PTB; accuracy undetermined.',
            'evaluation_heuristics': {'short_rr': '<0.6 * median RR; suspected extra/irregular only',
                't_on_methods_gap_ms': '>100; extra review hint, not a V1 flag or error label'},
            'optimization': 'None; test used only for fixed-baseline descriptive evaluation.'}
        audit_path = REPO/'scripts/reference_audit.json'
        if audit_path.exists():
            config['annotation_audit'] = json.loads(audit_path.read_text(encoding="utf-8"))
        write_json(config_path, config)
        for name in SOURCES:
            dest = out/'baseline_v1'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, dest)
        for name in ENGINE_SOURCES:
            dest = out/'baseline_v1'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, dest)
        for name in ['run_full_vcg_evaluation.py', 'vcg_evaluation_reporting.py',
                     'compare_vcg_annotations.py', 'reference_audit.md', 'reference_audit.json']:
            source = REPO/'scripts'/name
            if source.exists():
                shutil.copy2(source, out/'baseline_v1'/name)
    return out, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=REPO/'content/ptbdb/physionet.org/files/ptbdb/1.0.0')
    parser.add_argument('--output-dir', type=Path, default=REPO/'outputs/vcg_full_evaluation')
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--records', nargs='+', help='Optional smoke subset; default is every eligible record')
    parser.add_argument('--split-manifest', type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.seconds <= 0:
        parser.error('workers and seconds must be positive')
    inventory = discover_records(args.data_root)
    manifest, split_details = make_split(inventory, REPO/'vcg_12_features.csv', args.split_manifest)
    # Always split complete eligible cohort BEFORE smoke subset or failures.
    out, config = initialize_run(args, inventory, manifest, split_details)
    inventory.to_csv(out/'evaluation/dataset_inventory.csv', index=False)
    manifest.to_csv(out/'split_manifest.csv', index=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(out/'evaluation/run.log', encoding='utf-8'), logging.StreamHandler()])
    jobs = manifest.to_dict('records')
    if args.records:
        unknown = set(args.records)-set(manifest.record_id)
        if unknown:
            raise ValueError(f'Requested records not eligible: {unknown}')
        jobs = [r for r in jobs if r['record_id'] in set(args.records)]
    rows, pending = [], []
    for metadata in jobs:
        checkpoint = out/'checkpoints'/(metadata['record_id'].replace('/', '_')+'.json')
        if args.resume and checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            if row['record_id'] != metadata['record_id']:
                raise ValueError('Checkpoint identity mismatch')
            rows.append(row)
        else:
            pending.append(metadata)
    logging.info('Eligible=%d; requested=%d; resumed=%d; pending=%d; workers=%d',
                 len(manifest), len(jobs), len(rows), len(pending), args.workers)
    cfg_dict = config['config']
    kwargs = dict(data_root=str(args.data_root.resolve()), output=str(out), cfg_dict=cfg_dict, plots=not args.no_plots)
    if args.workers == 1:
        for metadata in tqdm(pending, desc='PTB V1', unit='record'):
            row = process_one(metadata, **kwargs)
            rows.append(row)
            logging.info('%s %s %d/12 %s', row['record_id'], row['qc_status'], row['feature_completeness'], row['error'])
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, metadata, **kwargs): metadata for metadata in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc='PTB V1', unit='record'):
                # Processing errors are captured inside process_one; infrastructure errors must be visible.
                row = future.result()
                rows.append(row)
                logging.info('%s %s %d/12 %s', row['record_id'], row['qc_status'], row['feature_completeness'], row['error'])
    frame = write_tables(out, pd.DataFrame(rows))
    from vcg_evaluation_reporting import write_summary_outputs
    write_summary_outputs(out, frame, inventory, config)
    write_json(out/'evaluation/completion.json', dict(
        completed_at=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
        requested_records=len(jobs), completed_records=len(frame),
        eligible_records=len(manifest), full_dataset_complete=len(frame) == len(manifest),
        processing_success=int(frame.processing_success.sum()),
        twelve_features=int(frame.extraction_success.sum()),
        plot_errors=int(frame.plot_error.fillna('').ne('').sum()),
        source_hashes_unchanged=all(sha256(REPO/name) == digest for name, digest in config['baseline_hashes'].items()),
        engine_hashes_unchanged=all(sha256(REPO/name) == digest for name, digest in config['engine_hashes'].items())))
    logging.info('Completed %d/%d eligible records: %s', len(frame), len(manifest), out)
    return frame


if __name__ == '__main__':
    main()
