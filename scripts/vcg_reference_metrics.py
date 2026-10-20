"""Independent reference validation and frozen-V1 QRS/T evaluation.

No prediction, heuristic or second detector is accepted as a reference. Missing
references remain NaN and every cohort record remains in the evaluation tables.
Time metrics use continuous intervals between represented sample endpoints;
feature cuts include both endpoint samples, exactly as V1 does.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import vcg_pipeline as v1
from vcg_preprocessing import Config

BOUNDARIES = ('qrs_on', 'qrs_off', 't_on', 't_off')
LOOPS = {'QRS': ('qrs_on', 'qrs_off'), 'T': ('t_on', 't_off')}
EPSILON = 1e-9  # Native feature unit; ratios are unavailable below this value.
META = ('record_id', 'patient_id', 'label_name', 'split', 'qc_status', 'qc_flags')
ACCEPTED_COLUMNS = ['record_id', 'patient_id', 'split', 'boundary',
                    'reference_ms', 'reference_index', 'requested_ref_ms',
                    'quantization_error_ms', 'reviewer', 'reference_source',
                    'annotation_row', 'reference_selection']
AUDIT_COLUMNS = ['annotation_row', 'record_id', 'reviewer', 'boundary',
                 'requested_ref_ms', 'reference_ms', 'reference_index',
                 'quantization_error_ms', 'candidate_accepted', 'accepted',
                 'acceptance_status', 'rejection_reason', 'reference_source',
                 'reference_selection']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _root(artifact_root):
    root = Path(artifact_root)
    return root / 'artifacts' if (root / 'artifacts').is_dir() else root


def _text(value):
    return '' if value is None or pd.isna(value) else str(value).strip()


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _true(value):
    return value is True or isinstance(value, np.bool_) and bool(value) or (
        isinstance(value, str) and value.strip().lower() == 'true')


def load_median(artifact_root, record_id):
    """Read only cached V1 median and validate its time grid and identity.

    Failed recordings return ``xyz=None``; their JSON/NPZ identity hashes remain
    available. ``signal_sha256`` is deliberately the complete NPZ file hash.
    """
    root = _root(artifact_root)
    safe = str(record_id).replace('/', '_')
    jp, npz = root / (safe + '.json'), root / (safe + '.npz')
    if not jp.is_file() or not npz.is_file():
        raise FileNotFoundError(f'Missing V1 artifact: {record_id}')
    artifact = json.loads(jp.read_text())
    if artifact.get('record_id') != record_id:
        raise ValueError('Artifact record_id mismatch')
    if artifact.get('time_reference') != 'median_relative_ms':
        raise ValueError('Artifact is not in median_relative_ms coordinates')
    fs = _number(artifact.get('fs'))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Artifact fs must be positive')
    result = dict(xyz=None, time_ms=None, fs=fs, anchor=None,
                  boundaries=artifact.get('boundaries', {}), artifact=artifact,
                  signal_sha256=sha256_file(npz), artifact_sha256=sha256_file(jp),
                  npz_path=str(npz), json_path=str(jp), record_id=record_id,
                  time_reference='median_relative_ms')
    with np.load(npz, allow_pickle=False) as arrays:
        if 'median_xyz' not in arrays:
            return result
        xyz = np.asarray(arrays['median_xyz'], dtype=float)
        time = np.asarray(arrays['time_ms'], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
        raise ValueError('Median XYZ must have shape (n>=2,3)')
    if time.ndim != 1 or len(time) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid median XYZ/time arrays')
    if not np.isfinite(time).all() or not np.allclose(np.diff(time), 1000 / fs,
                                                    rtol=0, atol=1e-7):
        raise ValueError('Median time grid does not match fs')
    anchor = artifact.get('median_anchor_sample')
    if anchor is None or int(anchor) != anchor or not 0 <= int(anchor) < len(time):
        raise ValueError('Median anchor missing or out of range')
    anchor = int(anchor)
    if not np.allclose(time, (np.arange(len(time)) - anchor) * 1000 / fs,
                       rtol=0, atol=1e-7):
        raise ValueError('Median time origin does not match saved anchor')
    result.update(xyz=xyz, time_ms=time, anchor=anchor)
    return result


def time_ms_to_index(value, time_ms, fs):
    """Nearest sample, ties earlier; reject outside domain and never clip.

    Returns ``(index, represented_ms, represented_ms - requested_ms)``.
    """
    value, fs = _number(value), _number(fs)
    time = np.asarray(time_ms, dtype=float)
    if not np.isfinite(value) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Time and fs must be finite; fs must be positive')
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError('Time grid must contain at least two finite samples')
    if not np.allclose(np.diff(time), 1000 / fs, rtol=0, atol=1e-7):
        raise ValueError('Time grid/fs mismatch')
    if value < time[0] or value > time[-1]:
        raise ValueError('Reference time outside saved median domain')
    index = int(np.argmin(np.abs(time - value)))
    represented = float(time[index])
    return index, represented, represented - value


def _annotation_metadata_reasons(row, original, median, allow_test):
    reasons = []
    for key in ('patient_id', 'split'):
        if _text(row.get(key)) != _text(original.get(key)):
            reasons.append(key + '_mismatch')
    if _text(original.get('split')) == 'test' and not allow_test:
        reasons.append('test_evaluation_disabled')
    if _text(row.get('annotation_scope')) != 'median_beat':
        reasons.append('unaligned_raw_or_wrong_annotation_scope')
    if _text(row.get('time_reference')) != 'median_relative_ms':
        reasons.append('wrong_time_reference')
    if _text(row.get('signal_sha256')) != median['signal_sha256']:
        reasons.append('signal_sha256_mismatch')
    if not np.isclose(_number(row.get('fs')), median['fs'], rtol=0, atol=1e-8):
        reasons.append('fs_mismatch')
    if not _true(row.get('independent_reference')):
        reasons.append('independent_reference_not_explicit_true')
    if _true(row.get('post_reveal_edit')):
        reasons.append('annotation_edited_after_prediction_reveal')
    if _text(row.get('reference_quality')).lower() not in ('accepted', 'good'):
        reasons.append('reference_quality_not_accepted')
    if _text(row.get('review_status')).lower() != 'confirmed':
        reasons.append('review_not_confirmed')
    for key in ('reviewer', 'reference_source'):
        if not _text(row.get(key)):
            reasons.append(key + '_missing')
    if _text(row.get('reference_source')).lower() in (
            'v1', 'prediction', 'algorithm', 'tangent', 'threshold', 'detector'):
        reasons.append('algorithm_output_is_not_independent_reference')
    if median['xyz'] is None:
        reasons.append('median_unavailable')
    return reasons


def validate_annotations(annotations, frame, artifact_root, allow_test=False):
    """Validate independently for each boundary; select one reference per edge.

    Multiple reviewer rows never become repeated evaluation samples. If more
    than one acceptable reference exists, exactly one candidate must have
    ``reference_selection=primary`` or ``adjudicated`` (adjudicated has priority).
    Duplicate entries from the same reviewer also need unambiguous selection.
    All unselected candidates remain in the audit for interreviewer analysis.
    """
    annotations = pd.DataFrame(annotations).copy()
    originals = frame.set_index('record_id', drop=False)
    if not originals.index.is_unique:
        raise ValueError('Cohort record_id must be unique')
    cache, audit = {}, []
    for position, (_, row) in enumerate(annotations.iterrows()):
        rid = _text(row.get('record_id'))
        metadata_reasons = []
        median = None
        if rid not in originals.index:
            metadata_reasons.append('unknown_record_id')
        else:
            if rid not in cache:
                try:
                    cache[rid] = load_median(artifact_root, rid)
                except (OSError, ValueError, KeyError) as exc:
                    cache[rid] = str(exc)
            if isinstance(cache[rid], str):
                metadata_reasons.append('invalid_artifact:' + cache[rid])
            else:
                median = cache[rid]
                metadata_reasons.extend(_annotation_metadata_reasons(
                    row, originals.loc[rid], median, allow_test))
        converted = {}
        for boundary in BOUNDARIES:
            value = _number(row.get(boundary + '_ref_ms'))
            reasons = list(metadata_reasons)
            status = _text(row.get(boundary + '_status')).lower()
            if status != 'visible':
                reasons.append('boundary_' + (status or 'status_missing'))
            if not np.isfinite(value):
                reasons.append('boundary_value_missing')
            ix, represented, quant = np.nan, np.nan, np.nan
            if median is not None and median['xyz'] is not None and np.isfinite(value):
                try:
                    ix, represented, quant = time_ms_to_index(
                        value, median['time_ms'], median['fs'])
                    converted[boundary] = (ix, represented)
                except ValueError as exc:
                    reasons.append(str(exc))
            audit.append(dict(annotation_row=position, record_id=rid,
                              reviewer=_text(row.get('reviewer')), boundary=boundary,
                              requested_ref_ms=value, reference_ms=represented,
                              reference_index=ix, quantization_error_ms=quant,
                              candidate_accepted=not reasons, accepted=False,
                              acceptance_status='rejected' if reasons else 'candidate',
                              rejection_reason='; '.join(reasons),
                              reference_source=_text(row.get('reference_source')),
                              reference_selection=_text(row.get('reference_selection')).lower()))
        # Only visible values participate in geometry checks. A missing T edge
        # never prevents acceptance of independently visible QRS edges.
        visible = {b: converted[b] for b in converted
                   if _text(row.get(b + '_status')).lower() == 'visible'}
        invalid = set()
        for first, last in (('qrs_on', 'qrs_off'), ('t_on', 't_off'), ('qrs_off', 't_on')):
            if first in visible and last in visible and visible[first][0] >= visible[last][0]:
                invalid.update((first, last))
        for entry in audit[-4:]:
            if entry['boundary'] in invalid:
                entry.update(candidate_accepted=False, acceptance_status='rejected',
                             rejection_reason=(entry['rejection_reason'] + '; invalid_boundary_order').strip('; '))
    audit = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
    if audit.empty:
        return pd.DataFrame(columns=ACCEPTED_COLUMNS), audit
    for _, group in audit[audit.candidate_accepted].groupby(['record_id', 'boundary']):
        selected = None
        if len(group) == 1:
            selected = group.index[0]
        else:
            for selection in ('adjudicated', 'primary'):
                candidates = group[group.reference_selection.eq(selection)]
                if len(candidates) == 1:
                    selected = candidates.index[0]
                    break
                if len(candidates) > 1:
                    break
        audit.loc[group.index, 'acceptance_status'] = 'requires_reference_selection'
        audit.loc[group.index, 'rejection_reason'] = 'multiple_references_require_unique_primary_or_adjudicated'
        if selected is not None:
            audit.loc[selected, ['accepted', 'acceptance_status', 'rejection_reason']] = [True, 'accepted', '']
            other = group.index.difference([selected])
            audit.loc[other, 'acceptance_status'] = 'valid_unselected_reference'
            audit.loc[other, 'rejection_reason'] = 'another_reference_explicitly_selected'
    # Partial rows from different raters can each be valid but form an invalid
    # selected chain. Reject offending selected edges rather than constructing
    # a reversed/overlapping reference or silently choosing a fallback rater.
    for _, selected_group in audit[audit.accepted].groupby('record_id'):
        selected_edges = selected_group.set_index('boundary')
        bad = set()
        for first, last in (('qrs_on','qrs_off'),('t_on','t_off'),('qrs_off','t_on')):
            if first in selected_edges.index and last in selected_edges.index:
                if selected_edges.loc[first,'reference_index'] >= selected_edges.loc[last,'reference_index']:
                    bad.update((first,last))
        bad_indices = selected_group.index[selected_group.boundary.isin(bad)]
        audit.loc[bad_indices,'accepted'] = False
        audit.loc[bad_indices,'acceptance_status'] = 'selected_reference_geometry_rejected'
        audit.loc[bad_indices,'rejection_reason'] = 'selected_boundaries_from_partial_rows_have_invalid_order'
    accepted = audit[audit.accepted].copy()
    if accepted.empty:
        return pd.DataFrame(columns=ACCEPTED_COLUMNS), audit
    accepted['patient_id'] = accepted.record_id.map(originals.patient_id)
    accepted['split'] = accepted.record_id.map(originals.split)
    return accepted[ACCEPTED_COLUMNS].reset_index(drop=True), audit


def interreviewer_disagreement(audit):
    columns = ['record_id', 'boundary', 'reviewer_a', 'reviewer_b',
               'reference_a_ms', 'reference_b_ms', 'difference_ms',
               'absolute_difference_ms', 'status']
    records = []
    if audit.empty:
        return pd.DataFrame(columns=columns)
    for (rid, boundary), group in audit[audit.candidate_accepted].groupby(['record_id', 'boundary']):
        # Repeated revisions by one reviewer are not independent raters.
        valid = group.groupby('reviewer').filter(lambda x: len(x) == 1)
        for (_, first), (_, second) in itertools.combinations(valid.iterrows(), 2):
            delta = float(first.reference_ms - second.reference_ms)
            records.append(dict(record_id=rid, boundary=boundary,
                                reviewer_a=first.reviewer, reviewer_b=second.reviewer,
                                reference_a_ms=first.reference_ms, reference_b_ms=second.reference_ms,
                                difference_ms=delta, absolute_difference_ms=abs(delta),
                                status='independent_rater_disagreement_not_algorithm_error'))
    return pd.DataFrame(records, columns=columns)


def temporal_overlap(pred_start, pred_end, ref_start, ref_end):
    """Continuous endpoint metric; durations do not add one sample period."""
    p0, p1, r0, r1 = map(float, (pred_start, pred_end, ref_start, ref_end))
    if not np.isfinite([p0, p1, r0, r1]).all() or p1 <= p0 or r1 <= r0:
        raise ValueError('Intervals must have finite strictly ordered endpoints')
    overlap = max(0.0, min(p1, r1) - max(p0, r0))
    pred_duration, ref_duration = p1 - p0, r1 - r0
    union = pred_duration + ref_duration - overlap
    return dict(predicted_duration_ms=pred_duration, reference_duration_ms=ref_duration,
                duration_error_ms=pred_duration-ref_duration, temporal_iou=overlap/union,
                intersection_ms=overlap, union_ms=union,
                missed_ms=ref_duration-overlap, excess_ms=pred_duration-overlap,
                reference_coverage=overlap/ref_duration,
                start_error_ms=p0-r0, end_error_ms=p1-r1)


def _groups(frame):
    yield 'all', 'all', frame
    yield 'train_valid', 'train_valid', frame[frame.split.isin(['train', 'valid'])]
    for column in ('label_name', 'split', 'qc_status'):
        for key, group in frame.groupby(column, dropna=False):
            yield column, str(key), group
    for key, group in frame.groupby(['label_name', 'split'], dropna=False):
        yield 'label_split', '|'.join(map(str, key)), group
    # A record with multiple QC flags enters each corresponding descriptive group.
    for flag in sorted({f.strip() for value in frame.qc_flags.fillna('')
                        for f in str(value).split(';') if f.strip()}):
        mask = frame.qc_flags.fillna('').map(lambda value: flag in [x.strip() for x in str(value).split(';')])
        yield 'qc_flag', flag, frame[mask]


def _coverage(group):
    eligible = group.evaluation_eligible.astype(bool)
    pred, ref = group.prediction_available.astype(bool), group.reference_available.astype(bool)
    n, nr = len(group), int(ref.sum())
    return dict(n_total=n, n_reference=nr, n_prediction=int(pred.sum()),
                n_evaluated=int(eligible.sum()), n_missing_prediction=int((~pred).sum()),
                n_missing_reference=int((~ref).sum()),
                n_reference_missing_prediction=int((ref & ~pred).sum()),
                paired_coverage=float(eligible.mean()) if n else np.nan,
                reference_coverage_rate=float(ref.mean()) if n else np.nan,
                prediction_failure_rate_of_cohort=float((~pred).mean()) if n else np.nan,
                prediction_failure_rate_on_references=float((ref & ~pred).sum()/nr) if nr else np.nan)


def _error_stats(values):
    values = pd.Series(values, dtype=float).dropna().to_numpy()
    absolute = np.abs(values)
    result = dict(mae_ms=np.nan, median_absolute_error_ms=np.nan, rmse_ms=np.nan,
                  p90_absolute_error_ms=np.nan, p95_absolute_error_ms=np.nan,
                  mean_signed_error_ms=np.nan)
    result.update({f'within_{threshold}_ms_rate': np.nan for threshold in (10, 20, 40, 60)})
    if len(values):
        result.update(mae_ms=float(absolute.mean()), median_absolute_error_ms=float(np.median(absolute)),
                      rmse_ms=float(np.sqrt(np.mean(values**2))), p90_absolute_error_ms=float(np.percentile(absolute,90)),
                      p95_absolute_error_ms=float(np.percentile(absolute,95)), mean_signed_error_ms=float(values.mean()))
        result.update({f'within_{threshold}_ms_rate': float(np.mean(absolute <= threshold)) for threshold in (10,20,40,60)})
    return result


def _summarize_boundary(table):
    records = []
    for boundary, boundaries in table.groupby('boundary', sort=False):
        for group_by, group, subset in _groups(boundaries):
            stats = _error_stats(subset.loc[subset.evaluation_eligible, 'error_ms'])
            n_reference = int(subset.reference_available.sum())
            for threshold in (10,20,40,60):
                n_within = int((subset.evaluation_eligible & subset.absolute_error_ms.le(threshold)).sum())
                stats[f'within_{threshold}_ms_numerator'] = n_within
                stats[f'within_{threshold}_ms_rate_of_references'] = n_within/n_reference if n_reference else np.nan
            stats['within_threshold_rate_denominator'] = 'n_evaluated paired predictions/references'
            stats['within_threshold_rate_of_references_denominator'] = 'n_reference including missing predictions as not within tolerance'
            records.append(dict(boundary=boundary, group_by=group_by, group=group,
                                **_coverage(subset), **stats))
    return pd.DataFrame(records)


def _summarize_loop(table):
    records = []
    for loop, loops in table.groupby('loop', sort=False):
        for group_by, group, subset in _groups(loops):
            good = subset[subset.evaluation_eligible]
            values = {f'{metric}_{stat}': float(getattr(good[metric], stat)()) if len(good) else np.nan
                      for metric in ('temporal_iou','duration_error_ms','missed_ms','excess_ms','reference_coverage')
                      for stat in ('mean','median')}
            records.append(dict(loop=loop, group_by=group_by, group=group,
                                **_coverage(subset), **values))
    return pd.DataFrame(records)


def _summarize_features(table):
    records = []
    for feature, features in table.groupby('feature', sort=False):
        for group_by, group, subset in _groups(features):
            good = subset[subset.evaluation_eligible]
            row = dict(feature=feature, group_by=group_by, group=group, **_coverage(subset),
                       n_near_zero_reference=int(subset.near_zero_reference.sum()),
                       n_relative_evaluated=int(good.relative_feature_error_pct.notna().sum()))
            for col in ('absolute_feature_error','relative_feature_error_pct','feature_error'):
                vals = good[col].dropna()
                row[col+'_mean'] = float(vals.mean()) if len(vals) else np.nan
                row[col+'_median'] = float(vals.median()) if len(vals) else np.nan
                row[col+'_p95'] = float(vals.quantile(.95)) if len(vals) else np.nan
            records.append(row)
    return pd.DataFrame(records)


def _feature_values(median, indices, cfg):
    xyz, fs = median['xyz'], median['fs']
    empty = dict.fromkeys(v1.FEATURE_NAMES, np.nan)
    if xyz is None:
        return empty
    speed = v1.spatial_speed(xyz, fs, cfg)
    cuts, speeds = {}, {}
    for loop, (first, last) in LOOPS.items():
        a, b = indices.get(first), indices.get(last)
        if a is None or b is None or not (0 <= a < b < len(xyz)):
            cuts[loop], speeds[loop] = None, None
        else:
            cuts[loop] = xyz[a:b+1]
            speeds[loop] = speed[a:b+1]
    return v1.extract_features(cuts['QRS'], cuts['T'], fs, cfg, speeds=speeds)


def evaluate_references(frame, annotations, artifact_root, cfg=None, allow_test=False):
    """Compute error only from accepted reference pairs, retaining all failures."""
    cfg = Config() if cfg is None else cfg
    frame = frame.copy()
    for key in META:
        if key not in frame:
            frame[key] = ''
    accepted, audit = validate_annotations(annotations, frame, artifact_root, allow_test)
    reference = {(r.record_id, r.boundary): r for r in accepted.itertuples(index=False)}
    b_rows, l_rows, f_rows, baseline_rows = [], [], [], []
    for _, row in frame.iterrows():
        meta = {key: row[key] for key in META}
        rid = row.record_id
        median = load_median(artifact_root, rid)
        pred_indices, ref_indices, local = {}, {}, {}
        for boundary in BOUNDARIES:
            predicted = _number(row.get(boundary + '_ms'))
            ix = median['boundaries'].get(boundary)
            pred_ix = None
            if median['xyz'] is not None and ix is not None and np.isfinite(_number(ix)):
                if int(ix) != ix or not 0 <= int(ix) < len(median['time_ms']):
                    raise ValueError(f'Invalid V1 boundary index: {rid}/{boundary}')
                pred_ix = int(ix)
                artifact_ms = float(median['time_ms'][pred_ix])
                if not np.isclose(predicted, artifact_ms, rtol=0, atol=1e-7):
                    raise ValueError(f'CSV/artifact prediction mismatch: {rid}/{boundary}')
            elif np.isfinite(predicted):
                raise ValueError(f'CSV has boundary without cached median/index: {rid}/{boundary}')
            pred_indices[boundary] = pred_ix
            candidate = reference.get((rid, boundary))
            ref_ms = candidate.reference_ms if candidate else np.nan
            ref_indices[boundary] = int(candidate.reference_index) if candidate else None
            pred_available = pred_ix is not None
            ref_available = candidate is not None
            paired = pred_available and ref_available
            status = 'VERIFIED_REFERENCE' if ref_available else 'GROUND_TRUTH_UNAVAILABLE'
            if row.split == 'test' and not allow_test:
                status = 'TEST_EVALUATION_DISABLED'
            values = dict(**meta, boundary=boundary, predicted_ms=predicted,
                          reference_ms=ref_ms,
                          requested_ref_ms=candidate.requested_ref_ms if candidate else np.nan,
                          quantization_error_ms=candidate.quantization_error_ms if candidate else np.nan,
                          predicted_index=pred_ix, reference_index=ref_indices[boundary],
                          error_ms=predicted-ref_ms if paired else np.nan,
                          absolute_error_ms=abs(predicted-ref_ms) if paired else np.nan,
                          reference_status=status, prediction_available=pred_available,
                          reference_available=ref_available, evaluation_eligible=paired,
                          reviewer=candidate.reviewer if candidate else '',
                          reference_annotation_row=candidate.annotation_row if candidate else np.nan,
                          reference_selection=candidate.reference_selection if candidate else '',
                          reference_source=candidate.reference_source if candidate else '')
            b_rows.append(values)
            local[boundary] = values
        pred_features = _feature_values(median, pred_indices, cfg)
        ref_features = _feature_values(median, ref_indices, cfg)
        for feature in v1.FEATURE_NAMES:
            saved = _number(row.get(feature))
            recomputed = pred_features[feature]
            match = (not np.isfinite(saved) and not np.isfinite(recomputed)) or np.isclose(saved, recomputed, rtol=1e-8, atol=1e-10)
            baseline_rows.append(dict(record_id=rid, feature=feature, csv_value=saved,
                                      recomputed_value=recomputed, matches=bool(match)))
            if not match:
                raise ValueError(f'Frozen V1 feature does not reproduce CSV: {rid}/{feature}')
            loop = 'QRS' if feature.startswith('QRS') else 'T'
            first, last = LOOPS[loop]
            has_ref = all(ref_indices[b] is not None for b in (first, last))
            has_pred = np.isfinite(recomputed)
            ref_value = ref_features[feature]
            paired = bool(has_pred and has_ref and np.isfinite(ref_value))
            near_zero = bool(np.isfinite(ref_value) and abs(ref_value) <= EPSILON)
            error = recomputed-ref_value if paired else np.nan
            sources = sorted({local[b]['reference_source'] for b in (first,last) if local[b]['reference_source']})
            reviewers = sorted({local[b]['reviewer'] for b in (first,last) if local[b]['reviewer']})
            annotation_rows = sorted({int(local[b]['reference_annotation_row']) for b in (first,last)
                                      if np.isfinite(local[b]['reference_annotation_row'])})
            interval_kind = ('unavailable' if not annotation_rows else 'partial_reference' if not has_ref else
                             'single_annotation' if len(annotation_rows)==1 else 'composite_selected_edges')
            f_rows.append(dict(**meta, feature=feature, loop=loop, predicted=recomputed,
                               reference=ref_value, feature_error=error,
                               absolute_feature_error=abs(error) if paired else np.nan,
                               relative_feature_error_pct=abs(error)/abs(ref_value)*100 if paired and not near_zero else np.nan,
                               near_zero_reference=near_zero, prediction_available=bool(has_pred),
                               reference_available=bool(has_ref), evaluation_eligible=paired,
                               status=('VERIFIED_REFERENCE' if paired else
                                       'GROUND_TRUTH_UNAVAILABLE' if not has_ref else
                                       'REFERENCE_LOOP_TOO_SHORT_FOR_V1_FEATURES' if not np.isfinite(ref_value) else
                                       'PREDICTION_UNAVAILABLE'),
                               reference_source='; '.join(sources), reviewer='; '.join(reviewers),
                               reference_annotation_rows='; '.join(map(str,annotation_rows)),
                               reference_interval_kind=interval_kind))
        for loop, (first,last) in LOOPS.items():
            p0,p1,r0,r1 = (local[first]['predicted_ms'],local[last]['predicted_ms'],
                            local[first]['reference_ms'],local[last]['reference_ms'])
            has_pred = all(pred_indices[b] is not None for b in (first,last)) and p0<p1
            has_ref = all(ref_indices[b] is not None for b in (first,last)) and r0<r1
            paired = bool(has_pred and has_ref)
            metrics = dict.fromkeys(('predicted_duration_ms','reference_duration_ms','duration_error_ms',
                                    'temporal_iou','intersection_ms','union_ms','missed_ms','excess_ms',
                                    'reference_coverage','start_error_ms','end_error_ms'), np.nan)
            if has_pred:
                metrics['predicted_duration_ms'] = p1-p0
            if has_ref:
                metrics['reference_duration_ms'] = r1-r0
            if paired:
                metrics.update(temporal_overlap(p0,p1,r0,r1))
            sources = sorted({local[b]['reference_source'] for b in (first,last) if local[b]['reference_source']})
            reviewers = sorted({local[b]['reviewer'] for b in (first,last) if local[b]['reviewer']})
            annotation_rows = sorted({int(local[b]['reference_annotation_row']) for b in (first,last)
                                      if np.isfinite(local[b]['reference_annotation_row'])})
            interval_kind = ('unavailable' if not annotation_rows else 'partial_reference' if not has_ref else
                             'single_annotation' if len(annotation_rows)==1 else 'composite_selected_edges')
            l_rows.append(dict(**meta, loop=loop, predicted_start_ms=p0,predicted_end_ms=p1,
                               reference_start_ms=r0,reference_end_ms=r1,
                               predicted_start_index=pred_indices[first],predicted_end_index=pred_indices[last],
                               reference_start_index=ref_indices[first],reference_end_index=ref_indices[last],
                               predicted_sample_count=pred_indices[last]-pred_indices[first]+1 if has_pred else np.nan,
                               reference_sample_count=ref_indices[last]-ref_indices[first]+1 if has_ref else np.nan,
                               **metrics, prediction_available=bool(has_pred),reference_available=bool(has_ref),
                               evaluation_eligible=paired,
                               status='VERIFIED_REFERENCE' if paired else ('PREDICTION_UNAVAILABLE' if has_ref else 'GROUND_TRUTH_UNAVAILABLE'),
                               reference_source='; '.join(sources), reviewer='; '.join(reviewers),
                               reference_annotation_rows='; '.join(map(str,annotation_rows)),
                               reference_interval_kind=interval_kind))
    boundaries = pd.DataFrame(b_rows)
    loops = pd.DataFrame(l_rows)
    features = pd.DataFrame(f_rows)
    # Average recordings within patient before any across-patient summary. These
    # patient tables expose denominators and avoid pretending recordings are iid.
    patient_boundaries = boundaries.groupby(['patient_id','label_name','split','boundary'], dropna=False).agg(
        n_records=('record_id','size'),n_reference=('reference_available','sum'),
        n_evaluated=('evaluation_eligible','sum'),patient_mean_signed_error_ms=('error_ms','mean'),
        patient_mae_ms=('absolute_error_ms','mean')).reset_index()
    patient_features = features.groupby(['patient_id','label_name','split','feature'], dropna=False).agg(
        n_records=('record_id','size'),n_reference=('reference_available','sum'),
        n_evaluated=('evaluation_eligible','sum'),patient_mean_absolute_feature_error=('absolute_feature_error','mean'),
        patient_mean_relative_feature_error_pct=('relative_feature_error_pct','mean')).reset_index()
    return dict(boundary_error_per_record=boundaries,boundary_error_summary=_summarize_boundary(boundaries),
                loop_overlap_per_record=loops,loop_overlap_summary=_summarize_loop(loops),
                feature_reference_comparison=features,feature_error_summary=_summarize_features(features),
                annotation_acceptance_audit=audit,accepted_references=accepted,
                interreviewer_disagreement=interreviewer_disagreement(audit),
                patient_boundary_summary=patient_boundaries,patient_feature_summary=patient_features,
                baseline_reproduction_check=pd.DataFrame(baseline_rows))
