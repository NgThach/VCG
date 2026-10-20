"""Frozen-V1 feature response to one boundary displacement at a time.

This is a technical sensitivity experiment, not a boundary accuracy estimate.
``run_feature_sensitivity`` returns DataFrames under ``long``, ``scenarios``,
``summary``, ``rank`` and ``baseline_check``, plus a ``metadata`` dictionary.
Every input record contributes 16 scenarios and 192 long rows, including records
without a median beat or without T boundaries. Only a feature's own loop boundary
experiments enter its summaries/ranks. Inputs and Notebook 06 CSVs are read only.
"""
from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import vcg_pipeline as v1
from vcg_preprocessing import Config

EPSILON = 1e-9  # Added in the native unit of EACH feature; not a clinical threshold.
SHIFTS_MS = (-20, -10, 10, 20)
BOUNDARIES = ('qrs_on', 'qrs_off', 't_on', 't_off')
FEATURE_UNITS = {name: ('mV/s' if name.endswith(('MaxVel', 'MeanVel', 'StdVel'))
                        else 'mV^2' if name.endswith('Area') else 'mV')
                 for name in v1.FEATURE_NAMES}
BASELINE_RTOL = 1e-8
BASELINE_ATOL = 1e-10
SCHEMA_VERSION = 1


def _sha(path):
    path = Path(path)
    if not path.exists():
        return 'missing'
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical(value):
    return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _index(value):
    if value is None or not np.isfinite(value):
        return None
    if float(value) != int(value):
        raise ValueError(f'Boundary index is not an integer: {value}')
    return int(value)


def validate_scenario(boundaries, n, anchor, fs, changed_boundary, cfg=Config()):
    """Reject invalid geometry; never clamp, move another edge, or impose 60 ms.

    A QRS-only record can still undergo QRS experiments. All known cross-loop
    relations are checked, but missing T boundaries are explicitly reported in
    ``validation_scope`` rather than invented to make a full chain.
    """
    b = boundaries
    target_loop = 'QRS' if changed_boundary.startswith('qrs') else 'T'
    required = ('qrs_on', 'qrs_off') if target_loop == 'QRS' else (
        'qrs_on', 'qrs_off', 't_on', 't_peak', 't_off')
    absent = [key for key in required if b.get(key) is None]
    if absent:
        return False, 'missing_boundaries:' + ','.join(absent), 'unavailable'
    for key in ('qrs_on', 'qrs_off', 't_on', 't_peak', 't_off'):
        ix = b.get(key)
        if ix is not None and not 0 <= ix < n:
            return False, f'index_out_of_range:{key}', 'rejected'
    if anchor is None or not 0 <= anchor < n:
        return False, 'anchor_missing_or_out_of_range', 'rejected'
    if not b['qrs_on'] < anchor < b['qrs_off']:
        return False, 'qrs_on_anchor_qrs_off_order', 'rejected'
    min_samples = max(5, v1.ms_samples(cfg.derivative_window_ms, fs))
    if b['qrs_off'] - b['qrs_on'] + 1 < min_samples:
        return False, 'qrs_loop_too_short_for_v1_features', 'rejected'
    if b.get('t_on') is not None and not b['qrs_off'] < b['t_on']:
        return False, 'qrs_off_t_on_order', 'rejected'
    if b.get('t_on') is not None and b.get('t_peak') is not None:
        if not b['t_on'] < b['t_peak']:
            return False, 't_on_t_peak_order', 'rejected'
    if b.get('t_peak') is not None and b.get('t_off') is not None:
        if not b['t_peak'] < b['t_off']:
            return False, 't_peak_t_off_order', 'rejected'
    if b.get('t_on') is not None and b.get('t_off') is not None:
        if b['t_off'] - b['t_on'] + 1 < min_samples:
            return False, 't_loop_too_short_for_v1_features', 'rejected'
    complete_t = all(b.get(key) is not None for key in ('t_on', 't_peak', 't_off'))
    return True, '', 'full_qrs_t_chain' if complete_t else 'qrs_only_missing_t'


def compute_v1_features(xyz, boundaries, fs, cfg=Config(), full_speed=None):
    """Use V1 cuts and the SG derivative computed on the FULL median beat.

    Recomputing SG on a cropped loop changes endpoint derivatives and does not
    reproduce ``process_record``. The optional speed is a full-beat array only.
    """
    if full_speed is None:
        full_speed = v1.spatial_speed(xyz, fs, cfg)
    if len(full_speed) != len(xyz):
        raise ValueError('Full-median speed length differs from median XYZ')
    qrs = v1.cut_loop(xyz, boundaries.get('qrs_on'), boundaries.get('qrs_off'))
    tloop = v1.cut_loop(xyz, boundaries.get('t_on'), boundaries.get('t_off'))
    speeds = {}
    for name, loop, a, z in [('QRS', qrs, 'qrs_on', 'qrs_off'),
                              ('T', tloop, 't_on', 't_off')]:
        speeds[name] = None if loop is None else full_speed[
            boundaries[a]:boundaries[z] + 1]
    return v1.extract_features(qrs, tloop, fs, cfg, speeds=speeds)


def feature_change(baseline, perturbed, epsilon=EPSILON):
    """Absolute/relative changes in native units; a near-zero marker is retained."""
    baseline = float(baseline) if baseline is not None else np.nan
    perturbed = float(perturbed) if perturbed is not None else np.nan
    near_zero = bool(np.isfinite(baseline) and abs(baseline) <= epsilon)
    if not np.isfinite(baseline) or not np.isfinite(perturbed):
        return np.nan, np.nan, near_zero
    absolute = abs(perturbed - baseline)
    return absolute, absolute / (abs(baseline) + epsilon), near_zero


def _record_experiment(row, artifact_root, cfg):
    common = {key: row.get(key) for key in ('record_id', 'patient_id', 'label_name',
                                           'label', 'split')}
    common.update(qc_status=row.get('qc_status', ''), qc_flags=row.get('qc_flags', ''),
                  original_feature_completeness=row.get('feature_completeness', np.nan))
    safe = str(row['record_id']).replace('/', '_')
    json_path = artifact_root / (safe + '.json')
    npz_path = artifact_root / (safe + '.npz')
    artifact = json.loads(json_path.read_text()) if json_path.exists() else {}
    if artifact and artifact.get('record_id') != row['record_id']:
        raise ValueError(f'Artifact record mismatch for {row["record_id"]}')
    if artifact and artifact.get('time_reference') != 'median_relative_ms':
        raise ValueError(f'Unsupported artifact time reference for {row["record_id"]}')
    fs = artifact.get('fs') or row.get('fs')
    anchor = _index(artifact.get('median_anchor_sample'))
    b = {name: _index(artifact.get('boundaries', {}).get(name)) for name in
         ('qrs_on', 'qrs_off', 't_on', 't_peak', 't_off')}
    xyz, saved_time = None, None
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as arrays:
            if 'median_xyz' in arrays:
                xyz = np.array(arrays['median_xyz'], dtype=float)
                saved_time = np.array(arrays['time_ms'], dtype=float)
    unavailable = ''
    baseline = dict.fromkeys(v1.FEATURE_NAMES, np.nan)
    if not artifact:
        unavailable = 'artifact_json_missing'
    elif xyz is None:
        unavailable = 'median_beat_unavailable:' + str(artifact.get('failure_stage', 'unknown'))
    elif (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
          or not np.isfinite(fs) or fs <= 0):
        raise ValueError(f'Invalid saved median or sample rate: {row["record_id"]}')
    else:
        expected_time = (np.arange(len(xyz)) - anchor) * 1000.0 / fs
        if saved_time.shape != expected_time.shape or not np.allclose(
                saved_time, expected_time, rtol=0, atol=1e-7):
            raise ValueError(f'Median time reference mismatch: {row["record_id"]}')
        for name, ix in b.items():
            csv_ms = row.get(name + '_ms', np.nan)
            if ix is not None and pd.notna(csv_ms) and not np.isclose(
                    saved_time[ix], csv_ms, rtol=0, atol=1e-7):
                raise ValueError(f'CSV/artifact boundary mismatch: {row["record_id"]} {name}')
        speed = v1.spatial_speed(xyz, fs, cfg)
        baseline = compute_v1_features(xyz, b, fs, cfg, full_speed=speed)
    checks = []
    for feature in v1.FEATURE_NAMES:
        csv_value = row.get(feature, np.nan)
        csv_value = float(csv_value) if pd.notna(csv_value) else np.nan
        value = baseline[feature]
        match = bool(np.isclose(value, csv_value, rtol=BASELINE_RTOL,
                                atol=BASELINE_ATOL, equal_nan=True))
        checks.append(dict(common, feature=feature, unit=FEATURE_UNITS[feature],
                           csv_value=csv_value, recomputed_value=value,
                           absolute_difference=abs(value - csv_value), matches_csv=match,
                           status='numeric_match' if match and np.isfinite(value)
                           else 'both_unavailable' if match else 'mismatch',
                           reason=unavailable))
    scenarios, long = [], []
    for boundary in BOUNDARIES:
        for shift in SHIFTS_MS:
            shifted = b.copy()
            samples = int(round(shift * fs / 1000.0)) if fs else None
            eligible = bool(not unavailable and b[boundary] is not None
                            and any(np.isfinite(baseline[f]) for f in v1.FEATURE_NAMES
                                    if f.startswith('QRS' if boundary.startswith('qrs') else 'T')))
            valid, reason, scope = False, unavailable, 'unavailable'
            if not unavailable:
                if shifted[boundary] is None:
                    reason = 'missing_boundary:' + boundary
                else:
                    shifted[boundary] += samples
                    valid, reason, scope = validate_scenario(shifted, len(xyz), anchor,
                                                             fs, boundary, cfg)
            perturbed = (compute_v1_features(xyz, shifted, fs, cfg, full_speed=speed)
                         if valid else dict.fromkeys(v1.FEATURE_NAMES, np.nan))
            scenario = dict(common, boundary=boundary, shift_ms=shift,
                            shift_samples=samples, actual_shift_ms=(samples * 1000 / fs
                            if fs else np.nan), eligible_scenario=eligible,
                            valid_scenario=valid, status='valid' if valid else 'unavailable'
                            if not eligible else 'invalid_geometry', reason=reason,
                            validation_scope=scope, fs=fs,
                            minimum_loop_samples=(max(5, v1.ms_samples(
                                cfg.derivative_window_ms, fs)) if fs else np.nan))
            for name in ('qrs_on', 'qrs_off', 't_on', 't_peak', 't_off'):
                scenario[name + '_index'] = shifted[name]
                scenario[name + '_ms'] = ((shifted[name] - anchor) * 1000 / fs
                                           if shifted[name] is not None and anchor is not None
                                           and fs else np.nan)
            scenarios.append(scenario)
            for feature in v1.FEATURE_NAMES:
                value, changed = baseline[feature], perturbed[feature]
                absolute, relative, near_zero = feature_change(value, changed)
                feature_loop = 'QRS' if feature.startswith('QRS') else 'T'
                own_loop = feature_loop == ('QRS' if boundary.startswith('qrs') else 'T')
                valid_feature = bool(valid and np.isfinite(value) and np.isfinite(changed))
                long.append(dict(common, boundary=boundary, shift_ms=shift,
                    shift_samples=samples, actual_shift_ms=scenario['actual_shift_ms'],
                    feature=feature, feature_loop=feature_loop, own_loop_boundary=own_loop,
                    unit=FEATURE_UNITS[feature], baseline=value, perturbed=changed,
                    absolute_delta=absolute, relative_delta=relative, epsilon=EPSILON,
                    base_near_zero=near_zero, eligible_scenario=eligible,
                    eligible_feature=bool(eligible and np.isfinite(value)),
                    valid_scenario=valid, valid_feature=valid_feature,
                    status=scenario['status'] if not valid else 'valid' if valid_feature
                    else 'feature_unavailable', reason=reason if not valid else ''
                    if valid_feature else 'baseline_or_perturbed_feature_unavailable',
                    validation_scope=scope))
    return {'long': long, 'scenarios': scenarios, 'baseline_check': checks}


def _groups(own):
    for scope in ('all', 'train_valid'):
        sub = own if scope == 'all' else own[own['split'].isin(['train', 'valid'])]
        yield scope, 'all', 'ALL', 'ALL', sub
        for label, part in sub.groupby('label_name', sort=True):
            yield scope, 'label', label, 'ALL', part
        for split, part in sub.groupby('split', sort=True):
            yield scope, 'split', 'ALL', split, part
        for (label, split), part in sub.groupby(['label_name', 'split'], sort=True):
            yield scope, 'label_split', label, split, part


def _stats(part):
    valid = part[part['valid_feature']]
    absolute, relative = valid['absolute_delta'], valid['relative_delta']
    result = {'n_records_total': part['record_id'].nunique(),
              'n_records_eligible': part.loc[part.eligible_feature, 'record_id'].nunique(),
              'n_records_valid': valid['record_id'].nunique(),
              'n_scenarios_total': len(part),
              'n_eligible': int(part.eligible_feature.sum()),
              'n_valid': len(valid), 'n_invalid': len(part) - len(valid),
              'n_unavailable': int((~part.eligible_feature).sum()),
              'n_eligible_but_invalid': int((part.eligible_feature & ~part.valid_feature).sum()),
              'n_base_near_zero': int(part.base_near_zero.sum()),
              'n_base_near_zero_valid': int(valid.base_near_zero.sum())}
    for name, values in [('absolute_delta', absolute), ('relative_delta', relative)]:
        result.update({name + '_mean': values.mean(), name + '_median': values.median(),
                       name + '_p90': values.quantile(.9), name + '_p95': values.quantile(.95),
                       name + '_max': values.max()})
    return result


def summarize_sensitivity(long):
    """Own-loop scenarios only; keep attempted and unavailable denominators."""
    summary, ranks = [], []
    own = long[long.own_loop_boundary]
    for scope, grouping, label, split, part in _groups(own):
        common = dict(scope=scope, grouping=grouping, label_name=label, split=split,
                      own_loop_boundaries_only=True, epsilon=EPSILON)
        for (feature, boundary, shift), block in part.groupby(
                ['feature', 'boundary', 'shift_ms'], sort=True):
            summary.append(dict(common, feature=feature, boundary=boundary, shift_ms=shift,
                                unit=FEATURE_UNITS[feature], **_stats(block)))
        for feature, block in part.groupby('feature', sort=True):
            ranks.append(dict(common, feature=feature, unit=FEATURE_UNITS[feature],
                              **_stats(block)))
    summary, ranks = pd.DataFrame(summary), pd.DataFrame(ranks)
    if not ranks.empty:
        by = ['scope', 'grouping', 'label_name', 'split']
        ranks['relative_median_rank'] = ranks.groupby(by)['relative_delta_median'].rank(
            method='min', ascending=False)
        ranks['relative_p95_rank'] = ranks.groupby(by)['relative_delta_p95'].rank(
            method='min', ascending=False)
        ranks = ranks.sort_values(by + ['relative_median_rank', 'feature']).reset_index(drop=True)
    return summary, ranks


def run_feature_sensitivity(frame, artifact_root: Path, output_dir: Path,
                            cfg=Config(), resume=True):
    """Run/cache every record and return the five tables plus metadata.

    Cache keys include the source modules, configuration, entire input row and
    content hashes of BOTH saved artifacts. Changed inputs cannot reuse stale
    record results. Numeric mismatch against Notebook 06 fails the run explicitly.
    ``n_invalid`` includes unavailable attempts; ``n_eligible_but_invalid`` isolates
    rejected shifts among computable baseline features.
    """
    required = ['record_id', 'patient_id', 'label_name', 'label', 'split'] + v1.FEATURE_NAMES
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError('Missing sensitivity input columns: ' + ', '.join(missing))
    if frame.empty or frame.record_id.duplicated().any():
        raise ValueError('Sensitivity requires nonempty, unique record IDs')
    if not frame['split'].isin(['train', 'valid', 'test']).all():
        raise ValueError('Unexpected split; existing train/valid/test must be preserved')
    artifact_root, output_dir = Path(artifact_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / 'sensitivity_checkpoints'
    cache_dir.mkdir(exist_ok=True)
    sources = {p.name: _sha(p) for p in [Path(__file__), REPO / 'vcg_pipeline.py',
                                        REPO / 'vcg_preprocessing.py']}
    experiment = dict(schema_version=SCHEMA_VERSION, config=asdict(cfg), shifts_ms=SHIFTS_MS,
        boundaries=BOUNDARIES, epsilon=EPSILON, epsilon_units='native unit of each feature',
        near_zero_rule='abs(baseline) <= epsilon', baseline_rtol=BASELINE_RTOL,
        baseline_atol=BASELINE_ATOL, feature_units=FEATURE_UNITS, source_sha256=sources,
        derivative_rule='SG on full median XYZ, then slice speed for each changed loop',
        ranking_rule='Only features of the displaced boundary loop enter summaries/ranks',
        validation='No clamping; qrs_on < anchor < qrs_off < t_on < t_peak < t_off; '
        'QRS-only experiments allow unavailable T; minimum samples equals V1 feature gate',
        interpretation='Technical boundary sensitivity; not independent boundary accuracy')
    experiment_hash = _digest(experiment)
    tables = {key: [] for key in ('long', 'scenarios', 'baseline_check')}
    cache_hits, record_hashes = 0, {}
    for row in frame.to_dict('records'):
        safe = str(row['record_id']).replace('/', '_')
        signature = _digest(dict(experiment_hash=experiment_hash, input_row=row,
            artifact_json_sha256=_sha(artifact_root / (safe + '.json')),
            artifact_npz_sha256=_sha(artifact_root / (safe + '.npz'))))
        record_hashes[row['record_id']] = signature
        checkpoint = cache_dir / (safe + '.json.gz')
        saved = None
        if resume and checkpoint.exists():
            try:
                with gzip.open(checkpoint, 'rt', encoding='utf-8') as handle:
                    candidate = json.load(handle)
                if candidate.get('signature') == signature:
                    saved = candidate['tables']
                    cache_hits += 1
            except (OSError, ValueError, KeyError):
                saved = None
        if saved is None:
            saved = _clean(_record_experiment(row, artifact_root, cfg))
            temporary = checkpoint.with_suffix('.tmp')
            with gzip.open(temporary, 'wt', encoding='utf-8') as handle:
                json.dump(dict(signature=signature, tables=saved), handle,
                          ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            os.replace(temporary, checkpoint)
        for key in tables:
            tables[key].extend(saved[key])
    result = {key: pd.DataFrame(value) for key, value in tables.items()}
    # Cache JSON nulls must return to numeric NaN for identical cold/resumed tables.
    for key, numeric_columns in {
        'long': ['baseline', 'perturbed', 'absolute_delta', 'relative_delta', 'shift_samples',
                 'actual_shift_ms', 'epsilon', 'original_feature_completeness'],
        'baseline_check': ['csv_value', 'recomputed_value', 'absolute_difference'],
        'scenarios': ['shift_samples', 'actual_shift_ms', 'fs', 'minimum_loop_samples'] +
                     [name + suffix for name in ('qrs_on', 'qrs_off', 't_on', 't_peak',
                                                  't_off') for suffix in ('_ms', '_index')]
    }.items():
        for column in numeric_columns:
            result[key][column] = pd.to_numeric(result[key][column], errors='coerce')
    result['baseline_check'].to_csv(output_dir / 'sensitivity_baseline_check.csv', index=False)
    bad = result['baseline_check'][~result['baseline_check'].matches_csv]
    if len(bad):
        pairs = bad[['record_id', 'feature']].head(10).to_dict('records')
        raise ValueError(f'Frozen V1 baseline differs from input CSV ({len(bad)} values): {pairs}')
    expected_scenarios = len(frame) * len(BOUNDARIES) * len(SHIFTS_MS)
    if len(result['scenarios']) != expected_scenarios or len(result['long']) != (
            expected_scenarios * len(v1.FEATURE_NAMES)):
        raise RuntimeError('Sensitivity denominator changed unexpectedly')
    result['summary'], result['rank'] = summarize_sensitivity(result['long'])
    for key, filename in [('long', 'feature_sensitivity.csv'),
                           ('scenarios', 'sensitivity_scenarios.csv'),
                           ('summary', 'feature_sensitivity_summary.csv'),
                           ('rank', 'feature_sensitivity_rank.csv')]:
        result[key].to_csv(output_dir / filename, index=False)
    metadata = dict(experiment, experiment_hash=experiment_hash,
                    input_frame_hash=_digest(frame.to_dict('records')),
                    artifact_root=str(artifact_root.resolve()), record_signatures=record_hashes,
                    n_records=len(frame), n_scenarios=expected_scenarios,
                    n_long_rows=len(result['long']), cache_hits=cache_hits,
                    n_baseline_numeric_matches=int(result['baseline_check'].status.eq('numeric_match').sum()),
                    n_baseline_both_unavailable=int(result['baseline_check'].status.eq('both_unavailable').sum()))
    (output_dir / 'sensitivity_config.json').write_text(
        json.dumps(_clean(metadata), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    result['metadata'] = metadata
    return result
