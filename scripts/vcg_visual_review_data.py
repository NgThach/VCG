"""Read-only export of saved Notebook06 V1 signals for offline visual review.

Saved median XYZ and saved predictions are always the display source. Re-running
``delineate_loops`` checks provenance and supplies explicitly labelled missing
intermediates; it never replaces a saved boundary. A failed record remains a
record in both the manifest and the JavaScript payload set.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vcg_pipeline import cut_loop, delineate_loops, ms_samples, robust_sigma
from vcg_preprocessing import Config

SCHEMA_VERSION = "saved-v1-visual-review-1"
BOUNDARIES = ("qrs_on", "qrs_off", "t_on", "t_peak", "t_off")
INDEX_FIELDS = BOUNDARIES + ("qrs_speed_peak", "t_rise_index", "t_fall_index",
    "t_tangent_on", "t_tangent_off", "t_threshold_on", "t_threshold_off")
ARRAY_FIELDS = ("speed", "smooth_speed", "magnitude", "t_magnitude", "t_slope")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean(value):
    """Strict JSON: no NaN, no Infinity, no zero substituted for absent data."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or value is pd.NA:
        return None
    return value


def _json(value):
    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=True)


def _write_json(path, value):
    Path(path).write_text(json.dumps(_clean(value), ensure_ascii=False,
        allow_nan=False, indent=2) + "\n", encoding="utf-8")


def sample_time_ms(sample, fs, anchor):
    """A median-local sample uses this record's fs and median-local anchor."""
    if sample is None:
        return None
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be finite and positive")
    return (int(sample) - int(anchor)) * 1000.0 / float(fs)


def _index(value, n):
    return (value is not None and isinstance(value, (int, np.integer))
            and 0 <= int(value) < n)


def _equal(saved, recomputed):
    if saved is None or recomputed is None:
        # The saved JSON encodes V1's scalar NaN as null. They are the same
        # unavailable value, not a changed prediction or intermediate.
        def missing_scalar(value):
            return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
        return missing_scalar(saved) and missing_scalar(recomputed), None
    if isinstance(saved, (str, bool)):
        return saved == recomputed, None
    try:
        a, b = np.asarray(saved, float), np.asarray(recomputed, float)
        if a.shape != b.shape:
            return False, None
        same = bool(np.array_equal(a, b, equal_nan=True))
        finite = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
        return same, delta
    except (ValueError, TypeError):
        return saved == recomputed, None


def _base_meta(row, artifact=None):
    artifact = artifact or {}
    return dict(record_id=row["record_id"], patient_id=row.get("patient_id"),
        label_name=row.get("label_name"), label=row.get("label"), split=row.get("split"),
        fs=artifact.get("fs", row.get("fs")), qc_status=row.get("qc_status", "unknown"),
        qc_flags="" if pd.isna(row.get("qc_flags", "")) else row.get("qc_flags", ""),
        failure_stage=artifact.get("failure_stage") or row.get("failure_stage") or "",
        time_reference="median_relative_ms", signal_units="mV", velocity_units="mV/s",
        anchor_sample=artifact.get("median_anchor_sample"),
        prediction_source="Saved Notebook06 V1 artifacts; never replaced by recomputation",
        manual_review_interpretation="Visual review; not independently verified ground truth")


def _unavailable_payload(row, artifact, reason, provenance):
    meta = _base_meta(row, artifact)
    meta["segmentation_status"] = "median_missing"
    return dict(schema_version=SCHEMA_VERSION, record_id=row["record_id"],
        available=False, reason=reason, meta=meta, time_ms=[], xyz=[], speed=[],
        smooth_speed=[], magnitude=[], t_magnitude=[], t_slope=None,
        boundaries=artifact.get("boundaries", {}), boundary_times_ms={},
        qrs_search=None, pq=artifact.get("pq_indices"), qrs_loop_indices=None,
        t_loop_indices=None, parity={"status": "not_available", "reason": reason},
        provenance=provenance)


def build_visual_record(row, artifact_root, cfg=None):
    """Return payload, per-field parity rows and integrity issues for one record.

    CSV decimal serialization is checked at 1e-9 ms; saved sample indices and the
    stored time grid are compared exactly. All values displayed still come from
    saved arrays/JSON even if recomputation disagrees.
    """
    cfg = cfg or Config()
    row = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    safe = str(row["record_id"]).replace("/", "_")
    artroot = Path(artifact_root)
    if (artroot / "artifacts").is_dir():
        artroot = artroot / "artifacts"
    jp, zp = artroot / (safe + ".json"), artroot / (safe + ".npz")
    provenance = {"artifact_json": str(jp), "artifact_npz": str(zp),
        "artifact_json_sha256": _sha(jp) if jp.is_file() else None,
        "artifact_npz_sha256": _sha(zp) if zp.is_file() else None,
        "config_v1": asdict(cfg), "source": "Saved V1 median, PQ and predictions",
        "loop_cut": "xyz[on:off+1], separately for QRS and T; ST gap excluded",
        "normalization": "Saved representative median XYZ already centered on PQ; no recentering"}
    art = json.loads(jp.read_text(encoding="utf-8"))
    if art.get("record_id") != row["record_id"]:
        raise ValueError("Artifact record_id differs from input record_id")
    with np.load(zp, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    if "median_xyz" not in arrays:
        reason = art.get("error") or "No saved representative median beat"
        return _unavailable_payload(row, art, reason, provenance), [], []
    xyz, time = arrays["median_xyz"], arrays["time_ms"]
    n, fs = len(xyz), float(art["fs"])
    anchor = int(art["median_anchor_sample"])
    issues = []
    def issue(message):
        issues.append({"record_id": row["record_id"], "issue": message})
    if xyz.shape != (n, 3) or time.shape != (n,) or not np.isfinite(xyz).all() or not np.isfinite(time).all():
        raise ValueError("Saved median XYZ/time must be finite, aligned and (n,3)/(n,)")
    expected_time = (np.arange(n) - anchor) * 1000.0 / fs
    if not np.array_equal(time, expected_time):
        issue("Saved time grid differs from exact (sample-anchor)*1000/fs")
    if not _index(anchor, n):
        raise ValueError("Saved median anchor outside median")
    if float(row.get("fs", fs)) != fs:
        issue("CSV sampling frequency differs from artifact sampling frequency")
    if art.get("time_reference") != "median_relative_ms":
        issue("Artifact time reference is not median_relative_ms")
    boundaries = dict(art.get("boundaries", {}))
    times = {}
    for name in INDEX_FIELDS:
        value = boundaries.get(name)
        if value is None:
            times[name] = None
        elif _index(value, n):
            times[name] = float(time[value])
        else:
            times[name] = None
            issue(f"Saved {name} is not a valid median-local sample")
        column = name + "_ms"
        if column in row:
            csv_value = row[column]
            csv_missing = csv_value is None or pd.isna(csv_value)
            if (times[name] is None) != csv_missing:
                issue(f"CSV/artifact missingness mismatch: {column}")
            elif times[name] is not None and not np.isclose(times[name], float(csv_value), atol=1e-9, rtol=0):
                issue(f"CSV/artifact prediction mismatch: {column}")
    pq = tuple(int(v) for v in art["pq_indices"])
    if not 0 <= pq[0] < pq[1] <= n:
        raise ValueError("Saved PQ slice outside median")
    rr = np.asarray(art.get("rr_seconds", []), float)
    if not len(rr) or not np.isfinite(rr).all():
        raise ValueError("No valid saved RR for exact delineation parity check")
    rep = dict(xyz=xyz, time_ms=time, anchor=anchor, pq=pq,
               rr_seconds=float(np.median(rr)))
    parity = []
    recomputed = {}
    recompute_error = ""
    try:
        recomputed = delineate_loops(rep, fs, cfg)
        for key, saved_value in boundaries.items():
            if key not in recomputed:
                continue
            match, delta = _equal(saved_value, recomputed[key])
            parity.append(dict(record_id=row["record_id"], field=key, kind="boundary_or_scalar",
                matches_exactly=match, max_abs_difference=delta,
                saved_value=_json(saved_value), recomputed_value=_json(recomputed[key])))
        for key in ARRAY_FIELDS:
            if key not in arrays:
                continue
            match, delta = _equal(arrays[key], recomputed.get(key))
            parity.append(dict(record_id=row["record_id"], field=key, kind="intermediate_array",
                matches_exactly=match, max_abs_difference=delta,
                saved_value="saved NPZ array", recomputed_value="same full-median V1 function"))
    except Exception as exc:
        recompute_error = f"{type(exc).__name__}: {exc}"
        issue("Delineation parity could not be checked: " + recompute_error)
    signal_arrays, array_sources = {}, {}
    for key in ARRAY_FIELDS:
        if key in arrays:
            values, source = arrays[key], "saved_v1_npz"
        elif recomputed.get(key) is not None:
            values, source = recomputed[key], "recomputed_intermediate_same_full_median_v1"
        else:
            values, source = None, "unavailable"
        if values is not None and np.asarray(values).shape != (n,):
            issue(f"Saved/recomputed {key} array shape differs from median")
            values, source = None, "invalid_shape"
        signal_arrays[key], array_sources[key] = values, source
    raw_magnitude = np.linalg.norm(xyz, axis=1)
    # The raw magnitude is also directly derivable from saved XYZ. Prefer saved
    # V1 magnitude if present, but report any difference rather than silently fix.
    if signal_arrays["magnitude"] is None:
        signal_arrays["magnitude"] = raw_magnitude
        array_sources["magnitude"] = "norm_of_saved_median_xyz"
    qlo = max(0, anchor + int(cfg.qrs_search_ms[0] * fs / 1000))
    qhi = min(n - 1, anchor + int(cfg.qrs_search_ms[1] * fs / 1000))
    smooth = signal_arrays["smooth_speed"]
    masks = None
    if smooth is not None and boundaries.get("q_threshold") is not None:
        raw = np.asarray(smooth[qlo:qhi+1]) > boundaries["q_threshold"]
        closed = binary_closing(raw, structure=np.ones(ms_samples(15, fs), bool))
        peak = boundaries.get("qrs_speed_peak")
        selected = np.zeros_like(closed)
        if _index(peak, n) and qlo <= peak <= qhi:
            left = right = peak - qlo
            while left > 0 and closed[left-1]:
                left -= 1
            while right < len(closed)-1 and closed[right+1]:
                right += 1
            selected[left:right+1] = True
        masks = dict(start_sample=qlo, above_threshold=raw, closed=closed,
                     selected_component=selected)
    def loop_indices(on, off):
        a, b = boundaries.get(on), boundaries.get(off)
        if not (_index(a, n) and _index(b, n)):
            return None
        loop = cut_loop(xyz, a, b)
        if loop is None:
            return None
        if not np.array_equal(loop, xyz[a:b+1]):
            raise AssertionError("V1 cut_loop does not match inclusive saved indices")
        return [int(a), int(b)]
    qindices, tindices = loop_indices("qrs_on", "qrs_off"), loop_indices("t_on", "t_off")
    if qindices is not None and tindices is not None and qindices[1] >= tindices[0]:
        issue("Saved QRS/T loop intervals overlap or are unordered")
    missing = [name for name in ("qrs_on", "qrs_off", "t_on", "t_off") if times[name] is None]
    status = "complete_bounds" if qindices is not None and tindices is not None else "missing_bounds"
    if times["t_peak"] is None:
        status = "t_peak_missing"
    if issues:
        status = "integrity_warning_" + status
    meta = _base_meta(row, art)
    meta.update(segmentation_status=status, n_samples=n,
                time_start_ms=float(time[0]), time_end_ms=float(time[-1]))
    matches = all(v["matches_exactly"] for v in parity) if parity else None
    reason = "; ".join([("Missing saved boundaries: " + ", ".join(missing)) if missing else ""] +
                       [i["issue"] for i in issues]).strip("; ")
    payload = dict(schema_version=SCHEMA_VERSION, record_id=row["record_id"], available=True,
        reason=reason, meta=meta, xyz=xyz, time_ms=time, raw_magnitude=raw_magnitude,
        **signal_arrays, boundaries=boundaries, boundary_times_ms=times,
        qrs_search=[qlo, qhi], pq=pq, qrs_loop_indices=qindices, t_loop_indices=tindices,
        qrs_masks=masks, parity=dict(status="matches_exactly" if matches else "difference_or_unavailable",
            all_available_fields_match_exactly=matches, n_fields=len(parity),
            n_differences=sum(not v["matches_exactly"] for v in parity), error=recompute_error),
        provenance={**provenance, "array_sources": array_sources,
            "rr_seconds_used_for_parity": rep["rr_seconds"],
            "time_grid_matches_exactly": bool(np.array_equal(time, expected_time)),
            "csv_time_tolerance_ms": 1e-9})
    return _clean(payload), parity, issues


def prepare_visual_review_data(frame, artifact_root, output_dir, cfg=None, resume=True):
    """Export every input record and return manifest/records/parity/validation.

    ``output_dir`` is the actual ``all_cases`` directory. Cache reuse requires
    unchanged input row, artifacts, V1 config, exporter and baseline source.
    Existing manual review files are neither read nor written by this module.
    """
    cfg = cfg or Config()
    if frame.empty or frame.record_id.duplicated().any():
        raise ValueError("A nonempty cohort with unique record_id is required")
    out, artroot = Path(output_dir), Path(artifact_root)
    if (artroot / "artifacts").is_dir():
        artroot = artroot / "artifacts"
    data_dir, cache_dir = out / "signal_data", out / "signal_checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    sources = {p.name: _sha(p) for p in [Path(__file__), ROOT / "vcg_pipeline.py", ROOT / "vcg_preprocessing.py"]}
    manifest, all_parity, all_issues = [], [], []
    cache_hits = 0
    for row in frame.to_dict("records"):
        rid, safe = row["record_id"], str(row["record_id"]).replace("/", "_")
        jp, zp = artroot / (safe + ".json"), artroot / (safe + ".npz")
        signature = hashlib.sha256(_json(dict(schema=SCHEMA_VERSION, input=row,
            config=asdict(cfg), sources=sources, artifact_json=_sha(jp) if jp.is_file() else None,
            artifact_npz=_sha(zp) if zp.is_file() else None)).encode()).hexdigest()
        checkpoint, path = cache_dir / (safe + ".json"), data_dir / (safe + ".js")
        cached = None
        if resume and checkpoint.is_file() and path.is_file():
            try:
                candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
                if candidate["signature"] == signature and candidate["manifest"]["payload_sha256"] == _sha(path):
                    cached = candidate
                    cache_hits += 1
            except (OSError, ValueError, KeyError):
                pass
        if cached:
            item = dict(cached["manifest"], cached=True)
            manifest.append(item)
            all_parity.extend(cached["parity"])
            all_issues.extend(cached["issues"])
            continue
        try:
            payload, parity, issues = build_visual_record(row, artroot, cfg)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            payload = _unavailable_payload(row, {}, reason, dict(artifact_json=str(jp), artifact_npz=str(zp)))
            payload["meta"]["segmentation_status"] = "load_or_validation_error"
            parity, issues = [], [{"record_id": rid, "issue": reason}]
        # Script callback works from file:// without fetch/CORS or a web server.
        path.write_text("window.VCGVisualReview.receiveRecord(" + _json(rid) + "," + _json(payload) + ");\n", encoding="utf-8")
        meta, available = payload["meta"], payload["available"]
        item = dict(row)
        item.update(has_median=available, has_qrs_bounds=payload["qrs_loop_indices"] is not None,
            has_t_bounds=payload["t_loop_indices"] is not None,
            has_t_peak=payload.get("boundary_times_ms", {}).get("t_peak") is not None,
            qrs_loop_available=payload["qrs_loop_indices"] is not None,
            t_loop_available=payload["t_loop_indices"] is not None,
            segmentation_status=meta["segmentation_status"], reason=payload["reason"],
            anchor_sample=meta.get("anchor_sample"), fs=meta.get("fs"),
            n_samples=meta.get("n_samples"), time_start_ms=meta.get("time_start_ms"),
            time_end_ms=meta.get("time_end_ms"), payload_path=str(path.relative_to(out)),
            payload_sha256=_sha(path), signal_sha256=_sha(zp) if zp.is_file() else None,
            source_json_sha256=_sha(jp) if jp.is_file() else None,
            parity_status=payload["parity"]["status"],
            parity_differences=payload["parity"].get("n_differences", 0),
            integrity_issue_count=len(issues), cached=False)
        for name in INDEX_FIELDS:
            item[name + "_sample"] = payload.get("boundaries", {}).get(name)
        item = _clean(item)
        manifest.append(item)
        all_parity.extend(parity)
        all_issues.extend(issues)
        _write_json(checkpoint, dict(signature=signature, manifest=item, parity=parity, issues=issues))
    manifest = pd.DataFrame(manifest)
    parity_columns = ["record_id", "field", "kind", "matches_exactly", "max_abs_difference", "saved_value", "recomputed_value"]
    parity = pd.DataFrame(all_parity, columns=parity_columns)
    issues = pd.DataFrame(all_issues, columns=["record_id", "issue"])
    manifest.to_csv(out / "visual_signal_manifest.csv", index=False)
    parity.to_csv(out / "visual_v1_parity.csv", index=False)
    parity.loc[~parity.matches_exactly.astype(bool)].to_csv(out / "visual_v1_parity_differences.csv", index=False)
    issues.to_csv(out / "visual_signal_integrity_issues.csv", index=False)
    validation = dict(schema_version=SCHEMA_VERSION, n_records=len(manifest),
        n_unique_records=manifest.record_id.nunique(), n_medians=int(manifest.has_median.sum()),
        n_missing_medians=int((~manifest.has_median).sum()), n_qrs_loops=int(manifest.has_qrs_bounds.sum()),
        n_t_loops=int(manifest.has_t_bounds.sum()), n_payloads=len(manifest),
        n_parity_fields=len(parity), n_parity_differences=int((~parity.matches_exactly.astype(bool)).sum()),
        n_integrity_issues=len(issues), cache_hits=cache_hits,
        all_payloads_exist=all((out / p).is_file() for p in manifest.payload_path),
        all_ids_preserved=set(manifest.record_id) == set(frame.record_id),
        baseline_and_manual_review_modified=False, config_v1=asdict(cfg), source_sha256=sources,
        prediction_source="Saved V1 indices and arrays, not recomputed predictions",
        exact_time_rule="(median_sample - median_anchor_sample) * 1000 / record_fs",
        steps_05_reused="6-8 saved representative/PQ; 9-15 exact full-median delineate_loops parity; 16-19 saved inclusive cuts",
        review_status="Visual assessment is not independent validated ground truth")
    _write_json(out / "visual_signal_validation.json", validation)
    return dict(manifest=manifest, records=manifest.to_dict("records"), parity=parity,
                issues=issues, validation=validation)
