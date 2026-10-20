#!/usr/bin/env python3
"""Compare frozen V1 candidates with explicitly confirmed independent references.

Usage::

    python scripts/compare_vcg_annotations.py --output-dir outputs/vcg_full_evaluation \
        --annotations reviewed_boundaries.csv
    python scripts/compare_vcg_annotations.py --output-dir outputs/vcg_full_evaluation \
        --qrs-reference reviewed_qrs_events.csv --tolerance-ms 50

Boundary CSV columns: record_id, reviewer, review_status, independent_reference,
time_reference, reference_anchor_sample, qrs_on_ms, qrs_off_ms, t_on_ms, t_off_ms.
Only confirmed rows with a named reviewer and independent_reference=true qualify.
The optional usable_reference field can explicitly exclude a row. Partial boundary
references are supported: only finite, independently annotated boundaries enter
their respective denominators. Fields beginning with pred_ are never references.

time_reference must explicitly be median_relative_ms or raw_recording_ms. Raw
references require an integer reference_anchor_sample present in the artifact's
used_samples. They are converted by subtracting that observed beat's anchor time.
Their scope is comparison of a median estimate with one observed beat; they do
not establish accuracy of delineation on every individual beat.
When saved median time limits are available (JSON, otherwise NPZ time_ms), every
finite boundary reference must lie within that observed/median beat window.
The comparison records whether window limits were available for validation.

QRS event CSV columns: record_id, sample, reviewer, review_status,
independent_reference, complete_event_reference, coverage_start_ms,
coverage_end_ms (plus optional usable_reference). complete_event_reference=true
attests that every event in the explicit half-open coverage interval was reviewed.
Every row for a recording must qualify, and coverage/reviewer must be consistent.
Predictions are restricted to that same interval. A partial list of event references
cannot establish false positives. Use one reviewer/reference set per recording in
each invocation. Matching uses sorted,
chronological one-to-one matching within the declared tolerance. It maximizes
matched event count for ordered timestamps; it does not minimize timing error
among alternative matchings. Timing MAE is conditional on matched events.

No prediction is used as reference. Pending, unconfirmed, and invalid references
are excluded with reasons in audit CSVs. Duplicate prediction IDs, reference IDs,
or natural reference keys raise errors instead of multiplying observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


BOUNDARIES = ("qrs_on", "qrs_off", "t_on", "t_off")
TOLERANCES_MS = (10, 20, 40, 60)
COMPARISON_COLUMNS = [
    "record_id", "patient_id", "label_name", "split", "reference_row",
    "reference_id", "reviewer", "time_reference", "reference_anchor_sample",
    "comparison_scope", "window_validation", "boundary", "reference_ms", "prediction_ms",
    "signed_error_ms", "absolute_error_ms", "matched", "missing_prediction",
    *[f"within_{t}_ms" for t in TOLERANCES_MS],
]
SUMMARY_COLUMNS = [
    "label_name", "boundary", "comparison_scope", "valid_references",
    "matched", "missing_prediction", "prediction_coverage_pct", "mae_ms",
    "mean_signed_error_ms", *[f"within_{t}_ms_pct_of_references" for t in TOLERANCES_MS],
]
AUDIT_COLUMNS = ["reference_row", "record_id", "accepted", "reason"]
EVENT_COLUMNS = [
    "record_id", "patient_id", "label_name", "split", "n_reference", "n_predicted",
    "tp", "fp", "fn", "precision", "recall", "f1", "timing_mae_ms",
    "timing_error_sum_ms", "tolerance_ms", "matching_method", "comparison_scope",
    "coverage_start_ms", "coverage_end_ms",
]


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _true(value: Any) -> bool:
    return _text(value).lower() in {"true", "1", "1.0"}


def _integer(value: Any) -> int | None:
    number = _finite(value)
    return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None


def _sample_key(value: Any) -> Any:
    integer = _integer(value)
    return integer if integer is not None else _text(value)


def safe_record_id(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", record_id)


def _review_rejection(row: pd.Series) -> str | None:
    if _text(row.get("review_status")).lower() != "confirmed":
        return "review_status_not_confirmed"
    if not _text(row.get("reviewer")):
        return "reviewer_missing"
    if not _true(row.get("independent_reference")):
        return "independent_reference_not_explicitly_true"
    if not _blank(row.get("usable_reference")) and not _true(row.get("usable_reference")):
        return "usable_reference_not_true"
    return None


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "record_id" not in predictions:
        raise ValueError("Predictions require record_id.")
    out = predictions.copy()
    out["record_id"] = out["record_id"].map(_text)
    if (out.record_id == "").any() or out.record_id.duplicated().any():
        raise ValueError("Prediction record_id must be nonempty and unique.")
    return out.set_index("record_id", drop=False)


def _validate_reference_keys(references: pd.DataFrame, events: bool = False) -> pd.DataFrame:
    if "record_id" not in references:
        raise ValueError("References require record_id.")
    refs = references.copy().reset_index(drop=True)
    refs["record_id"] = refs.record_id.map(_text)
    if (refs.record_id == "").any():
        raise ValueError("Reference record_id must be nonempty.")
    if "reference_id" in refs:
        ids = refs.reference_id.map(_text)
        nonempty = ids[ids != ""]
        if nonempty.duplicated().any():
            raise ValueError("Duplicate reference_id.")
    if events:
        keys = [(r.record_id, _sample_key(r.get("sample"))) for _, r in refs.iterrows()]
    else:
        keys = [
            (r.record_id, _text(r.get("time_reference")),
             _sample_key(r.get("reference_anchor_sample")), _text(r.get("reviewer")))
            for _, r in refs.iterrows()
        ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate natural reference keys.")
    return refs


def _artifact(output_dir: Path, record_id: str) -> dict[str, Any] | None:
    path = output_dir / "artifacts" / (safe_record_id(record_id) + ".json")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Artifact is not an object: {path}")
    if value.get("record_id", record_id) != record_id:
        raise ValueError(f"Artifact record_id mismatch: {path}")
    return value


def _fs(prediction: pd.Series, artifact: dict[str, Any] | None = None) -> float:
    fs = _finite(prediction.get("fs", prediction.get("fs_hz")))
    if artifact is not None:
        stored = _finite(artifact.get("fs"))
        if np.isfinite(fs) and np.isfinite(stored) and not np.isclose(fs, stored, rtol=0, atol=1e-8):
            raise ValueError("Sampling frequency differs between prediction and artifact.")
    return fs


def _median_window(output_dir: Path, record_id: str, artifact: dict[str, Any] | None) -> tuple[float, float] | None:
    """Prefer explicit JSON limits; use the exact saved time array as fallback."""
    if artifact is not None:
        start = _finite(artifact.get("median_time_start_ms"))
        end = _finite(artifact.get("median_time_end_ms"))
        if np.isfinite(start) and np.isfinite(end):
            if start >= end:
                raise ValueError(f"Invalid median time window in artifact: {record_id}")
            return start, end
    path = output_dir / "artifacts" / (safe_record_id(record_id) + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as arrays:
            if "time_ms" not in arrays:
                return None
            time = np.asarray(arrays["time_ms"], dtype=float)
        if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or not (np.diff(time) > 0).all():
            raise ValueError(f"Invalid median time array in artifact: {record_id}")
        return float(time[0]), float(time[-1])
    return None


def compare_boundaries(
    predictions: pd.DataFrame, references: pd.DataFrame, output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return accepted reference comparisons, stratified summary, and row audit."""
    pred = _validate_predictions(predictions)
    refs = _validate_reference_keys(references)
    comparisons: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for index, row in refs.iterrows():
        rid = row.record_id
        reason = _review_rejection(row)
        if reason is None and rid not in pred.index:
            reason = "record_not_in_prediction_manifest"
        frame = _text(row.get("time_reference"))
        anchor = None
        offset = 0.0
        scope = "median_estimate_vs_expert_same_median"
        window_validation = "no_window_data"
        artifact = None
        if reason is None and frame not in {"median_relative_ms", "raw_recording_ms"}:
            reason = "time_reference_missing_or_unsupported"
        values = {name: _finite(row.get(name + "_ms")) for name in BOUNDARIES}
        if reason is None and not any(np.isfinite(v) for v in values.values()):
            reason = "no_finite_boundary_reference"
        supplied = [values[name] for name in BOUNDARIES if np.isfinite(values[name])]
        if reason is None and any(a >= b for a, b in zip(supplied, supplied[1:])):
            reason = "reference_boundary_order_invalid"
        if reason is None:
            artifact = _artifact(Path(output_dir), rid)
        if reason is None and frame == "raw_recording_ms":
            anchor = _integer(row.get("reference_anchor_sample"))
            fs = _fs(pred.loc[rid], artifact)
            if anchor is None:
                reason = "raw_reference_requires_integer_anchor_sample"
            elif not np.isfinite(fs) or fs <= 0:
                reason = "raw_reference_requires_prediction_fs"
            elif artifact is None:
                reason = "raw_reference_requires_retained_anchor_artifact"
            elif anchor not in artifact.get("used_samples", []):
                reason = "reference_anchor_not_in_retained_beats"
            elif any(v < 0 for v in values.values() if np.isfinite(v)):
                reason = "raw_reference_time_negative"
            else:
                offset = anchor * 1000.0 / fs
                scope = "median_estimate_vs_selected_observed_beat"
        if reason is None:
            window = _median_window(Path(output_dir), rid, artifact)
            if window is not None:
                window_validation = "validated_with_saved_median_window"
                if any(not window[0] - 1e-8 <= value - offset <= window[1] + 1e-8
                       for value in values.values() if np.isfinite(value)):
                    reason = "reference_boundary_outside_selected_beat_window"
        audit.append({"reference_row": index + 2, "record_id": rid,
                      "accepted": reason is None, "reason": reason or "accepted"})
        if reason is not None:
            continue
        prediction = pred.loc[rid]
        for name, value in values.items():
            if not np.isfinite(value):
                continue
            reference = value - offset
            estimate = _finite(prediction.get(name + "_ms"))
            matched = bool(np.isfinite(estimate))
            error = estimate - reference if matched else float("nan")
            item = {
                "record_id": rid, "patient_id": _text(prediction.get("patient_id")),
                "label_name": _text(prediction.get("label_name")), "split": _text(prediction.get("split")),
                "reference_row": index + 2, "reference_id": _text(row.get("reference_id")),
                "reviewer": _text(row.get("reviewer")), "time_reference": frame,
                "reference_anchor_sample": anchor, "comparison_scope": scope, "window_validation": window_validation,
                "boundary": name, "reference_ms": reference, "prediction_ms": estimate,
                "signed_error_ms": error, "absolute_error_ms": abs(error),
                "matched": matched, "missing_prediction": not matched,
            }
            item.update({f"within_{t}_ms": bool(matched and abs(error) <= t + 1e-10)
                         for t in TOLERANCES_MS})
            comparisons.append(item)
    comparison = pd.DataFrame(comparisons, columns=COMPARISON_COLUMNS)
    summaries = []
    for keys, group in comparison.groupby(["label_name", "boundary", "comparison_scope"], dropna=False):
        matched = int(group.matched.sum())
        item = dict(zip(["label_name", "boundary", "comparison_scope"], keys))
        item.update(valid_references=len(group), matched=matched, missing_prediction=len(group)-matched,
                    prediction_coverage_pct=100.0*matched/len(group),
                    mae_ms=group.absolute_error_ms.mean(), mean_signed_error_ms=group.signed_error_ms.mean())
        item.update({f"within_{t}_ms_pct_of_references": 100.0*group[f"within_{t}_ms"].mean()
                     for t in TOLERANCES_MS})
        summaries.append(item)
    return comparison, pd.DataFrame(summaries, columns=SUMMARY_COLUMNS), pd.DataFrame(audit, columns=AUDIT_COLUMNS)


def match_qrs_events(predicted_s: Any, reference_s: Any, tolerance_ms: float = 50.0) -> dict[str, Any]:
    """Chronological one-to-one matching; no event can contribute twice."""
    if not np.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("QRS tolerance must be finite and positive.")
    predicted = np.sort(np.asarray(predicted_s, dtype=float))
    reference = np.sort(np.asarray(reference_s, dtype=float))
    if predicted.ndim != 1 or reference.ndim != 1 or not (np.isfinite(predicted).all() and np.isfinite(reference).all()):
        raise ValueError("QRS timestamps must be finite one-dimensional arrays.")
    i = j = 0
    signed = []
    pairs = []
    tolerance_s = tolerance_ms / 1000.0
    while i < len(predicted) and j < len(reference):
        delta = predicted[i] - reference[j]
        if abs(delta) <= tolerance_s + 1e-12:
            signed.append(delta*1000.0)
            pairs.append((float(predicted[i]), float(reference[j])))
            i += 1
            j += 1
        elif delta < -tolerance_s:
            i += 1
        else:
            j += 1
    tp = len(signed)
    fp, fn = len(predicted)-tp, len(reference)-tp
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else float("nan"),
            "recall": tp/(tp+fn) if tp+fn else float("nan"),
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float("nan"),
            "timing_mae_ms": float(np.mean(np.abs(signed))) if tp else float("nan"),
            "timing_error_sum_ms": float(np.sum(np.abs(signed))),
            "pairs": pairs, "signed_errors_ms": signed}


def compare_qrs_references(
    predictions: pd.DataFrame, references: pd.DataFrame, output_dir: Path, tolerance_ms: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare independently reviewed raw-recording event samples with anchors."""
    pred = _validate_predictions(predictions)
    refs = _validate_reference_keys(references, events=True)
    if not np.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("QRS tolerance must be finite and positive.")
    accepted: dict[str, list[int]] = {}
    intervals: dict[str, tuple[float, float, str]] = {}
    audit = []
    for index, row in refs.iterrows():
        rid = row.record_id
        reason = _review_rejection(row)
        sample = _integer(row.get("sample"))
        if reason is None and rid not in pred.index:
            reason = "record_not_in_prediction_manifest"
        if reason is None and sample is None:
            reason = "reference_sample_not_nonnegative_integer"
        if reason is None and not (_fs(pred.loc[rid]) > 0):
            reason = "event_reference_requires_prediction_fs"
        if reason is None and not _true(row.get("complete_event_reference")):
            reason = "complete_event_reference_not_explicitly_true"
        if reason is None and _artifact(Path(output_dir), rid) is None:
            reason = "event_reference_requires_anchor_artifact"
        start = _finite(row.get("coverage_start_ms"))
        end = _finite(row.get("coverage_end_ms"))
        if reason is None and not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end):
            reason = "event_reference_requires_explicit_valid_coverage_interval"
        if reason is None:
            fs = _fs(pred.loc[rid])
            loaded_ms = _finite(pred.loc[rid].get("seconds_loaded"))*1000
            if not np.isfinite(loaded_ms):
                loaded_ms = _finite(pred.loc[rid].get("samples_loaded"))/fs*1000
            if not np.isfinite(loaded_ms):
                reason = "event_reference_requires_loaded_duration"
            elif end > loaded_ms + 1e-8:
                reason = "event_reference_coverage_exceeds_loaded_recording"
            elif not start <= sample*1000/fs < end:
                reason = "reference_event_outside_declared_coverage"
            else:
                interval = (start, end, _text(row.get("reviewer")))
                if rid in intervals and intervals[rid] != interval:
                    raise ValueError(f"Conflicting coverage or reviewers within event reference set: {rid}")
                intervals[rid] = interval
        audit.append({"reference_row": index+2, "record_id": rid,
                      "accepted": reason is None, "reason": reason or "accepted"})
        if reason is None:
            accepted.setdefault(rid, []).append(sample)
    # A partly unconfirmed event list is not a complete reference event set.
    rejected_records = {row["record_id"] for row in audit if not row["accepted"]}
    for row in audit:
        if row["record_id"] in rejected_records and row["accepted"]:
            row.update(accepted=False, reason="event_reference_set_contains_unconfirmed_or_invalid_rows")
    accepted = {rid: samples for rid, samples in accepted.items() if rid not in rejected_records}
    results = []
    for rid, samples in accepted.items():
        prediction = pred.loc[rid]
        artifact = _artifact(Path(output_dir), rid)
        fs = _fs(prediction, artifact)
        anchors = [] if artifact is None else artifact.get("anchors", [])
        if any(_integer(value) is None for value in anchors) or len(anchors) != len(set(anchors)):
            raise ValueError(f"Artifact anchors must be unique nonnegative integers: {rid}")
        start, end, _ = intervals[rid]
        anchor_times = np.asarray(anchors)/fs
        anchor_times = anchor_times[(anchor_times*1000 >= start) & (anchor_times*1000 < end)]
        metrics = match_qrs_events(anchor_times, np.asarray(samples)/fs, tolerance_ms)
        item = {"record_id": rid, "patient_id": _text(prediction.get("patient_id")),
                "label_name": _text(prediction.get("label_name")), "split": _text(prediction.get("split")),
                "n_reference": len(samples), "n_predicted": len(anchor_times),
                "tolerance_ms": tolerance_ms, "matching_method": "chronological_one_to_one",
                "comparison_scope": "qrs_events_in_explicitly_fully_annotated_interval",
                "coverage_start_ms": start, "coverage_end_ms": end}
        item.update({key: value for key, value in metrics.items() if key not in {"pairs", "signed_errors_ms"}})
        results.append(item)
    details = pd.DataFrame(results, columns=EVENT_COLUMNS)
    summaries = []
    for label, group in details.groupby("label_name", dropna=False):
        tp, fp, fn = (int(group[column].sum()) for column in ["tp", "fp", "fn"])
        summaries.append({"label_name": label, "n_records_with_valid_reference": len(group),
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": tp/(tp+fp) if tp+fp else np.nan,
                          "recall": tp/(tp+fn) if tp+fn else np.nan,
                          "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                          "timing_mae_ms": group.timing_error_sum_ms.sum()/tp if tp else np.nan,
                          "tolerance_ms": tolerance_ms, "matching_method": "chronological_one_to_one"})
    columns = ["label_name", "n_records_with_valid_reference", "tp", "fp", "fn", "precision",
               "recall", "f1", "timing_mae_ms", "tolerance_ms", "matching_method"]
    return details, pd.DataFrame(summaries, columns=columns), pd.DataFrame(audit, columns=AUDIT_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="An existing full-evaluation run.")
    parser.add_argument("--annotations", type=Path, help="Independent reviewed boundary references.")
    parser.add_argument("--qrs-reference", type=Path, help="Independent reviewed raw QRS event references.")
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    args = parser.parse_args(argv)
    if args.annotations is None and args.qrs_reference is None:
        parser.error("Provide --annotations or --qrs-reference.")
    if not args.output_dir.is_dir():
        parser.error("--output-dir must be an existing run directory.")
    predictions = pd.read_csv(args.output_dir / "features" / "all_features.csv")
    evaluation = args.output_dir / "evaluation"
    evaluation.mkdir(exist_ok=True)
    if args.annotations is not None:
        comparison, summary, audit = compare_boundaries(predictions, pd.read_csv(args.annotations), args.output_dir)
        comparison.to_csv(evaluation / "boundary_reference_comparison.csv", index=False)
        summary.to_csv(evaluation / "reference_summary.csv", index=False)
        audit.to_csv(evaluation / "boundary_reference_audit.csv", index=False)
        print(f"Boundary references: {int(audit.accepted.sum())}/{len(audit)} accepted; "
              f"{len(comparison)} boundary references, {int(comparison.matched.sum())} matched predictions.")
    if args.qrs_reference is not None:
        comparison, summary, audit = compare_qrs_references(
            predictions, pd.read_csv(args.qrs_reference), args.output_dir, args.tolerance_ms)
        comparison.to_csv(evaluation / "qrs_reference_comparison.csv", index=False)
        summary.to_csv(evaluation / "qrs_reference_summary.csv", index=False)
        audit.to_csv(evaluation / "qrs_reference_audit.csv", index=False)
        print(f"QRS references: {int(audit.accepted.sum())}/{len(audit)} accepted event rows; "
              f"{len(comparison)} recordings evaluated with {args.tolerance_ms:g} ms tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
