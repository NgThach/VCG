"""Scientific figures for independently accepted QRS/T references (Notebook 08).

No missing reference is replaced by a prediction or a zero error.  Accuracy
figures are created only for eligible, finite reference/prediction pairs.  The
separate diagnostic gallery is explicitly labelled as unverified V1 output.
All XYZ overlays use the *same saved median array* for both interval cuts.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import textwrap
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_qc_audit_matplotlib'))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from vcg_pipeline import FEATURE_NAMES
from vcg_preprocessing import Config

BOUNDARIES = ("qrs_on", "qrs_off", "t_on", "t_off")
DISPLAY = {"qrs_on": "QRS-on", "qrs_off": "QRS-off", "t_on": "T-on", "t_off": "T-off"}
REFERENCE = "#12805c"
PREDICTION = "#c46d18"
MISSED = "#a62c91"
EXCESS = "#dc3434"


def _truth(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "1.0", "yes"))


def _eligible(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Do not infer acceptance from finite numbers alone."""
    if frame.empty or "evaluation_eligible" not in frame:
        return frame.iloc[:0].copy()
    valid = _truth(frame["evaluation_eligible"])
    for column in columns:
        if column not in frame:
            return frame.iloc[:0].copy()
        valid &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    return frame.loc[valid].copy()


def _table(results: dict, name: str) -> pd.DataFrame:
    table = results.get(name)
    return table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _safe(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record_id))


def _artifacts(root: Path) -> Path:
    root = Path(root)
    return root / "artifacts" if (root / "artifacts").is_dir() else root


def _load_artifact(root: Path, record_id: str) -> tuple[dict, dict]:
    root = _artifacts(root)
    path = root / (_safe(record_id) + ".json")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("record_id") != record_id:
        raise ValueError(f"Artifact identity mismatch: {path}")
    with np.load(root / (_safe(record_id) + ".npz"), allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    return artifact, arrays


def _source_note(frame: pd.DataFrame) -> str:
    sources = sorted(set(frame.get("reference_source", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"})
    reviewers = sorted(set(frame.get("reviewer", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"})
    patients = frame["patient_id"].nunique() if "patient_id" in frame else "?"
    records = frame["record_id"].nunique() if "record_id" in frame else len(frame)
    source = "; ".join(sources) or "Accepted independent reference (source not supplied in plot table)"
    reviewer = "; ".join(reviewers) or "See accepted annotation table"
    return (f"Reference: {source}. Reviewer(s): {reviewer}. "
            f"n={len(frame)} paired observations; {records} recordings, {patients} patients. "
            "Recordings may repeat within patients; plots are descriptive, without an independence assumption. "
            "Priority review samples do not establish cohort-wide error.")


def _finish(fig, frame: pd.DataFrame, target: Path, title: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=12, y=.985)
    note = "\n".join(textwrap.wrap(_source_note(frame), width=155))
    fig.text(.035, .015, note, fontsize=7, va="bottom", color="#37434c")
    fig.tight_layout(rect=(0, .11, 1, .95))
    fig.savefig(target, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _status_row(family: str, status: str, n: int, path: Path | None, output_dir: Path,
                reason: str = "") -> dict:
    return {"plot_family": family, "status": status, "n_paired_observations": int(n),
            "plot_path": path.relative_to(output_dir).as_posix() if path else "",
            "reason": reason}


def _prepare_current_reference_outputs(output_dir: Path, tables: tuple[pd.DataFrame, ...]) -> dict:
    """Preserve older reference figures outside the current result namespace.

    Withdrawing/rejecting a reference must not leave its old accuracy PNG or
    highest/lowest gallery looking current. Only this module's generated paths
    are moved; reviewer annotations and diagnostic figures are never touched.
    An identical input fingerprint keeps the same current namespace.
    """
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for table in tables:
        digest.update(table.to_json(orient="split", index=False, double_precision=15).encode())
    fingerprint = digest.hexdigest()
    state_path = output_dir / "reference_plots_input_state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    generated = [output_dir / "plots" / "verified_reference",
                 output_dir / "reference_loop_gallery.html",
                 output_dir / "reference_loop_gallery_manifest.csv",
                 output_dir / "all_reference_loop_gallery.html",
                 output_dir / "all_reference_loop_gallery_manifest.csv",
                 output_dir / "all_reference_boundary_gallery.html",
                 output_dir / "all_reference_boundary_gallery_manifest.csv"]
    current = [p for p in generated if p.is_file() or p.is_dir() and any(p.rglob("*"))]
    archived = ""
    if current and previous.get("fingerprint") != fingerprint:
        old = previous.get("fingerprint", "legacy-unfingerprinted")[:24]
        base = output_dir / "history" / "reference_plots" / old
        destination = base
        version = 1
        while destination.exists():
            destination = base.with_name(base.name + f"-{version}")
            version += 1
        destination.mkdir(parents=True)
        for path in current:
            path.rename(destination / path.name)
        if previous:
            (destination / "reference_plots_input_state.json").write_text(
                json.dumps(previous, indent=2) + "\n", encoding="utf-8")
        archived = destination.relative_to(output_dir).as_posix()
    state = {"fingerprint": fingerprint, "archived_previous_run": archived,
             "n_boundary_pairs": len(tables[0]), "n_loop_pairs": len(tables[1]),
             "n_feature_pairs": len(tables[2]),
             "interpretation": "Only plots_status.csv describes current figure availability; history is superseded output"}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def _index_interval(row: pd.Series, prefix: str, t: np.ndarray) -> tuple[int, int]:
    indices = []
    for side in ("start", "end"):
        key = f"{prefix}_{side}_index"
        value = row.get(key, np.nan)
        if pd.notna(value):
            index = int(value)
            if float(value) != index or not 0 <= index < len(t):
                raise ValueError(f"Invalid saved {key}: {value}")
        else:
            value = float(row[f"{prefix}_{side}_ms"])
            # np.argmin resolves an exact tie toward the earlier sample.
            index = int(np.argmin(np.abs(t - value)))
        expected = float(row[f"{prefix}_{side}_ms"])
        if not np.isclose(t[index], expected, atol=1e-7, rtol=0):
            raise ValueError("Overlay interval must use represented sample times, not unsnapped annotations")
        indices.append(index)
    if indices[0] >= indices[1]:
        raise ValueError("Overlay requires an ordered, nonzero-duration interval")
    return tuple(indices)


def _cut_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive point runs whose segments satisfy a temporal membership mask."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return [(int(chunk[0]), int(chunk[-1] + 1)) for chunk in np.split(indices, splits)]


def _overlay_loop(row: pd.Series, artifact_root: Path, path: Path) -> None:
    """XYZ figures display curves and endpoint membership, not a spatial accuracy metric."""
    artifact, arrays = _load_artifact(artifact_root, str(row["record_id"]))
    if "median_xyz" not in arrays:
        raise ValueError("An accepted loop comparison requires a saved median beat")
    xyz, t = arrays["median_xyz"], arrays["time_ms"]
    if xyz.shape != (len(t), 3):
        raise ValueError("Inconsistent saved XYZ shape")
    pa, pz = _index_interval(row, "predicted", t)
    ra, rz = _index_interval(row, "reference", t)
    pred = xyz[pa:pz + 1]
    ref = xyz[ra:rz + 1]
    # Segment membership makes missed/excess duration continuous [on,off].
    pred_segments = (np.arange(len(t) - 1) >= pa) & (np.arange(len(t) - 1) < pz)
    ref_segments = (np.arange(len(t) - 1) >= ra) & (np.arange(len(t) - 1) < rz)
    missed = _cut_runs(ref_segments & ~pred_segments)
    excess = _cut_runs(pred_segments & ~ref_segments)
    fig = plt.figure(figsize=(15, 9.4))
    ax_time = fig.add_subplot(2, 3, 1)
    ax_3d = fig.add_subplot(2, 3, 2, projection="3d")
    projections = [(fig.add_subplot(2, 3, j + 3), ij) for j, ij in enumerate(((0, 1), (0, 2), (1, 2)))]
    timeline = fig.add_subplot(2, 3, 6)
    for axis, color in enumerate(("#246ba2", "#ba4533", "#42844d")):
        ax_time.plot(t, xyz[:, axis], lw=.9, color=color, label="XYZ"[axis])
    ax_time.axvspan(t[ra], t[rz], color=REFERENCE, alpha=.10, label="Reference interval")
    ax_time.axvspan(t[pa], t[pz], color=PREDICTION, alpha=.10, label="Predicted interval")
    for a, z, color, name in ((ra, rz, REFERENCE, "Ref"), (pa, pz, PREDICTION, "Pred")):
        ax_time.axvline(t[a], color=color, ls="--", lw=1, label=f"{name} start {t[a]:.1f} ms")
        ax_time.axvline(t[z], color=color, ls=":", lw=1, label=f"{name} end {t[z]:.1f} ms")
    pad = max((max(t[pz], t[rz]) - min(t[pa], t[ra])) * .15, 15)
    left, right = (max(t[0], min(t[pa], t[ra]) - pad), min(t[-1], max(t[pz], t[rz]) + pad))
    ax_time.set_xlim(left, right)
    local = xyz[(t >= left) & (t <= right)]
    margin = max(float(np.ptp(local)) * .12, .005)
    ax_time.set_ylim(float(local.min()) - margin, float(local.max()) + margin)
    ax_time.set(title="Same saved median Frank XYZ", xlabel="Median-relative time (ms)", ylabel="mV")
    ax_time.legend(fontsize=6.5, ncol=2)
    ax_time.grid(alpha=.18)

    for curve, color, label, style in ((ref, REFERENCE, "Reference", "-"), (pred, PREDICTION, "Predicted", "--")):
        ax_3d.plot(*curve.T, color=color, lw=1.5, ls=style, label=label)
        ax_3d.scatter(*curve[0], marker="o", s=34, color=color, label=label + " start")
        ax_3d.scatter(*curve[-1], marker="x", s=42, color=color, label=label + " end")
        for ax, (i, j) in projections:
            ax.plot(curve[:, i], curve[:, j], color=color, lw=1.5, ls=style, label=label)
            ax.scatter(curve[0, i], curve[0, j], marker="o", s=30, color=color)
            ax.scatter(curve[-1, i], curve[-1, j], marker="x", s=38, color=color)
    for runs, color, label in ((missed, MISSED, "Missed reference signal"), (excess, EXCESS, "Excess predicted signal")):
        for k, (a, z) in enumerate(runs):
            curve = xyz[a:z + 1]
            ax_3d.plot(*curve.T, color=color, lw=3, alpha=.85, label=label if k == 0 else None)
            for ax, (i, j) in projections:
                ax.plot(curve[:, i], curve[:, j], color=color, lw=3, alpha=.85, label=label if k == 0 else None)
    points = np.vstack((ref, pred))
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) / 2, .005) * 1.12
    ax_3d.set(xlabel="X (mV)", ylabel="Y (mV)", zlabel="Z (mV)", title="3D interval cuts from one median")
    ax_3d.set_xlim(center[0] - radius, center[0] + radius)
    ax_3d.set_ylim(center[1] - radius, center[1] + radius)
    ax_3d.set_zlim(center[2] - radius, center[2] + radius)
    ax_3d.set_box_aspect((1, 1, 1))
    ax_3d.legend(fontsize=6, loc="best")
    for ax, (i, j) in projections:
        ax.set(title="XYZ"[i] + "XYZ"[j] + " projection", xlabel="XYZ"[i] + " (mV)", ylabel="XYZ"[j] + " (mV)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.18)
        ax.legend(fontsize=6.2, loc="best")
    timeline.plot((t[ra], t[rz]), (1, 1), color=REFERENCE, lw=8, solid_capstyle="butt", label="Reference")
    timeline.plot((t[pa], t[pz]), (0, 0), color=PREDICTION, lw=8, solid_capstyle="butt", label="Prediction")
    for runs, color, y in ((missed, MISSED, 1), (excess, EXCESS, 0)):
        for a, z in runs:
            timeline.plot((t[a], t[z]), (y, y), color=color, lw=8, solid_capstyle="butt")
    timeline.set(yticks=[0, 1], yticklabels=["Prediction", "Reference"], ylim=(-.5, 1.8), xlabel="Median-relative time (ms)",
                 title=f"Temporal IoU={row['temporal_iou']:.3f}; missed={row['missed_ms']:.1f} ms; excess={row['excess_ms']:.1f} ms")
    timeline.text(.02, .92, "Curves use inclusive endpoint samples.\nDurations and IoU use continuous [start,end].\nColors show temporal omission / excess;\nno geometric accuracy metric is claimed.", transform=timeline.transAxes, fontsize=7, va="top")
    _finish(fig, pd.DataFrame([row]), path,
            f"{row['record_id']} | {row.get('label_name', '')} | {row.get('split', '')} | {row['loop']} accepted-reference comparison")


def _overlay_boundaries(frame: pd.DataFrame, artifact_root: Path, path: Path) -> None:
    """Display every accepted boundary, including isolated/partial references.

    A reference for one edge does not authorize an interval comparison.  Each
    panel therefore contains only the accepted edge of its own boundary type.
    """
    record_id = str(frame.record_id.iloc[0])
    _, arrays = _load_artifact(artifact_root, record_id)
    if "median_xyz" not in arrays:
        raise ValueError("Accepted median-relative boundary pairs require a saved median beat")
    xyz, time = arrays["median_xyz"], arrays["time_ms"]
    if xyz.shape != (len(time), 3):
        raise ValueError("Inconsistent saved XYZ shape")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    for ax, name in zip(axes.flat, BOUNDARIES):
        for axis, color in enumerate(("#246ba2", "#ba4533", "#42844d")):
            ax.plot(time, xyz[:, axis], color=color, lw=.9, label="XYZ"[axis])
        pair = frame.loc[frame.boundary.eq(name)]
        if pair.empty:
            ax.text(.5, .90, "No accepted reference/prediction pair for this boundary",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
            ax.set_title(DISPLAY[name] + " — unavailable")
        else:
            row = pair.iloc[0]
            reference, predicted = float(row.reference_ms), float(row.predicted_ms)
            for stamp, color, label, style in ((reference, REFERENCE, "Reference", "-"),
                                                (predicted, PREDICTION, "V1", "--")):
                index = int(np.argmin(np.abs(time - stamp)))
                if not np.isclose(time[index], stamp, atol=1e-7, rtol=0):
                    raise ValueError("Boundary overlay requires represented median sample times")
                ax.axvline(stamp, color=color, ls=style, lw=1.8, label=f"{label} {stamp:.1f} ms")
                ax.scatter(np.repeat(stamp, 3), xyz[index], color=color, s=24, zorder=4)
            ax.axvspan(min(reference, predicted), max(reference, predicted), color=PREDICTION, alpha=.14,
                       label=f"Absolute shift {row.absolute_error_ms:.1f} ms")
            pad = max(abs(predicted - reference) * .25, 35.)
            left, right = (max(time[0], min(reference, predicted) - pad),
                           min(time[-1], max(reference, predicted) + pad))
            ax.set_xlim(left, right)
            local = xyz[(time >= left) & (time <= right)]
            margin = max(float(np.ptp(local)) * .12, .005)
            ax.set_ylim(float(local.min()) - margin, float(local.max()) + margin)
            direction = "later" if row.error_ms > 1e-7 else "earlier" if row.error_ms < -1e-7 else "same sample"
            ax.set_title(f"{DISPLAY[name]} | V1 − ref = {row.error_ms:+.1f} ms ({direction})")
        ax.set(xlabel="Median-relative time (ms)", ylabel="Frank XYZ (mV)")
        ax.grid(alpha=.18)
        ax.legend(fontsize=7, ncol=2, loc="best")
    _finish(fig, frame, path,
            f"{record_id} | {frame.iloc[0].get('label_name', '')} | {frame.iloc[0].get('split', '')} | all accepted boundary pairs")


def create_reference_plots(results: dict, artifact_root: Path, output_dir: Path, cfg=None) -> dict:
    """Create only supported accuracy figures; always record availability in CSV.

    Expected result tables are the Notebook 08 metric-engine outputs.  An
    ``evaluation_eligible=True`` flag is mandatory even when numeric values exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots" / "verified_reference"
    boundary = _table(results, "boundary_error_per_record")
    loops = _table(results, "loop_overlap_per_record")
    features = _table(results, "feature_reference_comparison")
    paired = _eligible(boundary, ("predicted_ms", "reference_ms", "error_ms", "absolute_error_ms")).reset_index(drop=True)
    loop_pairs = _eligible(loops, ("predicted_duration_ms", "reference_duration_ms", "temporal_iou", "duration_error_ms")).reset_index(drop=True)
    feature_pairs = _eligible(features, ("predicted", "reference", "absolute_feature_error"))
    for table, keys in ((paired, ["record_id", "boundary"]), (loop_pairs, ["record_id", "loop"])):
        if not table.empty and table.duplicated(keys).any():
            raise ValueError(f"Accepted reference plot pairs must be unique on {keys}; adjudicate duplicate references first")
    input_state = _prepare_current_reference_outputs(out, (paired, loop_pairs, feature_pairs))
    plots.mkdir(parents=True, exist_ok=True)
    rows = []
    boundary_gallery = all_gallery = None
    boundary_gallery_table = pd.DataFrame()
    all_gallery_table = pd.DataFrame()
    missing = "GROUND_TRUTH_UNAVAILABLE: no finite, independently accepted reference/prediction pairs"

    def unavailable(name, n=0, reason=missing):
        rows.append(_status_row(name, "unavailable", n, None, out, reason))

    if paired.empty:
        unavailable("boundary_absolute_error_distribution")
        unavailable("boundary_error_by_HC_MI")
        unavailable("boundary_bland_altman")
        unavailable("all_reference_boundary_gallery")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, name in zip(axes.flat, BOUNDARIES):
            group = paired.loc[paired.boundary.eq(name)]
            if len(group):
                ax.hist(group.absolute_error_ms, bins=min(25, max(5, int(np.ceil(np.sqrt(len(group)))))), color="#337ca8", edgecolor="white")
            else:
                ax.text(.5, .5, "No eligible pairs", ha="center", transform=ax.transAxes)
            ax.set(title=f"{DISPLAY[name]} (n={len(group)})", xlabel="Absolute boundary error (ms)", ylabel="Recordings")
        path = plots / "boundary_absolute_error_distribution.png"
        _finish(fig, paired, path, "Absolute boundary error against accepted independent references")
        rows.append(_status_row("boundary_absolute_error_distribution", "created", len(paired), path, out))

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, name in zip(axes.flat, BOUNDARIES):
            group = paired.loc[paired.boundary.eq(name)]
            values, labels = [], []
            for label in ("HC", "MI"):
                part = group.loc[group.get("label_name", pd.Series(index=group.index, dtype=str)).eq(label)]
                if len(part):
                    values.append(part.error_ms.to_numpy())
                    labels.append(f"{label} (n={len(part)})")
            if values:
                ax.boxplot(values, tick_labels=labels, showfliers=True, patch_artist=True,
                           boxprops={"facecolor": "#d4e6ee"}, medianprops={"color": "#9c3f26"})
            else:
                ax.text(.5, .5, "HC/MI paired groups unavailable", ha="center", transform=ax.transAxes)
            ax.axhline(0, color=".4", ls="--", lw=.8)
            ax.set(title=DISPLAY[name], ylabel="Predicted − reference (ms)")
        path = plots / "boundary_error_by_HC_MI.png"
        _finish(fig, paired, path, "Signed boundary error by diagnosis (positive = later prediction)")
        rows.append(_status_row("boundary_error_by_HC_MI", "created", len(paired), path, out))

        ba_pairs = paired.groupby("boundary").filter(lambda x: len(x) >= 2)
        if ba_pairs.empty:
            unavailable("boundary_bland_altman", len(paired), "At least two eligible pairs per boundary are required for descriptive limits of agreement")
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for ax, name in zip(axes.flat, BOUNDARIES):
                group = ba_pairs.loc[ba_pairs.boundary.eq(name)]
                _bland_altman(ax, group.get("reference_ms", []), group.get("predicted_ms", []), f"{DISPLAY[name]} (n={len(group)})", "Boundary time")
            path = plots / "boundary_bland_altman.png"
            _finish(fig, ba_pairs, path, "Boundary agreement: mean difference ± 1.96 SD (descriptive, no clinical acceptance claim)")
            rows.append(_status_row("boundary_bland_altman", "created", len(ba_pairs), path, out))

        boundary_gallery_table = paired.copy()
        boundary_cards = []
        for record_id, group in paired.groupby("record_id", sort=True):
            path = plots / "boundary_overlays" / f"{_safe(record_id)}.png"
            _overlay_boundaries(group, artifact_root, path)
            relative_path = path.relative_to(out).as_posix()
            boundary_gallery_table.loc[group.index, "plot_path"] = relative_path
            sources = sorted(set(group.get("reference_source", pd.Series(dtype=str)).dropna().astype(str)))
            loop_names = sorted({"QRS" if name.startswith("qrs") else "T" for name in group.boundary})
            boundary_cards.append({
                "record_id": record_id, "patient_id": group.iloc[0].get("patient_id", ""),
                "label_name": group.iloc[0].get("label_name", ""), "split": group.iloc[0].get("split", ""),
                "loop": ";".join(loop_names), "n_pairs": len(group),
                "n_shifted_pairs": int(group.absolute_error_ms.gt(1e-7).sum()),
                "error_extent_ms": float(group.absolute_error_ms.max()), "temporal_iou": np.nan,
                "reference_source": "; ".join(sources), "plot_path": relative_path,
                "detail": "; ".join(f"{DISPLAY[row.boundary]}: V1 {row.predicted_ms:.1f}, ref {row.reference_ms:.1f}, delta {row.error_ms:+.1f} ms"
                                    for _, row in group.iterrows()),
            })
        boundary_gallery_table.to_csv(out / "all_reference_boundary_gallery_manifest.csv", index=False)
        boundary_gallery = out / "all_reference_boundary_gallery.html"
        _write_complete_reference_gallery(pd.DataFrame(boundary_cards), boundary_gallery, "boundary")
        rows.append(_status_row("all_reference_boundary_gallery", "created", len(paired), boundary_gallery, out,
                                f"All {len(paired)} eligible boundary pairs in {paired.record_id.nunique()} recordings; partial references included"))

    if loop_pairs.empty:
        for name in ("temporal_iou", "duration_scatter", "duration_bland_altman", "best_worst_reference_gallery", "all_reference_loop_gallery"):
            unavailable(name)
        gallery = None
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.7))
        for ax, loop in zip(axes, ("QRS", "T")):
            group = loop_pairs.loc[loop_pairs.loop.eq(loop)]
            if len(group):
                ax.hist(group.temporal_iou, bins=np.linspace(0, 1, 21), color="#2a8b73", edgecolor="white")
            ax.set(title=f"{loop} (n={len(group)})", xlabel="Temporal intersection / union (unitless)", ylabel="Recordings", xlim=(0, 1))
        path = plots / "temporal_iou.png"
        _finish(fig, loop_pairs, path, "Continuous-interval Temporal IoU of predicted and reference loop cuts")
        rows.append(_status_row("temporal_iou", "created", len(loop_pairs), path, out))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.7))
        for ax, loop in zip(axes, ("QRS", "T")):
            group = loop_pairs.loc[loop_pairs.loop.eq(loop)]
            for label, color in (("HC", "#276c9e"), ("MI", "#bd5d36")):
                part = group.loc[group.get("label_name", pd.Series(index=group.index, dtype=str)).eq(label)]
                if len(part):
                    ax.scatter(part.reference_duration_ms, part.predicted_duration_ms, color=color, s=20, alpha=.7, label=f"{label} n={len(part)}")
            if len(group):
                hi = max(group.reference_duration_ms.max(), group.predicted_duration_ms.max()) * 1.05
                ax.plot([0, hi], [0, hi], "--", color=".5", lw=.9, label="Identity")
                ax.set_xlim(0, hi)
                ax.set_ylim(0, hi)
                ax.legend(fontsize=7)
            ax.set(title=f"{loop} (n={len(group)})", xlabel="Reference duration (ms)", ylabel="Predicted duration (ms)")
            ax.set_aspect("equal", adjustable="box")
        path = plots / "duration_scatter.png"
        _finish(fig, loop_pairs, path, "Predicted versus reference loop duration")
        rows.append(_status_row("duration_scatter", "created", len(loop_pairs), path, out))

        ba_loops = loop_pairs.groupby("loop").filter(lambda x: len(x) >= 2)
        if ba_loops.empty:
            unavailable("duration_bland_altman", len(loop_pairs), "At least two accepted pairs per loop required")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.7))
            for ax, loop in zip(axes, ("QRS", "T")):
                group = ba_loops.loc[ba_loops.loop.eq(loop)]
                _bland_altman(ax, group.reference_duration_ms, group.predicted_duration_ms, f"{loop} (n={len(group)})", "Duration")
            path = plots / "duration_bland_altman.png"
            _finish(fig, ba_loops, path, "Duration agreement: mean difference ± 1.96 SD (descriptive)")
            rows.append(_status_row("duration_bland_altman", "created", len(ba_loops), path, out))
        # Render every eligible recording/loop pair once.  The extremes below
        # reuse this complete population rather than choosing what to visualize.
        complete_rows = []
        for _, row in loop_pairs.sort_values(["record_id", "loop"]).iterrows():
            path = plots / "loop_overlays" / f"{_safe(row.record_id)}_{row.loop}.png"
            _overlay_loop(row, artifact_root, path)
            start_error = float(row.predicted_start_ms - row.reference_start_ms)
            end_error = float(row.predicted_end_ms - row.reference_end_ms)
            item = row.to_dict()
            item.update(plot_path=path.relative_to(out).as_posix(), start_error_ms=start_error,
                        end_error_ms=end_error, error_extent_ms=max(abs(start_error), abs(end_error)),
                        n_pairs=1, n_shifted_pairs=int(max(abs(start_error), abs(end_error)) > 1e-7),
                        detail=(f"Start: V1 {row.predicted_start_ms:.1f}, ref {row.reference_start_ms:.1f}, delta {start_error:+.1f} ms; "
                                f"end: V1 {row.predicted_end_ms:.1f}, ref {row.reference_end_ms:.1f}, delta {end_error:+.1f} ms; "
                                f"IoU {row.temporal_iou:.3f}; missed {row.missed_ms:.1f} ms; excess {row.excess_ms:.1f} ms"))
            complete_rows.append(item)
        all_gallery_table = pd.DataFrame(complete_rows)
        all_gallery_table.to_csv(out / "all_reference_loop_gallery_manifest.csv", index=False)
        all_gallery = out / "all_reference_loop_gallery.html"
        _write_complete_reference_gallery(all_gallery_table, all_gallery, "loop")
        rows.append(_status_row("all_reference_loop_gallery", "created", len(all_gallery_table), all_gallery, out,
                                f"All {len(all_gallery_table)} eligible recording/loop pairs; no highest/lowest subset selection"))

        gallery_rows = []
        for loop in ("QRS", "T"):
            group = loop_pairs.loc[loop_pairs.loop.eq(loop)].sort_values(["temporal_iou", "record_id"], ascending=[False, True])
            selected = pd.concat([group.head(3).assign(rank_group="highest_IoU"), group.tail(3).assign(rank_group="lowest_IoU")])
            for _, row in selected.drop_duplicates(["record_id", "loop", "rank_group"]).iterrows():
                path = plots / "loop_overlays" / f"{_safe(row.record_id)}_{loop}.png"
                gallery_rows.append({"record_id": row.record_id, "loop": loop, "rank_group": row.rank_group,
                                     "temporal_iou": row.temporal_iou, "missed_ms": row.missed_ms,
                                     "excess_ms": row.excess_ms, "reference_source": row.get("reference_source", ""),
                                     "plot_path": path.relative_to(out).as_posix()})
        gallery_table = pd.DataFrame(gallery_rows)
        gallery_table.to_csv(out / "reference_loop_gallery_manifest.csv", index=False)
        gallery = out / "reference_loop_gallery.html"
        _write_reference_gallery(gallery_table, gallery)
        rows.append(_status_row("best_worst_reference_gallery", "created", len(gallery_rows), gallery, out,
                                "Extremes describe accepted reviewed cases only; low IoU is not an adjudication of reference quality"))

    relative = _eligible(feature_pairs, ("relative_feature_error_pct",))
    if relative.empty:
        unavailable("feature_reference_change_heatmap", len(feature_pairs),
                    "No accepted finite relative feature differences; missing / near-zero references are excluded")
    else:
        columns = ["All", "HC", "MI"]
        matrix = np.full((len(FEATURE_NAMES), len(columns)), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        for i, name in enumerate(FEATURE_NAMES):
            for j, label in enumerate(columns):
                group = relative.loc[relative.feature.eq(name)]
                if label != "All":
                    group = group.loc[group.get("label_name", pd.Series(index=group.index, dtype=str)).eq(label)]
                counts[i, j] = len(group)
                if len(group):
                    matrix[i, j] = group.relative_feature_error_pct.median()
        fig, ax = plt.subplots(figsize=(9.2, 8))
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="YlOrRd", vmin=0)
        fig.colorbar(im, ax=ax, label="Median absolute relative feature difference (%)")
        ax.set(yticks=range(len(FEATURE_NAMES)), yticklabels=FEATURE_NAMES, xticks=range(3), xticklabels=columns)
        for i in range(len(FEATURE_NAMES)):
            for j in range(3):
                label = f"{matrix[i,j]:.2f}%\nn={counts[i,j]}" if np.isfinite(matrix[i,j]) else "Unavailable\nn=0"
                ax.text(j, i, label, ha="center", va="center", fontsize=7)
        path = plots / "feature_reference_change_heatmap.png"
        _finish(fig, relative, path, "Feature change from reference boundary cuts (not ML feature importance)")
        rows.append(_status_row("feature_reference_change_heatmap", "created", len(relative), path, out,
                                "Near-zero reference denominators are excluded; see feature comparison status/epsilon"))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "plots_status.csv", index=False)
    return {"manifest": manifest, "gallery": gallery, "all_gallery": all_gallery,
            "boundary_gallery": boundary_gallery, "all_gallery_manifest": all_gallery_table,
            "boundary_gallery_manifest": boundary_gallery_table,
            "n_visualized_loop_pairs": len(all_gallery_table), "n_visualized_boundary_pairs": len(boundary_gallery_table),
            "n_visualized_boundary_records": paired.record_id.nunique() if not paired.empty else 0,
            "n_evaluated_pairs": len(paired),
            "n_evaluated_loop_pairs": len(loop_pairs), "n_evaluated_feature_pairs": len(feature_pairs),
            "input_state": input_state}


def _bland_altman(ax, reference, predicted, title: str, unit_name: str) -> None:
    reference, predicted = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    if len(reference) < 2:
        ax.text(.5, .5, "At least two eligible pairs required", ha="center", transform=ax.transAxes)
    else:
        delta = predicted - reference
        bias, sd = float(np.mean(delta)), float(np.std(delta, ddof=1))
        ax.scatter((reference + predicted) / 2, delta, s=20, alpha=.65, color="#337ca8")
        for value, label, style in ((bias, f"Bias {bias:.2f} ms", "-"), (bias + 1.96*sd, f"Upper {bias+1.96*sd:.2f} ms", "--"), (bias - 1.96*sd, f"Lower {bias-1.96*sd:.2f} ms", "--")):
            ax.axhline(value, color="#a34d28", ls=style, lw=.9, label=label)
        ax.legend(fontsize=7)
    ax.set(title=title, xlabel=f"Mean predicted/reference {unit_name.lower()} (ms)", ylabel="Predicted − reference (ms)")
    ax.grid(alpha=.15)


def _write_reference_gallery(frame: pd.DataFrame, target: Path) -> None:
    cards = []
    for _, row in frame.iterrows():
        title = f"{row.record_id} | {row.loop} | {row.rank_group} | IoU={row.temporal_iou:.3f}"
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>Reference: {html.escape(str(row.reference_source))}. '
                     f'Missed {row.missed_ms:.1f} ms; excess {row.excess_ms:.1f} ms.</p>'
                     f'<a href="{html.escape(row.plot_path, quote=True)}"><img loading="lazy" src="{html.escape(row.plot_path, quote=True)}" alt="{html.escape(title, quote=True)}"></a></article>')
    target.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Accepted reference loop comparisons</title>'
                      '<style>body{font:15px system-ui;margin:2rem auto;max-width:1400px;color:#263743}article{margin:2rem 0;border-top:1px solid #ccc}img{max-width:100%}h2{font-size:19px}</style>'
                      '<h1>Loop comparisons against independently accepted references</h1><p>Highest and lowest Temporal IoU within reviewed cases only. '
                      'Priority sampling and repeated recordings limit cohort generalization. Spatial curves illustrate interval membership, not a geometric accuracy metric.</p>'
                      + ''.join(cards) + '</html>', encoding="utf-8")


def _write_complete_reference_gallery(frame: pd.DataFrame, target: Path, kind: str) -> None:
    """Filterable complete population; numeric sorting never removes cases."""
    boundary = kind == "boundary"
    title = "All accepted reference boundary comparisons" if boundary else "All accepted reference loop comparisons"
    n_pairs = int(frame.n_pairs.sum())
    n_shifted = int(frame.n_shifted_pairs.sum())
    n_shifted_cards = int(frame.error_extent_ms.gt(1e-7).sum())
    pairs_label = "boundary pairs" if boundary else "recording/loop pairs"
    overview = (f"{n_pairs} eligible {pairs_label}; {frame.record_id.nunique()} recordings; "
                f"{n_shifted} pairs differ by more than 1e-7 ms (numerical equality tolerance). "
                f"{n_shifted_cards} of {len(frame)} displayed comparisons contain a nonzero shift. "
                "A nonzero shift is a measured difference, not a clinical failure threshold.")
    cards = []
    for _, row in frame.iterrows():
        heading = f"{row.record_id} | {row.get('label_name', '')} | {row.get('split', '')} | {row.loop}"
        iou = float(row.temporal_iou)
        attrs = {"record": row.record_id, "label": row.get("label_name", ""), "split": row.get("split", ""),
                 "loop": row.loop, "error": row.error_extent_ms, "iou": iou if np.isfinite(iou) else "",
                 "pairs": row.n_pairs, "shifted": row.n_shifted_pairs,
                 "source": row.get("reference_source", "")}
        data = " ".join(f'data-{key}="{html.escape(str(value), quote=True)}"' for key, value in attrs.items())
        url = html.escape(str(row.plot_path), quote=True)
        cards.append(f'<article {data}><h2>{html.escape(heading)}</h2>'
                     f'<p><b>Largest absolute edge shift: {row.error_extent_ms:.1f} ms.</b> '
                     f'{html.escape(str(row.detail))}</p>'
                     f'<p>Reference source(s): {html.escape(str(row.get("reference_source", "")))}</p>'
                     f'<a href="{url}"><img loading="lazy" src="{url}" alt="{html.escape(heading, quote=True)}"></a></article>')
    # JSON and HTML escaping are separate: no source metadata is executed as JS.
    iou_control = ('<label>Maximum IoU <input id="iou" type="number" min="0" max="1" step="0.01" value="1"></label>'
                   if not boundary else '<input id="iou" type="hidden" value="1">')
    labels = sorted(set(frame.get("label_name", pd.Series(dtype=str)).fillna("").astype(str)) - {""})
    label_options = ''.join(f'<option>{html.escape(label)}</option>' for label in labels)
    target.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title}</title><style>body{{font:15px system-ui;margin:1.5rem auto;max-width:1450px;color:#263743;line-height:1.5;padding:0 12px}}'
        'article{margin:2rem 0;border-top:1px solid #ccc}article[hidden]{display:none}img{max-width:100%;height:auto}h2{font-size:19px}'
        '.controls{position:sticky;top:0;background:#f8fafb;padding:10px;z-index:10;border:1px solid #ccd6dc;display:flex;flex-wrap:wrap;gap:10px}'
        'select,input{padding:5px;max-width:180px}input[type=number]{width:72px}#count{font-weight:600;width:100%}</style>'
        f'<h1>{title}</h1><p><b>{html.escape(overview)}</b></p>'
        '<p>Every eligible pair is visualized. Positive V1 − reference means V1 is later; negative means earlier. '
        'Only independently accepted annotations are used. These reviewed cases and repeated recordings do not establish cohort-wide accuracy.</p>'
        '<div class="controls"><label>Diagnosis <select id="label"><option value="">All</option>' + label_options + '</select></label>'
        '<label>Split <select id="split"><option value="">All</option><option>train</option><option>valid</option><option>test</option></select></label>'
        '<label>Loop <select id="loop"><option value="">All</option><option>QRS</option><option>T</option></select></label>'
        '<label>Minimum edge shift (ms) <input id="error" type="number" min="0" step="1" value="0"></label>' + iou_control +
        '<label>Find recording/source <input id="query" type="search" placeholder="record ID or source"></label>'
        '<label>Order <select id="sort"><option value="error_desc">Largest edge shift</option><option value="error_asc">Smallest edge shift</option>'
        '<option value="iou_asc">Lowest IoU</option><option value="record">Recording ID</option></select></label>'
        '<span id="count"></span></div><main id="cases">' + ''.join(cards) + '</main>'
        '<script>const ids=["label","split","loop","error","iou","query","sort"],c=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));'
        'const cards=Array.from(document.querySelectorAll("article")),holder=document.getElementById("cases");'
        'function filter(){let n=0,p=0,s=0;const q=c.query.value.trim().toLowerCase();'
        'cards.forEach(a=>{const d=a.dataset;let show=(!c.label.value||d.label===c.label.value)&&(!c.split.value||d.split===c.split.value)'
        '&&(!c.loop.value||d.loop.split(";").includes(c.loop.value))&&Number(d.error)>=Number(c.error.value||0)'
        '&&(d.iou===""||Number(d.iou)<=Number(c.iou.value||1))&&(!q||(d.record+" "+d.source).toLowerCase().includes(q));'
        'a.hidden=!show;if(show){n++;p+=Number(d.pairs);s+=Number(d.shifted);}});'
        'cards.sort((a,b)=>{let d=0;if(c.sort.value==="error_desc")d=Number(b.dataset.error)-Number(a.dataset.error);'
        'else if(c.sort.value==="error_asc")d=Number(a.dataset.error)-Number(b.dataset.error);'
        'else if(c.sort.value==="iou_asc")d=(a.dataset.iou===""?Infinity:Number(a.dataset.iou))-(b.dataset.iou===""?Infinity:Number(b.dataset.iou));'
        'return d||a.dataset.record.localeCompare(b.dataset.record)||a.dataset.loop.localeCompare(b.dataset.loop);}).forEach(a=>holder.appendChild(a));'
        'document.getElementById("count").textContent=n+" / "+cards.length+" comparisons shown; "+p+" eligible pairs; "+s+" pairs with nonzero shift";}'
        'ids.forEach(id=>c[id].addEventListener("input",filter));filter();</script></html>', encoding="utf-8")


def _boundary_time(boundaries: dict, key: str, time: np.ndarray) -> float | None:
    index = boundaries.get(key)
    if index is None or not np.isfinite(index) or int(index) != index or not 0 <= index < len(time):
        return None
    return float(time[int(index)])


def _v1_explanation(artifact: dict, arrays: dict) -> str:
    if "median_xyz" not in arrays:
        qc = artifact.get("beat_qc", {})
        return ("V1 did not create a representative median beat. Median-relative boundary or loop comparisons are unavailable. "
                "Review the saved detector / beat-selection diagnostics and raw XYZ before arranging an independent raw-beat annotation. "
                f"Saved beat selection: {json.dumps(qc, ensure_ascii=False)}")
    t = arrays["time_ms"]
    b = artifact.get("boundaries", {})
    def stamp(key):
        value = _boundary_time(b, key, t)
        return f"{value:.1f} ms" if value is not None else "unavailable"
    pieces = [f"Saved V1 QRS cut: {stamp('qrs_on')} to {stamp('qrs_off')}. "
              f"It selects the speed peak at {stamp('qrs_speed_peak')} and applies its frozen velocity threshold "
              f"{b.get('q_threshold', 'unavailable')} mV/s within the configured search window.",
              f"Saved T method: {b.get('t_method', 'unavailable')}; T-on {stamp('t_on')}, T-peak {stamp('t_peak')}, T-off {stamp('t_off')}.",
              f"Rising / falling tangent contacts: {stamp('t_rise_index')} / {stamp('t_fall_index')}. "
              f"ST level: {b.get('st_level', 'unavailable')} mV; T-end baseline: {b.get('t_end_baseline', 'unavailable')} mV.",
              f"Threshold candidates: T-on {stamp('t_threshold_on')}, T-off {stamp('t_threshold_off')}; "
              f"tangent candidates: T-on {stamp('t_tangent_on')}, T-off {stamp('t_tangent_off')}."]
    pieces.append("These candidates share the same median signal and baseline processing; they are not independent ground truth. "
                  "QC, unusual duration, ST level, or method disagreement alone does not establish an incorrect boundary.")
    return " ".join(pieces)


def create_diagnostic_case_gallery(frame: pd.DataFrame, priority: pd.DataFrame,
                                   artifact_root: Path, output_dir: Path, cfg=None,
                                   max_cases: int = 36) -> dict:
    """List every priority case; render a new diagnostic subset without altering V1.

    Existing Notebook 07 PNGs are linked read-only for all available cases.  New
    figures reproduce the saved thresholds, contacts and XYZ loops.  Test cases
    remain explicitly descriptive and cannot be used for algorithm selection.
    """
    from scripts.vcg_qc_audit_gallery import _median_panels
    out = Path(output_dir)
    plotdir = out / "plots" / "diagnostic_cases"
    plotdir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or Config()
    if "record_id" not in priority:
        raise ValueError("Priority table requires record_id")
    ids = priority.record_id.astype(str)
    if ids.duplicated().any():
        raise ValueError("Priority records must be unique")
    base = frame.set_index("record_id", drop=False)
    if base.index.duplicated().any():
        raise ValueError("Input frame must have one row per recording")
    order = priority.sort_values(["priority_score", "record_id"], ascending=[False, True]) if "priority_score" in priority else priority
    # Select high priority plus at least one example per semicolon-delimited reason.
    selected = []
    for _, row in order.iterrows() if max_cases > 0 else []:
        if str(row.get("split", base.loc[row.record_id].get("split", ""))) == "test":
            continue
        reasons = str(row.get("priority_reasons", row.get("reasons", ""))).split(";")
        for reason in reasons:
            if reason.strip() and not any(reason.strip() in str(pr.get("priority_reasons", "")).split(";") for pr in selected):
                selected.append(row)
                break
        if len(selected) >= max_cases:
            break
    chosen = {str(row.record_id) for row in selected}
    for _, row in order.iterrows():
        if len(chosen) >= max_cases:
            break
        if str(row.get("split", base.loc[row.record_id].get("split", ""))) != "test":
            chosen.add(str(row.record_id))
    manifest = []
    old = REPO / "outputs" / "vcg_qc_audit_v1"
    sourcehash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for _, case in order.iterrows():
        record_id = str(case.record_id)
        if record_id not in base.index:
            raise ValueError(f"Priority record is absent from the cohort: {record_id}")
        record = base.loc[record_id]
        artifact, arrays = _load_artifact(artifact_root, record_id)
        has_median = "median_xyz" in arrays
        old_path = old / "plots" / (str(record.label_name) if has_median else "failure_cases") / (_safe(record_id) + ".png")
        new_path = plotdir / (_safe(record_id) + ".png")
        rendered = False
        if record_id in chosen and has_median:
            fingerprint = hashlib.sha256((sourcehash + json.dumps(artifact, sort_keys=True, allow_nan=True) +
                          hashlib.sha256((_artifacts(artifact_root) / (_safe(record_id) + ".npz")).read_bytes()).hexdigest()).encode()).hexdigest()
            checkpoint = new_path.with_suffix(".json")
            previous = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            if not new_path.exists() or previous.get("fingerprint") != fingerprint:
                fig = plt.figure(figsize=(17, 11.8))
                _median_panels(fig, artifact, arrays, cfg)
                title = f"DIAGNOSTIC ONLY — {record_id} | {record.label_name} | {record.get('split', '')} | prediction display; reference not shown"
                fig.suptitle(title, fontsize=11, y=.99)
                fig.text(.03, .015, "Frozen V1 output: threshold / tangent candidates are model diagnostics, not ground truth. Median-relative time in ms; Frank XYZ in mV.", fontsize=8)
                fig.subplots_adjust(left=.055, right=.965, bottom=.065, top=.93, hspace=.30, wspace=.22)
                fig.savefig(new_path, dpi=112, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                checkpoint.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
                rendered = True
        path = new_path if record_id in chosen and has_median else old_path if old_path.exists() else None
        manifest.append({"record_id": record_id, "patient_id": record.get("patient_id", ""),
                         "label_name": record.label_name, "split": record.get("split", ""),
                         "priority_reasons": "" if pd.isna(case.get("priority_reasons", case.get("reasons", ""))) else str(case.get("priority_reasons", case.get("reasons", ""))),
                         "priority_score": case.get("priority_score", np.nan), "has_median": has_median,
                         "new_diagnostic_selected": record_id in chosen, "new_plot_rendered": rendered,
                         "plot_path": os.path.relpath(path, out) if path else "",
                         "notebook07_plot": os.path.relpath(old_path, out) if old_path.exists() else "",
                         "evaluation_mode": "diagnostic_only", "interpretation": _v1_explanation(artifact, arrays),
                         "test_scope": "descriptive only; no tuning" if record.get("split") == "test" else "train/valid review priority"})
    table = pd.DataFrame(manifest)
    table.to_csv(out / "diagnostic_gallery_manifest.csv", index=False)
    target = out / "diagnostic_gallery.html"
    _write_diagnostic_html(table, target)
    return {"manifest": table, "gallery": target, "n_new_figures": int(table.new_plot_rendered.sum()) if len(table) else 0,
            "n_cases": len(table)}


def _write_diagnostic_html(frame: pd.DataFrame, target: Path) -> None:
    cards = []
    for _, row in frame.iterrows():
        title = f"{row.record_id} | {row.label_name} | {row.split}"
        img = (f'<a href="{html.escape(row.plot_path, quote=True)}"><img loading="lazy" src="{html.escape(row.plot_path, quote=True)}" alt="{html.escape(title, quote=True)}"></a>'
               if row.plot_path else '<p>Diagnostic image unavailable; inspect the saved artifact and review tool.</p>')
        cards.append(f'<article data-label="{html.escape(str(row.label_name), quote=True)}" '
                     f'data-split="{html.escape(str(row.split), quote=True)}" '
                     f'data-reasons="{html.escape(str(row.priority_reasons), quote=True)}">'
                     f'<h2>{html.escape(title)}</h2><p><b>Priority indicators:</b> {html.escape(str(row.priority_reasons))}</p>'
                     f'<p>{html.escape(str(row.interpretation))}</p><p><em>{html.escape(str(row.test_scope))}</em></p>{img}</article>')
    reasons = sorted({reason.strip() for value in frame.get("priority_reasons", pd.Series(dtype=str)).fillna("").astype(str) for reason in value.split(";") if reason.strip()})
    options = ''.join(f'<option value="{html.escape(reason, quote=True)}">{html.escape(reason)}</option>' for reason in reasons)
    target.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Notebook 08 diagnostic cases</title>'
                      '<style>body{font:15px system-ui;margin:2rem auto;max-width:1400px;color:#263743;line-height:1.5}article{margin:2rem 0;border-top:1px solid #ccc}img{max-width:100%}h2{font-size:19px}select{padding:6px;margin-right:1rem}.controls{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #ddd}</style>'
                      '<h1>Diagnostic cases: frozen V1 predictions</h1><p>These plots do not establish boundary error. Saved Notebook 07 images are reused read-only where available. '
                      'Candidates share preprocessing and are not independent references. Use the blinded annotation tool before viewing these diagnostic predictions.</p>'
                      '<div class="controls"><label>Diagnosis <select id="label"><option value="">All</option><option>HC</option><option>MI</option></select></label>'
                      '<label>Split <select id="split"><option value="">All</option><option>train</option><option>valid</option><option>test</option></select></label>'
                      '<label>Indicator <select id="reason"><option value="">All</option>' + options + '</select></label><span id="count"></span></div>'
                      + ''.join(cards) + '<script>const label=document.getElementById("label"),split=document.getElementById("split"),reason=document.getElementById("reason"),count=document.getElementById("count");function filter(){let n=0;document.querySelectorAll("article").forEach(a=>{let visible=(!label.value||a.dataset.label===label.value)&&(!split.value||a.dataset.split===split.value)&&(!reason.value||a.dataset.reasons.split(";").map(x=>x.trim()).includes(reason.value));a.hidden=!visible;if(visible)n++;});count.textContent=n+" cases";}document.querySelectorAll("select").forEach(x=>x.addEventListener("change",filter));filter();</script></html>', encoding="utf-8")
