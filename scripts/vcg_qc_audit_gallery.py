"""Read-only diagnostic gallery for the frozen V1 QRS/T audit (Notebook 07).

Saved arrays and indices are displayed without rerunning delineation. The five
records without a median beat use only a fresh *raw XYZ read*, together with
saved detector / beat-selection diagnostics. A plotted boundary is a prediction,
not a verified reference. Plot fingerprints include this source, input artifacts,
record metadata, Config, and (for raw failure strips) the raw signal files.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from collections import Counter
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
from vcg_preprocessing import Config

COLORS = {"qrs_on": "#1372b4", "qrs_off": "#1372b4", "t_on": "#d66c14",
          "t_peak": "#b34591", "t_off": "#d66c14"}
NAMES = {"qrs_on": "QRS-on", "qrs_off": "QRS-off", "t_on": "T-on",
         "t_peak": "T-peak", "t_off": "T-off"}


def _flags(value: Any) -> list[str]:
    if value is None or isinstance(value, float) and not np.isfinite(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(x).strip() for x in value if str(x).strip()})
    return sorted({s.strip() for s in str(value).split(";") if s.strip()})


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _idx(b: dict, name: str, n: int) -> int | None:
    value = b.get(name)
    if value is None or not np.isfinite(value) or int(value) != value:
        return None
    value = int(value)
    return value if 0 <= value < n else None


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


def _span(ax: Any, t: np.ndarray, b: dict, prefix: str) -> None:
    a, z = _idx(b, prefix + "_on", len(t)), _idx(b, prefix + "_off", len(t))
    if a is not None and z is not None and a < z:
        ax.axvspan(t[a], t[z], color=COLORS[prefix + "_on"], alpha=.095)


def _boundaries(ax: Any, t: np.ndarray, b: dict, keys: tuple = tuple(NAMES)) -> None:
    for key in keys:
        i = _idx(b, key, len(t))
        if i is not None:
            ax.axvline(t[i], color=COLORS[key], ls=":" if key.endswith("peak") else "--",
                       lw=1, label=f"{NAMES[key]} {t[i]:.0f} ms")


def _legend(ax: Any, ncol: int = 3) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6.6, ncol=ncol, loc="best", framealpha=.87)


def _annotated_tangent(ax: Any, t: np.ndarray, mag: np.ndarray, slope: np.ndarray | None,
                       b: dict, side: str, xlo: float, xhi: float) -> None:
    contact = _idx(b, "t_rise_index" if side == "on" else "t_fall_index", len(t))
    baseline = b.get("st_level" if side == "on" else "t_end_baseline")
    boundary = _idx(b, "t_tangent_" + side, len(t))
    color = "#087f8c" if side == "on" else "#9b4dca"
    if contact is None:
        return
    x, y = float(t[contact]), float(mag[contact])
    ax.scatter([x], [y], s=40, color=color, marker="o", zorder=6,
               label=f"T-{side} contact {x:.0f} ms")
    ax.annotate("rise contact" if side == "on" else "fall contact", (x, y),
                xytext=(5, 9 if side == "on" else -17), textcoords="offset points",
                fontsize=7, color=color)
    if slope is None or not np.isfinite(slope[contact]) or abs(slope[contact]) < 1e-12:
        return
    k = float(slope[contact])  # mV / s; the x axis is milliseconds.
    if boundary is None or not _finite(baseline):
        # An unresolved tangent must not be displayed as an accepted boundary.
        short = np.array([max(xlo, x - 18), min(xhi, x + 18)])
        ax.plot(short, y + k * (short - x) / 1000, color=color, lw=1.1, ls=":",
                label=f"T-{side} local slope; intersection unresolved")
        return
    crossing = x + (float(baseline) - y) / k * 1000
    xx = np.array([max(xlo, min(x, crossing) - 12), min(xhi, max(x, crossing) + 12)])
    ax.plot(xx, y + k * (xx - x) / 1000, color=color, lw=1.5,
            label=f"T-{side} tangent ({k:.2f} mV/s)")
    ax.scatter([crossing], [baseline], s=48, marker="D", facecolors="white",
               edgecolors=color, linewidths=1.3, zorder=7,
               label=f"T-{side} intersection {crossing:.1f} ms")


def _median_panels(fig: Any, artifact: dict, arrays: dict, cfg: Config) -> None:
    xyz, t = arrays["median_xyz"], arrays["time_ms"]
    n = len(t)
    if xyz.shape != (n, 3):
        raise ValueError("Saved median XYZ and time array are inconsistent")
    b = artifact.get("boundaries", {})
    fs = float(artifact["fs"])
    axa = fig.add_subplot(2, 2, 1)
    axb = fig.add_subplot(2, 2, 2)
    axc = fig.add_subplot(2, 2, 3)
    axd = fig.add_subplot(2, 2, 4, projection="3d")
    for j, color in enumerate(("#176da0", "#b44534", "#37834e")):
        axa.plot(t, xyz[:, j], lw=1.05, color=color, label="XYZ"[j])
    _span(axa, t, b, "qrs")
    _span(axa, t, b, "t")
    _boundaries(axa, t, b)
    axa.axvline(0, color=".45", lw=.7, ls=":")
    pq = artifact.get("pq_indices")
    if pq is not None and len(pq) == 2 and 0 <= pq[0] < pq[1] <= n:
        axa.axvspan(t[pq[0]], t[pq[1] - 1], color=".4", alpha=.1, label="PQ reference window")
    axa.set(title="A. Saved representative median Frank XYZ", xlabel="Time relative to QRS anchor (ms)", ylabel="mV")
    axa.grid(alpha=.16)
    _legend(axa, 4)

    speed, smooth = arrays.get("speed"), arrays.get("smooth_speed")
    if speed is not None:
        axb.plot(t, speed, lw=.8, color=".65", label="Spatial speed (SG derivative)")
    if smooth is not None:
        axb.plot(t, smooth, lw=1.25, color="#176da0", label="5 ms smoothed spatial speed")
    if _finite(b.get("q_threshold")):
        axb.axhline(b["q_threshold"], color="#ba5a17", ls="--", lw=1, label=f"Threshold {b['q_threshold']:.3f} mV/s")
    axb.axvline(0, color="black", lw=.9, ls=":", label="QRS anchor 0 ms")
    _boundaries(axb, t, b, ("qrs_on", "qrs_off"))
    qpeak = _idx(b, "qrs_speed_peak", n)
    if qpeak is not None and smooth is not None:
        axb.scatter([t[qpeak]], [smooth[qpeak]], color="#b44534", s=26, zorder=5,
                    label=f"Selected speed peak {t[qpeak]:.0f} ms")
    for limit in cfg.qrs_search_ms:
        axb.axvline(limit, color=".6", ls=":", lw=.7)
    axb.set_xlim(max(t[0], cfg.qrs_search_ms[0] - 15), min(t[-1], cfg.qrs_search_ms[1] + 15))
    _span(axb, t, b, "qrs")
    axb.set(title="B. QRS velocity / threshold; dotted gray = search limits", xlabel="Time relative to QRS anchor (ms)", ylabel="mV/s")
    axb.grid(alpha=.16)
    _legend(axb, 2)

    mag = arrays.get("t_magnitude", arrays.get("magnitude"))
    if mag is None:
        raise ValueError("Saved delineation has no magnitude array")
    qoff = _idx(b, "qrs_off", n)
    search = b.get("t_search", [None, None])
    left = max(0, qoff - int(.015 * fs)) if qoff is not None else max(0, int(artifact.get("median_anchor_sample", 0)))
    right = n - 1
    if search and search[1] is not None:
        right = min(n - 1, int(search[1]) + int(.020 * fs))
    xlo, xhi = float(t[left]), float(t[right])
    axc.plot(t, mag, lw=1.4, color="#223745", label="Smoothed T magnitude")
    rawmag = arrays.get("magnitude")
    if rawmag is not None:
        axc.plot(t, rawmag, lw=.55, color=".7", alpha=.7, label="Unsmoothed magnitude")
    refs = [("st_level", "ST reference", "#d66c14"), ("t_end_baseline", "TP reference (PQ fallback)", "#7c3aed")]
    for key, title, color in refs:
        if _finite(b.get(key)):
            actual_title = "TP reference" if key == "t_end_baseline" and b.get("tp") is not None else title
            axc.axhline(b[key], color=color, ls="--", lw=.9, label=f"{actual_title} {b[key]:.4f} mV")
    if pq is not None and len(pq) == 2:
        pq_level = float(np.median(mag[int(pq[0]):int(pq[1])]))
        axc.axhline(pq_level, color=".4", ls=":", lw=.9, label=f"PQ magnitude reference {pq_level:.4f} mV")
    tp = b.get("tp")
    if tp is not None and 0 <= tp[0] < tp[1] <= n:
        axc.axvspan(t[tp[0]], t[tp[1] - 1], color="#7c3aed", alpha=.08, label="TP reference window")
    _span(axc, t, b, "t")
    _boundaries(axc, t, b, ("t_on", "t_peak", "t_off"))
    peak = _idx(b, "t_peak", n)
    if peak is not None:
        axc.scatter([t[peak]], [mag[peak]], color=COLORS["t_peak"], marker="^", s=42, zorder=6)
    for side in ("on", "off"):
        _annotated_tangent(axc, t, mag, arrays.get("t_slope"), b, side, xlo, xhi)
    for key, color, ls, title in [("t_threshold_off", "#4b4667", "-.", "Threshold T-off"),
                                  ("t_threshold_on", "#747f46", ":", "Threshold T-on")]:
        i = _idx(b, key, n)
        if i is not None:
            axc.axvline(t[i], color=color, ls=ls, lw=.95, label=f"{title} {t[i]:.0f} ms")
    if _finite(b.get("t_end_threshold")):
        axc.axhline(b["t_end_threshold"], color="#4b4667", lw=.7, ls="-.", label="T-end threshold level")
    axc.set_xlim(xlo, xhi)
    visible = mag[left:right + 1]
    refvals = [b[k] for k in ("st_level", "t_end_baseline", "t_end_threshold") if _finite(b.get(k))]
    ymax = max([float(np.max(visible))] + refvals) if len(visible) else float(np.max(mag))
    ymin = min(0, float(np.min(visible))) if len(visible) else 0
    pad = max((ymax - ymin) * .12, .002)
    axc.set_ylim(ymin - pad, ymax + pad)
    axc.set(title=f"C. T magnitude / contacts / extrapolation ({b.get('t_method', 'unresolved')})",
            xlabel="Time relative to QRS anchor (ms)", ylabel="Magnitude (mV)")
    axc.grid(alpha=.16)
    _legend(axc, 2)

    allpoints = []
    for prefix, color in (("qrs", COLORS["qrs_on"]), ("t", COLORS["t_on"])):
        a, z = _idx(b, prefix + "_on", n), _idx(b, prefix + "_off", n)
        if a is not None and z is not None and a < z:
            loop = xyz[a:z + 1]
            allpoints.append(loop)
            axd.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, lw=1.35, label=prefix.upper() + " loop")
            axd.scatter(*loop[0], color=color, marker="o", s=35, label=prefix.upper() + " start")
            axd.scatter(*loop[-1], color=color, marker="x", s=42, label=prefix.upper() + " end")
    if allpoints:
        points = np.vstack(allpoints)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(float(np.ptp(points, axis=0).max()) / 2, .005) * 1.10
        axd.set_xlim(center[0] - radius, center[0] + radius)
        axd.set_ylim(center[1] - radius, center[1] + radius)
        axd.set_zlim(center[2] - radius, center[2] + radius)
        axd.set_box_aspect((1, 1, 1))
    else:
        axd.text2D(.1, .5, "No valid boundary pair; loop unavailable", transform=axd.transAxes)
    axd.set(title="D. Predicted QRS / T loops in saved PQ coordinates", xlabel="X (mV)", ylabel="Y (mV)", zlabel="Z (mV)")
    axd.view_init(elev=22, azim=40)
    _legend(axd, 2)


def _failure_panels(fig: Any, artifact: dict, arrays: dict, raw_path: Path, cfg: Config) -> None:
    # Read only raw XYZ; no preprocessing, detection, beat selection or delineation.
    from vcg_pipeline import load_xyz
    raw = load_xyz(raw_path, seconds=min(8., cfg.seconds))
    xyz, fs = raw["xyz"], raw["fs"]
    t = np.arange(len(xyz)) / fs
    anchors = np.asarray(artifact.get("anchors", arrays.get("anchors", [])), dtype=int)
    axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    for j in range(3):
        axes[0].plot(t, xyz[:, j], lw=.65, label="XYZ"[j])
    for k, anchor in enumerate(anchors[anchors < len(xyz)]):
        axes[0].axvline(anchor / fs, color="#d66c14", alpha=.4, lw=.65, label="Saved QRS anchors" if k == 0 else None)
    axes[0].set(title="A. Raw Frank XYZ first 8 s — NO median beat", xlabel="Raw recording time (s)", ylabel="mV")
    _legend(axes[0], 4)
    for key, color, label in (("detection_envelope", "#176da0", "Saved detection envelope"),
                               ("detection_threshold", "#b44534", "Saved adaptive threshold")):
        a = arrays.get(key)
        if a is not None:
            stop = min(len(a), len(xyz))
            axes[1].plot(np.arange(stop) / fs, a[:stop], color=color, lw=.9, label=label)
    axes[1].set(title="B. Saved detector diagnostics — no detector rerun", xlabel="Raw recording time (s)", ylabel="Envelope (mV)")
    _legend(axes[1], 2)
    if len(anchors) > 1:
        rr = np.diff(anchors) / fs
        med = float(np.median(rr))
        axes[2].plot(anchors[1:] / fs, rr * 1000, "o-", color="#176da0", markersize=3, lw=.8, label="Saved anchor RR intervals")
        axes[2].axhline(med * 1000, color="black", ls=":", label=f"Median RR {med * 1000:.1f} ms")
        for ratio in (.70, 1.35):
            axes[2].axhline(ratio * med * 1000, color="#b44534", ls="--", lw=.8,
                            label=f"Beat-selection {ratio:.2f} × median")
        axes[2].set_ylim(bottom=0)
    axes[2].set(title="C. RR intervals over saved recording; cutoffs are V1 rules", xlabel="Raw recording time (s)", ylabel="RR (ms)")
    _legend(axes[2], 2)
    counts = Counter(r.get("reason", "unknown") for r in artifact.get("beat_qc", []))
    axes[3].axis("off")
    lines = ["D. NO REPRESENTATIVE MEDIAN BEAT", "QRS/T boundaries and loops are unavailable.", "",
             "Failure stage: " + artifact.get("failure_stage", "unknown"),
             *textwrap.wrap(artifact.get("error", ""), 76), "", "Saved beat-selection reasons:"]
    lines.extend(f"  {reason}: {count}" for reason, count in sorted(counts.items()))
    lines += ["", f"Used beats: {artifact.get('n_used', 0)} / {len(artifact.get('beat_qc', []))}",
              "Saved input quality mask counts:"]
    lines.extend(f"  {name}: {count}" for name, count in artifact.get("input_quality_mask_counts", {}).items())
    lines += ["", "These diagnostics do not establish which anchor is a true beat.",
              "No median waveform or predicted boundaries are fabricated."]
    axes[3].text(.015, .98, "\n".join(lines), va="top", ha="left", fontsize=9, transform=axes[3].transAxes)
    for ax in axes[:3]:
        ax.grid(alpha=.16)


def _raw_hashes(path: Path) -> dict[str, str]:
    import wfdb
    header = wfdb.rdheader(str(path))
    files = {path.with_suffix(".hea")} | {path.parent / s for s in header.file_name}
    return {str(p): _sha(p) for p in sorted(files)}


def _render_one(task: tuple) -> dict:
    row, source_root, output_root, data_root, cfg_dict, source_hash, resume = task
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_dict.items()})
    root, out = Path(source_root), Path(output_root)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["record_id"])
    jp, npz = root / (safe + ".json"), root / (safe + ".npz")
    artifact = json.loads(jp.read_text(encoding="utf-8"))
    if artifact.get("record_id") != row["record_id"]:
        raise ValueError(f"Artifact ID mismatch: {jp}")
    with np.load(npz, allow_pickle=False) as source_arrays:
        arrays = {key: source_arrays[key] for key in source_arrays.files}
    has_median = "median_xyz" in arrays
    directory = "failure_cases" if not has_median else row["label_name"]
    if directory not in {"HC", "MI", "failure_cases"}:
        raise ValueError(f"Unexpected label for gallery: {row['label_name']}")
    path = out / "plots" / directory / (safe + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = out / "plot_checkpoints" / (safe + ".json")
    fingerprint_input = {"source_sha256": source_hash, "artifact_json_sha256": _sha(jp),
                         "artifact_npz_sha256": _sha(npz), "row": row, "config": cfg_dict,
                         "raw_sha256": _raw_hashes(Path(data_root) / row["record_id"]) if not has_median else {}}
    fingerprint = hashlib.sha256(json.dumps(_clean(fingerprint_input), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    result = {"record_id": row["record_id"], "patient_id": row["patient_id"],
              "label_name": row["label_name"], "split": row.get("split", ""),
              "qc_status": row.get("qc_status", ""), "qc_flags": row.get("qc_flags", ""),
              "feature_completeness": row.get("feature_completeness"), "has_median": has_median,
              "plot_path": path.relative_to(out).as_posix(), "source_json": os.path.relpath(jp, out),
              "source_npz": os.path.relpath(npz, out), "fingerprint": fingerprint,
              "suspicion_indicators": row.get("suspicion_indicators", ""),
              "screening_route": row.get("screening_route", "")}
    if resume and cache.exists() and path.exists():
        previous = json.loads(cache.read_text())
        if previous.get("fingerprint") == fingerprint and previous.get("plot_sha256") == _sha(path):
            result["rendered"] = False
            return result
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    flags = "; ".join(_flags(row.get("qc_flags"))) or "none (not independently verified)"
    title_lines = [f"{row['record_id']} | patient {row['patient_id']} | {row['label_name']} (label {row.get('label', '?')}) | split {row.get('split', '')}",
                   f"QC {row.get('qc_status', '')} | finite features {row.get('feature_completeness', '?')}/12 | "
                   + ("saved median beat" if has_median else "FAILED: no median beat"),
                   *textwrap.wrap("QC flags: " + flags, 158)]
    title_height = .030 * len(title_lines)
    fig = plt.figure(figsize=(17, 11.4 + max(0, len(title_lines) - 3) * .27))
    try:
        if has_median:
            _median_panels(fig, artifact, arrays, cfg)
        else:
            _failure_panels(fig, artifact, arrays, Path(data_root) / row["record_id"], cfg)
        fig.suptitle("\n".join(title_lines), fontsize=9, y=.99, va="top")
        fig.text(.015, .011, "Frozen V1 predictions. QC flags and screening indicators are not reference labels. "
                 "Median times are relative to its QRS anchor; failure strips use raw recording seconds.", fontsize=8, color="#434b52")
        fig.subplots_adjust(left=.055, right=.965, bottom=.065, top=max(.72, .98 - title_height), hspace=.26, wspace=.22)
        tmp = path.with_suffix(".tmp.png")
        fig.savefig(tmp, dpi=112, facecolor="white", bbox_inches="tight")
        tmp.replace(path)
    finally:
        plt.close(fig)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cache, {"fingerprint": fingerprint, "plot_sha256": _sha(path), "inputs": fingerprint_input})
    result["rendered"] = True
    return result


def _make_html(rows: list[dict], target: Path) -> None:
    esc = lambda value: html.escape(str(value if value is not None else ""), quote=True)
    def select(name: str, title: str, values: list[str]) -> str:
        opts = '<option value="">Tất cả</option>' + "".join(f'<option value="{esc(v)}">{esc(v)}</option>' for v in values)
        return f'<label>{title}<select id="{name}">{opts}</select></label>'
    filters = select("label", "HC / MI", sorted({r["label_name"] for r in rows}))
    filters += select("split", "Split", sorted({r["split"] for r in rows}))
    filters += select("status", "QC status", sorted({r["qc_status"] for r in rows}))
    filters += select("flag", "QC flag (khớp chính xác)", sorted({f for r in rows for f in _flags(r["qc_flags"])}))
    filters += select("complete", "Số features hữu hạn", [str(v) for v in sorted({int(r["feature_completeness"] or 0) for r in rows})])
    filters += select("route", "Screening route", sorted({r["screening_route"] for r in rows if r["screening_route"]}))
    filters += '<label>Không có QC flag<select id="noqc"><option value="">Tất cả</option><option value="yes">Chỉ không có flag</option><option value="no">Có ít nhất một flag</option></select></label>'
    filters += '<label>Median beat<select id="median"><option value="">Tất cả</option><option value="yes">Có median</option><option value="no">Lỗi / không có median</option></select></label>'
    cards = []
    for r in rows:
        metadata = {"label": r["label_name"], "split": r["split"], "status": r["qc_status"],
                    "flags": _flags(r["qc_flags"]), "complete": str(int(r["feature_completeness"] or 0)),
                    "route": r["screening_route"], "median": "yes" if r["has_median"] else "no",
                    "search": " ".join(str(r[k]) for k in ("record_id", "patient_id", "suspicion_indicators"))}
        flags = r["qc_flags"] or "Không có QC flag; chưa xác minh biên độc lập"
        cards.append(f'''<article class="card" data-meta="{esc(json.dumps(metadata, ensure_ascii=False))}" id="{esc(r['record_id'].replace('/', '_'))}">
<h2>{esc(r['record_id'])} <span>{esc(r['label_name'])}</span></h2>
<p>{esc(r['patient_id'])} · {esc(r['split'])} · {esc(r['qc_status'])} · {esc(r['feature_completeness'])}/12 features</p>
<p class="flags">{esc(flags)}</p><p class="route">{esc(r['screening_route'])} · {esc(r['suspicion_indicators'])}</p>
<a href="{esc(r['plot_path'])}" target="_blank"><img src="{esc(r['plot_path'])}" loading="lazy" decoding="async" alt="Four diagnostic panels for {esc(r['record_id'])}"></a>
<p><a href="{esc(r['plot_path'])}" target="_blank">Ảnh đầy đủ</a> · <a href="{esc(r['source_json'])}">Trace JSON</a> · <a href="{esc(r['source_npz'])}">Saved arrays NPZ</a></p></article>''')
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VCG V1 — QC / delineation audit</title>
<style>body{font:14px system-ui,sans-serif;background:#eef2f5;color:#15202b;margin:0}header{padding:20px 24px;background:#102d40;color:white}h1{font-size:23px;margin:0 0 8px}header p{max-width:1100px;line-height:1.5;margin:5px 0}nav{display:flex;flex-wrap:wrap;gap:12px;position:sticky;top:0;background:#fff;padding:13px 24px;box-shadow:0 2px 8px #0002;z-index:2}label{display:flex;flex-direction:column;gap:4px;font-size:12px}select,input,button{padding:7px;border:1px solid #bac5cd;border-radius:4px;background:white;max-width:320px}button{align-self:end;cursor:pointer}#count{padding:12px 24px;margin:0}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(650px,100%),1fr));gap:16px;padding:0 20px 24px}.card{background:#fff;border-radius:8px;padding:12px;min-width:0;box-shadow:0 1px 5px #0001}.card[hidden]{display:none}.card h2{font-size:16px;margin:0;overflow-wrap:anywhere}.card h2 span{font-size:12px;background:#e6eff4;padding:3px 7px;border-radius:5px}.card p{margin:7px 0;font-size:12px}.flags,.route{overflow-wrap:anywhere;color:#52616e}.card img{width:100%;height:auto;border:1px solid #e3e7eb}a{color:#12698e}footer{padding:16px 24px;color:#52616e}</style>
<header><h1>Kiểm toán QC và phân đoạn QRS/T — V1 giữ nguyên</h1><p>NUM_RECORDS recordings. Tất cả biểu đồ dùng các mảng và biên dự đoán đã lưu; các ca lỗi không có median được hiển thị bằng XYZ gốc, detector và beat-selection diagnostics. Không có nhãn “đúng”, “sai” hay “QC false positive” được suy ra tự động từ các hình này.</p><p>Thời gian median: ms tương đối theo QRS anchor. Các dải tín hiệu ca lỗi: giây theo recording gốc. Quan sát test chỉ để mô tả, không dùng để chọn V2 hay điều chỉnh threshold.</p><p><a style="color:#bde9ff" href="qc_audit_report.md">Báo cáo audit</a> · <a style="color:#bde9ff" href="manual_review_template.csv">Manual review template</a> · <a style="color:#bde9ff" href="gallery_manifest.csv">Danh mục hình / trace</a></p></header>
<nav>FILTERS<label>Tìm record / patient / indicator<input id="search" placeholder="patient001 hoặc tên indicator"></label><button id="reset">Xóa bộ lọc</button></nav><p id="count" aria-live="polite"></p><main>CARDS</main>
<footer>Ảnh được tải khi cuộn tới; mọi asset đều là file local. Click ảnh để mở kích thước đầy đủ. Screening indicators chỉ gợi ý ưu tiên review, chưa phải ground truth.</footer>
<script>const cards=[...document.querySelectorAll('.card')].map(el=>({el,m:JSON.parse(el.dataset.meta)}));const ids=['label','split','status','flag','complete','route','noqc','median','search'];const controls=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));function filter(){let count=0;for(const {el,m}of cards){let yes=true;for(const key of ['label','split','status','complete','route','median'])if(controls[key].value&&m[key]!==controls[key].value)yes=false;if(controls.flag.value&&!m.flags.includes(controls.flag.value))yes=false;if(controls.noqc.value==='yes'&&m.flags.length)yes=false;if(controls.noqc.value==='no'&&!m.flags.length)yes=false;if(controls.search.value&&!m.search.toLowerCase().includes(controls.search.value.trim().toLowerCase()))yes=false;el.hidden=!yes;if(yes)count++;}document.getElementById('count').textContent=`Hiển thị ${count} / ${cards.length} recordings; chưa được xác minh biên độc lập.`;}Object.values(controls).forEach(el=>el.addEventListener('input',filter));document.getElementById('reset').addEventListener('click',()=>{Object.values(controls).forEach(el=>el.value='');filter();});filter();</script></html>'''
    document = document.replace("NUM_RECORDS", str(len(rows))).replace("FILTERS", filters).replace("CARDS", "\n".join(cards))
    target.write_text(document, encoding="utf-8")


def create_audit_gallery(frame: pd.DataFrame, artifact_root: Path, outdir: Path,
                         data_root: Path, cfg: Config | None = None,
                         workers: int = 4, resume: bool = True) -> dict[str, Any]:
    """Create all requested four-panel record diagnostics and a local HTML gallery.

    A non-448 subset is supported for smoke tests. The caller validates the full
    study population. ``artifact_root`` accepts the actual artifacts folder or
    its evaluation parent (which must contain an ``artifacts`` child). Nothing
    in the source feature CSVs, V1, or artifacts is written. All supplied records
    must have their own saved JSON and NPZ.
    """
    required = {"record_id", "patient_id", "label_name", "qc_status", "qc_flags", "feature_completeness", "split"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Gallery metadata missing columns: {sorted(missing)}")
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs in gallery metadata")
    cfg = Config() if cfg is None else cfg
    outdir, artifact_root, data_root = Path(outdir).resolve(), Path(artifact_root).resolve(), Path(data_root).resolve()
    if artifact_root.name != "artifacts" and (artifact_root / "artifacts").is_dir():
        artifact_root = artifact_root / "artifacts"
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"No artifact directory: {artifact_root}")
    if outdir == artifact_root or outdir == artifact_root.parent:
        raise ValueError("Gallery output must be separate from baseline artifact root")
    outdir.mkdir(parents=True, exist_ok=True)
    source_hash = _sha(Path(__file__))
    # Only fields used for display participate; unrelated float columns cannot
    # invalidate all plots when the independent audit tables are extended.
    fields = sorted(required | {"label", "processing_success", "suspicion_indicators", "screening_route"})
    records = [_clean({k: row.get(k, "") for k in fields}) for row in frame.to_dict("records")]
    tasks = [(row, str(artifact_root), str(outdir), str(data_root), asdict(cfg), source_hash, bool(resume)) for row in records]
    rows = []
    if workers <= 1 or len(tasks) <= 1:
        rows = [_render_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            pending = {pool.submit(_render_one, task): task[0]["record_id"] for task in tasks}
            for job in as_completed(pending):
                try:
                    rows.append(job.result())
                except Exception as exc:
                    raise RuntimeError(f"Could not render {pending[job]}: {exc}") from exc
    rows.sort(key=lambda r: (r["label_name"], r["record_id"]))
    manifest = outdir / "gallery_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    target = outdir / "qc_review_gallery.html"
    _make_html(rows, target)
    cache_metadata = outdir / "gallery_config.json"
    _write_json(cache_metadata, {"source_sha256": source_hash, "config": asdict(cfg), "n_records": len(rows),
                               "n_plots_rendered": sum(r["rendered"] for r in rows),
                               "n_saved_medians": sum(r["has_median"] for r in rows),
                               "artifact_root": artifact_root, "data_root": data_root,
                               "note": "No independent boundary adjudication is implied by this gallery."})
    return {"gallery": target, "manifest_csv": manifest, "plots_dir": outdir / "plots",
            "cache_metadata": cache_metadata, "n_records": len(rows),
            "n_rendered": sum(r["rendered"] for r in rows)}


def validate_audit_gallery(outdir: Path, expected_ids: list[str] | None = None) -> dict[str, int]:
    """Verify record coverage, linked local assets, PNGs and plot-cache integrity.

    This checks delivered artifacts, not correctness of V1 boundary predictions.
    """
    import struct
    outdir = Path(outdir).resolve()
    manifest = pd.read_csv(outdir / "gallery_manifest.csv").fillna("")
    if manifest.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs in delivered gallery manifest")
    if expected_ids is not None and set(manifest.record_id) != set(expected_ids):
        raise ValueError("Gallery coverage differs from expected recording IDs")
    document = (outdir / "qc_review_gallery.html").read_text(encoding="utf-8")
    cards = re.findall(r'<article class="card" data-meta="([^"]+)"', document)
    if len(cards) != len(manifest):
        raise ValueError("HTML card count differs from manifest")
    if document.count('loading="lazy"') != len(manifest):
        raise ValueError("Every gallery record must have a lazy-loaded image")
    if re.search(r'<(?:script|img|link)[^>]+(?:src|href)=["\']https?://', document, flags=re.I):
        raise ValueError("Unexpected remote asset in local gallery")
    for row in manifest.to_dict("records"):
        for key in ("plot_path", "source_json", "source_npz"):
            if not (outdir / row[key]).is_file():
                raise FileNotFoundError(outdir / row[key])
        path = outdir / row["plot_path"]
        with path.open("rb") as f:
            header = f.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG signature: {path}")
        width, height = struct.unpack(">II", header[16:24])
        if width < 1200 or height < 800:
            raise ValueError(f"Unexpected diagnostic image dimensions: {path}: {width}x{height}")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["record_id"])
        checkpoint = json.loads((outdir / "plot_checkpoints" / (safe + ".json")).read_text())
        if checkpoint.get("fingerprint") != row["fingerprint"] or checkpoint.get("plot_sha256") != _sha(path):
            raise ValueError(f"Plot cache does not match delivered artifact: {path}")
    return {"n_records": len(manifest), "n_cards": len(cards),
            "n_medians": int(manifest.has_median.map(lambda x: str(x).lower() == "true").sum()),
            "n_no_median": int(manifest.has_median.map(lambda x: str(x).lower() == "false").sum())}
