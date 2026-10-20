"""Every-record visual inspection of the original saved V1 QRS/T cuts.

The full median XYZ, QRS velocity, T magnitude and exact inclusive loop cuts
follow Notebook 05 steps 6–19. Candidates are secondary guides, never evidence
of accuracy. No delineation, baseline subtraction or prediction is changed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/vcg_qc_audit_matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vcg_preprocessing import Config
from scripts.vcg_qc_audit_gallery import _annotated_tangent

COLORS = {"v1": "#b45114", "tangent": "#187da0", "threshold": "#782cad"}
METHODS = {"v1": ("t_on", "t_off"),
           "tangent": ("t_tangent_on", "t_tangent_off"),
           "threshold": ("t_threshold_on", "t_threshold_off")}
NOTE = "Vùng tô và loop là biên V1 đã lưu; chưa phải phân đoạn đã được xác nhận đúng."
LOOP_COLORS = {"qrs": "#176da0", "t": "#c46614"}
XYZ_COLORS = ("#176da0", "#b44534", "#37834e")


def _safe(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record_id))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index(value, n: int) -> int | None:
    if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return None
    if int(value) != value or not 0 <= int(value) < n:
        return None
    return int(value)


def candidate_positions(boundaries: dict, time_ms: np.ndarray) -> dict:
    """Keep unavailable candidates missing, including index 0 as valid."""
    positions = {}
    for key in ("qrs_on", "qrs_off", "qrs_speed_peak", "t_peak", *sum(METHODS.values(), ())):
        index = _index(boundaries.get(key), len(time_ms))
        positions[key] = {"index": index, "ms": float(time_ms[index]) if index is not None else None}
    return positions


def interval_segments(n: int, first: tuple[int, int], second: tuple[int, int]) -> dict:
    """Classify temporal segments relative to two alternatives, not correctness."""
    for interval in (first, second):
        if not 0 <= interval[0] < interval[1] < n:
            raise ValueError("Expected ordered, in-range interval endpoints")
    index = np.arange(n - 1)
    a = (index >= first[0]) & (index < first[1])
    b = (index >= second[0]) & (index < second[1])
    return {"common": a & b, "v1_only": a & ~b, "alternative_only": b & ~a}


def _runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [(int(chunk[0]), int(chunk[-1] + 1))
            for chunk in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)]


def _pair(positions, method):
    a, z = (positions[key]["index"] for key in METHODS[method])
    return (a, z) if a is not None and z is not None and a < z else None


def _axes_style(ax, title, ylabel="mV"):
    ax.set(title=title, xlabel="Thời gian theo QRS anchor (ms)", ylabel=ylabel)
    ax.grid(alpha=.18)


def _legend(ax, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, ncol=ncol, loc="best", framealpha=.91)


def _endpoint_lines(ax, positions, method, alpha=1):
    for side, key in zip(("on", "off"), METHODS[method]):
        value = positions[key]["ms"]
        if value is not None:
            ax.axvline(value, color=COLORS[method], ls="--" if side == "on" else ":",
                       alpha=alpha, lw=1.2, label=f"{method} T-{side}: {value:.1f} ms")


def validate_saved_median(artifact, arrays):
    """Check the saved timebase; never replace it or infer missing boundaries."""
    t = np.asarray(arrays["time_ms"], dtype=float)
    xyz = np.asarray(arrays["median_xyz"], dtype=float)
    if t.ndim != 1 or len(t) < 2 or xyz.shape != (len(t), 3):
        raise ValueError("Saved median XYZ/time array shapes are inconsistent")
    if not np.isfinite(t).all() or not np.isfinite(xyz).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Saved median XYZ/time arrays contain invalid data")
    fs = float(artifact["fs"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Saved sampling frequency is invalid")
    if not np.allclose(np.diff(t), 1000. / fs, rtol=1e-7, atol=1e-7):
        raise ValueError("Saved time_ms spacing does not match this recording's fs")
    anchor = artifact.get("median_anchor_sample")
    if anchor is not None:
        anchor = _index(anchor, len(t))
        if anchor is None or not np.allclose(t, (np.arange(len(t)) - anchor) * 1000. / fs,
                                             rtol=1e-7, atol=1e-7):
            raise ValueError("Saved time_ms does not match this recording's median anchor")
    return t, xyz, fs


def v1_loop_points(xyz, boundaries, prefix):
    """Notebook 05 step 17: exact independent inclusive cuts, with no filler."""
    if prefix not in ("qrs", "t"):
        raise ValueError("Expected qrs or t loop")
    a, z = (_index(boundaries.get(prefix + side), len(xyz)) for side in ("_on", "_off"))
    if a is None or z is None or a >= z:
        return None
    return xyz[a:z + 1]


def _finite(value):
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)


def _saved_window(value, n):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    a, z = value
    if not _finite(a) or not _finite(z) or int(a) != a or int(z) != z or not 0 <= a < z <= n:
        return None
    return int(a), int(z)


def _actual_boundaries(ax, positions, prefixes=("qrs", "t"), labels=True):
    for prefix in prefixes:
        for side in ("on", "off"):
            key = prefix + "_" + side
            value = positions[key]["ms"]
            if value is not None:
                ax.axvline(value, color=LOOP_COLORS[prefix], ls="-" if side == "on" else "--",
                           lw=1.65, zorder=4,
                           label=f"V1 {prefix.upper()}-{side}: {value:.1f} ms" if labels else None)
        a, z = (positions[prefix + "_" + side]["index"] for side in ("on", "off"))
        if a is not None and z is not None and a < z:
            ax.axvspan(positions[prefix + "_on"]["ms"], positions[prefix + "_off"]["ms"],
                       color=LOOP_COLORS[prefix], alpha=.10, zorder=0)


def _pq_context(ax, t, pq, label=True):
    if pq is not None:
        ax.axvspan(t[pq[0]], t[pq[1] - 1], color=".4", alpha=.10,
                   label="Cửa sổ PQ đã dùng đặt gốc XYZ" if label else None)
    ax.axhline(0, color=".5", lw=.45)


def _plot_xyz(ax, t, xyz):
    for j, color in enumerate(XYZ_COLORS):
        ax.plot(t, xyz[:, j], lw=1.05, color=color, label="XYZ"[j])


def _loop_limits(loops):
    valid = [loop for loop in loops.values() if loop is not None]
    if not valid:
        return np.zeros(3), .01
    points = np.vstack(valid)
    low, high = np.minimum(points.min(axis=0), 0), np.maximum(points.max(axis=0), 0)
    return (high + low) / 2, max(float(np.max(high - low)) / 2, .005) * 1.13


def _plot_single_loop(ax, loop, prefix, limits):
    center, radius = limits
    if loop is None:
        ax.text2D(.04, .55, "Thiếu cặp biên V1 hợp lệ.\nKhông tạo loop thay thế.",
                  transform=ax.transAxes, color="#a63224", fontsize=10)
    else:
        segments = np.stack((loop[:-1], loop[1:]), axis=1)
        collection = Line3DCollection(segments, cmap="Blues" if prefix == "qrs" else "Oranges",
                                      norm=Normalize(0, 1), linewidth=1.8)
        collection.set_array(np.linspace(.35, 1, len(segments)))
        ax.add_collection3d(collection)
        color = LOOP_COLORS[prefix]
        ax.scatter(*loop[0], color=color, edgecolors="black", linewidths=.5, s=44, marker="o", label="Bắt đầu (o)")
        ax.scatter(*loop[-1], color=color, s=60, linewidths=1.8, marker="x", label="Kết thúc (×)")
        for index in sorted(set((len(loop) // 3, 2 * len(loop) // 3))):
            last = min(len(loop) - 1, index + max(1, len(loop) // 18))
            if last > index:
                delta = loop[last] - loop[index]
                ax.quiver(*loop[index], *delta, color=color, linewidth=1.5,
                          arrow_length_ratio=.5, normalize=False)
        ax.legend(fontsize=7, loc="upper right")
    ax.scatter(0, 0, 0, marker="+", s=35, color=".35")
    ax.set(xlim=(center[0] - radius, center[0] + radius),
           ylim=(center[1] - radius, center[1] + radius),
           zlim=(center[2] - radius, center[2] + radius),
           xlabel="X (mV)", ylabel="Y (mV)", zlabel="Z (mV)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=40)


def _loop_axis(ax, xyz, positions, prefix="t"):
    """Legacy private helper now draws only the chosen V1 loop."""
    boundaries = {key: val["index"] for key, val in positions.items()}
    loops = {name: v1_loop_points(xyz, boundaries, name) for name in ("qrs", "t")}
    _plot_single_loop(ax, loops[prefix], prefix, _loop_limits(loops))


def _projection(ax, loops, first, second, limits):
    center, radius = limits
    for prefix, curve in loops.items():
        if curve is None:
            continue
        color = LOOP_COLORS[prefix]
        ax.plot(curve[:, first], curve[:, second], color=color, lw=1.25, label=prefix.upper() + " V1")
        ax.scatter(curve[0, first], curve[0, second], color=color, s=25, marker="o")
        ax.scatter(curve[-1, first], curve[-1, second], color=color, s=35, marker="x")
        index = len(curve) // 2
        last = min(len(curve) - 1, index + max(1, len(curve) // 12))
        if last > index:
            ax.annotate("", xy=(curve[last, first], curve[last, second]),
                        xytext=(curve[index, first], curve[index, second]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1})
    ax.axhline(0, color=".65", lw=.5)
    ax.axvline(0, color=".65", lw=.5)
    ax.set(xlim=(center[first] - radius, center[first] + radius),
           ylim=(center[second] - radius, center[second] + radius),
           xlabel="XYZ"[first] + " (mV)", ylabel="XYZ"[second] + " (mV)",
           title="Chiếu Frank " + "XYZ"[first] + "–" + "XYZ"[second] + " · cùng gốc PQ")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.18)
    _legend(ax)


def _plot_median(artifact, arrays, row, path, cfg):
    t, xyz, fs = validate_saved_median(artifact, arrays)
    n = len(t)
    b = artifact.get("boundaries", {})
    positions = candidate_positions(b, t)
    pq = _saved_window(artifact.get("pq_indices"), n)
    mag = arrays.get("t_magnitude")
    rawmag = arrays.get("magnitude")
    if rawmag is None:
        # This is the exact Notebook 05 magnitude, not a new delineation.
        rawmag = np.linalg.norm(xyz, axis=1)
    if len(rawmag) != n:
        raise ValueError("Saved magnitude length differs from median")
    if mag is not None and len(mag) != n:
        raise ValueError("Saved smoothed T magnitude length differs from median")
    loops = {name: v1_loop_points(xyz, b, name) for name in ("qrs", "t")}
    limits = _loop_limits(loops)
    fig = plt.figure(figsize=(20, 18.4))
    grid = fig.add_gridspec(5, 6, height_ratios=(1.0, .80, 1.0, 1.35, .85))
    axa = fig.add_subplot(grid[0, :])
    qxyz = fig.add_subplot(grid[1, :3])
    txyz = fig.add_subplot(grid[1, 3:])
    qspeed = fig.add_subplot(grid[2, :3], sharex=qxyz)
    tmag = fig.add_subplot(grid[2, 3:], sharex=txyz)
    qloop = fig.add_subplot(grid[3, :3], projection="3d")
    tloop = fig.add_subplot(grid[3, 3:], projection="3d")
    projections = [fig.add_subplot(grid[4, first:first + 2]) for first in (0, 2, 4)]
    try:
        _plot_xyz(axa, t, xyz)
        _actual_boundaries(axa, positions)
        _pq_context(axa, t, pq)
        axa.axvline(0, color=".3", ls=":", lw=.7, label="QRS anchor: 0 ms")
        axa.set_xlim(t[0], t[-1])
        _axes_style(axa, "A. TOÀN BỘ MEDIAN XYZ THẬT · xanh = QRS V1 · cam = T V1 · giữ cả tín hiệu ngoài hai vùng cắt")
        axa.title.set_fontweight("bold")
        _legend(axa, 5)

        anchor = artifact.get("median_anchor_sample")
        if anchor is None:
            # Synthetic compatibility only; actual artifacts always save anchor.
            anchor = int(np.argmin(np.abs(t)))
        qlo = max(0, int(anchor) + int(cfg.qrs_search_ms[0] * fs / 1000))
        qhi = min(n - 1, int(anchor) + int(cfg.qrs_search_ms[1] * fs / 1000))
        qxmin, qxmax = max(t[0], t[qlo] - 15), min(t[-1], t[qhi] + 15)
        for ax in (qxyz, qspeed):
            _actual_boundaries(ax, positions, ("qrs",))
            _pq_context(ax, t, pq)
            for j, index in enumerate((qlo, qhi)):
                ax.axvline(t[index], color=".45", ls=":", lw=.9,
                           label="Giới hạn cửa sổ tìm QRS" if j == 0 else None)
            ax.axvline(0, color=".45", ls=":", lw=.65)
            ax.set_xlim(qxmin, qxmax)
        _plot_xyz(qxyz, t, xyz)
        _axes_style(qxyz, "B1. QRS phóng to: XYZ và mốc V1; vùng PQ xám")
        _legend(qxyz, 3)
        for key, label, color, width in (("speed", "Spatial speed (đạo hàm SG 11 ms)", ".65", .85),
                                        ("smooth_speed", "Speed dùng phân đoạn (mean 5 ms)", "#176da0", 1.2)):
            values = arrays.get(key)
            if values is not None and len(values) == n:
                qspeed.plot(t, values, lw=width, color=color, label=label)
        if _finite(b.get("q_threshold")):
            qspeed.axhline(b["q_threshold"], color="#b45114", ls="--", lw=1.,
                          label=f"Ngưỡng QRS V1: {b['q_threshold']:.3f} mV/s")
        peak = positions["qrs_speed_peak"]["index"]
        smooth = arrays.get("smooth_speed")
        if peak is not None and smooth is not None and len(smooth) == n:
            qspeed.scatter(t[peak], smooth[peak], color="#b44534", marker="^", s=38, zorder=6,
                           label=f"Speed peak hạt giống: {t[peak]:.1f} ms")
        _axes_style(qspeed, "B2. Cùng khoảng QRS: vận tốc / ngưỡng / speed peak", "mV/s")
        _legend(qspeed, 2)

        candidate_times = [v["ms"] for key, v in positions.items()
                           if key.startswith("t_") and v["ms"] is not None]
        qoff = positions["qrs_off"]["ms"]
        # Full A retains the whole QRS tail; the T zoom begins 5 ms before
        # QRS-off so a large QRS excursion cannot hide the much smaller T.
        low = max(t[0], min([qoff - 5. if qoff is not None else 0.]
                           + [value - 20. for value in candidate_times]))
        search = b.get("t_search")
        search_end = _index(search[-1], n) if isinstance(search, (list, tuple)) and len(search) == 2 else None
        high = min(t[-1], max(candidate_times + ([float(t[search_end])] if search_end is not None else [float(t[-1])])) + 25.)
        if high <= low:
            low, high = float(t[0]), float(t[-1])
        for ax in (txyz, tmag):
            _actual_boundaries(ax, positions, ("t",))
            ax.set_xlim(low, high)
            tp = _saved_window(b.get("tp"), n)
            if tp is not None:
                ax.axvspan(t[tp[0]], t[tp[1] - 1], color="#7c3aed", alpha=.08,
                           label="Cửa sổ TP đã lưu")
        _plot_xyz(txyz, t, xyz)
        visible_xyz = xyz[(t >= low) & (t <= high)]
        xyz_low, xyz_high = float(np.min(visible_xyz)), float(np.max(visible_xyz))
        xyz_pad = max(.002, .12 * (xyz_high - xyz_low))
        txyz.set_ylim(xyz_low - xyz_pad, xyz_high + xyz_pad)
        txyz.axhline(0, color=".5", lw=.5, label="Gốc tọa độ PQ chung")
        if positions["t_peak"]["ms"] is not None:
            txyz.axvline(positions["t_peak"]["ms"], color="#b93872", ls=":", lw=.9, label="T peak đã chọn")
        _axes_style(txyz, "C1. Sóng T phóng to: XYZ thật; giữ tín hiệu trước/sau biên")
        _legend(txyz, 3)
        tmag.plot(t, rawmag, color=".65", lw=.8, label="Magnitude gốc ||XYZ||")
        if mag is not None:
            tmag.plot(t, mag, color="#263844", lw=1.25, label="Magnitude làm mượt 21 ms đã lưu")
        else:
            tmag.text(.02, .88, "Thiếu magnitude làm mượt đã lưu; không tự phân đoạn lại.", transform=tmag.transAxes, fontsize=8)
        for key, label, color in (("st_level", "Mức ST", "#b45114"),
                                   ("t_end_baseline", "Baseline TP" if b.get("tp") is not None else "Baseline cuối T (PQ fallback)", "#7c3aed"),
                                   ("t_threshold", "Ngưỡng T-on", "#747f46"),
                                   ("t_end_threshold", "Ngưỡng T-off", "#782cad")):
            if _finite(b.get(key)):
                tmag.axhline(b[key], color=color, ls="-.", lw=.65, label=f"{label}: {b[key]:.4f} mV")
        if pq is not None and mag is not None:
            pq_mag = float(np.median(mag[pq[0]:pq[1]]))
            tmag.axhline(pq_mag, color=".35", ls=":", lw=.8, label=f"Magnitude nền PQ: {pq_mag:.4f} mV")
        for method in ("tangent", "threshold"):
            for side in ("on", "off"):
                value = positions[f"t_{method}_{side}"]["ms"]
                if value is not None:
                    tmag.axvline(value, color=COLORS[method], ls=":" if method == "tangent" else "-.",
                                 lw=.70, alpha=.72, label=f"candidate {method} T-{side}: {value:.1f}")
        peak = positions["t_peak"]["index"]
        if peak is not None and mag is not None:
            tmag.scatter(t[peak], mag[peak], color="#b93872", marker="^", s=40, zorder=6,
                         label=f"T peak: {t[peak]:.1f} ms")
        if mag is not None:
            for side in ("on", "off"):
                _annotated_tangent(tmag, t, mag, arrays.get("t_slope"), b, side, low, high)
        visible = (t >= low) & (t <= high)
        values = rawmag[visible] if mag is None else mag[visible]
        refs = [b[key] for key in ("st_level", "t_end_baseline", "t_end_threshold", "t_threshold") if _finite(b.get(key))]
        ymin, ymax = min(0., float(np.min(values))), max([float(np.max(values))] + refs)
        pad = max(.002, (ymax - ymin) * .10)
        tmag.set_ylim(ymin - pad, ymax + pad)
        _axes_style(tmag, "C2. Magnitude / ST / TP · nét V1 đậm; candidate mảnh là thông tin bổ sung", "Magnitude (mV)")
        _legend(tmag, 3)

        for ax, prefix in ((qloop, "qrs"), (tloop, "t")):
            _plot_single_loop(ax, loops[prefix], prefix, limits)
            pair = [positions[prefix + "_" + side]["ms"] for side in ("on", "off")]
            if loops[prefix] is None:
                detail = "thiếu cặp biên hợp lệ"
            else:
                detail = f"{pair[0]:.1f} → {pair[1]:.1f} ms; {len(loops[prefix])} mẫu, giữ cả mẫu cuối"
            ax.set_title(f"D. {prefix.upper()} loop V1 · {detail}\nMàu nhạt → đậm và mũi tên: hướng thời gian", fontsize=10)
        for ax, (first, second) in zip(projections, ((0, 1), (0, 2), (1, 2))):
            _projection(ax, loops, first, second, limits)
        missing = [key for key in ("qrs_on", "qrs_off", "t_on", "t_off") if positions[key]["index"] is None]
        segmentation = "Đủ bốn biên đã lưu" if not missing else "Thiếu biên: " + ", ".join(missing)
        flags = str(row.get("qc_flags", "") or "Không có QC flag")
        patient = row.get("patient_id", str(row["record_id"]).split("/")[0])
        status = row.get("status", row.get("qc_status", "chưa xác minh"))
        title = (f"{row['record_id']} | patient={patient} | {row['label_name']} | {row['split']} | fs={fs:g} Hz | QC={status}\n"
                 f"{segmentation} | T method: {b.get('t_method', 'unresolved')} | median XYZ giữ nguyên gốc PQ")
        fig.suptitle(title + "\n" + "\n".join(textwrap.wrap("QC flags: " + flags, width=180)), fontsize=10.2, y=.991)
        fig.text(.035, .013, NOTE + " Hai loop dùng CÙNG gốc PQ, tỉ lệ trục và góc nhìn; không lấy khoảng ST giữa hai loop.", fontsize=9)
        fig.subplots_adjust(left=.055, right=.975, bottom=.055, top=.915, hspace=.50, wspace=.43)
        tmp = path.with_suffix(".tmp.png")
        fig.savefig(tmp, dpi=130, facecolor="white")
        tmp.replace(path)
    finally:
        plt.close(fig)

def _render_one_impl(task):
    row, artifact_root, output_dir, cfg_values, source_hash, resume = task
    root, out = Path(artifact_root), Path(output_dir)
    rid, safe = row["record_id"], _safe(row["record_id"])
    jp, npz = root / (safe + ".json"), root / (safe + ".npz")
    artifact = json.loads(jp.read_text())
    if artifact.get("record_id") != rid:
        raise ValueError(f"Artifact identity mismatch: {jp}")
    with np.load(npz, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    has_median = "median_xyz" in arrays
    path, cache = out / "plots" / (safe + ".png"), out / "cache" / (safe + ".json")
    original_failure = ROOT / "outputs/vcg_qc_audit_v1/plots/failure_cases" / (safe + ".png")
    failure_hash = _sha(original_failure) if not has_median and original_failure.is_file() else None
    fingerprint = hashlib.sha256(json.dumps({"json": _sha(jp), "npz": _sha(npz), "renderer": source_hash,
                    "row": row, "config": cfg_values, "failure_plot": failure_hash}, sort_keys=True).encode()).hexdigest()
    b = artifact.get("boundaries", {})
    if has_median:
        validate_saved_median(artifact, arrays)
    loops = {prefix: v1_loop_points(arrays["median_xyz"], b, prefix) if has_median else None
             for prefix in ("qrs", "t")}
    missing = [key for key in ("qrs_on", "qrs_off", "t_on", "t_off")
               if _index(b.get(key), len(arrays.get("time_ms", []))) is None]
    segmentation = ("missing_median" if not has_median else
                    "complete_boundaries" if not missing and all(curve is not None for curve in loops.values())
                    else "missing_or_invalid_boundaries")
    result = {"record_id": rid, "patient_id": row["patient_id"], "label_name": row["label_name"],
              "split": row["split"], "has_median": has_median, "plot_path": path.relative_to(out).as_posix(),
              "render_status": "rendered" if has_median else "copied_failure_diagnostic",
              "sha256": "", "fingerprint": fingerprint, "fs": artifact.get("fs"),
              "segmentation_status": segmentation,
              "has_qrs_bounds": loops["qrs"] is not None, "has_t_bounds": loops["t"] is not None,
              "has_t_peak": has_median and _index(b.get("t_peak"), len(arrays["time_ms"])) is not None,
              "qrs_n_samples": len(loops["qrs"]) if loops["qrs"] is not None else None,
              "t_n_samples": len(loops["t"]) if loops["t"] is not None else None,
              "render_error": "", "diagnostic_content_version": "saved_v1_xyz_qrs_t_v2"}
    if resume and cache.is_file() and path.is_file():
        previous = json.loads(cache.read_text())
        if previous.get("fingerprint") == fingerprint and previous.get("sha256") == _sha(path):
            result.update(render_status="cached", sha256=previous["sha256"])
            return result
    if has_median:
        cfg = Config(**{key: tuple(value) if isinstance(value, list) else value for key, value in cfg_values.items()})
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                             "axes.titlesize": 10, "axes.labelsize": 9,
                             "xtick.labelsize": 8, "ytick.labelsize": 8})
        _plot_median(artifact, arrays, row, path, cfg)
    elif original_failure.is_file():
        # Existing raw eight-second diagnostic explicitly states NO MEDIAN.
        # Copy into this result tree; the Notebook 07 file is never changed.
        shutil.copy2(original_failure, path)
    else:
        # Subset/synthetic use: no raw arrays or an invented median to display.
        fig = plt.figure(figsize=(12, 7))
        fig.text(.07, .86, rid + " — KHÔNG CÓ MEDIAN", fontsize=18, color="#a63224")
        fig.text(.07, .65, "Không có QRS/T loop hay biên để đánh giá.\nKhông dựng waveform giả từ candidate.", fontsize=14)
        fig.text(.07, .4, "Lỗi đã lưu: " + str(artifact.get("error", "Không có median_xyz trong artifact")), fontsize=10)
        fig.text(.07, .12, NOTE, fontsize=11)
        fig.savefig(path, dpi=135, facecolor="white")
        plt.close(fig)
        result["render_status"] = "missing_median_notice"
    result["sha256"] = _sha(path)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _render_error(task, exc):
    """Keep a visible, traceable card even when a source or plot is broken."""
    row, _, output_dir, _, source_hash, _ = task
    out = Path(output_dir)
    rid = row["record_id"]
    path = out / "plots" / (_safe(rid) + ".png")
    reason = type(exc).__name__ + ": " + str(exc)
    fig = plt.figure(figsize=(13, 8))
    try:
        fig.text(.06, .9, rid + " — KHÔNG LOAD/RENDER ĐƯỢC", fontsize=17, color="#a63224")
        fig.text(.06, .80, f"patient={row['patient_id']} | {row['label_name']} | {row['split']}", fontsize=12)
        fig.text(.06, .67, "Recording vẫn nằm trong cohort; không dựng tín hiệu hoặc biên giả.", fontsize=12)
        fig.text(.06, .52, "Nguyên nhân:\n" + "\n".join(textwrap.wrap(reason, 100)), fontsize=11)
        fig.text(.06, .12, NOTE, fontsize=11)
        fig.savefig(path, dpi=130, facecolor="white")
    finally:
        plt.close(fig)
    return {"record_id": rid, "patient_id": row["patient_id"], "label_name": row["label_name"],
            "split": row["split"], "has_median": False, "plot_path": path.relative_to(out).as_posix(),
            "render_status": "render_error", "sha256": _sha(path), "fingerprint": source_hash,
            "fs": None, "segmentation_status": "load_or_render_error", "has_qrs_bounds": False,
            "has_t_bounds": False, "has_t_peak": False, "qrs_n_samples": None, "t_n_samples": None,
            "render_error": reason, "diagnostic_content_version": "saved_v1_xyz_qrs_t_v2"}


def _render_one(task):
    try:
        return _render_one_impl(task)
    except Exception as exc:
        return _render_error(task, exc)


def render_all_case_panels(frame: pd.DataFrame, artifact_root: Path, output_dir: Path,
                           cfg: Config | None = None, workers: int = 4, resume: bool = True) -> pd.DataFrame:
    """Render every supplied record, with strict identity/coverage/cache validation.

    ``output_dir`` is the ``all_cases`` folder. ``plot_path`` is relative to it.
    Subsets are supported for tests; the CLI defaults to all 448 saved records.
    """
    required = {"record_id", "patient_id", "label_name", "split"}
    if required - set(frame):
        raise ValueError(f"Missing record metadata: {sorted(required - set(frame))}")
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs")
    if workers < 1:
        raise ValueError("workers must be at least one")
    root, out = Path(artifact_root).resolve(), Path(output_dir).resolve()
    if (root / "artifacts").is_dir():
        root = root / "artifacts"
    if out == root or out in root.parents or root in out.parents:
        raise ValueError("Output must be separate from baseline artifacts")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    # Include the shared tangent helper in the cache identity.
    source_hash = hashlib.sha256(Path(__file__).read_bytes() +
                                (ROOT / "scripts/vcg_qc_audit_gallery.py").read_bytes()).hexdigest()
    fields = sorted(required | ({"qc_flags", "status", "qc_status"} & set(frame)))
    records = [{key: "" if pd.isna(value) else str(value) for key, value in row.items()}
               for row in frame[fields].to_dict("records")]
    tasks = [(row, str(root), str(out), asdict(cfg or Config()), source_hash, bool(resume)) for row in records]
    if workers == 1 or len(tasks) <= 1:
        rows = [_render_one(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(_render_one, task): task[0]["record_id"] for task in tasks}
            for job in as_completed(pending):
                try:
                    rows.append(job.result())
                except Exception as exc:
                    task = next(task for task in tasks if task[0]["record_id"] == pending[job])
                    rows.append(_render_error(task, exc))
                if len(rows) % 50 == 0:
                    print(f"All-case panels: {len(rows)}/{len(tasks)}", flush=True)
    manifest = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["record_id", "patient_id", "label_name", "split", "has_median", "plot_path", "render_status", "sha256", "fingerprint"])
    validate_case_panels(manifest, out, frame.record_id.tolist())
    manifest.to_csv(out / "all_case_panel_manifest.csv", index=False)
    (out / "panel_render_config.json").write_text(json.dumps({"records": len(manifest),
        "medians": int(manifest.has_median.sum()), "cached": int(manifest.render_status.eq("cached").sum()),
        "render_errors": int(manifest.render_status.eq("render_error").sum()),
        "source_sha256": source_hash, "interpretation": NOTE,
        "diagnostic_content_version": "saved_v1_xyz_qrs_t_v2",
        "loop_slice": "xyz[on:off + 1], each loop independently; common PQ origin and axis limits",
        "delineation_rerun": False}, ensure_ascii=False, indent=2) + "\n")
    return manifest


def validate_case_panels(manifest: pd.DataFrame, output_dir: Path, expected_ids=None):
    import struct
    if manifest.record_id.duplicated().any():
        raise ValueError("Duplicate rendered record IDs")
    if expected_ids is not None and (len(manifest) != len(expected_ids) or set(manifest.record_id) != set(expected_ids)):
        raise ValueError("Panel coverage differs from expected records")
    for row in manifest.to_dict("records"):
        path = Path(output_dir) / row["plot_path"]
        if not path.is_file() or _sha(path) != row["sha256"]:
            raise ValueError(f"Missing or modified panel: {path}")
        header = path.read_bytes()[:24]
        if header[:8] != b"\x89PNG\r\n\x1a\n" or min(struct.unpack(">II", header[16:24])) < 800:
            raise ValueError(f"Invalid/undersized panel PNG: {path}")
    return {"n_records": len(manifest), "n_medians": int(manifest.has_median.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/vcg_boundary_reference_eval_v1/all_cases")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "outputs/vcg_full_evaluation/artifacts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    from scripts.vcg_qc_audit import load_inputs
    frame, *_ = load_inputs()
    if args.limit is not None:
        frame = frame.head(args.limit)
    result = render_all_case_panels(frame, args.artifact_root, args.output_dir,
                                   workers=args.workers, resume=not args.no_resume)
    print(json.dumps({"records": len(result), "medians": int(result.has_median.sum()),
                      "status": result.render_status.value_counts().to_dict()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
