"""Descriptive reporting for the frozen PTB Frank-VCG evaluation.

This module does not change signals, boundaries, labels, or QC decisions. All
denominators in the label summaries include unsuccessful included recordings.
The generated report deliberately separates technical coverage from accuracy.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "QRS_Arc", "QRS_Area", "QRS_MaxVec", "QRS_MaxCentroidDistance",
    "QRS_MaxVel", "QRS_MeanVel", "QRS_StdVel", "T_MaxVec", "T_Area",
    "T_MaxVel", "T_MeanVel", "T_StdVel",
]
LABELS = ("HC", "MI")
COLORS = {"HC": "#157eab", "MI": "#cf6339"}
BOOL_COLUMNS = [
    "processing_success", "extraction_success", "qrs_boundary_valid",
    "qrs_algorithm_range_valid", "t_boundary_valid", "t_algorithm_range_valid",
]
NUMERIC_COLUMNS = FEATURE_NAMES + [
    "feature_completeness", "n_qrs_detected", "n_beats_used", "n_beats_excluded",
    "median_rr_ms", "qrs_on_ms", "qrs_off_ms", "qrs_duration_ms", "t_on_ms",
    "t_peak_ms", "t_off_ms", "t_duration_ms", "st_level", "tp_level",
    "t_on_method_gap_ms", "t_off_method_gap_ms",
]
TEXT_COLUMNS = [
    "patient_id", "record_id", "split", "diagnosis", "label_name", "qc_status",
    "qc_flags", "input_qc_flags", "evaluation_flags", "qrs_qc_flags",
    "t_qc_flags", "t_method", "plot_path", "loop_plot_path", "detection_plot_path", "failure_stage", "error",
]


def _bool_series(series: pd.Series) -> pd.Series:
    """Treat missing/unrecognized values as false, never bool('False')."""
    return series.map(
        lambda value: value is True or isinstance(value, (bool, np.bool_)) and bool(value)
        or isinstance(value, (int, float, np.number)) and pd.notna(value) and value == 1
        or isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}
    ).astype(bool)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in TEXT_COLUMNS:
        if column not in out:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    for column in BOOL_COLUMNS:
        if column not in out:
            out[column] = False
        out[column] = _bool_series(out[column])
    for column in NUMERIC_COLUMNS:
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out.loc[~np.isfinite(out[column]), column] = np.nan
    if "label" in out:
        numeric_label = pd.to_numeric(out["label"], errors="coerce")
        out.loc[out["label_name"].eq("") & numeric_label.eq(0), "label_name"] = "HC"
        out.loc[out["label_name"].eq("") & numeric_label.eq(1), "label_name"] = "MI"
    # Finite feature counts are measured here, independently of a success flag.
    out["feature_completeness"] = np.isfinite(out[FEATURE_NAMES]).sum(axis=1)
    out["qrs_detected"] = out["n_qrs_detected"].fillna(0).gt(0)
    out["qrs_pair_available"] = out[["qrs_on_ms", "qrs_off_ms"]].notna().all(axis=1)
    out["t_pair_available"] = out[["t_on_ms", "t_off_ms"]].notna().all(axis=1)
    out["all_features_finite"] = out["feature_completeness"].eq(len(FEATURE_NAMES))
    out["_flags"] = out["qc_flags"].map(_flags)
    out["_n_flags"] = out["_flags"].map(len)
    return out


def _flags(value: Any) -> list[str]:
    if value is None or isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    return sorted({item.strip() for item in re.split(r";\s*", str(value)) if item.strip()})


def _pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator * 100) if denominator else np.nan


def _display_pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({_pct(numerator, denominator):.1f}%)" if denominator else "0/0 (không có mẫu)"


def _groups(frame: pd.DataFrame):
    for label in LABELS:
        selected = frame.loc[frame["label_name"].eq(label)]
        yield label, "all", selected
        for split in ("train", "val", "test"):
            yield label, split, selected.loc[selected["split"].eq(split)]
    yield "ALL", "all", frame
    for split in ("train", "val", "test"):
        yield "ALL", split, frame.loc[frame["split"].eq(split)]


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, completeness_rows, flag_rows, statistic_rows = [], [], [], []
    metrics = {
        "qrs_detected": "qrs_detected", "qrs_pair_available": "qrs_pair_available",
        "qrs_boundary_valid": "qrs_boundary_valid",
        "qrs_algorithm_range_valid": "qrs_algorithm_range_valid",
        "t_pair_available": "t_pair_available", "t_boundary_valid": "t_boundary_valid",
        "t_algorithm_range_valid": "t_algorithm_range_valid",
        "processing_success": "processing_success", "extraction_success": "extraction_success",
        "all_12_features": "all_features_finite",
    }
    all_flags = sorted({flag for flags in frame["_flags"] for flag in flags})
    for label, split, subset in _groups(frame):
        count = len(subset)
        summary = {"label_name": label, "split": split, "n_recordings": count,
                   "n_patients": subset["patient_id"].nunique()}
        for name, column in metrics.items():
            n = int(subset[column].sum())
            summary[f"n_{name}"] = n
            summary[f"pct_{name}"] = _pct(n, count)
        for status in ("auto_candidate", "needs_review", "failed"):
            n = int(subset["qc_status"].eq(status).sum())
            summary[f"n_{status}"] = n
            summary[f"pct_{status}"] = _pct(n, count)
        n_flags = int(subset["_n_flags"].gt(0).sum())
        summary.update(n_any_qc_flag=n_flags, pct_any_qc_flag=_pct(n_flags, count))
        overlap = int((subset["all_features_finite"] & subset["qc_status"].eq("needs_review")).sum())
        summary.update(n_all12_and_needs_review=overlap,
                       pct_all12_and_needs_review=_pct(overlap, count))
        summary_rows.append(summary)
        for feature in FEATURE_NAMES:
            values = subset[feature].dropna()
            finite_n = len(values)
            completeness_rows.append({
                "label_name": label, "split": split, "feature": feature,
                "n_recordings": count, "n_finite": finite_n,
                "n_missing_or_nonfinite": count - finite_n,
                "finite_pct": _pct(finite_n, count),
                "missing_pct": _pct(count - finite_n, count),
            })
            statistic_rows.append({
                "label_name": label, "split": split, "feature": feature,
                "n_recordings": count, "n_finite": finite_n,
                "mean": values.mean(), "std": values.std(ddof=1),
                "median": values.median(), "min": values.min(), "max": values.max(),
                "missing_pct": _pct(count - finite_n, count),
            })
        for flag in all_flags:
            flag_n = int(subset["_flags"].map(lambda items: flag in items).sum())
            flag_rows.append({"label_name": label, "split": split, "qc_flag": flag,
                              "n_recordings_with_flag": flag_n,
                              "n_recordings": count, "pct_recordings": _pct(flag_n, count)})
    patient_rows = []
    for (patient, label, split), subset in frame.groupby(
        ["patient_id", "label_name", "split"], dropna=False, sort=True
    ):
        row = {"patient_id": patient, "label_name": label, "split": split,
               "n_recordings": len(subset),
               "record_ids": "; ".join(subset["record_id"].tolist())}
        for name, column in metrics.items():
            row[f"n_{name}"] = int(subset[column].sum())
            row[f"any_{name}"] = bool(subset[column].any())
            row[f"all_{name}"] = bool(subset[column].all())
        row["n_needs_review"] = int(subset["qc_status"].eq("needs_review").sum())
        row["n_failed"] = int(subset["qc_status"].eq("failed").sum())
        row["qc_flags"] = "; ".join(sorted({f for fs in subset["_flags"] for f in fs}))
        patient_rows.append(row)
    patient_columns = ["patient_id", "label_name", "split", "n_recordings", "record_ids"]
    for name in metrics:
        patient_columns += [f"n_{name}", f"any_{name}", f"all_{name}"]
    patient_columns += ["n_needs_review", "n_failed", "qc_flags"]
    return (
        pd.DataFrame(summary_rows), pd.DataFrame(completeness_rows),
        pd.DataFrame(flag_rows, columns=["label_name", "split", "qc_flag", "n_recordings_with_flag", "n_recordings", "pct_recordings"]),
        pd.DataFrame(patient_rows, columns=patient_columns), pd.DataFrame(statistic_rows),
    )


def _units(feature: str) -> str:
    return "mV²" if "Area" in feature else "mV/s" if "Vel" in feature else "mV"


def _draw_box(ax: Any, frame: pd.DataFrame, feature: str) -> None:
    arrays = [frame.loc[frame["label_name"].eq(label), feature].dropna().to_numpy() for label in LABELS]
    valid = [(index + 1, values) for index, values in enumerate(arrays) if len(values)]
    if valid:
        box = ax.boxplot([values for _, values in valid], positions=[index for index, _ in valid],
                         widths=0.48, patch_artist=True, showfliers=True,
                         flierprops={"markersize": 2.5, "alpha": 0.45})
        for patch, (position, _) in zip(box["boxes"], valid):
            patch.set_facecolor(COLORS[LABELS[position - 1]])
            patch.set_alpha(0.65)
    else:
        ax.text(.5, .5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([1, 2], [f"{label}\nn={len(values)}" for label, values in zip(LABELS, arrays)])
    ax.set_xlim(.4, 2.6)
    ax.set_title(feature)
    ax.set_ylabel(_units(feature))
    ax.grid(axis="y", alpha=.18)


def _plots(outdir: Path, frame: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    target = outdir / "plots" / "summary"
    target.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig: Any, name: str) -> None:
        fig.savefig(target / name, dpi=135, bbox_inches="tight")
        plt.close(fig)
        paths.append((target / name).relative_to(outdir).as_posix())

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    for feature in FEATURE_NAMES:
        fig, ax = plt.subplots(figsize=(4.7, 4.1))
        _draw_box(ax, frame, feature)
        fig.suptitle("All included recordings with finite values; QC retained", fontsize=9, y=1.01)
        fig.tight_layout()
        save(fig, f"feature_{feature}.png")
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax, feature in zip(axes.flat, FEATURE_NAMES):
        _draw_box(ax, frame, feature)
    fig.suptitle("Descriptive HC / MI comparison — all finite candidates, including review flags", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    save(fig, "feature_boxplots.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, field, title in zip(axes, ("qrs_duration_ms", "t_duration_ms"), ("QRS duration", "T duration")):
        pooled = frame[field].dropna().to_numpy()
        bins = np.histogram_bin_edges(pooled, bins=min(35, max(5, int(np.sqrt(len(pooled)))))) if len(pooled) else np.arange(2)
        for label in LABELS:
            values = frame.loc[frame["label_name"].eq(label), field].dropna()
            if len(values):
                ax.hist(values, bins=bins, alpha=.45, color=COLORS[label], label=f"{label} n={len(values)}")
        if not len(pooled):
            ax.text(.5, .5, "No finite durations", ha="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="ms", ylabel="Recordings")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Detected durations; availability and algorithm-range checks are separate")
    fig.tight_layout()
    save(fig, "duration_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    overall = summary.loc[summary["split"].eq("all") & summary["label_name"].isin(LABELS)].set_index("label_name")
    bottom = np.zeros(2)
    statuses = (("auto_candidate", "#4c956c"), ("needs_review", "#e5ab39"), ("failed", "#b34545"))
    for status, color in statuses:
        values = [float(overall.loc[label, f"pct_{status}"]) if pd.notna(overall.loc[label, f"pct_{status}"]) else 0 for label in LABELS]
        axes[0].bar(LABELS, values, bottom=bottom, label=status, color=color)
        bottom += values
    axes[0].set(title="Automatic QC status", ylabel="% of all included label recordings", ylim=(0, 105))
    axes[0].legend(fontsize=8)
    positions = np.arange(4)
    fields = ("qrs_detected", "qrs_pair_available", "t_pair_available", "all_12_features")
    for index, label in enumerate(LABELS):
        values = [float(overall.loc[label, f"pct_{field}"]) if pd.notna(overall.loc[label, f"pct_{field}"]) else 0 for field in fields]
        axes[1].bar(positions + (index - .5) * .36, values, .36, label=label, color=COLORS[label])
    axes[1].set_xticks(positions, ["QRS anchor", "QRS on/off", "T on/off", "12 features"])
    axes[1].set(title="Technical coverage, not annotation accuracy", ylabel="% of all included label recordings", ylim=(0, 105))
    axes[1].legend()
    fig.tight_layout()
    save(fig, "qc_and_coverage_rates.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(13)
    for index, label in enumerate(LABELS):
        counts = frame.loc[frame["label_name"].eq(label), "feature_completeness"].value_counts().reindex(positions, fill_value=0)
        ax.bar(positions + (index - .5) * .37, counts, .37, color=COLORS[label], label=label)
    ax.set(xticks=positions, xlabel="Number of finite features (out of 12)", ylabel="Recordings", title="Feature completeness, including unsuccessful recordings")
    ax.legend()
    fig.tight_layout()
    save(fig, "feature_completeness.png")
    return paths


def _review_groups(row: pd.Series) -> list[str]:
    groups = []
    flags = " ".join([row["qc_flags"], row["input_qc_flags"], row["evaluation_flags"]]).lower()
    if row["qc_status"] == "auto_candidate" and not row["_flags"]:
        groups.append("no_qc")
    if row["qrs_qc_flags"] or re.search(r"qrs|anchor|pq_overlap", flags):
        groups.append("qrs_concerns")
    if row["t_qc_flags"] or re.search(r"(?:^|[ ;])t_|tangent|tp_|st_level", flags):
        groups.append("t_concerns")
    if re.search(r"tangent|methods_disagree|method_gap|st_level", flags):
        groups.append("tangent_concerns")
    if row["_n_flags"] >= 3:
        groups.append("many_qc")
    return groups


GROUP_TITLES = {
    "no_qc": "Không có cờ QC tự động",
    "qrs_concerns": "Có cờ liên quan QRS",
    "t_concerns": "Có cờ liên quan T",
    "tangent_concerns": "Có cờ tiếp tuyến / mức ST / bất đồng phương pháp",
    "many_qc": "Có ít nhất 3 cờ QC",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _gallery(outdir: Path, frame: pd.DataFrame, plot_paths: list[str]) -> None:
    cards = []
    for _, row in frame.iterrows():
        groups = _review_groups(row)
        images = []
        for field, title in (("plot_path", "Delineation"), ("loop_plot_path", "QRS/T loops"),
                             ("detection_plot_path", "QRS detection")):
            path = row[field]
            if path:
                images.append(f'<a href="{_escape(path)}" target="_blank" rel="noopener"><img loading="lazy" src="{_escape(path)}" alt="{title}: {_escape(row["record_id"])}"></a>')
        if not images:
            images.append('<div class="empty">Không có hình: xem failure_stage / error.</div>')
        search_text = " ".join(str(row[c]) for c in ("patient_id", "record_id", "label_name", "split", "qc_flags", "error")).lower()
        finite_features = int(row["feature_completeness"])
        details = f'QRS on/off: {"có" if row["qrs_pair_available"] else "thiếu"}; T on/off: {"có" if row["t_pair_available"] else "thiếu"}; features: {finite_features}/12'
        failure = f'<p class="failure">{_escape(row["failure_stage"])}: {_escape(row["error"])}</p>' if row["error"] or row["failure_stage"] else ""
        cards.append(
            f'<article class="card" data-label="{_escape(row["label_name"])}" data-split="{_escape(row["split"])}" data-qc="{_escape(row["qc_status"])}" data-groups="{_escape(" ".join(groups))}" data-search="{_escape(search_text)}">'
            f'<h3>{_escape(row["record_id"])}</h3><p>{_escape(row["patient_id"])} · {_escape(row["label_name"])} · {_escape(row["split"])} · <b>{_escape(row["qc_status"])}</b></p>'
            f'<p>{_escape(details)}</p><p class="flags">{_escape(row["qc_flags"] or "Không có cờ QC tự động")}</p>{failure}<div class="images">{"".join(images)}</div></article>'
        )
    options = lambda values: '<option value="">Tất cả</option>' + "".join(f'<option value="{_escape(value)}">{_escape(value)}</option>' for value in values)
    groups = '<option value="">Tất cả</option>' + "".join(f'<option value="{key}">{_escape(title)}</option>' for key, title in GROUP_TITLES.items())
    summary_images = "".join(f'<a href="{_escape(path)}"><img loading="lazy" src="{_escape(path)}" alt="{_escape(Path(path).stem)}"></a>' for path in plot_paths if Path(path).name in {"feature_boxplots.png", "duration_distributions.png", "qc_and_coverage_rates.png", "feature_completeness.png"})
    content = """<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PTB Frank VCG evaluation</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#203040;background:#f5f7fa;margin:0}main{max-width:1500px;margin:auto;padding:25px}h1{font-size:27px}a{color:#126a96}.note{background:#fff3d7;border-left:4px solid #d3a13c;padding:12px 16px}.controls{display:flex;flex-wrap:wrap;gap:12px;background:#fff;padding:15px;position:sticky;top:0;z-index:2;box-shadow:0 2px 8px #ddd}label{display:grid;font-size:12px;gap:4px}select,input{padding:7px;font:inherit;max-width:310px}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,600px),1fr));gap:18px}.card{background:#fff;border:1px solid #dbe2e9;border-radius:8px;padding:15px;min-width:0}.card h3{margin:0;overflow-wrap:anywhere}.card p{margin:6px 0;overflow-wrap:anywhere}.flags,.failure{font-size:12px}.failure{color:#a33131}.images img{width:100%;height:auto}.empty{padding:40px;background:#f0f2f5;text-align:center}.summaries img{width:min(100%,720px);vertical-align:top}.summaries a{display:inline-block;width:49%;min-width:min(100%,500px)}[hidden]{display:none!important}#count{font-weight:bold}
</style><main><h1>PTB: delineation QRS/T và 12 đặc trưng Frank VCG</h1>
<p class="note">Đây là đánh giá độ bao phủ và kiểm tra QC của thuật toán đã cố định. Không có chuẩn chú giải độc lập tương ứng để kết luận độ chính xác của QRS/T. <b>auto_candidate</b> là ứng viên qua QC tự động; <b>needs_review</b> là cần xem lại, không đồng nghĩa thuật toán sai. Tất cả ứng viên được giữ trong gallery.</p>
<p><a href="full_evaluation_report.md">Báo cáo đầy đủ</a> · <a href="evaluation/qc_by_label.csv">QC theo nhãn</a> · <a href="evaluation/feature_statistics.csv">Thống kê đặc trưng</a> · <a href="evaluation/patient_summary.csv">Theo bệnh nhân</a></p>
<details><summary>Biểu đồ tổng hợp</summary><div class="summaries">SUMMARY_IMAGES</div></details>
<div class="controls"><label>Nhãn<select id="label">LABEL_OPTIONS</select></label><label>Split<select id="split">SPLIT_OPTIONS</select></label><label>QC<select id="qc">QC_OPTIONS</select></label><label>Nhóm ví dụ<select id="group">GROUP_OPTIONS</select></label><label>Tìm record / patient / cờ<input id="search" type="search" placeholder="patient, record, flag..."></label><button id="reset">Đặt lại</button></div>
<p id="count" aria-live="polite"></p><div class="cards">CARDS</div><script>
const cards=[...document.querySelectorAll('.card')];const ids=['label','split','qc','group','search'];function update(){const v=Object.fromEntries(ids.map(id=>[id,document.getElementById(id).value.trim().toLowerCase()]));let n=0;cards.forEach(c=>{const d=c.dataset;const ok=(!v.label||d.label.toLowerCase()===v.label)&&(!v.split||d.split.toLowerCase()===v.split)&&(!v.qc||d.qc.toLowerCase()===v.qc)&&(!v.group||d.groups.split(' ').includes(v.group))&&(!v.search||d.search.includes(v.search));c.hidden=!ok;if(ok)n++;});document.getElementById('count').textContent=n+' / '+cards.length+' recordings';}ids.forEach(id=>document.getElementById(id).addEventListener('input',update));document.getElementById('reset').addEventListener('click',()=>{ids.forEach(id=>document.getElementById(id).value='');update();});update();
</script></main></html>"""
    for key, value in {
        "SUMMARY_IMAGES": summary_images, "LABEL_OPTIONS": options(LABELS),
        "SPLIT_OPTIONS": options(("train", "val", "test")),
        "QC_OPTIONS": options(("auto_candidate", "needs_review", "failed")),
        "GROUP_OPTIONS": groups, "CARDS": "\n".join(cards),
    }.items():
        content = content.replace(key, value)
    (outdir / "index.html").write_text(content, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def _num(value: Any) -> str:
    try:
        return f"{float(value):.2f}" if np.isfinite(float(value)) else "NA"
    except (ValueError, TypeError):
        return "NA"


def _annotation_audit_text(annotation: Any) -> str:
    if not isinstance(annotation, dict):
        return str(annotation)
    selected = {}
    for key in ("audit_date", "challenge2006_reference", "accuracy_limits", "status", "notes"):
        if key in annotation:
            selected[key] = annotation[key]
    if isinstance(annotation.get("files"), dict):
        selected["local_reference_files"] = {
            key: annotation["files"][key]
            for key in ("annotators_file_present", "detected_independent_local_boundary_reference_files",
                        "original_release_has_independent_four_boundary_annotations")
            if key in annotation["files"]
        }
    if not selected:
        selected = {key: value for key, value in annotation.items() if key != "sources"}
    text = "```json\n" + json.dumps(selected, ensure_ascii=False, indent=2, default=str) + "\n```"
    sources = annotation.get("sources", [])
    if isinstance(sources, list):
        links = []
        for source in sources:
            if isinstance(source, dict) and str(source.get("url", "")).startswith(("https://", "http://")):
                links.append(f'[{source.get("id", "Nguồn audit")}]({source["url"]})')
        if links:
            text += "\n\nNguồn được ghi trong audit: " + ", ".join(links) + "."
    return text


def _report(outdir: Path, frame: pd.DataFrame, inventory: pd.DataFrame,
            summary: pd.DataFrame, flags: pd.DataFrame, patients: pd.DataFrame,
            config: dict[str, Any]) -> None:
    label_sets = {label: frame.loc[frame["label_name"].eq(label)] for label in LABELS}
    overall_rows = []
    for label, subset in label_sets.items():
        n = len(subset)
        overall_rows.append([
            label, n, subset["patient_id"].nunique(),
            _display_pct(int(subset["qrs_detected"].sum()), n),
            _display_pct(int(subset["qrs_pair_available"].sum()), n),
            _display_pct(int(subset["t_pair_available"].sum()), n),
            _display_pct(int(subset["all_features_finite"].sum()), n),
            _display_pct(int(subset["qc_status"].eq("auto_candidate").sum()), n),
            _display_pct(int(subset["qc_status"].eq("needs_review").sum()), n),
            _display_pct(int((subset["all_features_finite"] & subset["qc_status"].eq("needs_review")).sum()), n),
            _display_pct(int(subset["qc_status"].eq("failed").sum()), n),
        ])
    coverage_table = _markdown_table(
        ["Nhãn", "Recordings", "Patients", "Phát hiện ≥1 QRS", "Đủ QRS-on/off", "Đủ T-on/off", "12/12 hữu hạn", "auto_candidate", "needs_review", "Đủ 12 & needs_review", "failed"],
        overall_rows,
    )
    validation_rows = []
    for label, subset in label_sets.items():
        n = len(subset)
        validation_rows.append([
            label, _display_pct(int(subset["qrs_boundary_valid"].sum()), n),
            _display_pct(int(subset["qrs_algorithm_range_valid"].sum()), n),
            _display_pct(int(subset["t_boundary_valid"].sum()), n),
            _display_pct(int(subset["t_algorithm_range_valid"].sum()), n),
        ])
    split_rows = []
    for split in ("train", "val", "test"):
        for label in LABELS:
            subset = frame.loc[frame["split"].eq(split) & frame["label_name"].eq(label)]
            split_rows.append([split, label, len(subset), subset["patient_id"].nunique(),
                               _display_pct(int(subset["all_features_finite"].sum()), len(subset)),
                               _display_pct(int(subset["qc_status"].eq("needs_review").sum()), len(subset))])
    patient_rows = []
    for label in LABELS:
        subset = patients.loc[patients["label_name"].eq(label)]
        n = len(subset)
        patient_rows.append([
            label, n,
            _display_pct(int(subset["any_all_12_features"].sum()), n),
            _display_pct(int(subset["all_all_12_features"].sum()), n),
            _display_pct(int(subset["any_qrs_pair_available"].sum()), n),
            _display_pct(int(subset["all_qrs_pair_available"].sum()), n),
            _display_pct(int(subset["any_t_pair_available"].sum()), n),
            _display_pct(int(subset["all_t_pair_available"].sum()), n),
        ])
    top_flag_rows = []
    for label in LABELS:
        selected = flags.loc[flags["label_name"].eq(label) & flags["split"].eq("all") & flags["n_recordings_with_flag"].gt(0)]
        for _, row in selected.sort_values(["n_recordings_with_flag", "qc_flag"], ascending=[False, True]).head(15).iterrows():
            top_flag_rows.append([label, row["qc_flag"], _display_pct(int(row["n_recordings_with_flag"]), int(row["n_recordings"]))])
    duration_rows = []
    for label, subset in label_sets.items():
        for field in ("qrs_duration_ms", "t_duration_ms"):
            values = subset[field].dropna()
            duration_rows.append([label, field, len(values), _num(values.mean()), _num(values.std()),
                                  _num(values.median()), _num(values.min()), _num(values.max())])
    tangent_rows = []
    for label, subset in label_sets.items():
        n = len(subset)
        for title, pattern in (
            ("T-on: tiếp tuyến không giải được", r"t_on_tangent_unresolved"),
            ("T-off: tiếp tuyến không giải được", r"t_off_tangent_unresolved"),
            ("Mức ST cần xem lại", r"st_level_requires_review"),
            ("T-end: hai phương pháp bất đồng", r"t_end_methods_disagree"),
            ("Không về baseline ở T-off", r"t_off_no_baseline_return"),
        ):
            selected = subset["qc_flags"].str.contains(pattern, regex=True, na=False)
            tangent_rows.append([label, title, _display_pct(int(selected.sum()), n)])
    gap_rows = []
    for label, subset in label_sets.items():
        for field in ("t_on_method_gap_ms", "t_off_method_gap_ms"):
            values = subset[field].dropna().abs()
            gap_rows.append([label, field, len(values), _num(values.median()), _num(values.max())])
    examples = []
    for group, title in GROUP_TITLES.items():
        for label, subset in label_sets.items():
            selected = subset.loc[subset.apply(lambda row: group in _review_groups(row), axis=1)] if len(subset) else subset
            links = []
            for _, row in selected.head(4).iterrows():
                link = f'[{row["record_id"]}]({row["plot_path"]})' if row["plot_path"] else f'`{row["record_id"]}`'
                links.append(f'{link} ({row["split"]}; {row["qc_status"]}; {int(row["_n_flags"])} cờ)')
            examples.append([title, label, len(selected), "; ".join(links) or "Không có recording trong nhóm này"])
    inventory_text = f"Inventory có {len(inventory)} recording"
    if "patient_id" in inventory:
        inventory_text += f" của {inventory['patient_id'].nunique()} bệnh nhân"
    inventory_text += "."
    eligible_count = None
    excluded_rows = []
    if "included" in inventory:
        included = _bool_series(inventory["included"])
        eligible_count = int(included.sum())
        inventory_text += f" {int(included.sum())} dòng thuộc cohort đánh giá, {int((~included).sum())} dòng ngoài cohort."
        excluded = inventory.loc[~included]
        if "diagnosis" in excluded:
            for diagnosis, subset in excluded.groupby("diagnosis", dropna=False):
                excluded_rows.append([diagnosis, len(subset),
                                      subset["patient_id"].nunique() if "patient_id" in subset else "NA"])
    for field in ("missing_signal", "missing_metadata"):
        if field in inventory:
            inventory_text += f" {field}: {int(_bool_series(inventory[field]).sum())}."
    split_conflicts = int(frame.groupby("patient_id")["split"].nunique().gt(1).sum()) if len(frame) else 0
    failure_counts = frame.loc[frame["qc_status"].eq("failed"), "failure_stage"].replace("", "unspecified").value_counts()
    failure_text = "; ".join(f"{key}: {value}" for key, value in failure_counts.items()) or "Không có recording mang trạng thái failed."
    difference_rows = []
    for title, criterion in (
        ("Đủ T-on/off", lambda subset: subset["t_pair_available"]),
        ("ST cần xem lại", lambda subset: subset["qc_flags"].str.contains("st_level_requires_review", regex=False)),
        ("needs_review", lambda subset: subset["qc_status"].eq("needs_review")),
    ):
        percentages = [_pct(int(criterion(label_sets[label]).sum()), len(label_sets[label])) for label in LABELS]
        difference_rows.append([title, _num(percentages[0]), _num(percentages[1]),
                                _num(percentages[1] - percentages[0])])
    review_text = "Template chú giải chưa được tạo trong thư mục kết quả này."
    review_path = outdir / "annotations" / "manual_review_template.csv"
    if review_path.exists():
        review = pd.read_csv(review_path)
        rows = []
        if all(column in review for column in ("label_name", "patient_id")):
            for label in LABELS:
                selected = review.loc[review["label_name"].eq(label)]
                confirmed = int(selected.get("review_status", pd.Series("", index=selected.index)).eq("confirmed").sum())
                rows.append([label, len(selected), selected["patient_id"].nunique(), confirmed])
        review_text = f"[Template review](annotations/manual_review_template.csv) có **{len(review)} recording**"
        if "patient_id" in review:
            review_text += f" của **{review['patient_id'].nunique()} bệnh nhân**"
        review_text += ". Đây là danh sách đề xuất review, chưa phải chuẩn đã chú giải."
        if rows:
            review_text += "\n\n" + _markdown_table(["Nhãn", "Recordings được chọn", "Patients", "review_status=confirmed"], rows)
    annotation_text = _annotation_audit_text(config.get("annotation_audit", {}))
    split_details = config.get("split_details", config.get("split_method", "Xem config.json hoặc manifest của lần chạy."))
    if not isinstance(split_details, str):
        split_details = json.dumps(split_details, ensure_ascii=False, indent=2, default=str)
    baseline = config.get("baseline_hashes", config.get("baseline", {}))
    baseline_text = "```json\n" + json.dumps(baseline, ensure_ascii=False, indent=2, default=str) + "\n```" if baseline else "Hash và cấu hình baseline được ghi trong manifest/config của lần chạy."
    scope_text = f"Báo cáo này chứa **{len(frame)} recording đã chạy**, thuộc **{frame['patient_id'].nunique()} bệnh nhân**."
    if eligible_count is not None:
        scope_text += f" Cohort HC/MI trong inventory có {eligible_count} recording."
        if len(frame) != eligible_count:
            scope_text += " **Đây là lần chạy một phần / smoke test; các tỷ lệ chỉ áp dụng cho những recording đã chạy, chưa đại diện toàn cohort.**"
        else:
            scope_text += " Đã có dòng kết quả cho toàn bộ cohort được chọn, bao gồm các trường hợp thất bại."
    lines = [
        "# Đánh giá PTB HC/MI bằng Frank XYZ VCG",
        "",
        "## Kết luận có thể rút ra",
        "",
        "Các tỷ lệ dưới đây đo **khả năng chạy, trả về ranh giới và trích đặc trưng**, cùng tần suất cảnh báo QC của thuật toán đã cố định. Chúng **không đo độ chính xác delineation**: chưa có chuẩn chú giải độc lập tương ứng cho QRS-on/off và T-on/off trên Frank XYZ đại diện. Một cờ QC là dấu hiệu cần kiểm tra; số cờ cao không tự chứng minh thuật toán sai. Các recording cùng bệnh nhân không phải quan sát độc lập.",
        "", scope_text,
        "",
        coverage_table,
        "",
        "Mẫu số của mọi tỷ lệ HC/MI trong bảng tổng hợp là **toàn bộ recording đã chạy thuộc nhãn đó**, bao gồm recording xử lý thất bại, thiếu ranh giới hoặc thiếu đặc trưng. Không loại ứng viên vì có nhiều QC. Các bảng CSV có thêm từng split; phân tích test chỉ mô tả, không dùng để chỉnh tham số.",
        "",
        "## Cohort, dữ liệu thiếu và split",
        "", inventory_text, "",
        _markdown_table(["Chẩn đoán ngoài HC/MI", "Recordings ngoài cohort", "Patients"], excluded_rows) if excluded_rows else "Không có phân nhóm chẩn đoán bị loại trong inventory của lần chạy.", "",
        "Chỉ Frank X/Y/Z được đưa vào pipeline; nhãn HC/MI dùng cho mô tả kết quả, không dùng để điều chỉnh delineation từng nhóm. Recording ngoài HC/MI trong inventory không nằm trong mẫu số HC/MI.",
        "", str(split_details), "",
        f"Số bệnh nhân xuất hiện ở nhiều split: **{split_conflicts}**. Nếu số này khác 0, split không đáp ứng độc lập theo bệnh nhân và cần sửa trước ML.",
        "", _markdown_table(["Split", "Nhãn", "Recordings", "Patients", "12/12 hữu hạn", "needs_review"], split_rows),
        "",
        "## 1. Bao nhiêu % recording HC/MI phát hiện được QRS?",
        "",
        "Cột **Phát hiện ≥1 QRS** ở bảng đầu dùng điều kiện `n_qrs_detected > 0`, trên tất cả recording theo nhãn. Phát hiện được ít nhất một mốc chưa bảo đảm đủ nhịp để tạo median representative beat, cũng chưa bảo đảm mốc trùng chú giải chuyên gia. Xem `n_qrs_detected`, `n_beats_used`, `n_beats_excluded` và các cờ đầu vào của từng record để phân biệt các bước này.",
        "",
        "## 2. Bao nhiêu % có đủ QRS-on và QRS-off?",
        "",
        "Cột **Đủ QRS-on/off** chỉ đếm hai giá trị hữu hạn được trả về. Các phép kiểm tra thứ tự và khoảng thuật toán được báo riêng dưới đây; chúng là kiểm tra tính hợp lệ nội bộ, không phải chuẩn lâm sàng. Giới hạn QRS 40–240 ms và việc bao quanh anchor là điều kiện của pipeline đang đánh giá.",
        "",
        "## 3. Bao nhiêu % có đủ T-on và T-off?",
        "",
        "Cột **Đủ T-on/off** ở bảng đầu chỉ đếm cặp mốc hữu hạn. Kiểm tra thứ tự QRS/T và khoảng T của thuật toán được báo riêng. Một cặp mốc có thể vẫn cần xem lại vì T thấp, nhiều lobe, ST khác mức PQ, TP không yên hoặc gần giới hạn vùng tìm.",
        "", _markdown_table(["Nhãn", "QRS đúng thứ tự", "QRS qua giới hạn thuật toán", "T đúng thứ tự", "T qua giới hạn thuật toán"], validation_rows),
        "", _markdown_table(["Nhãn", "Duration (ms)", "n hữu hạn", "Mean", "SD", "Median", "Min", "Max"], duration_rows),
        "",
        "## 4. Những bất thường / cờ QC nào xuất hiện và khác nhau theo nhãn ra sao?",
        "",
        "Bảng dưới liệt kê tối đa 15 cờ thường gặp của mỗi nhãn. Một recording có thể có nhiều cờ; không cộng các tỷ lệ để suy ra tổng recording lỗi. `needs_review` giữ nguyên ứng viên cho xem xét, `failed` phản ánh bước không hoàn tất theo chính sách của lần chạy.",
        "", _markdown_table(["Nhãn", "Cờ QC", "Recording / toàn nhãn"], top_flag_rows) if top_flag_rows else "Không ghi nhận cờ QC trong dữ liệu kết quả.",
        "", "Các bước thất bại: " + failure_text, "",
        "Xem toàn bộ cờ tại [qc_flags_by_label.csv](evaluation/qc_flags_by_label.csv); xem từng recording và hình trong [gallery](index.html). Chênh lệch tỷ lệ cờ HC/MI là mô tả dữ liệu và hành vi heuristic, không phải ước lượng sai số thực của thuật toán.",
        "", "Chênh lệch quan sát **MI − HC**, đơn vị điểm phần trăm (cùng mẫu số toàn recording đã chạy của mỗi nhãn):", "",
        _markdown_table(["Tiêu chí", "HC (%)", "MI (%)", "MI − HC (điểm %)"], difference_rows), "",
        "Các chênh lệch trên chỉ mang tính mô tả, chưa kiểm định ý nghĩa thống kê và chưa điều chỉnh tương quan giữa recording cùng bệnh nhân.",
        "",
        "## 5. Sai số ranh giới thực tế là bao nhiêu nếu có ground truth?",
        "",
        "**Chưa xác định được.** Không báo MAE, bias, sensitivity, PPV hay accuracy cho QRS/T khi chưa có mốc độc lập có thể đối chiếu cùng recording, cùng hệ quy chiếu và cùng định nghĩa. Không dùng chính ranh giới do thuật toán tạo, giới hạn duration hoặc kết quả xem hình của tác giả như ground truth.",
        "",
        "Audit nguồn chú giải của lần chạy:", "", annotation_text, "",
        "Xem [audit nguồn và giới hạn so sánh](baseline_v1/reference_audit.md); cấu trúc audit đầy đủ được giữ trong `config.json`. Số liệu lịch sử trong audit không được dùng thay cho kết quả mới của lần chạy này.", "",
        "Nếu tìm được QT duration ở một lead ECG chuẩn cho các ID PTB trùng nhau, chỉ có thể cân nhắc phân tích QT tương ứng sau khi kiểm tra lead, nhịp và định nghĩa. QT duration không cung cấp đầy đủ QRS-on/off và T-on/off trên Frank VCG, nên không được chuyển thành benchmark sai số bốn mốc. Không tạo bảng so sánh chuẩn giả.",
        "",
        "## 6. Phương pháp tiếp tuyến có thất bại có hệ thống trên MI không?",
        "",
        "**Chưa có bằng chứng để khẳng định thất bại có hệ thống.** Các cờ dưới đây mô tả khả năng giải giao điểm, sự khác biệt phương pháp và mức ST của baseline. T-on lấy tiếp tuyến ở vị trí đạo hàm dương lớn nhất trên sườn lên rồi cắt mức ST; T-off dùng sườn xuống cắt baseline TP (PQ nếu thiếu TP). Trên magnitude, mức ST/TP và hình dạng nhiều lobe có thể làm định nghĩa này nhạy, nhưng cần chuẩn độc lập hoặc review có giao thức để kết luận sai.",
        "", _markdown_table(["Nhãn", "Dấu hiệu cần xem", "Recording / toàn nhãn"], tangent_rows),
        "", _markdown_table(["Nhãn", "Khoảng cách hai phương pháp", "n có cả hai mốc", "Median |gap| (ms)", "Max |gap| (ms)"], gap_rows),
        "",
        "Khoảng cách tiếp tuyến–ngưỡng không phải sai số so với chuẩn và chỉ tính ở recording có cả hai ứng viên. Recording không giải được một phương pháp không nằm trong mẫu gap; vì vậy phải đọc bảng gap cùng tỷ lệ thiếu/cờ. Không chọn ngưỡng mới hoặc sửa V2 dựa trên tập test này.",
        "",
        "## 7. Bao nhiêu % có đủ 12 đặc trưng và bao nhiêu % cần review?",
        "",
        "Cột **12/12 hữu hạn** và **needs_review** của bảng đầu trả lời ở cấp recording. Có đủ 12 giá trị hữu hạn chỉ có nghĩa phép tính hoàn tất trên loop được cắt; không chứng minh các biên đúng. Thống kê và boxplot bao gồm mọi giá trị hữu hạn dù có QC, để tránh chỉ trình bày nhóm thuận lợi. NaN/Inf được xem là thiếu; không thay bằng 0.",
        "", "Cột **auto_candidate** báo độ bao phủ ứng viên qua bộ QC tự động. Cột **Đủ 12 & needs_review** cho thấy phần giao giữa hoàn tất trích đặc trưng và cần review, với mẫu số vẫn là toàn recording của nhãn; hai điều kiện này có thể đồng thời xảy ra.",
        "",
        "Ở cấp bệnh nhân, báo riêng **ít nhất một** và **tất cả** recording đạt tiêu chí; không gộp hai định nghĩa này thành một tỷ lệ:",
        "", _markdown_table(["Nhãn", "Patients", "≥1 record đủ 12", "Mọi record đủ 12", "≥1 record đủ QRS", "Mọi record đủ QRS", "≥1 record đủ T", "Mọi record đủ T"], patient_rows),
        "",
        "[feature_completeness.csv](evaluation/feature_completeness.csv) báo số hữu hạn/thiếu theo từng feature; [feature_statistics.csv](evaluation/feature_statistics.csv) báo mean, SD, median, min, max và missing %. Mean/SD dùng các giá trị hữu hạn, SD mẫu (`ddof=1`); nhóm có dưới 2 giá trị có SD = NA. Arc/vector/distance dùng mV, area mV², velocity mV/s. So sánh HC/MI là mô tả, chưa có kiểm định độc lập theo bệnh nhân hay hiệu năng phân loại.",
        "",
        "## 8. Có cần V2 trước khi làm ML không?",
        "",
        "**Cần một bước kiểm tra chất lượng có giao thức trước ML; chưa đủ căn cứ để kết luận phải thay toàn bộ thuật toán bằng V2.** Dùng kết quả này để lập danh sách review theo bệnh nhân, nhãn, split và loại cờ; chú giải độc lập mẫu đại diện của các recording có/không có cờ và đánh giá sai số bốn mốc. Giữ baseline hiện tại và manifest làm mốc so sánh. Nếu review xác nhận lỗi có kiểu lặp lại, phát triển V2 trên train/val rồi khóa lại trước một đánh giá test mới hoặc tập ngoài.",
        "",
        "Khi chuẩn bị ML, split theo bệnh nhân, chỉ fit imputation/scaling/feature selection trên train, và mô tả rõ chính sách cho missing/QC. Đánh giá riêng khả năng phân loại với khoảng bất định theo bệnh nhân. Nếu đã dùng test để quyết định thiết kế V2, cần giữ một tập test chưa xem khác cho kết luận xác nhận. Lần chạy này không tự sửa biên hay tinh chỉnh theo nhãn.",
        "",
        "## Ví dụ để xem lại",
        "",
        review_text, "",
        "Template chọn đa dạng bệnh nhân trước, bao phủ nhóm lỗi/QRS/T/QC khác và nhóm không có cờ; các trường hợp train/val có cờ chưa được đại diện có thể được bổ sung. Mẫu này hỗ trợ tìm kiểu lỗi, không phải mẫu xác suất để ước lượng accuracy toàn cohort.", "",
        "Giao thức chú giải: người review đặt bốn mốc trên tín hiệu/median beat hiển thị **ẩn dự đoán và nhãn chẩn đoán** ở lượt đầu; giữ các cột `pred_*` riêng và chỉ mở sau khi chốt mốc tham chiếu. Điền người chú giải, hệ thời gian, mốc anchor nếu dùng tín hiệu gốc, lý do không dùng được và trạng thái xác nhận; chỉ đặt `independent_reference=true` khi mốc thực sự được tạo độc lập. Review lại hoặc phân xử các bất đồng. Các ảnh có overlay dự đoán trong gallery dùng cho kiểm tra sau chú giải, không dùng để tạo chuẩn độc lập.", "",
        "[Script đối chiếu](baseline_v1/compare_vcg_annotations.py) chỉ nhận tham chiếu có người review, đã xác nhận và khai báo độc lập. Script hỗ trợ `median_relative_ms`; mốc `raw_recording_ms` phải chỉ rõ anchor thuộc nhịp đã dùng và chỉ cho phép kết luận so median với nhịp quan sát đó. Các cột `pred_*` không được dùng làm chuẩn. Khi chưa có dữ liệu tham chiếu đạt điều kiện, bảng comparison rỗng là kết quả đúng.", "",
        "Các ví dụ là những recording đầu tiên theo thứ tự kết quả trong từng nhóm, để điều hướng kiểm tra; không phải mẫu ngẫu nhiên hoặc ước lượng prevalence. Một recording có thể thuộc nhiều nhóm. Không có cờ QC vẫn cần được lấy mẫu để tìm lỗi mà heuristic bỏ sót.",
        "", _markdown_table(["Nhóm", "Nhãn", "Tổng ứng viên", "Ví dụ"], examples),
        "",
        "## Artifacts và tái lập",
        "",
        "- [Gallery lọc theo nhãn/split/QC](index.html), chứa tất cả kết quả, kể cả ứng viên cần review và recording thất bại.",
        "- [QC và độ bao phủ](evaluation/qc_by_label.csv); [cờ QC](evaluation/qc_flags_by_label.csv).",
        "- [Theo bệnh nhân](evaluation/patient_summary.csv); [feature completeness](evaluation/feature_completeness.csv); [feature statistics](evaluation/feature_statistics.csv).",
        "- [12 boxplot HC/MI](plots/summary/feature_boxplots.png); [duration](plots/summary/duration_distributions.png); [QC và độ bao phủ](plots/summary/qc_and_coverage_rates.png).",
        "",
        "Hash baseline:", "", baseline_text, "",
        "Giới hạn: representative median beat làm giảm biến thiên giữa các nhịp; kết quả này không thay cho đánh giá từng nhịp, không phải chẩn đoán MI, và không chứng minh khả năng tổng quát ngoài cohort PTB được chọn. Các phép kiểm tra QC hỗ trợ review nhưng không xác nhận đúng sinh lý của loop.",
        "",
    ]
    (outdir / "full_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_outputs(outdir: Path, frame: pd.DataFrame, inventory: pd.DataFrame,
                          config: dict[str, Any]) -> dict[str, Any]:
    """Write summary CSVs, static plots, a filterable gallery and Vietnamese report.

    ``frame`` must contain one row per included recording, including failures.
    Plot paths in rows are relative to ``outdir``. Missing optional fields are
    represented as empty/NA; no reference annotations or accuracy are inferred.
    """
    outdir = Path(outdir)
    evaluation_dir = outdir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_frame(frame)
    summary, completeness, flags, patients, statistics = _summaries(prepared)
    artifacts = {
        "qc_by_label": summary, "feature_completeness": completeness,
        "qc_flags_by_label": flags, "patient_summary": patients,
        "feature_statistics": statistics,
    }
    for name, table in artifacts.items():
        table.to_csv(evaluation_dir / f"{name}.csv", index=False)
    plot_paths = _plots(outdir, prepared, summary)
    _gallery(outdir, prepared, plot_paths)
    _report(outdir, prepared, inventory, summary, flags, patients, config)
    return {"n_recordings": len(prepared), "n_patients": prepared["patient_id"].nunique(),
            "summary_csvs": [f"evaluation/{name}.csv" for name in artifacts],
            "summary_plots": plot_paths, "gallery": "index.html",
            "report": "full_evaluation_report.md"}
