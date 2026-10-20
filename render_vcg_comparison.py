import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, butter, sosfiltfilt, find_peaks
from sklearn.decomposition import PCA
import wfdb

# Thiết lập matplotlib không cần display
import matplotlib
matplotlib.use('Agg')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

DATA_ROOT = Path("content/ptbdb/physionet.org/files/ptbdb/1.0.0")
rec_mi_path = DATA_ROOT / "patient001/s0010_re"
rec_hc_path = DATA_ROOT / "patient104/s0306lre"

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

ARTIFACT_DIR = Path("/Users/nguyenbaothach/.gemini/antigravity-ide/brain/07eacadf-4f57-41ec-94cd-bb4bb5821f5f")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "QRS_Arc", "QRS_Area", "QRS_MaxVec", "QRS_MaxCentroidDistance",
    "QRS_MaxVel", "QRS_MeanVel", "QRS_StdVel",
    "T_MaxVec", "T_Area", "T_MaxVel", "T_MeanVel", "T_StdVel",
]

def get_xyz_channels(channel_names):
    aliases = {"X": {"x", "vx", "frankx", "frank_x"},
               "Y": {"y", "vy", "franky", "frank_y"},
               "Z": {"z", "vz", "frankz", "frank_z"}}
    normalized = {name: str(name).lower().replace("_", "").replace("-", "") for name in channel_names}
    res = {}
    for ax, acts in aliases.items():
        hits = [orig for orig, norm in normalized.items() if norm in acts]
        res[ax] = hits[0]
    return res

def choose_detection_channel(channel_names):
    low = {str(n).lower(): n for n in channel_names}
    for pref in ["ii", "i", "v2"]:
        if pref in low: return low[pref]
    xyz_set = set(get_xyz_channels(channel_names).values())
    cands = [n for n in channel_names if n not in xyz_set]
    return cands[0] if cands else channel_names[0]

def load_ptb_record(record_path, max_seconds=30):
    header = wfdb.rdheader(str(record_path))
    xyz_names = get_xyz_channels(header.sig_name)
    r_name = choose_detection_channel(header.sig_name)
    selected = [xyz_names[ax] for ax in "XYZ"] + [r_name]
    sampto = min(header.sig_len, int(max_seconds * header.fs)) if max_seconds else header.sig_len
    rec = wfdb.rdrecord(str(record_path), channel_names=selected, sampto=sampto)
    cols = {name: rec.p_signal[:, i].astype(float) for i, name in enumerate(rec.sig_name)}
    xyz = np.column_stack([cols[xyz_names[ax]] for ax in "XYZ"])
    return {
        "xyz": xyz, "r_signal": cols[r_name], "r_channel": r_name,
        "fs": float(rec.fs), "xyz_names": xyz_names, "header": header
    }

def safe_savgol_window(n_samples, desired_window, polyorder):
    if n_samples <= polyorder + 1:
        raise ValueError("Tín hiệu quá ngắn cho Savitzky–Golay")
    window = min(int(desired_window), n_samples if n_samples % 2 == 1 else n_samples - 1)
    if window % 2 == 0: window -= 1
    return max(polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4, window)

def preprocess_xyz(xyz, window_samples=1201, polyorder=2):
    n = len(xyz)
    w = safe_savgol_window(n, window_samples, polyorder)
    filtered = np.column_stack([
        savgol_filter(xyz[:, i], window_length=w, polyorder=polyorder, mode="interp")
        for i in range(3)
    ])
    return filtered, w

def detect_r_peaks(ecg_signal, fs):
    ecg = np.asarray(ecg_signal, dtype=float)
    nyquist = fs / 2
    sos = butter(2, [max(1.0, 5.0) / nyquist, min(nyquist - 1.0, 20.0) / nyquist], btype="bandpass", output="sos")
    bandpassed = sosfiltfilt(sos, ecg)
    distance = int(0.3 * fs)
    prominence = 0.4 * np.ptp(bandpassed)
    cand_peaks, _ = find_peaks(bandpassed, distance=distance, prominence=prominence)
    if len(cand_peaks) == 0:
        cand_peaks, _ = find_peaks(bandpassed, distance=distance)
    refined = []
    half_window = int(0.08 * fs)
    for p in cand_peaks:
        start = max(0, p - half_window)
        end = min(len(ecg), p + half_window + 1)
        refined.append(start + np.argmax(np.abs(ecg[start:end])))
    return np.unique(refined), bandpassed

def extract_beats(xyz, r_peaks, fs, pre_r_ms=300, post_r_ms=500):
    pre = int(round(pre_r_ms * fs / 1000))
    post = int(round(post_r_ms * fs / 1000))
    beats, kept_peaks = [], []
    for peak in np.asarray(r_peaks, dtype=int):
        start, stop = peak - pre, peak + post + 1
        if start >= 0 and stop <= len(xyz):
            beats.append(xyz[start:stop])
            kept_peaks.append(peak)
    return np.stack(beats), np.asarray(kept_peaks), pre

def build_representative_beat(beats, fs, baseline_ms=100):
    rep = np.median(np.asarray(beats), axis=0)
    baseline_n = max(1, min(len(rep), int(baseline_ms * fs / 1000)))
    return rep - np.median(rep[:baseline_n], axis=0)

def compute_vector_magnitude(xyz):
    return np.linalg.norm(np.asarray(xyz), axis=1)

def _nearest_below_run(values, start, direction, threshold, run_length, limit):
    indices = range(start, limit - 1, -1) if direction < 0 else range(start, limit + 1)
    count = 0
    for idx in indices:
        count = count + 1 if values[idx] <= threshold else 0
        if count >= run_length:
            return idx + run_length - 1 if direction < 0 else idx - run_length + 1
    return limit

def delineate_qrs_t(representative_xyz, fs, r_index):
    xyz = np.asarray(representative_xyz, dtype=float)
    n = len(xyz)
    speed = np.linalg.norm(np.gradient(xyz, 1 / fs, axis=0), axis=1)
    smooth_window = safe_savgol_window(n, max(5, int(0.021 * fs) | 1), 2)
    activity = savgol_filter(speed, smooth_window, 2)

    q_lo = max(0, r_index - int(0.15 * fs))
    q_hi = min(n - 1, r_index + int(0.18 * fs))
    baseline_zone = np.r_[activity[:max(1, q_lo)], activity[min(n, r_index + int(0.45 * fs)):]]
    baseline = float(np.median(baseline_zone)) if baseline_zone.size else float(np.median(activity))
    q_peak = float(np.max(activity[q_lo:q_hi + 1]))
    q_threshold = baseline + 0.12 * max(q_peak - baseline, 0)
    run = max(2, int(0.012 * fs))
    qrs_on = _nearest_below_run(activity, r_index, -1, q_threshold, run, q_lo)
    qrs_off = _nearest_below_run(activity, r_index, +1, q_threshold, run, q_hi)

    magnitude = compute_vector_magnitude(xyz)
    t_search_lo = min(n - 2, qrs_off + int(0.04 * fs))
    t_search_hi = min(n - 1, r_index + int(0.45 * fs))
    t_peak = t_search_lo + int(np.argmax(magnitude[t_search_lo:t_search_hi + 1]))
    t_baseline = float(np.median(np.r_[magnitude[:max(1, qrs_on)], magnitude[max(t_search_hi, 0):]]))
    t_threshold = t_baseline + 0.10 * max(float(magnitude[t_peak]) - t_baseline, 0)
    t_run = max(2, int(0.015 * fs))
    t_on = _nearest_below_run(magnitude, t_peak, -1, t_threshold, t_run, t_search_lo)
    t_off = _nearest_below_run(magnitude, t_peak, +1, t_threshold, t_run, t_search_hi)

    qrs_on = int(np.clip(qrs_on, 0, r_index - 1))
    qrs_off = int(np.clip(qrs_off, r_index + 1, t_search_lo - 1))
    t_on = int(np.clip(t_on, qrs_off + 1, t_peak))
    t_off = int(np.clip(t_off, t_peak + 1, n - 1))
    return {"qrs_on": qrs_on, "qrs_off": qrs_off, "t_on": t_on, "t_off": t_off,
            "activity": activity, "magnitude": magnitude}

def project_loop_pca(loop_3d):
    loop = np.asarray(loop_3d, dtype=float)
    model = PCA(n_components=2)
    projected = model.fit_transform(loop)
    return projected, model, model.explained_variance_ratio_

def compute_arc(loop_2d):
    return float(np.linalg.norm(np.diff(np.asarray(loop_2d), axis=0), axis=1).sum())

def compute_area(loop_2d):
    points = np.asarray(loop_2d, dtype=float)
    u, v = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))

def compute_velocity_features(loop_3d, fs):
    velocity = np.gradient(np.asarray(loop_3d, dtype=float), 1 / fs, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return {"max": float(np.max(speed)), "mean": float(np.mean(speed)), "std": float(np.std(speed, ddof=0))}

def extract_vcg_features(qrs_loop, t_loop, fs):
    qrs_2d, _, _ = project_loop_pca(qrs_loop)
    t_2d, _, _ = project_loop_pca(t_loop)
    qrs_vel = compute_velocity_features(qrs_loop, fs)
    t_vel = compute_velocity_features(t_loop, fs)
    centroid = np.mean(qrs_loop, axis=0)
    return {
        "QRS_Arc": compute_arc(qrs_2d),
        "QRS_Area": compute_area(qrs_2d),
        "QRS_MaxVec": float(np.linalg.norm(qrs_loop, axis=1).max()),
        "QRS_MaxCentroidDistance": float(np.linalg.norm(qrs_loop - centroid, axis=1).max()),
        "QRS_MaxVel": qrs_vel["max"],
        "QRS_MeanVel": qrs_vel["mean"],
        "QRS_StdVel": qrs_vel["std"],
        "T_MaxVec": float(np.linalg.norm(t_loop, axis=1).max()),
        "T_Area": compute_area(t_2d),
        "T_MaxVel": t_vel["max"],
        "T_MeanVel": t_vel["mean"],
        "T_StdVel": t_vel["std"],
    }

# Load and process data
data = {}
for code, path, lbl in [("HC", rec_hc_path, "Healthy Control"), ("MI", rec_mi_path, "Myocardial Infarction")]:
    raw = load_ptb_record(path, max_seconds=30)
    filt, _ = preprocess_xyz(raw["xyz"])
    rp, bp = detect_r_peaks(raw["r_signal"], raw["fs"])
    beats, kept_p, r_idx = extract_beats(filt, rp, raw["fs"])
    rep = build_representative_beat(beats, raw["fs"])
    bnd = delineate_qrs_t(rep, raw["fs"], r_idx)
    qrs_3d = rep[bnd["qrs_on"]:bnd["qrs_off"]+1]
    t_3d = rep[bnd["t_on"]:bnd["t_off"]+1]
    feat = extract_vcg_features(qrs_3d, t_3d, raw["fs"])
    data[code] = {
        "raw": raw, "filt": filt, "rp": rp, "bp": bp, "beats": beats, "r_idx": r_idx,
        "rep": rep, "bnd": bnd, "qrs_3d": qrs_3d, "t_3d": t_3d, "feat": feat, "name": lbl
    }

hc = data["HC"]
mi = data["MI"]
fs = hc["raw"]["fs"]

# =========================================================================
# FIGURE 1: Tín hiệu thời gian Frank X, Y, Z và Vector Magnitude M(t)
# =========================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex='row')
fig.suptitle("SO SÁNH TÍN HIỆU THỜI GIAN FRANK VCG VÀ VECTOR MAGNITUDE: HC VS MI", fontsize=16, fontweight='bold', y=0.98)

# Thời gian 4 giây cho sóng liên tục
t_4s = np.arange(int(4 * fs)) / fs

# Màu sắc chủ đạo
c_hc = "#1e40af" # Xanh dương đậm
c_mi = "#b91c1c" # Đỏ đậm

for col_idx, (d, color, title_prefix) in enumerate([(hc, c_hc, "Người Khỏe Mạnh (HC) - patient104"), 
                                                   (mi, c_mi, "Bệnh Nhân Nhồi Máu Cơ Tim (MI) - patient001")]):
    # Frank X
    axes[0, col_idx].plot(t_4s, d["filt"][:len(t_4s), 0], color=color, lw=1.2, label=f"Filtered X")
    axes[0, col_idx].plot(t_4s, d["raw"]["xyz"][:len(t_4s), 0], color='gray', alpha=0.35, lw=0.8, label="Raw X")
    axes[0, col_idx].set_title(f"{title_prefix}\nĐạo trình Frank X (Phải -> Trái)", fontweight='bold')
    axes[0, col_idx].set_ylabel("Điện thế (mV)")
    axes[0, col_idx].legend(loc="upper right", frameon=True)
    
    # Frank Y
    axes[1, col_idx].plot(t_4s, d["filt"][:len(t_4s), 1], color=color, lw=1.2)
    axes[1, col_idx].set_title("Đạo trình Frank Y (Đầu -> Chân)", fontweight='bold')
    axes[1, col_idx].set_ylabel("Điện thế (mV)")
    
    # Frank Z
    axes[2, col_idx].plot(t_4s, d["filt"][:len(t_4s), 2], color=color, lw=1.2)
    axes[2, col_idx].set_title("Đạo trình Frank Z (Sau -> Trước)", fontweight='bold')
    axes[2, col_idx].set_ylabel("Điện thế (mV)")
    axes[2, col_idx].set_xlabel("Thời gian (giây)")

    # Row 4: Representative Beat Vector Magnitude M(t)
    beat_t_ms = (np.arange(len(d["rep"])) - d["r_idx"]) / fs * 1000.0
    mag = compute_vector_magnitude(d["rep"])
    ax_mag = axes[3, col_idx]
    ax_mag.plot(beat_t_ms, mag, color=color, lw=2.0, label="M(t) = ||[X, Y, Z]||")
    
    # Tô màu dải QRS và T
    q_start = beat_t_ms[d["bnd"]["qrs_on"]]
    q_end = beat_t_ms[d["bnd"]["qrs_off"]]
    t_start = beat_t_ms[d["bnd"]["t_on"]]
    t_end = beat_t_ms[d["bnd"]["t_off"]]
    
    ax_mag.axvspan(q_start, q_end, color="#10b981", alpha=0.25, label=f"Vòng QRS [{q_start:.0f} - {q_end:.0f} ms]")
    ax_mag.axvspan(t_start, t_end, color="#f59e0b", alpha=0.25, label=f"Vòng T [{t_start:.0f} - {t_end:.0f} ms]")
    ax_mag.axvline(0, color="black", ls="--", lw=1.0, alpha=0.7, label="Đỉnh R (t=0)")
    
    ax_mag.set_title("Vector Magnitude M(t) trên Nhịp Đại Diện (Representative Beat)", fontweight='bold')
    ax_mag.set_xlabel("Thời gian so với đỉnh R (ms)")
    ax_mag.set_ylabel("Biên độ vector ||M|| (mV)")
    ax_mag.legend(loc="upper right", frameon=True)
    
    # Chú thích đỉnh MaxVec
    max_q = d["feat"]["QRS_MaxVec"]
    ax_mag.annotate(f"MaxVec QRS: {max_q:.3f} mV", 
                    xy=(0, max_q), xytext=(0, max_q * 1.08),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                    fontweight='bold', color=color)

plt.tight_layout()
p1 = OUT_DIR / "01_vcg_signals_time_domain.png"
plt.savefig(p1, dpi=300)
shutil.copy(p1, ARTIFACT_DIR / p1.name)
plt.close()
print(f"Saved {p1}")

# =========================================================================
# FIGURE 2: Mô hình Không gian 3D & Các Mặt Phẳng Chiếu (Frontal, Horizontal, Sagittal, PCA)
# =========================================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle("MÔ HÌNH HÌNH HỌC VCG 3D VÀ CÁC MẶT PHẲNG CHIẾU: NGƯỜI BÌNH THƯỜNG (HC) VS NHỒI MÁU CƠ TIM (MI)", fontsize=16, fontweight='bold', y=0.98)

# 1. 3D Loop cho HC
ax_3d_hc = fig.add_subplot(3, 4, 1, projection='3d')
ax_3d_hc.plot(hc["qrs_3d"][:, 0], hc["qrs_3d"][:, 1], hc["qrs_3d"][:, 2], color="#2563eb", lw=2.5, label="QRS loop")
ax_3d_hc.plot(hc["t_3d"][:, 0], hc["t_3d"][:, 1], hc["t_3d"][:, 2], color="#f59e0b", lw=2.0, ls="--", label="T loop")
ax_3d_hc.scatter(0, 0, 0, color="black", s=30, label="Gốc (Origin)")
ax_3d_hc.set_title("HC (Khỏe mạnh) - 3D VCG Loop\n(Quỹ đạo rộng, elip mượt mà)", fontweight='bold', fontsize=11)
ax_3d_hc.set_xlabel("X (mV)"); ax_3d_hc.set_ylabel("Y (mV)"); ax_3d_hc.set_zlabel("Z (mV)")
ax_3d_hc.legend(loc="upper left")

# 2. 3D Loop cho MI
ax_3d_mi = fig.add_subplot(3, 4, 2, projection='3d')
ax_3d_mi.plot(mi["qrs_3d"][:, 0], mi["qrs_3d"][:, 1], mi["qrs_3d"][:, 2], color="#dc2626", lw=2.5, label="QRS loop")
ax_3d_mi.plot(mi["t_3d"][:, 0], mi["t_3d"][:, 1], mi["t_3d"][:, 2], color="#f59e0b", lw=2.0, ls="--", label="T loop")
ax_3d_mi.scatter(0, 0, 0, color="black", s=30, label="Gốc (Origin)")
ax_3d_mi.set_title("MI (Nhồi máu) - 3D VCG Loop\n(Co cụm, thể tích suy giảm nặng)", fontweight='bold', fontsize=11)
ax_3d_mi.set_xlabel("X (mV)"); ax_3d_mi.set_ylabel("Y (mV)"); ax_3d_mi.set_zlabel("Z (mV)")
ax_3d_mi.legend(loc="upper left")

# 3. Mặt phẳng tối ưu PCA 2D cho HC
qrs_2d_hc, _, _ = project_loop_pca(hc["qrs_3d"])
t_2d_hc, _, _ = project_loop_pca(hc["t_3d"])
ax_pca_hc = fig.add_subplot(3, 4, 3)
ax_pca_hc.fill(qrs_2d_hc[:, 0], qrs_2d_hc[:, 1], color="#3b82f6", alpha=0.25, label=f"Area: {hc['feat']['QRS_Area']:.5f}")
ax_pca_hc.plot(qrs_2d_hc[:, 0], qrs_2d_hc[:, 1], color="#1d4ed8", lw=2, label=f"Arc: {hc['feat']['QRS_Arc']:.3f}")
ax_pca_hc.scatter(qrs_2d_hc[0, 0], qrs_2d_hc[0, 1], color="green", s=35, zorder=5, label="Start")
# Centroid
c_hc = np.mean(qrs_2d_hc, axis=0)
ax_pca_hc.scatter(c_hc[0], c_hc[1], color="red", marker="x", s=50, zorder=5, label="Centroid")
ax_pca_hc.set_title("HC - Mặt phẳng tối ưu PCA (2D)\n(Diện tích lớn, đối xứng tốt)", fontweight='bold', fontsize=11)
ax_pca_hc.set_xlabel("PC 1 (mV)"); ax_pca_hc.set_ylabel("PC 2 (mV)")
ax_pca_hc.axis("equal"); ax_pca_hc.legend(fontsize=8, loc="upper right")

# 4. Mặt phẳng tối ưu PCA 2D cho MI
qrs_2d_mi, _, _ = project_loop_pca(mi["qrs_3d"])
t_2d_mi, _, _ = project_loop_pca(mi["t_3d"])
ax_pca_mi = fig.add_subplot(3, 4, 4)
ax_pca_mi.fill(qrs_2d_mi[:, 0], qrs_2d_mi[:, 1], color="#ef4444", alpha=0.25, label=f"Area: {mi['feat']['QRS_Area']:.5f}")
ax_pca_mi.plot(qrs_2d_mi[:, 0], qrs_2d_mi[:, 1], color="#b91c1c", lw=2, label=f"Arc: {mi['feat']['QRS_Arc']:.3f}")
ax_pca_mi.scatter(qrs_2d_mi[0, 0], qrs_2d_mi[0, 1], color="green", s=35, zorder=5, label="Start")
c_mi = np.mean(qrs_2d_mi, axis=0)
ax_pca_mi.scatter(c_mi[0], c_mi[1], color="red", marker="x", s=50, zorder=5, label="Centroid")
ax_pca_mi.set_title("MI - Mặt phẳng tối ưu PCA (2D)\n(Diện tích co rút 58 lần!)", fontweight='bold', fontsize=11)
ax_pca_mi.set_xlabel("PC 1 (mV)"); ax_pca_mi.set_ylabel("PC 2 (mV)")
ax_pca_mi.axis("equal"); ax_pca_mi.legend(fontsize=8, loc="upper right")

# Hàng 2 & 3: Chiếu lên 3 mặt phẳng giải phẫu (Frontal X-Y, Horizontal X-Z, Sagittal Z-Y)
# 1. Frontal (X-Y) - Cột 1: HC, Cột 2: MI
ax_fr_hc = fig.add_subplot(3, 3, 4)
ax_fr_hc.plot(hc["rep"][:, 0], hc["rep"][:, 1], color="#94a3b8", lw=0.8, alpha=0.5, label="Toàn nhịp")
ax_fr_hc.plot(hc["qrs_3d"][:, 0], hc["qrs_3d"][:, 1], color="#2563eb", lw=2.2, label="QRS")
ax_fr_hc.plot(hc["t_3d"][:, 0], hc["t_3d"][:, 1], color="#f59e0b", lw=1.8, ls="--", label="T")
ax_fr_hc.set_title("Mặt phẳng Trán (Frontal: X-Y) — HC", fontweight='bold')
ax_fr_hc.set_xlabel("X (Trái - Phải) [mV]"); ax_fr_hc.set_ylabel("Y (Đầu - Chân) [mV]")
ax_fr_hc.legend(loc="lower right", fontsize=8)

ax_fr_mi = fig.add_subplot(3, 3, 5)
ax_fr_mi.plot(mi["rep"][:, 0], mi["rep"][:, 1], color="#94a3b8", lw=0.8, alpha=0.5)
ax_fr_mi.plot(mi["qrs_3d"][:, 0], mi["qrs_3d"][:, 1], color="#dc2626", lw=2.2, label="QRS")
ax_fr_mi.plot(mi["t_3d"][:, 0], mi["t_3d"][:, 1], color="#f59e0b", lw=1.8, ls="--", label="T")
ax_fr_mi.set_title("Mặt phẳng Trán (Frontal: X-Y) — MI", fontweight='bold')
ax_fr_mi.set_xlabel("X (Trái - Phải) [mV]"); ax_fr_mi.set_ylabel("Y (Đầu - Chân) [mV]")
ax_fr_mi.legend(loc="lower right", fontsize=8)

# So sánh đè nét trực tiếp Frontal
ax_fr_comp = fig.add_subplot(3, 3, 6)
ax_fr_comp.plot(hc["qrs_3d"][:, 0], hc["qrs_3d"][:, 1], color="#2563eb", lw=2.2, label="HC QRS")
ax_fr_comp.plot(mi["qrs_3d"][:, 0], mi["qrs_3d"][:, 1], color="#dc2626", lw=2.2, label="MI QRS")
ax_fr_comp.plot(hc["t_3d"][:, 0], hc["t_3d"][:, 1], color="#60a5fa", lw=1.5, ls=":", label="HC T")
ax_fr_comp.plot(mi["t_3d"][:, 0], mi["t_3d"][:, 1], color="#f87171", lw=1.5, ls=":", label="MI T")
ax_fr_comp.set_title("ĐỐI SÁNH ĐÈ NÉT: Frontal (X-Y)", fontweight='bold', color="#4338ca")
ax_fr_comp.set_xlabel("X (mV)"); ax_fr_comp.set_ylabel("Y (mV)")
ax_fr_comp.legend(loc="lower right", fontsize=8)

# 2. Horizontal (X-Z)
ax_hz_hc = fig.add_subplot(3, 3, 7)
ax_hz_hc.plot(hc["qrs_3d"][:, 0], hc["qrs_3d"][:, 2], color="#2563eb", lw=2.2, label="QRS")
ax_hz_hc.plot(hc["t_3d"][:, 0], hc["t_3d"][:, 2], color="#f59e0b", lw=1.8, ls="--", label="T")
ax_hz_hc.set_title("Mặt phẳng Ngang (Horizontal: X-Z) — HC", fontweight='bold')
ax_hz_hc.set_xlabel("X (Trái - Phải) [mV]"); ax_hz_hc.set_ylabel("Z (Sau - Trước) [mV]")
ax_hz_hc.legend(loc="lower right", fontsize=8)

ax_hz_mi = fig.add_subplot(3, 3, 8)
ax_hz_mi.plot(mi["qrs_3d"][:, 0], mi["qrs_3d"][:, 2], color="#dc2626", lw=2.2, label="QRS")
ax_hz_mi.plot(mi["t_3d"][:, 0], mi["t_3d"][:, 2], color="#f59e0b", lw=1.8, ls="--", label="T")
ax_hz_mi.set_title("Mặt phẳng Ngang (Horizontal: X-Z) — MI", fontweight='bold')
ax_hz_mi.set_xlabel("X (Trái - Phải) [mV]"); ax_hz_mi.set_ylabel("Z (Sau - Trước) [mV]")
ax_hz_mi.legend(loc="lower right", fontsize=8)

# So sánh đè nét Horizontal
ax_hz_comp = fig.add_subplot(3, 3, 9)
ax_hz_comp.plot(hc["qrs_3d"][:, 0], hc["qrs_3d"][:, 2], color="#2563eb", lw=2.2, label="HC QRS")
ax_hz_comp.plot(mi["qrs_3d"][:, 0], mi["qrs_3d"][:, 2], color="#dc2626", lw=2.2, label="MI QRS")
ax_hz_comp.plot(hc["t_3d"][:, 0], hc["t_3d"][:, 2], color="#60a5fa", lw=1.5, ls=":", label="HC T")
ax_hz_comp.plot(mi["t_3d"][:, 0], mi["t_3d"][:, 2], color="#f87171", lw=1.5, ls=":", label="MI T")
ax_hz_comp.set_title("ĐỐI SÁNH ĐÈ NÉT: Horizontal (X-Z)", fontweight='bold', color="#4338ca")
ax_hz_comp.set_xlabel("X (mV)"); ax_hz_comp.set_ylabel("Z (mV)")
ax_hz_comp.legend(loc="lower right", fontsize=8)

plt.tight_layout()
p2 = OUT_DIR / "02_vcg_loops_3d_and_planes.png"
plt.savefig(p2, dpi=300)
shutil.copy(p2, ARTIFACT_DIR / p2.name)
plt.close()
print(f"Saved {p2}")

# =========================================================================
# FIGURE 3: So sánh định lượng 12 đặc trưng toán học và Radar Chart
# =========================================================================
fig, (ax_bar, ax_radar) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("ĐỐI SÁNH ĐỊNH LƯỢNG 12 ĐẶC TRƯNG HÌNH HỌC VÀ ĐỘNG HỌC VCG: HC VS MI", fontsize=16, fontweight='bold', y=0.98)

# 1. Bar chart giá trị chuẩn hóa
df_comp = pd.DataFrame({
    "HC": [hc["feat"][k] for k in FEATURE_NAMES],
    "MI": [mi["feat"][k] for k in FEATURE_NAMES]
}, index=FEATURE_NAMES)

# Chuẩn hóa về [0, 1] theo giá trị lớn hơn giữa HC và MI để cùng biểu diễn
df_norm = df_comp.div(df_comp.max(axis=1), axis=0)

x_pos = np.arange(len(FEATURE_NAMES))
width = 0.38

bars_hc = ax_bar.barh(x_pos - width/2, df_norm["HC"], width, label="Healthy Control (HC)", color="#2563eb", alpha=0.85)
bars_mi = ax_bar.barh(x_pos + width/2, df_norm["MI"], width, label="Myocardial Infarction (MI)", color="#dc2626", alpha=0.85)

ax_bar.set_yticks(x_pos)
ax_bar.set_yticklabels(FEATURE_NAMES, fontweight='bold')
ax_bar.set_xlabel("Giá trị tương đối chuẩn hóa (Normalized Value)")
ax_bar.set_title("So sánh tương quan 12 đặc trưng toán học VCG", fontweight='bold')
ax_bar.legend(loc="lower right", frameon=True)
ax_bar.invert_yaxis()

# Thêm giá trị phần trăm tỷ lệ MI/HC
for i, feat in enumerate(FEATURE_NAMES):
    val_hc = df_comp.loc[feat, "HC"]
    val_mi = df_comp.loc[feat, "MI"]
    pct = (val_mi / val_hc) * 100 if val_hc != 0 else 0
    ax_bar.text(max(df_norm.loc[feat, "HC"], df_norm.loc[feat, "MI"]) + 0.02, i, 
                f"MI = {pct:.1f}% HC", va='center', fontsize=9, fontweight='bold', 
                color="#b91c1c" if pct < 50 else "#1e40af")

ax_bar.set_xlim(0, 1.25)

# 2. Polar Radar Chart
ax_radar.remove()
ax_polar = fig.add_subplot(1, 2, 2, polar=True)

angles = np.linspace(0, 2 * np.pi, len(FEATURE_NAMES), endpoint=False).tolist()
angles += angles[:1] # Đóng vòng

vals_hc = df_norm["HC"].tolist() + [df_norm["HC"].iloc[0]]
vals_mi = df_norm["MI"].tolist() + [df_norm["MI"].iloc[0]]

ax_polar.plot(angles, vals_hc, color="#2563eb", linewidth=2, label="Healthy Control (HC)")
ax_polar.fill(angles, vals_hc, color="#3b82f6", alpha=0.25)

ax_polar.plot(angles, vals_mi, color="#dc2626", linewidth=2, label="Myocardial Infarction (MI)")
ax_polar.fill(angles, vals_mi, color="#ef4444", alpha=0.25)

ax_polar.set_xticks(angles[:-1])
ax_polar.set_xticklabels(FEATURE_NAMES, fontsize=9, fontweight='bold')
ax_polar.set_title("Radar Profile: Diện tích bao phủ đặc trưng\n(HC bao trùm hoàn toàn MI)", fontweight='bold', pad=20)
ax_polar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
p3 = OUT_DIR / "03_vcg_12_features_quant_comparison.png"
plt.savefig(p3, dpi=300)
shutil.copy(p3, ARTIFACT_DIR / p3.name)
plt.close()
print(f"Saved {p3}")

print("\n--- HOÀN THÀNH TOÀN BỘ CÁC BIỂU ĐỒ SO SÁNH ---")
