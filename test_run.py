import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, sosfiltfilt, find_peaks
from sklearn.decomposition import PCA
import wfdb

DATA_ROOT = Path("content/ptbdb/physionet.org/files/ptbdb/1.0.0")
rec_mi_path = DATA_ROOT / "patient001/s0010_re"
rec_hc_path = DATA_ROOT / "patient104/s0306lre"

SG_WINDOW_SAMPLES = 1201
SG_POLYORDER = 2
PRE_R_MS = 300
POST_R_MS = 500

FEATURE_NAMES = [
    "QRS_Arc", "QRS_Area", "QRS_MaxVec", "QRS_MaxCentroidDistance",
    "QRS_MaxVel", "QRS_MeanVel", "QRS_StdVel",
    "T_MaxVec", "T_Area", "T_MaxVel", "T_MeanVel", "T_StdVel",
]

def get_xyz_channels(channel_names):
    aliases = {"X": {"x", "vx", "frankx", "frank_x"},
               "Y": {"y", "vy", "franky", "frank_y"},
               "Z": {"z", "vz", "frankz", "frank_z"}}
    normalized = {name: re.sub(r"[^a-z0-9_]", "", str(name).lower()) for name in channel_names}
    result = {}
    for axis, accepted in aliases.items():
        hits = [orig for orig, norm in normalized.items() if norm in accepted]
        result[axis] = hits[0]
    return result

def choose_detection_channel(channel_names):
    lower_to_orig = {str(name).lower(): name for name in channel_names}
    for pref in ["ii", "i", "v2"]:
        if pref in lower_to_orig:
            return lower_to_orig[pref]
    xyz_names = set(get_xyz_channels(channel_names).values())
    candidates = [name for name in channel_names if name not in xyz_names]
    return candidates[0] if candidates else channel_names[0]

def load_ptb_record(record_path, max_seconds=30):
    header = wfdb.rdheader(str(record_path))
    xyz_names = get_xyz_channels(header.sig_name)
    r_name = choose_detection_channel(header.sig_name)
    selected = [xyz_names[axis] for axis in "XYZ"] + [r_name]
    sampto = min(header.sig_len, int(max_seconds * header.fs)) if max_seconds else header.sig_len
    rec = wfdb.rdrecord(str(record_path), channel_names=selected, sampto=sampto)
    cols = {name: rec.p_signal[:, i].astype(float) for i, name in enumerate(rec.sig_name)}
    xyz = np.column_stack([cols[xyz_names[axis]] for axis in "XYZ"])
    return {
        "xyz": xyz, "r_signal": cols[r_name], "r_channel": r_name,
        "fs": float(rec.fs), "xyz_names": xyz_names, "header": header
    }

def safe_savgol_window(n_samples, desired_window, polyorder):
    if n_samples <= polyorder + 1:
        raise ValueError("Tín hiệu quá ngắn cho Savitzky–Golay")
    window = min(int(desired_window), n_samples if n_samples % 2 == 1 else n_samples - 1)
    if window % 2 == 0:
        window -= 1
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
    high = min(nyquist - 1.0, 20.0)
    low = max(1.0, 5.0)
    sos = butter(2, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
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
        local_idx = np.argmax(np.abs(ecg[start:end]))
        refined.append(start + local_idx)
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
    representative = np.median(np.asarray(beats), axis=0)
    baseline_n = max(1, min(len(representative), int(baseline_ms * fs / 1000)))
    return representative - np.median(representative[:baseline_n], axis=0)

def compute_vector_magnitude(xyz):
    return np.linalg.norm(np.asarray(xyz), axis=1)

def _nearest_below_run(values, start, direction, threshold, run_length, limit):
    if direction < 0:
        indices = range(start, limit - 1, -1)
    else:
        indices = range(start, limit + 1)
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

def build_qrs_loop(representative_xyz, boundaries):
    return np.asarray(representative_xyz)[boundaries["qrs_on"]:boundaries["qrs_off"] + 1]

def build_t_loop(representative_xyz, boundaries):
    return np.asarray(representative_xyz)[boundaries["t_on"]:boundaries["t_off"] + 1]

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
    features = {
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
    return features

for name, p in [("MI", rec_mi_path), ("HC", rec_hc_path)]:
    raw = load_ptb_record(p, max_seconds=30)
    filt, _ = preprocess_xyz(raw["xyz"])
    rp, bp = detect_r_peaks(raw["r_signal"], raw["fs"])
    beats, kept_p, r_idx = extract_beats(filt, rp, raw["fs"])
    rep = build_representative_beat(beats, raw["fs"])
    bounds = delineate_qrs_t(rep, raw["fs"], r_idx)
    qrs = build_qrs_loop(rep, bounds)
    t = build_t_loop(rep, bounds)
    feat = extract_vcg_features(qrs, t, raw["fs"])
    print(f"[{name}] QRS shape: {qrs.shape}, T shape: {t.shape}")
    print(f"[{name}] QRS_MaxVec: {feat['QRS_MaxVec']:.5f}, T_MaxVec: {feat['T_MaxVec']:.5f}")
print("SUCCESS!")
