import os
import re
import math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, butter, sosfiltfilt, find_peaks
from sklearn.decomposition import PCA
import wfdb

# Thiết lập thẩm mỹ cho Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

DATA_ROOT = Path("content/ptbdb/physionet.org/files/ptbdb/1.0.0")
rec_mi_path = DATA_ROOT / "patient001/s0010_re"
rec_hc_path = DATA_ROOT / "patient104/s0306lre"

SG_WINDOW_SAMPLES = 1201
SG_POLYORDER = 2
PRE_R_MS = 300
POST_R_MS = 500

def get_xyz_channels(channel_names):
    aliases = {"X": {"x", "vx", "frankx", "frank_x"},
               "Y": {"y", "vy", "franky", "frank_y"},
               "Z": {"z", "vz", "frankz", "frank_z"}}
    normalized = {name: re.sub(r"[^a-z0-9_]", "", str(name).lower()) for name in channel_names}
    result = {}
    for axis, accepted in aliases.items():
        hits = [orig for orig, norm in normalized.items() if norm in accepted]
        if len(hits) != 1:
            raise ValueError(f"Không xác định duy nhất {axis}")
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

def safe_savgol_window(n_samples, desired_window=1201, polyorder=2):
    w = min(int(desired_window), n_samples if n_samples % 2 == 1 else n_samples - 1)
    if w % 2 == 0: w -= 1
    return max(polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4, w)

def preprocess_xyz(xyz, window_samples=1201, polyorder=2):
    n_samples = len(xyz)
    w = safe_savgol_window(n_samples, window_samples, polyorder)
    filtered = np.column_stack([
        savgol_filter(xyz[:, i], window_length=w, polyorder=polyorder, mode="interp")
        for i in range(3)
    ])
    return filtered, w

def detect_r_peaks(ecg_signal, fs):
    ecg = np.asarray(ecg_signal, dtype=float)
    nyq = fs / 2.0
    sos = butter(2, [max(1.0, 5.0) / nyq, min(nyq - 1.0, 20.0) / nyq], btype="bandpass", output="sos")
    bandpassed = sosfiltfilt(sos, ecg)
    distance = int(0.3 * fs)
    prominence = 0.4 * np.ptp(bandpassed)
    cand_peaks, _ = find_peaks(bandpassed, distance=distance, prominence=prominence)
    if len(cand_peaks) == 0:
        cand_peaks, _ = find_peaks(bandpassed, distance=distance)
    refined = []
    half_w = int(0.08 * fs)
    for p in cand_peaks:
        s = max(0, p - half_w)
        e = min(len(ecg), p + half_w + 1)
        refined.append(s + np.argmax(np.abs(ecg[s:e])))
    return np.unique(refined), bandpassed

def extract_beats(xyz, r_peaks, fs, pre_r_ms=300, post_r_ms=500):
    pre = int(round(pre_r_ms * fs / 1000.0))
    post = int(round(post_r_ms * fs / 1000.0))
    beats, kept_peaks = [], []
    for p in r_peaks:
        if p - pre >= 0 and p + post <= len(xyz):
            beats.append(xyz[p - pre: p + post])
            kept_peaks.append(p)
    return np.asarray(beats), kept_peaks, pre

def build_representative_beat(beats, fs, baseline_ms=100):
    rep = np.median(beats, axis=0)
    b_n = max(1, min(len(rep), int(baseline_ms * fs / 1000.0)))
    base = np.median(rep[:b_n], axis=0)
    return rep - base

def compute_vector_magnitude(xyz):
    return np.linalg.norm(xyz, axis=1)

def _nearest_below_run(vals, start, direction, thresh, run_len, limit):
    idx_range = range(start, limit - 1, -1) if direction < 0 else range(start, limit + 1)
    cnt = 0
    for idx in idx_range:
        if vals[idx] <= thresh:
            cnt += 1
            if cnt >= run_len:
                return idx + (run_len - 1 if direction < 0 else -(run_len - 1))
        else:
            cnt = 0
    return limit

def delineate_qrs_t(rep_xyz, fs, r_idx):
    mag = compute_vector_magnitude(rep_xyz)
    dt = 1.0 / fs
    vel = np.linalg.norm(np.gradient(rep_xyz, dt, axis=0), axis=1)
    s_vel = savgol_filter(vel, window_length=safe_savgol_window(len(vel), int(0.03 * fs), 2), polyorder=2)
    
    b_len = max(5, int(0.1 * fs))
    q_thresh = np.median(s_vel[:b_len]) + 1.5 * np.std(s_vel[:b_len])
    run_q = max(2, int(0.006 * fs))
    qrs_on = _nearest_below_run(s_vel, r_idx, -1, q_thresh, run_q, max(0, r_idx - int(0.12 * fs)))
    qrs_off = _nearest_below_run(s_vel, r_idx, 1, q_thresh, run_q, min(len(rep_xyz) - 1, r_idx + int(0.14 * fs)))
    
    st_start = min(len(rep_xyz) - 1, qrs_off + int(0.04 * fs))
    t_search_end = min(len(rep_xyz) - 1, r_idx + int(0.48 * fs))
    t_peak = st_start + np.argmax(mag[st_start: t_search_end + 1]) if t_search_end > st_start else st_start
    t_thresh = np.median(mag[:b_len]) + 0.15 * max(0.0, mag[t_peak] - np.median(mag[:b_len]))
    run_t = max(2, int(0.01 * fs))
    t_on = _nearest_below_run(mag, t_peak, -1, t_thresh, run_t, qrs_off)
    t_off = _nearest_below_run(mag, t_peak, 1, t_thresh, run_t, min(len(rep_xyz) - 1, r_idx + int(0.5 * fs)))
    
    return {"qrs_on": qrs_on, "qrs_off": qrs_off, "t_on": t_on, "t_off": t_off, "r_idx": r_idx}

def project_loop_pca(loop_3d):
    pca = PCA(n_components=2)
    loop_2d = pca.fit_transform(loop_3d)
    return loop_2d, pca

def compute_arc(loop_2d):
    return float(np.linalg.norm(np.diff(loop_2d, axis=0), axis=1).sum())

def compute_area(loop_2d):
    u = np.append(loop_2d[:, 0], loop_2d[0, 0])
    v = np.append(loop_2d[:, 1], loop_2d[0, 1])
    return float(0.5 * abs(np.dot(u[:-1], v[1:]) - np.dot(u[1:], v[:-1])))

def extract_vcg_features(qrs_3d, t_3d, fs):
    qrs_2d, _ = project_loop_pca(qrs_3d)
    t_2d, _ = project_loop_pca(t_3d)
    dt = 1.0 / fs
    q_v = np.linalg.norm(np.gradient(qrs_3d, dt, axis=0), axis=1)
    t_v = np.linalg.norm(np.gradient(t_3d, dt, axis=0), axis=1)
    c = np.mean(qrs_3d, axis=0)
    return {
        "QRS_Arc": compute_arc(qrs_2d),
        "QRS_Area": compute_area(qrs_2d),
        "QRS_MaxVec": float(np.max(compute_vector_magnitude(qrs_3d))),
        "QRS_MaxCentroidDistance": float(np.max(np.linalg.norm(qrs_3d - c, axis=1))),
        "QRS_MaxVel": float(np.max(q_v)),
        "QRS_MeanVel": float(np.mean(q_v)),
        "QRS_StdVel": float(np.std(q_v)),
        "T_MaxVec": float(np.max(compute_vector_magnitude(t_3d))),
        "T_Area": compute_area(t_2d),
        "T_MaxVel": float(np.max(t_v)),
        "T_MeanVel": float(np.mean(t_v)),
        "T_StdVel": float(np.std(t_v)),
    }

print("Đang xử lý dữ liệu của Bệnh nhân MI (patient001) và Người khỏe mạnh HC (patient104)...")
d_mi = load_ptb_record(rec_mi_path, max_seconds=15)
d_hc = load_ptb_record(rec_hc_path, max_seconds=15)

f_mi, _ = preprocess_xyz(d_mi["xyz"])
f_hc, _ = preprocess_xyz(d_hc["xyz"])

rp_mi, bp_mi = detect_r_peaks(d_mi["r_signal"], d_mi["fs"])
rp_hc, bp_hc = detect_r_peaks(d_hc["r_signal"], d_hc["fs"])

beats_mi, _, r_idx_mi = extract_beats(f_mi, rp_mi, d_mi["fs"])
beats_hc, _, r_idx_hc = extract_beats(f_hc, rp_hc, d_hc["fs"])

rep_mi = build_representative_beat(beats_mi, d_mi["fs"])
rep_hc = build_representative_beat(beats_hc, d_hc["fs"])

bnd_mi = delineate_qrs_t(rep_mi, d_mi["fs"], r_idx_mi)
bnd_hc = delineate_qrs_t(rep_hc, d_hc["fs"], r_idx_hc)

qrs_mi = rep_mi[bnd_mi["qrs_on"]:bnd_mi["qrs_off"]+1]
t_mi = rep_mi[bnd_mi["t_on"]:bnd_mi["t_off"]+1]
qrs_hc = rep_hc[bnd_hc["qrs_on"]:bnd_hc["qrs_off"]+1]
t_hc = rep_hc[bnd_hc["t_on"]:bnd_hc["t_off"]+1]

feat_mi = extract_vcg_features(qrs_mi, t_mi, d_mi["fs"])
feat_hc = extract_vcg_features(qrs_hc, t_hc, d_hc["fs"])

print("Xử lý hoàn tất!")
print("MI features:", feat_mi)
print("HC features:", feat_hc)
