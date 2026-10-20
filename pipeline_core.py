def get_xyz_channels(channel_names):
    """Input: danh sách channel. Output: mapping trục X/Y/Z sang tên channel thật."""
    aliases = {"X": {"x", "vx", "frankx", "frank_x"},
               "Y": {"y", "vy", "franky", "frank_y"},
               "Z": {"z", "vz", "frankz", "frank_z"}}
    normalized = {name: re.sub(r"[^a-z0-9_]", "", str(name).lower()) for name in channel_names}
    result = {}
    for axis, accepted in aliases.items():
        hits = [original for original, norm in normalized.items() if norm in accepted]
        if len(hits) != 1:
            raise ValueError(f"Không xác định duy nhất {axis}; hits={hits}, channels={channel_names}")
        result[axis] = hits[0]
    return result


def choose_detection_channel(channel_names):
    """Input: danh sách channel. Output: tên ECG lead dùng để phát hiện R peak."""
    lower_to_original = {str(name).lower(): name for name in channel_names}
    for preferred in ["ii", "i", "v2"]:
        if preferred in lower_to_original:
            return lower_to_original[preferred]
    xyz_names = set(get_xyz_channels(channel_names).values())
    candidates = [name for name in channel_names if name not in xyz_names]
    if not candidates:
        raise ValueError("Không có ECG lead để R-peak detection")
    return candidates[0]


def load_ptb_record(record_path, max_seconds=None):
    """Input: record path và giới hạn giây. Output: XYZ, ECG detection lead, fs và metadata kênh."""
    header = wfdb.rdheader(str(record_path))
    xyz_names = get_xyz_channels(header.sig_name)
    r_name = choose_detection_channel(header.sig_name)
    selected = [xyz_names[axis] for axis in "XYZ"] + [r_name]
    sampto = None if max_seconds is None else min(header.sig_len, int(max_seconds * header.fs))
    record = wfdb.rdrecord(str(record_path), channel_names=selected, sampto=sampto)
    columns = {name: record.p_signal[:, i].astype(float) for i, name in enumerate(record.sig_name)}
    xyz = np.column_stack([columns[xyz_names[axis]] for axis in "XYZ"])
    return {"xyz": xyz, "r_signal": columns[r_name], "r_channel": r_name,
            "fs": float(record.fs), "xyz_names": xyz_names, "units": record.units}


demo_raw = load_ptb_record(demo_mi["record_path"], max_seconds=ANALYSIS_SECONDS)
seconds_to_plot = min(8, len(demo_raw["xyz"]) / demo_raw["fs"])
n_plot = int(seconds_to_plot * demo_raw["fs"])
t = np.arange(n_plot) / demo_raw["fs"]

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for axis_i, (axis_name, ax) in enumerate(zip("XYZ", axes)):
    ax.plot(t, demo_raw["xyz"][:n_plot, axis_i], linewidth=0.8)
    ax.set_ylabel(axis_name)
    ax.grid(alpha=0.2)
axes[-1].set_xlabel("Thời gian (giây)")
fig.suptitle(f"Frank X/Y/Z gốc — MI demo: {demo_mi['record_id']}")
plt.tight_layout()
plt.show()

# ---------------------------
def safe_savgol_window(n_samples, desired_window, polyorder):
    """Input: chiều dài, window mong muốn, polyorder. Output: window lẻ hợp lệ."""
    if n_samples <= polyorder + 1:
        raise ValueError("Tín hiệu quá ngắn cho Savitzky–Golay")
    window = min(int(desired_window), n_samples if n_samples % 2 == 1 else n_samples - 1)
    if window % 2 == 0:
        window -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        window = minimum
    if window > n_samples:
        raise ValueError("Không tạo được Savitzky–Golay window hợp lệ")
    return window


def preprocess_xyz(xyz, window_samples=1201, polyorder=2):
    """Input: mảng (n,3). Output: XYZ đã Savitzky-Golay và window thực dùng."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz phải có shape (n_samples, 3), nhận {xyz.shape}")
    window = safe_savgol_window(len(xyz), window_samples, polyorder)
    filtered = np.column_stack([
        savgol_filter(xyz[:, i], window_length=window, polyorder=polyorder, mode="interp")
        for i in range(3)
    ])
    return filtered, window


demo_filtered, used_window = preprocess_xyz(demo_raw["xyz"], SG_WINDOW_SAMPLES, SG_POLYORDER)
print("Savitzky–Golay window thực dùng:", used_window, "samples")
start = int(2 * demo_raw["fs"])
stop = min(len(demo_filtered), start + int(2 * demo_raw["fs"]))
t_small = np.arange(start, stop) / demo_raw["fs"]

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for i, (name, ax) in enumerate(zip("XYZ", axes)):
    ax.plot(t_small, demo_raw["xyz"][start:stop, i], alpha=0.45, label="raw")
    ax.plot(t_small, demo_filtered[start:stop, i], linewidth=1.5, label="filtered")
    ax.set_ylabel(name)
    ax.legend()
axes[-1].set_xlabel("Thời gian (giây)")
fig.suptitle("Raw vs Savitzky–Golay filtered")
plt.tight_layout()
plt.show()

# ---------------------------
def detect_r_peaks(ecg_signal, fs):
    """Input: ECG 1D và fs. Output: index R peaks và tín hiệu band-pass dùng cho QC."""
    ecg = np.asarray(ecg_signal, dtype=float)
    if len(ecg) < int(2 * fs):
        raise ValueError("Tín hiệu quá ngắn để phát hiện R peak")
    nyquist = fs / 2
    high = min(20.0, 0.95 * nyquist)
    low = min(5.0, 0.5 * high)
    sos = butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    bandpassed = sosfiltfilt(sos, np.nan_to_num(ecg, nan=np.nanmedian(ecg)))
    envelope = np.abs(bandpassed)
    threshold = np.median(envelope) + 2.5 * np.median(np.abs(envelope - np.median(envelope)))
    peaks, _ = find_peaks(envelope, distance=max(1, int(0.30 * fs)), prominence=max(threshold, 1e-12))

    radius = max(1, int(0.08 * fs))
    baseline = np.median(ecg)
    refined = []
    for peak in peaks:
        lo, hi = max(0, peak - radius), min(len(ecg), peak + radius + 1)
        refined.append(lo + int(np.argmax(np.abs(ecg[lo:hi] - baseline))))
    refined = np.array(sorted(set(refined)), dtype=int)
    if len(refined) < 3:
        raise ValueError(f"Chỉ tìm thấy {len(refined)} R peaks")
    return refined, bandpassed


demo_r_peaks, demo_r_bandpassed = detect_r_peaks(demo_raw["r_signal"], demo_raw["fs"])
print(f"Tìm thấy {len(demo_r_peaks)} R peaks trên {demo_raw['r_channel']}")

n_qc = min(len(demo_raw["r_signal"]), int(10 * demo_raw["fs"]))
t_qc = np.arange(n_qc) / demo_raw["fs"]
visible_peaks = demo_r_peaks[demo_r_peaks < n_qc]
plt.figure(figsize=(13, 4))
plt.plot(t_qc, demo_raw["r_signal"][:n_qc], linewidth=0.8, label=demo_raw["r_channel"])
plt.scatter(visible_peaks / demo_raw["fs"], demo_raw["r_signal"][visible_peaks], color="red", zorder=3)
for number, peak in enumerate(visible_peaks, 1):
    plt.annotate(f"R{number}", (peak / demo_raw["fs"], demo_raw["r_signal"][peak]), fontsize=8)
plt.xlabel("Thời gian (giây)")
plt.title("QC R-peak detection")
plt.legend()
plt.show()

# ---------------------------
def extract_beats(xyz, r_peaks, fs, pre_r_ms=300, post_r_ms=500):
    """Input: XYZ, R peaks, fs và cửa sổ ms. Output: beats, peaks được giữ và R index."""
    pre = int(round(pre_r_ms * fs / 1000))
    post = int(round(post_r_ms * fs / 1000))
    beats, kept_peaks = [], []
    for peak in np.asarray(r_peaks, dtype=int):
        start, stop = peak - pre, peak + post + 1
        if start >= 0 and stop <= len(xyz):
            beats.append(xyz[start:stop])
            kept_peaks.append(peak)
    if not beats:
        raise ValueError("Không có beat hợp lệ sau khi loại biên")
    return np.stack(beats), np.asarray(kept_peaks), pre


demo_beats, demo_kept_peaks, demo_r_index = extract_beats(
    demo_filtered, demo_r_peaks, demo_raw["fs"], PRE_R_MS, POST_R_MS
)
print("beats shape:", demo_beats.shape, "= (số beat, số thời điểm, X/Y/Z)")
beat_time_ms = (np.arange(demo_beats.shape[1]) - demo_r_index) / demo_raw["fs"] * 1000

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for axis_i, ax in enumerate(axes):
    for beat in demo_beats:
        ax.plot(beat_time_ms, beat[:, axis_i], alpha=0.20, linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("XYZ"[axis_i])
axes[-1].set_xlabel("Thời gian so với R (ms)")
fig.suptitle("Các beat chồng lên nhau")
plt.tight_layout()
plt.show()

# ---------------------------
def build_representative_beat(beats, fs, baseline_ms=100):
    """Input: tensor beats và fs. Output: median beat đã căn baseline."""
    representative = np.median(np.asarray(beats), axis=0)
    baseline_n = max(1, min(len(representative), int(baseline_ms * fs / 1000)))
    representative = representative - np.median(representative[:baseline_n], axis=0)
    return representative


demo_rep = build_representative_beat(demo_beats, demo_raw["fs"])
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for axis_i, ax in enumerate(axes):
    for beat in demo_beats:
        ax.plot(beat_time_ms, beat[:, axis_i], color="gray", alpha=0.12, linewidth=0.6)
    ax.plot(beat_time_ms, demo_rep[:, axis_i], color="crimson", linewidth=2, label="representative")
    ax.set_ylabel("XYZ"[axis_i])
    ax.legend()
axes[-1].set_xlabel("Thời gian so với R (ms)")
fig.suptitle("Tất cả beat (mờ) và representative beat (đậm)")
plt.tight_layout()
plt.show()

# ---------------------------
def compute_vector_magnitude(xyz):
    """Input: XYZ (n,3). Output: M(t), norm Euclid tại từng thời điểm."""
    return np.linalg.norm(np.asarray(xyz), axis=1)


demo_magnitude = compute_vector_magnitude(demo_rep)
plt.figure(figsize=(11, 4))
plt.plot(beat_time_ms, demo_magnitude)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Thời gian so với R (ms)")
plt.ylabel("M(t)")
plt.title("Độ lớn tổng hợp của vector VCG")
plt.show()

# ---------------------------
def _nearest_below_run(values, start, direction, threshold, run_length, limit):
    """Input: trace, điểm bắt đầu, hướng và ngưỡng. Output: index biên có đoạn dưới ngưỡng."""
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
    """Input: representative XYZ, fs, R index. Output: bốn boundary và trace QC."""
    # TODO: validate delineation method against reference paper/code
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
    if t_search_hi <= t_search_lo + 2:
        raise ValueError("Không đủ vùng tìm T sau QRS")
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
    if not (qrs_on < qrs_off < t_on < t_off):
        raise ValueError(f"Boundary không hợp lệ: {qrs_on, qrs_off, t_on, t_off}")
    return {"qrs_on": qrs_on, "qrs_off": qrs_off, "t_on": t_on, "t_off": t_off,
            "activity": activity, "magnitude": magnitude}


demo_bounds = delineate_qrs_t(demo_rep, demo_raw["fs"], demo_r_index)
plt.figure(figsize=(12, 4))
plt.plot(beat_time_ms, demo_bounds["magnitude"], label="M(t)")
colors = {"qrs_on": "green", "qrs_off": "green", "t_on": "purple", "t_off": "purple"}
for name in ["qrs_on", "qrs_off", "t_on", "t_off"]:
    idx = demo_bounds[name]
    plt.axvline(beat_time_ms[idx], color=colors[name], linestyle="--", label=name)
plt.xlabel("Thời gian so với R (ms)")
plt.title("QC bằng mắt: ranh giới QRS và T trên representative beat")
plt.legend(ncol=4)
plt.show()
print({key: f"{beat_time_ms[value]:.1f} ms" for key, value in demo_bounds.items() if key in {"qrs_on", "qrs_off", "t_on", "t_off"}})

# ---------------------------
def build_qrs_loop(representative_xyz, boundaries):
    """Input: representative XYZ và boundary. Output: QRS loop có shape (n_qrs,3)."""
    return np.asarray(representative_xyz)[boundaries["qrs_on"]:boundaries["qrs_off"] + 1]


def build_t_loop(representative_xyz, boundaries):
    """Input: representative XYZ và boundary. Output: T loop có shape (n_t,3)."""
    return np.asarray(representative_xyz)[boundaries["t_on"]:boundaries["t_off"] + 1]


demo_qrs_loop = build_qrs_loop(demo_rep, demo_bounds)
demo_t_loop = build_t_loop(demo_rep, demo_bounds)
print("QRS loop shape:", demo_qrs_loop.shape)
print("T loop shape:", demo_t_loop.shape)

for name, loop, color in [("QRS", demo_qrs_loop, "crimson"), ("T", demo_t_loop, "purple")]:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color)
    ax.scatter(loop[0, 0], loop[0, 1], loop[0, 2], color="green", label="start")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"{name} loop 3D")
    ax.legend()
    plt.show()

# ---------------------------
def project_loop_pca(loop_3d):
    """Input: loop 3D. Output: loop 2D, PCA model và explained-variance ratio."""
    loop = np.asarray(loop_3d, dtype=float)
    if loop.ndim != 2 or loop.shape[1] != 3 or len(loop) < 3:
        raise ValueError(f"Loop không hợp lệ: {loop.shape}")
    model = PCA(n_components=2)
    projected = model.fit_transform(loop)
    return projected, model, model.explained_variance_ratio_


demo_qrs_2d, demo_qrs_pca, qrs_ratio = project_loop_pca(demo_qrs_loop)
demo_t_2d, demo_t_pca, t_ratio = project_loop_pca(demo_t_loop)
for name, points, ratio, color in [
    ("QRS", demo_qrs_2d, qrs_ratio, "crimson"), ("T", demo_t_2d, t_ratio, "purple")
]:
    plt.figure(figsize=(6, 6))
    plt.plot(points[:, 0], points[:, 1], color=color)
    plt.scatter(points[0, 0], points[0, 1], color="green", label="start")
    plt.axis("equal")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"{name} loop PCA 2D — variance={ratio.sum():.3f}")
    plt.legend(); plt.show()

# ---------------------------
def compute_arc(loop_2d):
    """Input: loop 2D có thứ tự thời gian. Output: tổng độ dài các đoạn liên tiếp."""
    return float(np.linalg.norm(np.diff(np.asarray(loop_2d), axis=0), axis=1).sum())


def compute_area(loop_2d):
    """Input: loop 2D. Output: diện tích polygon theo shoelace formula."""
    points = np.asarray(loop_2d, dtype=float)
    u, v = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))


def compute_velocity_features(loop_3d, fs):
    """Input: loop 3D và fs. Output: max, mean, std của speed."""
    velocity = np.gradient(np.asarray(loop_3d, dtype=float), 1 / fs, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return {"max": float(np.max(speed)), "mean": float(np.mean(speed)), "std": float(np.std(speed, ddof=0))}


def extract_vcg_features(qrs_loop, t_loop, fs):
    """Input: QRS loop, T loop, fs. Output: dictionary đúng 12 VCG features."""
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
    assert list(features) == FEATURE_NAMES
    return features


demo_features = extract_vcg_features(demo_qrs_loop, demo_t_loop, demo_raw["fs"])
display(pd.DataFrame([demo_features]))
