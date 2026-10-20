"""Shared VCG pipeline for notebooks 04/05; unchanged detection/delineation/features."""
from pathlib import Path
from dataclasses import asdict, replace
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d, percentile_filter, binary_closing
from sklearn.decomposition import PCA
import wfdb
from vcg_preprocessing import Config, preprocess_xyz as _preprocess_xyz, assess_input_quality

CFG=Config()
FEATURE_NAMES=['QRS_Arc','QRS_Area','QRS_MaxVec','QRS_MaxCentroidDistance','QRS_MaxVel','QRS_MeanVel','QRS_StdVel','T_MaxVec','T_Area','T_MaxVel','T_MeanVel','T_StdVel']

def preprocess_xyz(xyz,fs,cfg=CFG,return_diagnostics=False):
    return _preprocess_xyz(xyz,fs,cfg,return_diagnostics=return_diagnostics,preliminary_detector=detect_qrs_anchors)

def ms_samples(ms, fs):
    return max(1, int(round(ms * fs / 1000.0)))


def odd_window(ms, fs, n, poly=3):
    w = ms_samples(ms, fs) | 1
    w = max(w, (poly + 2) | 1)
    w = min(w, n if n % 2 else n - 1)
    if w <= poly:
        raise ValueError('Tín hiệu quá ngắn để làm mượt/đạo hàm.')
    return w


def robust_sigma(a):
    a = np.asarray(a, float)
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def load_xyz(record_path, seconds=30.0):
    """Chỉ đọc Frank XYZ. Không đọc ECG II hay bất kỳ lead ECG chuẩn nào."""
    record_path = Path(record_path)
    header = wfdb.rdheader(str(record_path))
    aliases = ({'x', 'vx', 'frankx'}, {'y', 'vy', 'franky'}, {'z', 'vz', 'frankz'})
    norm = [re.sub('[^a-z0-9]', '', str(n).lower()) for n in header.sig_name]
    channels = []
    for accepted in aliases:
        hits = [i for i, name in enumerate(norm) if name in accepted]
        if len(hits) != 1:
            raise ValueError(f'Không xác định duy nhất Frank XYZ: {header.sig_name}')
        channels.append(hits[0])
    stop = header.sig_len if seconds is None else min(header.sig_len, int(seconds * header.fs))
    rec = wfdb.rdrecord(str(record_path), channels=channels, sampto=stop)
    if any(str(u).lower() != 'mv' for u in rec.units):
        raise ValueError(f'Đơn vị cần là mV, nhận {rec.units}; hãy đổi đơn vị trước.')
    xyz = np.asarray(rec.p_signal, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('XYZ phải là (n,3), hữu hạn; không tự thay NaN bằng số 0.')
    if len(xyz) < 5 * header.fs:
        raise ValueError('Cần ít nhất 5 giây XYZ.')
    diagnosis = next((c.split(':', 1)[1].strip() for c in header.comments
                      if c.lower().startswith('reason for admission:')), 'UNKNOWN')
    return {'xyz': xyz, 'fs': float(header.fs), 'header': header,
            'diagnosis': diagnosis, 'channels': rec.sig_name, 'path': record_path}


def detect_qrs_anchors(morph, detection, fs, cfg=CFG):
    """Mốc QRS từ năng lượng XYZ. Mốc không được gọi là R của ECG II."""
    envelope = np.sqrt(np.maximum(0, uniform_filter1d(
        np.sum(detection ** 2, axis=1), size=ms_samples(cfg.energy_window_ms, fs))))
    if np.max(envelope) < 1e-6:
        raise ValueError('Không có hoạt động QRS đủ lớn trong XYZ.')
    # Ước lượng ngưỡng theo thời gian trên envelope ~100 Hz để giảm chi phí.
    stride = max(1, int(fs / 100))
    small = envelope[::stride]
    size = max(3, int(2.0 * fs / stride) | 1)
    floor = percentile_filter(small, percentile=20, size=size, mode='nearest')
    upper = percentile_filter(small, percentile=95, size=size, mode='nearest')
    adaptive = floor + 0.25 * (upper - floor)
    threshold = np.interp(np.arange(len(envelope)), np.arange(len(small)) * stride, adaptive)
    threshold = np.maximum(threshold, 0.03 * np.quantile(envelope, 0.95))
    candidates, _ = find_peaks(envelope, height=threshold,
                               distance=ms_samples(cfg.refractory_ms, fs),
                               width=ms_samples(15, fs))
    # Tinh chỉnh trong +/-60 ms: đỉnh magnitude sau trừ baseline cục bộ.
    refined, scores = [], []
    for p in candidates:
        a, b = max(0, p - ms_samples(60, fs)), min(len(morph), p + ms_samples(60, fs) + 1)
        ba, bb = max(0, p - ms_samples(180, fs)), max(1, p - ms_samples(100, fs))
        baseline = np.median(morph[ba:bb], axis=0) if bb > ba else np.median(morph[a:b], axis=0)
        local_mag = np.linalg.norm(morph[a:b] - baseline, axis=1)
        refined.append(a + int(np.argmax(local_mag)))
        scores.append(float(envelope[p]))
    # Việc tinh chỉnh có thể kéo 2 candidate lại gần nhau; áp dụng refractory lại.
    chosen = []
    for j in np.argsort(scores)[::-1]:
        p = refined[j]
        if all(abs(p - q) >= ms_samples(cfg.refractory_ms, fs) for q in chosen):
            chosen.append(p)
    anchors = np.asarray(sorted(chosen), dtype=int)
    if len(anchors) < 4:
        raise ValueError(f'Chỉ có {len(anchors)} mốc QRS; cần ít nhất 4 để tạo median beat.')
    rr = np.diff(anchors) / fs
    flags = []
    if np.any((rr < 0.30) | (rr > 2.0)):
        flags.append('rr_extreme_or_missed_beat')
    if robust_sigma(rr) / max(np.median(rr), 1e-9) > 0.15:
        flags.append('rr_irregular')
    if np.any(rr > 1.7 * np.median(rr)):
        flags.append('possible_missed_beat')
    return {'anchors': anchors, 'envelope': envelope, 'threshold': threshold,
            'rr': rr, 'flags': flags}


def spatial_speed(xyz, fs, cfg=CFG):
    """Đạo hàm SG ngắn 11 ms; cùng định nghĩa dùng cho QC và velocity features."""
    derivative = savgol_filter(xyz, odd_window(cfg.derivative_window_ms, fs, len(xyz)),
                               polyorder=3, deriv=1, delta=1.0 / fs, axis=0)
    return np.linalg.norm(derivative, axis=1)


def flattest_pq(xyz, fs, anchor, left_ms=-220, right_ms=-75, width_ms=25):
    lo = max(0, anchor + int(left_ms * fs / 1000))
    hi = min(len(xyz), anchor + int(right_ms * fs / 1000))
    width = ms_samples(width_ms, fs)
    if hi - lo < width:
        raise ValueError('Không đủ vùng PQ để tìm baseline.')
    speed = spatial_speed(xyz, fs)
    starts = np.arange(lo, hi - width + 1)
    cs = np.r_[0.0, np.cumsum(speed)]
    best = int(starts[np.argmin((cs[starts + width] - cs[starts]) / width)])
    return (best, best + width), np.median(xyz[best:best + width], axis=0)


def representative_beat(morph, anchors, fs, cfg=CFG, quality=None):
    """Keep clean morphology variants; only demonstrable data faults reject beats."""
    if quality is None:
        quality = assess_input_quality(morph, fs, cfg)
    rr = np.diff(anchors) / fs
    if len(rr)<2:
        raise ValueError('Không đủ mốc QRS để tạo representative beat.')
    typical_rr = float(np.median(rr))
    pre_s, post_s = min(0.28,0.40*typical_rr), min(0.65,0.78*typical_rr)
    pre, post = int(round(pre_s*fs)), int(round(post_s*fs))
    beats, kept, rows = [], [], []
    for i,p in enumerate(anchors):
        p=int(p);reason=''
        lo,hi=p-pre,p+post+1
        if i==0 or i==len(anchors)-1 or lo<0 or hi>len(morph):reason='edge'
        elif not (0.70*typical_rr <= (p-anchors[i-1])/fs <= 1.35*typical_rr
                  and 0.70*typical_rr <= (anchors[i+1]-p)/fs <= 1.35*typical_rr):reason='rr_outlier'
        elif (p-anchors[i-1])/fs < pre_s+0.10 or (anchors[i+1]-p)/fs < post_s+0.08:reason='adjacent_qrs_in_window'
        if not reason:
            qslice=slice(max(lo,p-ms_samples(80,fs)),min(hi,p+ms_samples(80,fs)))
            if quality['invalid_mask'][lo:hi].any():reason='invalid_data'
            elif quality['saturation_mask'][lo:hi].any():reason='near_adc_rail'
            elif quality['dropout_mask'][qslice].mean()>.5:reason='flat_data_over_qrs'
            else:
                impulses=np.flatnonzero(quality['spike_mask'][lo:hi])+lo
                if np.any(np.abs(impulses-p)>ms_samples(40,fs)):reason='isolated_impulse_away_from_qrs'
        if reason:
            rows.append({'sample':p,'reason':reason,'included':False,'correlation':np.nan,
                         'morphology_status':'not_assessed','pq_warning':''})
            continue
        beat=morph[lo:hi].copy()
        pq,base=flattest_pq(beat,fs,pre,left_ms=-min(220,pre_s*1000-5))
        normalized=beat-base
        velocity=spatial_speed(beat,fs,cfg)
        q0,q1=max(0,pre-ms_samples(120,fs)),min(len(beat),pre+ms_samples(160,fs))
        qamp=max(float(np.linalg.norm(normalized[q0:q1],axis=1).max()),1e-12)
        qspeed=max(float(velocity[q0:q1].max()),1e-12)
        pq_speed=float(velocity[pq[0]:pq[1]].mean())
        pq_radius=float(np.quantile(np.linalg.norm(normalized[pq[0]:pq[1]],axis=1),.95))
        warnings=[]
        if pq_speed>cfg.baseline_speed_fraction*qspeed or pq_radius>cfg.baseline_radius_fraction*qamp:
            warnings.append('pq_not_quiet_review')
        if quality['spike_mask'][max(lo,p-ms_samples(40,fs)):min(hi,p+ms_samples(40,fs))].any():
            warnings.append('sharp_qrs_or_impulse_review')
        if quality.get('motion_suspect_mask',np.zeros(len(morph),bool))[lo:hi].any():
            warnings.append('motion_or_ST_review')
        beats.append(normalized);kept.append(p)
        rows.append({'sample':p,'reason':'used','included':True,'correlation':np.nan,
                     'morphology_status':'pending','pq_on':pq[0],'pq_off':pq[1],
                     'pq_mean_speed_mV_s':pq_speed,'pq_radius95_mV':pq_radius,
                     'pq_warning':'; '.join(warnings)})
    if len(beats)<3:raise ValueError('Không đủ 3 beat sau kiểm tra biên/RR/lỗi dữ liệu.')
    beats=np.stack(beats);first=np.median(beats,axis=0)
    q0,q1=max(0,pre-ms_samples(120,fs)),min(len(first),pre+ms_samples(160,fs))
    template=first[q0:q1].ravel();template-=template.mean()
    correlations=[]
    for beat in beats:
        vector=beat[q0:q1].ravel();vector=vector-vector.mean()
        correlations.append(float(np.dot(vector,template)/max(np.linalg.norm(vector)*np.linalg.norm(template),1e-12)))
    flags=[]
    for row,corr in zip([r for r in rows if r['included']],correlations):
        row['correlation']=corr
        row['morphology_status']='different_morphology_retained' if corr<cfg.min_template_correlation else 'similar'
        if corr<cfg.min_template_correlation:flags.append('different_morphology_retained_review')
        if row['pq_warning']:flags.extend(row['pq_warning'].split('; '))
    rep=np.median(beats,axis=0)
    pq,base=flattest_pq(rep,fs,pre,left_ms=-min(220,pre_s*1000-5));rep-=base
    return {'xyz':rep,'beats':beats,'anchor':pre,'rr_seconds':typical_rr,
            'time_ms':(np.arange(len(rep))-pre)*1000/fs,'pq':pq,
            'beat_qc':pd.DataFrame(rows),'n_used':len(beats),'n_excluded':len(rows)-len(beats),
            'quality_flags':list(dict.fromkeys(flags)),'used_samples':np.asarray(kept)}


def quiet_boundary(values, start, limit, direction, threshold, run):
    """None khi không có đoạn yên; không trả biên cửa sổ như thể tìm được sóng."""
    count = 0
    for i in range(start, limit + direction, direction):
        count = count + 1 if values[i] <= threshold else 0
        if count >= run:
            return i + run - 1 if direction < 0 else i - run + 1
    return None


def delineate_loops(rep, fs, cfg=CFG):
    xyz, anchor, t, pq = rep['xyz'], rep['anchor'], rep['time_ms'], rep['pq']
    n = len(xyz)
    speed = spatial_speed(xyz, fs, cfg)
    speed_q = uniform_filter1d(speed, size=ms_samples(5, fs))
    mag = np.linalg.norm(xyz, axis=1)
    mag_t = np.maximum(0, savgol_filter(mag, odd_window(21, fs, n), 3))
    flags = []
    qlo = max(0, anchor + int(cfg.qrs_search_ms[0]*fs/1000))
    qhi = min(n-1, anchor + int(cfg.qrs_search_ms[1]*fs/1000))
    core_lo, core_hi = max(qlo, anchor-ms_samples(80, fs)), min(qhi, anchor+ms_samples(80, fs))
    qpeak = core_lo + int(np.argmax(speed_q[core_lo:core_hi+1]))
    qbase = float(np.median(speed_q[pq[0]:pq[1]]))
    qnoise = robust_sigma(speed_q[pq[0]:pq[1]])
    qthreshold = max(qbase + 4*qnoise, cfg.qrs_fraction * float(speed_q[qpeak]))
    # Đóng khe <= 15 ms để không cắt giữa các sườn Q/R/S của cùng một cụm.
    active = speed_q[qlo:qhi+1] > qthreshold
    active = binary_closing(active, structure=np.ones(ms_samples(15, fs), bool))
    left = right = qpeak-qlo
    while left > 0 and active[left-1]: left -= 1
    while right < len(active)-1 and active[right+1]: right += 1
    q_on = quiet_boundary(speed_q, qlo+left, qlo, -1, qthreshold, ms_samples(cfg.quiet_qrs_ms, fs))
    q_off = quiet_boundary(speed_q, qlo+right, qhi, 1, qthreshold, ms_samples(cfg.quiet_qrs_ms, fs))
    if q_on is None: flags.append('qrs_on_not_found')
    if q_off is None: flags.append('qrs_off_not_found')
    t_on = t_peak = t_off = None
    tthreshold = np.nan
    tlo = thi = None
    t_method = 'unresolved'
    rising = end_slope = None
    dm = None
    st = np.nan
    tangent_off = tangent_on = None
    threshold_on = threshold_off = None
    tp = None
    t_end_baseline = np.nan
    t_end_threshold = np.nan
    if q_on is not None and q_off is not None:
        dur = (q_off-q_on)*1000/fs
        if not (q_on < anchor < q_off) or not 40 <= dur <= 240:
            flags.append('qrs_duration_or_anchor_requires_review')
        if q_on <= pq[1]: flags.append('pq_overlaps_qrs')
        if (q_on-qlo) < ms_samples(8,fs) or (qhi-q_off) < ms_samples(8,fs):
            flags.append('qrs_near_search_limit')
        # T-search kết thúc trước QRS kế tiếp; RR của chính record quyết định giới hạn.
        tlo = q_off + ms_samples(20, fs)
        thi = min(n-1, anchor + int(min(0.60, 0.70*rep['rr_seconds'])*fs))
        if thi-tlo < ms_samples(100, fs):
            flags.append('t_search_too_short')
        else:
            mbase = float(np.median(mag_t[pq[0]:pq[1]]))
            mnoise = max(robust_sigma(mag_t[pq[0]:pq[1]]), 1e-6)
            prominence = max(4*mnoise, 0.015*float(np.max(mag_t[q_on:q_off+1])))
            ps, props = find_peaks(mag_t[tlo:thi+1], prominence=prominence,
                                    distance=ms_samples(40, fs))
            ps = ps + tlo
            valid = (ps >= q_off+ms_samples(50,fs)) & (ps <= thi-ms_samples(30,fs))
            if not valid.any():
                flags.append('t_peak_not_resolved')
            else:
                js = np.flatnonzero(valid)
                best = int(js[np.argmax(props['prominences'][js])])
                t_peak = int(ps[best])
                amp = float(mag_t[t_peak]-mbase)
                tthreshold = mbase + max(3*mnoise, cfg.t_fraction*amp)
                if amp <= 5*mnoise:
                    flags.append('t_low_amplitude')
                # Gộp các lobe rõ trong +/-100 ms; không cắt T hai pha giữa hai đỉnh.
                significant = ps[(np.abs(ps-t_peak) <= ms_samples(100,fs)) &
                                  (props['prominences'] >= 0.35*props['prominences'][best])]
                if len(significant)>1:flags.append('multiple_t_lobes_requires_review')
                first_peak, last_peak = int(significant.min()), int(significant.max())
                threshold_on = quiet_boundary(mag_t, first_peak, tlo, -1, tthreshold, ms_samples(cfg.quiet_t_ms,fs))
                # Baseline cuối T từ đoạn TP phẳng phía sau, tránh dùng PQ khi có drift.
                # Chỉ phục vụ delineation; XYZ/feature vẫn có gốc PQ chung.
                tp_lo, width = last_peak+ms_samples(100,fs), ms_samples(25,fs)
                t_end_baseline = mbase
                if thi-tp_lo >= width:
                    starts = np.arange(tp_lo,thi-width+2)
                    cs = np.r_[0.0,np.cumsum(speed_q)]
                    tp_start = int(starts[np.argmin((cs[starts+width]-cs[starts])/width)])
                    tp = (tp_start,tp_start+width)
                    t_end_baseline = float(np.median(mag_t[tp[0]:tp[1]]))
                    if np.mean(speed_q[tp[0]:tp[1]]) > 0.10*speed_q[qpeak]:
                        flags.append('tp_not_quiet')
                else:
                    flags.append('tp_window_too_short')
                end_noise = mnoise if tp is None else max(robust_sigma(mag_t[tp[0]:tp[1]]),1e-6)
                t_end_threshold = t_end_baseline+max(3*end_noise,cfg.t_fraction*max(mag_t[t_peak]-t_end_baseline,0))
                threshold_off = quiet_boundary(mag_t,last_peak,thi,1,t_end_threshold,ms_samples(cfg.quiet_t_ms,fs))
                t_method = 'tangent_on_and_end'
                # T-end: tiếp tuyến sườn xuống cắt mức baseline TP (PQ nếu thiếu TP).
                dm = np.gradient(mag_t, 1/fs)
                end_slope = last_peak + int(np.argmin(dm[last_peak:thi+1]))
                if dm[end_slope] < -1e-9:
                    candidate = int(round(end_slope + (t_end_baseline-mag_t[end_slope])/dm[end_slope]*fs))
                    if last_peak < candidate < thi:
                        tangent_off = candidate
                # T-on: định nghĩa thao tác bằng tiếp tuyến sườn lên cắt mức ST.
                # Đây không phải quy tắc lâm sàng duy nhất; ST chênh cần review.
                rising = tlo + int(np.argmax(dm[tlo:first_peak+1]))
                st = float(np.median(mag_t[tlo:min(first_peak,tlo+ms_samples(25,fs))]))
                if dm[rising] > 1e-9:
                    candidate = int(round(rising + (st-mag_t[rising])/dm[rising]*fs))
                    if tlo <= candidate < first_peak:
                        tangent_on = candidate
                t_on = tangent_on if tangent_on is not None else threshold_on
                t_off = tangent_off if tangent_off is not None else threshold_off
                t_method = ('tangent_on' if tangent_on is not None else 'threshold_on_candidate') + '+' + ('tangent_end' if tangent_off is not None else 'threshold_end_candidate')
                if tangent_on is None: flags.append('t_on_tangent_unresolved')
                if tangent_off is None: flags.append('t_off_tangent_unresolved')
                if st-mbase > max(5*mnoise, 0.15*amp):
                    flags.append('st_level_requires_review')
                if threshold_off is None:
                    flags.append('t_off_no_baseline_return')
                elif tangent_off is not None and abs(threshold_off-tangent_off)*1000/fs > 100:
                    flags.append('t_end_methods_disagree')
                if t_on is not None and t_off is not None:
                    if not (q_off < t_on < first_peak <= t_peak <= last_peak < t_off < n):
                        flags.append('t_boundaries_invalid')
                        t_on = t_off = None
                    elif not 60 <= (t_off-t_on)*1000/fs <= 400:
                        flags.append('t_duration_requires_review')
                    if t_off is not None and thi-t_off < ms_samples(15,fs): flags.append('t_near_search_limit')
                else:
                    flags.append('t_boundary_not_resolved')
    return {'qrs_on': q_on, 'qrs_off': q_off, 'qrs_speed_peak': qpeak,
            't_on': t_on, 't_peak': t_peak, 't_off': t_off,
            'speed': speed, 'smooth_speed': speed_q, 'magnitude': mag, 't_magnitude': mag_t,
            'q_threshold': qthreshold, 't_threshold': tthreshold,
            't_end_baseline': t_end_baseline, 't_end_threshold': t_end_threshold, 'tp': tp,
            't_rise_index': rising, 't_fall_index': end_slope, 't_slope': dm, 'st_level': st,
            't_search': (tlo, thi), 't_tangent_off': tangent_off, 't_tangent_on': tangent_on,
            't_threshold_on': threshold_on, 't_threshold_off': threshold_off,
            't_method': t_method, 'flags': list(dict.fromkeys(flags)), 'manual': False}


def apply_manual_boundaries(boundaries, rep, times_ms):
    """Sửa bằng mốc đã xem trên XYZ; không âm thầm ép chỉ số vào mảng."""
    out = dict(boundaries)
    t = rep['time_ms']
    names = ('qrs_on', 'qrs_off', 't_on', 't_off')
    if set(times_ms) != set(names):
        raise ValueError(f'Cần đúng bốn mốc {names}.')
    if any(not np.isfinite(v) or not t[0] <= v <= t[-1] for v in times_ms.values()):
        raise ValueError('Mốc manual phải hữu hạn và nằm trong beat.')
    for k in names: out[k] = int(np.argmin(np.abs(t-times_ms[k])))
    a,b,c,d = [out[k] for k in names]
    if not 0 <= a < rep['anchor'] < b < c < d < len(t):
        raise ValueError('Cần QRS_on < anchor < QRS_off < T_on < T_off.')
    out['automatic_flags'] = boundaries['flags']
    out['flags'] = []
    out['manual'] = True
    out['t_method'] = 'manual_visual_review'
    out['t_peak'] = c+int(np.argmax(out['t_magnitude'][c:d+1]))
    return out


def cut_loop(xyz, onset, offset):
    if onset is None or offset is None or not 0 <= onset < offset < len(xyz):
        return None
    return xyz[onset:offset+1]


def project_pca(loop):
    if loop is None or len(loop)<3:
        return None
    pca = PCA(n_components=2)
    xy = pca.fit_transform(loop)
    area = float(abs(np.dot(xy[:,0], np.roll(xy[:,1],-1))-np.dot(xy[:,1],np.roll(xy[:,0],-1)))/2)
    arc = float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
    return {'xy':xy, 'model':pca, 'area':area, 'arc':arc,
            'variance':float(pca.explained_variance_ratio_.sum())}


def extract_features(qrs, tloop, fs, cfg=CFG, speeds=None):
    """Candidate features; chỉ dùng ML nếu QC cho phép hoặc được review thủ công."""
    features = dict.fromkeys(FEATURE_NAMES, np.nan)
    for name, loop in [('QRS',qrs),('T',tloop)]:
        if loop is None or len(loop)<max(5,ms_samples(cfg.derivative_window_ms,fs)):
            continue
        p = project_pca(loop)
        v = spatial_speed(loop,fs,cfg) if speeds is None else speeds[name]
        features.update({name+'_Area':p['area'], name+'_MaxVec':float(np.linalg.norm(loop,axis=1).max()),
                         name+'_MaxVel':float(v.max()), name+'_MeanVel':float(v.mean()),
                         name+'_StdVel':float(v.std())})
        if name=='QRS':
            features['QRS_Arc']=p['arc']
            features['QRS_MaxCentroidDistance']=float(np.linalg.norm(loop-loop.mean(axis=0),axis=1).max())
    return features


def process_record(record_path, cfg=CFG, manual_ms=None):
    raw=load_xyz(record_path,cfg.seconds)
    fs=raw['fs']
    morph, detsig, preprocessing=preprocess_xyz(raw['xyz'],fs,cfg,return_diagnostics=True)
    detector=detect_qrs_anchors(morph,detsig,fs,cfg)
    rep=representative_beat(morph,detector['anchors'],fs,cfg,quality=preprocessing['input_quality'])
    boundaries=delineate_loops(rep,fs,cfg)
    if manual_ms is not None:
        boundaries=apply_manual_boundaries(boundaries,rep,manual_ms)
    qrs=cut_loop(rep['xyz'],boundaries['qrs_on'],boundaries['qrs_off'])
    tloop=cut_loop(rep['xyz'],boundaries['t_on'],boundaries['t_off'])
    flags=list(dict.fromkeys(detector['flags']+boundaries['flags']+rep['quality_flags']))
    if preprocessing['powerline_status']=='inconclusive':flags.append('powerline_inconclusive_review')
    baseline_residual={}
    tp=boundaries.get('tp')
    if tp is not None:
        pq=rep['pq'];pv=np.median(rep['xyz'][pq[0]:pq[1]],axis=0);tv=np.median(rep['xyz'][tp[0]:tp[1]],axis=0)
        baseline_residual={'pq_tp_vector_offset_mV':float(np.linalg.norm(tv-pv)),
                           'note':'Offset PQ–TP có thể là drift/tail T; không phải đo nhiễu chắc chắn.'}
    if rep['n_used'] < 5: flags.append('few_representative_beats')
    if rep['n_excluded']/len(detector['anchors']) > 0.4: flags.append('many_excluded_beats')
    speeds = {name: None if loop is None else boundaries['speed'][boundaries[a]:boundaries[z]+1]
              for name,loop,a,z in [('QRS',qrs,'qrs_on','qrs_off'),('T',tloop,'t_on','t_off')]}
    features=extract_features(qrs,tloop,fs,cfg,speeds=speeds)
    if not np.isfinite(list(features.values())).all(): flags.append('features_incomplete')
    status='manual_reviewed' if manual_ms is not None and not flags else ('auto_candidate' if not flags else 'needs_review')
    return {'raw':raw,'morph':morph,'detection_xyz':detsig,'detector':detector,'rep':rep,
            'boundaries':boundaries,'qrs_loop':qrs,'t_loop':tloop,'features':features,
            'flags':list(dict.fromkeys(flags)),'status':status,
            'eligible_for_features':not flags,'config':asdict(cfg),
            'preprocessing':preprocessing,'baseline_residual':baseline_residual}


def qc_row(result):
    b,rep=result['boundaries'],result['rep']
    row={'record_id':result['raw']['path'].parent.name+'/'+result['raw']['path'].name,
         'diagnosis':result['raw']['diagnosis'],'status':result['status'],
         'eligible_for_features':result['eligible_for_features'],
         'n_qrs_detected':len(result['detector']['anchors']),'n_beats_used':rep['n_used'],
         'median_rr_ms':rep['rr_seconds']*1000,'t_method':b['t_method'],
         'flags':'; '.join(result['flags'])}
    for k in ['qrs_on','qrs_off','t_on','t_peak','t_off']:
        row[k+'_ms']=np.nan if b[k] is None else float(rep['time_ms'][b[k]])
    row['qrs_duration_ms']=row['qrs_off_ms']-row['qrs_on_ms']
    row['t_duration_ms']=row['t_off_ms']-row['t_on_ms']
    for key in ['t_tangent_on','t_tangent_off','t_threshold_on','t_threshold_off']:
        row[key+'_ms']=np.nan if b[key] is None else float(rep['time_ms'][b[key]])
    preprocessing=result.get('preprocessing',{})
    for key in ['notch_applied','notch_mode','notch_channels','powerline_status','powerline_score_db','powerline_persistence','notch_reason','morphology_mode','notch_q']:
        row[key]=preprocessing.get(key,np.nan)
    for name,info in preprocessing.get('powerline',{}).get('channels',{}).items():
        row['powerline_'+name+'_status']=info.get('status')
        row['powerline_'+name+'_score_db']=info.get('score_db',np.nan)
    row.update(result.get('baseline_residual',{}))
    return row


def plot_detection(result, seconds=6):
    raw,fs=result['raw'],result['raw']['fs']
    stop=min(len(raw['xyz']),int(seconds*fs)); tt=np.arange(stop)/fs
    fig,axs=plt.subplots(4,1,figsize=(13,9),sharex=True,layout='constrained')
    for j,ax in enumerate(axs[:3]):
        ax.plot(tt,raw['xyz'][:stop,j],color='0.7',lw=.7,label='XYZ gốc')
        ax.plot(tt,result['morph'][:stop,j],lw=1,label='XYZ dùng cho loops')
        ax.set_ylabel('XYZ'[j]+' (mV)')
        for p in result['detector']['anchors']:
            if p<stop: ax.axvline(p/fs,color='#d94841',alpha=.5,lw=.8)
    d=result['detector']; axs[3].plot(tt,d['envelope'][:stop],label='Envelope năng lượng XYZ')
    axs[3].plot(tt,d['threshold'][:stop],ls='--',label='Ngưỡng thích nghi')
    visible=d['anchors'][d['anchors']<stop]
    axs[3].scatter(visible/fs,d['envelope'][visible],color='#d94841',label='Mốc QRS từ VCG')
    axs[0].legend(loc='upper right'); axs[3].legend(loc='upper right')
    axs[3].set_ylabel('Envelope (mV)'); axs[3].set_xlabel('Thời gian recording (s)')
    fig.suptitle(f"{raw['path'].parent.name}/{raw['path'].name} — detector chỉ dùng XYZ")
    return fig


def plot_delineation(result):
    rep,b=result['rep'],result['boundaries']; t=rep['time_ms']
    fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
    for j,color in enumerate(['#2563eb','#059669','#d97706']):
        axs[0].plot(t,rep['xyz'][:,j],color=color,label='Frank '+'XYZ'[j])
    for ax in axs:
        ax.axvline(0,color='0.4',ls='--',lw=1,label='Mốc QRS VCG' if ax is axs[0] else None)
        for a,z,color,label in [('qrs_on','qrs_off','#10b981','QRS candidate'),('t_on','t_off','#f97316','T candidate')]:
            if b[a] is not None and b[z] is not None:
                ax.axvspan(t[b[a]],t[b[z]],color=color,alpha=.17,label=label if ax is axs[0] else None)
                ax.axvline(t[b[a]],color=color,ls=':'); ax.axvline(t[b[z]],color=color,ls=':')
    axs[0].axvspan(t[rep['pq'][0]],t[rep['pq'][1]-1],color='0.4',alpha=.15,label='PQ dùng đặt baseline')
    if b['tp'] is not None:axs[0].axvspan(t[b['tp'][0]],t[b['tp'][1]-1],color='#6366f1',alpha=.12,label='TP kiểm tra T-end')
    axs[0].set_ylabel('Điện thế (mV)'); axs[0].legend(loc='upper right',ncol=3,fontsize=9)
    axs[0].set_title('1. Median beat: biên tô màu là kết quả ước lượng')
    axs[1].plot(t,b['smooth_speed'],color='#4f46e5',label='Tốc độ XYZ (SG 11 ms + mean 5 ms)')
    axs[1].axhline(b['q_threshold'],color='#dc2626',ls='--',label='Ngưỡng QRS')
    axs[1].set_ylabel('Tốc độ (mV/s)');axs[1].legend();axs[1].set_title('2. Tìm biên QRS từ cụm hoạt động nhanh')
    axs[2].plot(t,b['magnitude'],color='0.65',lw=.8,label='Magnitude')
    axs[2].plot(t,b['t_magnitude'],color='#0891b2',label='Magnitude làm mượt 21 ms')
    if np.isfinite(b['t_threshold']):axs[2].axhline(b['t_threshold'],color='#d97706',ls='--',label='Ngưỡng T')
    if np.isfinite(b['t_end_threshold']):axs[2].axhline(b['t_end_threshold'],color='#7c3aed',ls=':',label='Ngưỡng cuối T theo TP')
    if b['t_peak'] is not None:axs[2].scatter(t[b['t_peak']],b['t_magnitude'][b['t_peak']],color='#ea580c',label='T peak candidate')
    if b['t_threshold_off'] is not None:axs[2].axvline(t[b['t_threshold_off']],ls='--',color='0.4',label='T-end theo ngưỡng: kiểm tra chéo')
    for ix, edge, color, label in [('t_rise_index','t_tangent_on','#ea580c','Tiếp tuyến lên → mức ST'),('t_fall_index','t_tangent_off','#7c3aed','Tiếp tuyến xuống → baseline TP')]:
        if b[ix] is not None and b[edge] is not None:
            k=b[ix]; points=np.asarray([b[edge],k])
            yy=b['t_magnitude'][k]+b['t_slope'][k]*(points-k)/result['raw']['fs']
            axs[2].plot(t[points],yy,color=color,ls='-.',lw=1.5,label=label)
    axs[2].set_ylabel('Magnitude (mV)');axs[2].set_xlabel('Thời gian so với mốc QRS VCG (ms)');axs[2].legend()
    axs[2].set_title('3. T-on/end theo tiếp tuyến; ngưỡng baseline để kiểm tra chéo')
    for ax in axs:ax.grid(alpha=.2)
    flags='; '.join(result['flags']) or 'Không có cờ heuristic; vẫn cần kiểm tra bằng mắt'
    fig.suptitle(f"{result['raw']['path'].name} | {result['status']}\n{flags}",fontsize=11)
    return fig


def plot_loops(result):
    qrs,tloop=result['qrs_loop'],result['t_loop']
    fig=plt.figure(figsize=(15,9),layout='constrained')
    a3=fig.add_subplot(2,2,1,projection='3d');axy=fig.add_subplot(2,2,2)
    for loop,color,label in [(qrs,'#059669','QRS'),(tloop,'#ea580c','T')]:
        if loop is None:continue
        a3.plot(*loop.T,color=color,label=label)
        a3.scatter(*loop[0],color=color,marker='o');a3.scatter(*loop[-1],color=color,marker='x')
        axy.plot(loop[:,0],loop[:,1],color=color,label=label)
        axy.scatter(*loop[0,:2],color=color,marker='o');axy.scatter(*loop[-1,:2],color=color,marker='x')
    a3.set(xlabel='X (mV)',ylabel='Y (mV)',zlabel='Z (mV)',title='XYZ chung: vị trí tương đối QRS/T')
    a3.scatter(0,0,0,color='black',s=20)
    available=[x for x in [qrs,tloop] if x is not None]
    if available:
        allpts=np.vstack(available); low=allpts.min(axis=0);high=allpts.max(axis=0)
        mid=(low+high)/2;radius=max(float((high-low).max())/2,1e-3)*1.1
        a3.set_xlim(mid[0]-radius,mid[0]+radius);a3.set_ylim(mid[1]-radius,mid[1]+radius);a3.set_zlim(mid[2]-radius,mid[2]+radius);a3.set_box_aspect((1,1,1))
    axy.set(xlabel='X (mV)',ylabel='Y (mV)',title='Chiếu Frank X–Y cố định: bỏ Z');axy.set_aspect('equal',adjustable='datalim')
    for ax in [a3,axy]:
        if available:ax.legend()
    for pos,loop,color,label in [(3,qrs,'#059669','QRS'),(4,tloop,'#ea580c','T')]:
        ax=fig.add_subplot(2,2,pos);p=project_pca(loop)
        if p is None:
            ax.text(.5,.5,label+' chưa có biên đáng dùng',ha='center',transform=ax.transAxes)
        else:
            xy=p['xy'];ax.plot(*xy.T,color=color);ax.fill(*xy.T,color=color,alpha=.15)
            ax.scatter(*xy[0],color=color,marker='o');ax.scatter(*xy[-1],color=color,marker='x')
            ax.scatter(0,0,color='black',marker='+',label='Tâm sau PCA')
            ax.set_title(f"{label} PCA riêng | Area={p['area']:.5g} mV²\nArc={p['arc']:.4g} mV; variance={100*p['variance']:.1f}%")
            ax.legend(fontsize=8)
        ax.set(xlabel='PC1 (mV)',ylabel='PC2 (mV)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    fig.suptitle('o = đầu, × = cuối | PCA riêng không dùng để so vị trí QRS/T',fontsize=12)
    return fig
