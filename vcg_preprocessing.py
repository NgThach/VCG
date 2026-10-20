"""Conditional Frank VCG preprocessing. Thresholds are research heuristics.

Raw arrays are never edited. Morphology keeps a PQ reference; detector alone
uses a 5–30 Hz band-pass. No status is a clinical confirmation of interference.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.signal import (firwin, freqz, butter, sosfiltfilt, iirnotch,
                          filtfilt, welch, detrend)
from scipy.ndimage import median_filter, uniform_filter1d


@dataclass(frozen=True)
class Config:
    seconds: float = 30.0
    morphology_mode: str = 'fir'  # fir / legacy / none
    morphology_design_fs_hz: float = 1000.0
    fir_numtaps: int = 401
    fir_cutoff_hz: float = 175.0
    fir_kaiser_beta: float = 8.0
    fir_preserve_hz: float = 150.0
    fir_stopband_hz: tuple = (190.0, 200.0)
    notch_mode: str = 'auto'  # auto / on / off
    notch_hz: float = 50.0
    notch_q: float = 50.0
    morphology_highpass_hz: float | None = None  # optional comparison only
    morphology_highpass_order: int = 2
    legacy_lowpass_hz: float = 80.0
    legacy_lowpass_order: int = 4
    legacy_notch_q: float = 30.0
    psd_window_s: float = 4.0
    psd_hop_s: float = 2.0
    powerline_min_duration_s: float = 8.0
    powerline_band_hz: tuple = (49.5, 50.5)
    powerline_background_hz: tuple = ((45.0, 48.0), (52.0, 55.0))
    powerline_score_threshold_db: float = 8.0
    powerline_persistence_threshold: float = 0.60
    powerline_peak_concentration: float = 0.55
    powerline_inconsistent_persistence: float = 0.20
    quiet_lowpass_hz: float = 15.0  # QC mask only; never the morphology branch
    quiet_activity_smooth_ms: float = 50.0
    quiet_quantile: float = 0.35
    quiet_min_run_s: float = 0.12
    quiet_min_total_s: float = 1.0
    quiet_min_runs: int = 4
    quiet_min_span_s: float = 6.0
    quiet_qrs_exclusion_ms: tuple = (-120.0, 180.0)
    quiet_line_score_db: float = 8.0
    quiet_line_explained_fraction: float = 0.25
    quiet_frequency_step_hz: float = 0.05
    quiet_fit_condition_limit: float = 1e6
    input_flat_min_s: float = 0.30
    input_flat_tolerance_mV: float = 1e-9
    input_spike_mad_factor: float = 20.0
    input_spike_min_mV: float = 0.15
    input_saturation_abs_mV: float = 15.5  # near PTB nominal ADC rail; review only
    input_saturation_min_ms: float = 10.0
    input_motion_center_step_mV: float = 0.5
    input_motion_mad_factor: float = 6.0
    baseline_speed_fraction: float = 0.10
    baseline_radius_fraction: float = 0.05
    detector_band_hz: tuple = (5.0, 30.0)
    refractory_ms: float = 250.0
    energy_window_ms: float = 80.0
    derivative_window_ms: float = 11.0
    qrs_search_ms: tuple = (-180.0, 200.0)
    qrs_fraction: float = 0.07
    quiet_qrs_ms: float = 12.0
    quiet_t_ms: float = 20.0
    t_fraction: float = 0.08
    min_template_correlation: float = 0.85  # flag morphology variation; do not discard alone


def legacy_config(cfg=Config()):
    return replace(cfg, morphology_mode='legacy', notch_mode='on',
                   notch_q=cfg.legacy_notch_q, morphology_highpass_hz=None)


def _validate_xyz(xyz, fs):
    a = np.asarray(xyz, dtype=float)
    if not np.isfinite(fs) or fs < 100:
        raise ValueError('Cần fs hữu hạn >= 100 Hz.')
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2:
        raise ValueError('XYZ phải có shape (n,3), ít nhất 2 mẫu.')
    if not np.isfinite(a).all():
        raise ValueError('XYZ có NaN/Inf; không tự điền giá trị đo giả.')
    return a


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _robust_sigma(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


@lru_cache(maxsize=16)
def _fir_taps(fs, numtaps, cutoff, beta):
    return firwin(numtaps, cutoff, fs=fs, window=('kaiser', beta))


def design_morphology_fir(fs, cfg=Config()):
    if not np.isclose(fs, cfg.morphology_design_fs_hz, rtol=0, atol=1e-8):
        raise ValueError(f'FIR nghiên cứu được thiết kế cho {cfg.morphology_design_fs_hz:g} Hz; '
                         f'nhận fs={fs:g}. Cần cấu hình và kiểm chứng lại, không tự resample/cap cutoff.')
    if cfg.fir_numtaps < 3 or cfg.fir_numtaps % 2 == 0:
        raise ValueError('FIR cần số hệ số lẻ >= 3.')
    if not 0 < cfg.fir_preserve_hz < cfg.fir_cutoff_hz < fs/2:
        raise ValueError('Dải giữ/cutoff FIR không hợp lệ với Nyquist.')
    return _fir_taps(float(fs), cfg.fir_numtaps, cfg.fir_cutoff_hz, cfg.fir_kaiser_beta).copy()


def filter_frequency_response(fs, cfg=Config()):
    b = design_morphology_fir(fs, cfg)
    frequencies = np.unique(np.r_[np.linspace(0, fs/2, 16001),
        cfg.fir_preserve_hz, cfg.fir_cutoff_hz, *cfg.fir_stopband_hz, cfg.notch_hz])
    _, hf = freqz(b, [1.0], worN=frequencies, fs=fs)
    if not 0 < cfg.notch_hz < fs/2 or cfg.notch_q <= 0:
        raise ValueError('Notch center/Q không hợp lệ.')
    bn, an = iirnotch(cfg.notch_hz, cfg.notch_q, fs=fs)
    _, hn = freqz(bn, an, worN=frequencies, fs=fs)
    magnitude = lambda h: np.abs(h)
    db = lambda v: 20*np.log10(np.maximum(v, 1e-15))
    return pd.DataFrame({'frequency_hz':frequencies,
        'fir_one_pass_db':db(magnitude(hf)), 'fir_two_pass_db':db(magnitude(hf)**2),
        'notch_one_pass_db':db(magnitude(hn)), 'notch_two_pass_db':db(magnitude(hn)**2)})


def welch_xyz(xyz, fs, cfg=Config()):
    a = _validate_xyz(xyz, fs)
    size = min(len(a), int(round(cfg.psd_window_s*fs)))
    if size < 8: raise ValueError('Không đủ mẫu cho PSD.')
    f, p = welch(a, fs=fs, nperseg=size, noverlap=size//2,
                 window='hann', detrend='constant', scaling='density', axis=0)
    return f, p


def _band_density(f, p, bounds):
    selected = (f >= bounds[0]) & (f <= bounds[1])
    return np.mean(p[selected], axis=0) if np.any(selected) else np.full(p.shape[1], np.nan)


def _line_psd_metrics(f, p, cfg):
    narrow = (f >= cfg.powerline_band_hz[0]) & (f <= cfg.powerline_band_hz[1])
    bg = np.zeros(len(f), bool)
    for low, high in cfg.powerline_background_hz: bg |= (f >= low) & (f <= high)
    if narrow.sum()<3 or bg.sum()<4:
        return None
    # Both quantities are PSD density (mV²/Hz), not integrals with unequal widths.
    p50 = np.mean(p[narrow], axis=0)
    background = np.median(p[bg], axis=0)
    score = 10*np.log10(np.maximum(p50, 1e-24)/np.maximum(background, 1e-24))
    local = p[narrow]
    peak_ix = np.argmax(local, axis=0)
    peak_freq = f[narrow][peak_ix]
    concentration = np.max(local, axis=0)/np.maximum(local.sum(axis=0), 1e-24)
    return p50, background, score, peak_freq, concentration


def assess_input_quality(xyz, fs, cfg=Config()):
    a = np.asarray(xyz, float)
    if a.ndim != 2 or a.shape[1] != 3: raise ValueError('QC cần XYZ (n,3).')
    invalid = ~np.isfinite(a).all(axis=1)
    result = {'invalid_samples':int(invalid.sum()), 'flags':[], 'invalid_mask':invalid,
              'spike_mask':np.zeros(len(a),bool), 'dropout_mask':np.zeros(len(a),bool),
              'saturation_mask':np.zeros(len(a),bool), 'motion_suspect_mask':np.zeros(len(a),bool)}
    if invalid.any():
        result['flags'].append('invalid_values'); return result
    _validate_xyz(a, fs)
    flat = np.r_[False, np.all(np.abs(np.diff(a,axis=0))<=cfg.input_flat_tolerance_mV,axis=1)]
    for lo,hi in _runs(flat):
        if (hi-lo)/fs >= cfg.input_flat_min_s: result['dropout_mask'][lo:hi] = True
    if result['dropout_mask'].any(): result['flags'].append('all_axes_flat_segment_review')
    diff = np.diff(a,axis=0)
    thresholds = np.maximum(cfg.input_spike_min_mV,
        cfg.input_spike_mad_factor*np.array([_robust_sigma(diff[:,j]) for j in range(3)]))
    # Isolated up/down impulses, not every steep QRS slope.
    opposite = diff[:-1]*diff[1:] < 0
    strong = (np.abs(diff[:-1])>thresholds)&(np.abs(diff[1:])>thresholds)&opposite
    result['spike_mask'][1:-1] = np.any(strong,axis=1)
    if result['spike_mask'].any(): result['flags'].append('isolated_impulse_review')
    rail = np.any(np.abs(a)>=cfg.input_saturation_abs_mV,axis=1)
    for lo,hi in _runs(rail):
        if (hi-lo)*1000/fs >= cfg.input_saturation_min_ms: result['saturation_mask'][lo:hi] = True
    if result['saturation_mask'].any(): result['flags'].append('near_nominal_adc_rail_review')
    f,p = welch_xyz(a,fs,cfg)
    result.update({'psd_frequency_hz':f,'psd_density':p,
        'low_frequency_density':_band_density(f,p,(0.05,0.5)),
        'high_frequency_density':_band_density(f,p,(200,min(450,fs/2))),
        'rms_mV':np.sqrt(np.mean(a*a,axis=0))})
    # Low-frequency content may be drift, ST/T, or movement; flag, never auto-reject.
    blocks = [a[i:i+int(fs)] for i in range(0,len(a)-int(fs)+1,int(fs))]
    centers = np.array([np.median(block,axis=0) for block in blocks])
    result['baseline_center_range_mV'] = np.ptp(centers,axis=0) if len(centers) else np.full(3,np.nan)
    if len(centers)>2:
        jumps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
        limit=max(cfg.input_motion_center_step_mV,float(np.median(jumps)+cfg.input_motion_mad_factor*_robust_sigma(jumps)))
        for block in np.flatnonzero(jumps>limit):
            result['motion_suspect_mask'][int(block*fs):min(len(a),int((block+2)*fs))]=True
        if result['motion_suspect_mask'].any():
            result['flags'].append('low_frequency_transient_motion_or_ST_review')
        result['motion_center_jump_threshold_mV']=limit
    return result


def find_low_activity_intervals(xyz, fs, cfg, qrs_anchors):
    if qrs_anchors is None or len(qrs_anchors)<4:
        return [], {'reason':'Không có đủ mốc QRS sơ bộ để kiểm tra vùng ít hoạt động.'}
    a = _validate_xyz(xyz,fs)
    lf = sosfiltfilt(butter(4,cfg.quiet_lowpass_hz,fs=fs,output='sos'),a,axis=0)
    stride=max(1,int(fs/100))
    small=lf[::stride]
    baseline=median_filter(small,size=(max(3,int(fs/stride)|1),1),mode='nearest')
    base=np.column_stack([np.interp(np.arange(len(a)),np.arange(len(small))*stride,baseline[:,j]) for j in range(3)])
    magnitude=np.linalg.norm(lf-base,axis=1)
    speed=np.linalg.norm(np.gradient(lf,1/fs,axis=0),axis=1)
    # Assess a low-activity interval rather than isolated low samples. This also
    # prevents residual 50-Hz ripple from fragmenting every quiet interval.
    smoothing=max(1,int(round(cfg.quiet_activity_smooth_ms*fs/1000)))
    magnitude=uniform_filter1d(magnitude,size=smoothing)
    speed=uniform_filter1d(speed,size=smoothing)
    mask=(magnitude<=np.quantile(magnitude,cfg.quiet_quantile))&(speed<=np.quantile(speed,cfg.quiet_quantile))
    # Require low magnitude AND low slope; a gap between QRSs alone is not quiet.
    for anchor in qrs_anchors:
        lo=max(0,int(anchor+cfg.quiet_qrs_exclusion_ms[0]*fs/1000))
        hi=min(len(a),int(anchor+cfg.quiet_qrs_exclusion_ms[1]*fs/1000)+1)
        mask[lo:hi]=False
    guard=int(round(cfg.psd_window_s/4*fs));mask[:guard]=False;mask[-guard:]=False
    runs=[(lo,hi) for lo,hi in _runs(mask) if (hi-lo)/fs>=cfg.quiet_min_run_s]
    return runs, {'reason':'Các khoảng ít magnitude và ít slope sau low-pass QC; không phải PQ/TP chuẩn y khoa.',
                  'magnitude_threshold_mV':float(np.quantile(magnitude,cfg.quiet_quantile)),
                  'speed_threshold_mV_s':float(np.quantile(speed,cfg.quiet_quantile)),
                  'n_runs':len(runs),'total_s':sum(hi-lo for lo,hi in runs)/fs}


def _quiet_line_evidence(xyz, fs, runs, cfg):
    samples=sum(hi-lo for lo,hi in runs)
    span=(runs[-1][1]-runs[0][0])/fs if runs else 0
    if len(runs)<cfg.quiet_min_runs or samples/fs<cfg.quiet_min_total_s or span<cfg.quiet_min_span_s:
        return {'sufficient':False,'reason':'Không đủ khoảng ít hoạt động/độ dài/span để kiểm tra 50 Hz ngoài QRS.'}
    # Keep original timestamps; never concatenate gaps as continuous Welch samples.
    tt=np.concatenate([np.arange(lo,hi)/fs for lo,hi in runs])
    yy=np.concatenate([detrend(xyz[lo:hi],axis=0,type='linear') for lo,hi in runs])
    frequencies=np.arange(cfg.powerline_band_hz[0],cfg.powerline_band_hz[1]+cfg.quiet_frequency_step_hz/2,
                          cfg.quiet_frequency_step_hz)
    def fit(f):
        design=np.column_stack([np.sin(2*np.pi*f*tt),np.cos(2*np.pi*f*tt)])
        coef=np.linalg.lstsq(design,yy,rcond=None)[0]
        prediction=design@coef
        amplitude=np.linalg.norm(coef,axis=0)
        explained=np.maximum(0,1-np.sum((yy-prediction)**2,axis=0)/np.maximum(np.sum(yy**2,axis=0),1e-24))
        return amplitude,explained
    fits=[fit(f) for f in frequencies]
    amplitudes=np.array([x[0] for x in fits]);explained=np.array([x[1] for x in fits])
    ix=np.argmax(amplitudes,axis=0)
    line_amp=amplitudes[ix,np.arange(3)];fraction=explained[ix,np.arange(3)]
    # Jointly fit neighboring frequencies: gaps in the quiet mask can otherwise
    # leak one coherent 50-Hz sinusoid into independent neighboring projections.
    line_amp=np.zeros(3);bg=np.zeros(3);conditions=[]
    neighbors=[46.0,47.0,48.0,52.0,53.0,54.0]
    for j in range(3):
        frequency_list=[frequencies[ix[j]],*neighbors]
        design=np.column_stack([part for f in frequency_list for part in
                                [np.sin(2*np.pi*f*tt),np.cos(2*np.pi*f*tt)]])
        condition=float(np.linalg.cond(design));conditions.append(condition)
        if condition>cfg.quiet_fit_condition_limit:
            return {'sufficient':False,'reason':'Fit harmonic trên các khoảng rời rạc có condition quá lớn.'}
        coef=np.linalg.lstsq(design,yy[:,j],rcond=None)[0].reshape(-1,2)
        line_amp[j]=np.linalg.norm(coef[0])
        bg[j]=max(float(np.median(np.linalg.norm(coef[1:],axis=1))),1e-12)
    return {'sufficient':True,'reason':'Fit sin/cos trên nhiều đoạn ít hoạt động với timestamp thật; heuristic.',
            'amplitude_mV':line_amp,'score_db':20*np.log10(np.maximum(line_amp,1e-12)/bg),
            'explained_fraction':fraction,'peak_hz':frequencies[ix],
            'total_quiet_s':samples/fs,'quiet_span_s':span,'fit_condition':conditions}


def analyze_powerline_noise(xyz, fs, cfg=Config(), qrs_anchors=None):
    a=_validate_xyz(xyz,fs);f,p=welch_xyz(a,fs,cfg)
    if not cfg.powerline_band_hz[0] < cfg.notch_hz < cfg.powerline_band_hz[1]:
        raise ValueError('Notch center không nằm trong dải phân tích powerline; cần cấu hình nhất quán.')
    metrics=_line_psd_metrics(f,p,cfg)
    base={'status':'inconclusive','channels':{},'windows':pd.DataFrame(),
          'psd_frequency_hz':f,'psd_density':p,'quiet_intervals':[],
          'warnings':['Các ngưỡng PSD/persistence/quiet là heuristic chưa được kiểm chứng trên cohort.']}
    if metrics is None or len(a)/fs<cfg.powerline_min_duration_s:
        base.update(reason='Bản ghi quá ngắn/độ phân giải phổ chưa đủ.',score_db=np.nan,persistence=np.nan)
        for name in 'XYZ':base['channels'][name]={'status':'inconclusive','reason':base['reason']}
        return base
    size=int(round(cfg.psd_window_s*fs));hop=int(round(cfg.psd_hop_s*fs));rows=[]
    for start in range(0,len(a)-size+1,hop):
        wf,wp=welch_xyz(a[start:start+size],fs,cfg);m=_line_psd_metrics(wf,wp,cfg)
        if m is None:continue
        for j,name in enumerate('XYZ'):
            rows.append({'start_s':start/fs,'center_s':(start+size/2)/fs,'channel':name,
                         'score_db':float(m[2][j]),'peak_hz':float(m[3][j]),
                         'concentration':float(m[4][j])})
    windows=pd.DataFrame(rows)
    runs,quiet_info=find_low_activity_intervals(a,fs,cfg,qrs_anchors)
    evidence=_quiet_line_evidence(a,fs,runs,cfg)
    channels={}
    for j,name in enumerate('XYZ'):
        channel_windows=windows[windows.channel==name]
        significant=(channel_windows.score_db>=cfg.powerline_score_threshold_db)&(channel_windows.concentration>=cfg.powerline_peak_concentration)
        persistence=float(significant.mean()) if len(significant) else np.nan
        globally_high=bool(metrics[2][j]>=cfg.powerline_score_threshold_db and metrics[4][j]>=cfg.powerline_peak_concentration)
        quiet_positive=bool(evidence.get('sufficient') and evidence['score_db'][j]>=cfg.quiet_line_score_db
                            and evidence['explained_fraction'][j]>=cfg.quiet_line_explained_fraction)
        if globally_high and persistence>=cfg.powerline_persistence_threshold and quiet_positive:
            status='noise_detected';reason='Đỉnh hẹp, duy trì theo thời gian và có thành phần coherent trong vùng ít hoạt động.'
        elif globally_high and persistence<cfg.powerline_persistence_threshold:
            status='inconclusive';reason='Thành phần quanh 50 Hz nổi bật nhưng không duy trì đủ; có thể không ổn định.'
        elif globally_high and not evidence.get('sufficient'):
            status='inconclusive';reason='Có đỉnh PSD nhưng thiếu vùng ít hoạt động để phân biệt tim/nhiễu.'
        elif globally_high and not quiet_positive:
            status='noise_not_detected';reason='Không có bằng chứng coherent ngoài hoạt động tim; không tự Notch đỉnh PSD.'
        elif np.isfinite(persistence) and persistence>=cfg.powerline_inconsistent_persistence:
            status='inconclusive';reason='Bằng chứng phổ không nhất quán giữa toàn record và các đoạn.'
        else:
            status='noise_not_detected';reason='Không tìm thấy đỉnh hẹp/persistent đáng kể theo các tiêu chí thử nghiệm.'
        channels[name]={'status':status,'reason':reason,'score_db':float(metrics[2][j]),
            'p50_density':float(metrics[0][j]),'background_density':float(metrics[1][j]),
            'peak_hz':float(metrics[3][j]),'peak_concentration':float(metrics[4][j]),
            'persistence':persistence,'quiet_score_db':float(evidence['score_db'][j]) if evidence.get('sufficient') else np.nan,
            'quiet_explained_fraction':float(evidence['explained_fraction'][j]) if evidence.get('sufficient') else np.nan}
    states=[x['status'] for x in channels.values()]
    status='noise_detected' if 'noise_detected' in states else ('inconclusive' if 'inconclusive' in states else 'noise_not_detected')
    base.update(status=status,channels=channels,windows=windows,quiet_intervals=runs,
                quiet_info=quiet_info,quiet_evidence=evidence,score_db=float(np.max(metrics[2])),
                persistence=max(x['persistence'] for x in channels.values()),
                reason='; '.join(name+': '+channels[name]['reason'] for name in 'XYZ'))
    if not evidence.get('sufficient'):base['warnings'].append(evidence['reason'])
    return base


def decide_notch(analysis, cfg=Config()):
    if cfg.notch_mode not in {'auto','on','off'}:raise ValueError('notch_mode cần auto/on/off.')
    if cfg.notch_mode=='on':channels=list('XYZ');reason='Bật thủ công để thí nghiệm; không phải kết luận nhiễu.'
    elif cfg.notch_mode=='off':channels=[];reason='Tắt thủ công để thí nghiệm.'
    else:
        channels=[name for name,c in analysis['channels'].items() if c['status']=='noise_detected']
        reason='Auto chỉ lọc từng trục có bằng chứng đủ theo heuristic; trục inconclusive giữ nguyên.'
    return {'notch_applied':bool(channels),'notch_mode':cfg.notch_mode,'notch_channels':channels,
            'powerline_status':analysis['status'],'powerline_score_db':analysis.get('score_db',np.nan),
            'powerline_persistence':analysis.get('persistence',np.nan),'notch_reason':reason}


def apply_morphology_filter(xyz, fs, cfg=Config(), preliminary_detector=None):
    a=_validate_xyz(xyz,fs);raw=a.copy();quality=assess_input_quality(raw,fs,cfg)
    if cfg.morphology_mode=='fir':
        taps=design_morphology_fir(fs,cfg);padlen=3*len(taps)
        if len(raw)<=padlen+1:raise ValueError(f'FIR filtfilt cần > {padlen+1} mẫu; không tự rút ngắn bộ lọc.')
        morphology=filtfilt(taps,[1.0],raw,axis=0,padlen=padlen)
    elif cfg.morphology_mode=='legacy':
        cutoff=min(cfg.legacy_lowpass_hz,.40*fs)
        morphology=sosfiltfilt(butter(cfg.legacy_lowpass_order,cutoff,fs=fs,output='sos'),raw,axis=0)
    elif cfg.morphology_mode=='none':morphology=raw.copy()
    else:raise ValueError('morphology_mode cần fir/legacy/none.')
    anchors=None;preliminary_error=''
    if preliminary_detector is not None:
        try:
            preliminary_band=sosfiltfilt(butter(3,cfg.detector_band_hz,btype='bandpass',fs=fs,output='sos'),morphology,axis=0)
            anchors=preliminary_detector(morphology,preliminary_band,fs,cfg)['anchors']
        except ValueError as exc:preliminary_error=str(exc)
    analysis=analyze_powerline_noise(raw,fs,cfg,qrs_anchors=anchors)
    decision=decide_notch(analysis,cfg)
    if decision['notch_applied']:
        if not 0<cfg.notch_hz<fs/2 or cfg.notch_q<=0:raise ValueError('Notch center/Q không hợp lệ.')
        bn,an=iirnotch(cfg.notch_hz,cfg.notch_q,fs=fs)
        indices=['XYZ'.index(name) for name in decision['notch_channels']]
        morphology[:,indices]=filtfilt(bn,an,morphology[:,indices],axis=0)
    warnings=list(quality['flags'])+list(analysis['warnings'])
    if preliminary_error:warnings.append('preliminary_detector: '+preliminary_error)
    if analysis['status']=='inconclusive':warnings.append('powerline_inconclusive_review')
    if cfg.morphology_highpass_hz is not None:
        if cfg.morphology_highpass_hz!=.5:raise ValueError('Phương án high-pass thử nghiệm hiện chỉ cấu hình 0.5 Hz.')
        morphology=sosfiltfilt(butter(cfg.morphology_highpass_order,.5,btype='highpass',fs=fs,output='sos'),morphology,axis=0)
        warnings.append('optional_highpass_may_change_ST_T')
    if not np.isfinite(morphology).all():raise ValueError('Bộ lọc tạo giá trị không hữu hạn.')
    after_f,after_p=welch_xyz(morphology,fs,cfg)
    diagnostics={**decision,'input_quality':quality,'powerline':analysis,
        'morphology_mode':cfg.morphology_mode,'notch_q':cfg.notch_q,
        'rms_difference_uV':np.sqrt(np.mean((morphology-raw)**2,axis=0))*1000,
        'psd_after_frequency_hz':after_f,'psd_after_density':after_p,
        'warnings':list(dict.fromkeys(warnings)),'preliminary_anchors':anchors}
    return morphology,diagnostics


def preprocess_xyz(xyz, fs, cfg=Config(), return_diagnostics=False, preliminary_detector=None):
    a=_validate_xyz(xyz,fs)
    # Frozen legacy window experiments need the original two-array contract.
    if (cfg.morphology_mode=='legacy' and cfg.notch_mode in {'on','off'}
            and cfg.morphology_highpass_hz is None and not return_diagnostics):
        m=sosfiltfilt(butter(cfg.legacy_lowpass_order,min(cfg.legacy_lowpass_hz,.40*fs),fs=fs,output='sos'),a,axis=0)
        if cfg.notch_mode=='on':
            b,c=iirnotch(cfg.notch_hz,cfg.notch_q,fs=fs);m=filtfilt(b,c,m,axis=0)
    else:m,diagnostics=apply_morphology_filter(a,fs,cfg,preliminary_detector)
    lo,hi=cfg.detector_band_hz
    if not 0<lo<hi<fs/2:raise ValueError('Dải detector không hợp lệ; không tự đổi tần số.')
    detection=sosfiltfilt(butter(3,[lo,hi],btype='bandpass',fs=fs,output='sos'),m,axis=0)
    return (m,detection,diagnostics) if return_diagnostics else (m,detection)
