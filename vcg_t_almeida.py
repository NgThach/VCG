"""Experimental Frank-XYZ T delineation: quadratic-spline WT loops and TLS.

SPDX-License-Identifier: GPL-2.0-only

Paper equations: Martinez et al. (2004), II-A--D and Appendix;
Almeida et al. (2009), II-A and II-C. Implementation details were also checked
against ECG-kit/wavedet, commit c8e3de47c54a9214138143676d2aa546b0540dd2.
This is a documented reconstruction/adaptation, NOT a bit-exact Wavedet port.
See METHOD_REPRODUCTION.md for every departure and its scientific limits.

The numerical entry point intentionally accepts neither V1 T marks nor labels.
All sample indices are zero-based and local to the unchanged V1 median.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from scipy.signal import convolve, find_peaks, resample_poly

METHOD_VERSION = "almeida-reconstruction-0.1"
METHOD_CLASS = "paper_equations_with_documented_project_adaptations"


@dataclass(frozen=True)
class AlmeidaConfig:
    reference_fs: float = 250.0
    scales: tuple = (3, 4, 5)
    detection_rms_fraction: float = 0.25  # Martinez A.3
    significant_fraction: float = 0.125  # Martinez A.7
    onset_fraction: float = 0.25  # Martinez A.11
    single_lead_end_fraction: float = 0.40  # Martinez A.12
    projected_end_fraction: float = 0.25  # Almeida II-A footnote 1
    sigma_cse_ms: float = 15.3  # Almeida II-C3, both T boundaries
    stable_less_than_samples: int = 2  # native samples, NOT 2 ms
    max_iterations: int = 20  # engineering safeguard, not paper convergence
    search_start_s: float = 0.100  # reference twavef/twave3D
    search_max_s: float = 0.600
    next_qrs_guard_s: float = 0.240
    minimum_search_s: float = 0.100
    boundary_radius_s: float = 0.120  # reference searchon/searchoff calls
    sl_opposite_max_distance_s: float = 0.200  # reference twavef 2012
    ml_opposite_max_distance_s: float = 0.150  # reference twave3D
    biphasic_weak_multiplier: float = 2.5  # reference twavef/twave3D
    filter_resampling_kaiser_beta: float = 5.0  # explicit project interpolation
    numeric_floor: float = 1e-12
    tls_equal_singular_tolerance: float = 1e-8
    tls_review_energy_fraction: float = 0.60  # review only, not detection
    duration_review_ms: tuple = (60.0, 400.0)  # review only, never clip

    def __post_init__(self):
        if self.reference_fs != 250 or tuple(self.scales) != (3, 4, 5):
            raise ValueError("This reconstruction requires the verified 250-Hz bank and scales 3/4/5")
        if self.max_iterations < 1 or self.stable_less_than_samples != 2:
            raise ValueError("max_iterations >= 1; paper convergence is strictly < 2 samples")
        if not 0 < self.significant_fraction < 1:
            raise ValueError("Invalid significant-extremum fraction")


def config_id(cfg=AlmeidaConfig()):
    body = json.dumps({"method": METHOD_VERSION, **asdict(cfg)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def ms_to_samples(ms, fs):
    """Positive widths: nearest sample, halves upward, as MATLAB round."""
    return int(math.floor(float(ms) * float(fs) / 1000.0 + 0.5))


def sample_ms(index, anchor, fs):
    return None if index is None else (int(index) - int(anchor)) * 1000.0 / float(fs)


def _dilate(kernel, step):
    out = np.zeros((len(kernel) - 1) * step + 1)
    out[::step] = kernel
    return out


def reference_equivalent_filter(scale):
    """Martinez (4),(6),(7): G(2^(m-1)w) prod H(2^l w), no decimation.

    Returned causal-array representation is antisymmetric; its delay is
    (len(q)-1)/2. The paper's h[-2:1], g[-1:0] have the same shape after a
    time-origin shift. Only this known linear phase is removed below.
    """
    if int(scale) != scale or not 1 <= scale <= 5:
        raise ValueError("Verified quadratic-spline scales are 1 through 5")
    h = np.array([1., 3., 3., 1.]) / 8
    g = np.array([2., -2.])
    smooth = np.array([1.])
    for level in range(int(scale) - 1):
        smooth = np.convolve(smooth, _dilate(h, 2**level))
    return np.convolve(smooth, _dilate(g, 2**(int(scale) - 1)))


@lru_cache(maxsize=32)
def _adapted_filter(scale, fs, beta):
    if not np.isfinite(fs) or fs < 100:
        raise ValueError("fs must be finite and >=100 Hz; signals are never resampled")
    q = reference_equivalent_filter(scale)
    ratio = Fraction(float(fs) / 250).limit_denominator(10000)
    if abs(float(ratio) - fs / 250) > 1e-10:
        raise ValueError("Sampling frequency has no supported rational adaptation")
    if fs == 250:
        coeff, delay = q, (len(q) - 1) / 2
    else:
        # Martinez II-C specifies resampling equivalent impulse responses but
        # not the interpolator. This Kaiser polyphase choice is an adaptation.
        # Zeros give the interpolator room; 1/ratio preserves analogue gain.
        pad = 12
        coeff = resample_poly(np.pad(q, (pad, pad)), ratio.numerator,
                              ratio.denominator, window=("kaiser", beta)) / float(ratio)
        delay = (pad + (len(q) - 1) / 2) * float(ratio)
    return coeff, float(delay)


def wavelet_transform(xyz, fs, cfg=AlmeidaConfig()):
    """Undecimated quadratic-spline WT: output (N,3) for every scale.

    Reflect padding is used only to calculate WT, never to invent loop samples.
    A border-support mask makes padding-dependent positions inspectable. At
    1000 Hz all three equivalent filters have integer linear-phase delays.
    Fractional-delay alignment at other rates uses explicit linear interpolation.
    """
    a = np.asarray(xyz, float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 5 or not np.isfinite(a).all():
        raise ValueError("Finite XYZ with shape (N>=5,3) required")
    transformed, metadata = {}, {}
    for scale in cfg.scales:
        q, delay = _adapted_filter(scale, float(fs), cfg.filter_resampling_kaiser_beta)
        pad = int(math.ceil(max(delay, len(q) - 1 - delay))) + 2
        padded = np.pad(a, ((pad, pad), (0, 0)), mode="reflect")
        full = convolve(padded, q[:, None], mode="full", method="direct")
        loc = pad + delay + np.arange(len(a))
        result = np.column_stack([np.interp(loc, np.arange(len(full)), full[:, k])
                                  for k in range(3)])
        transformed[scale] = result
        # Use non-negligible tap support rather than the zero padding tail.
        nz = np.flatnonzero(abs(q) > max(np.max(abs(q)) * 1e-10, cfg.numeric_floor))
        support = int(math.ceil(max(delay - nz[0], nz[-1] - delay))) if len(nz) else 0
        metadata[str(scale)] = dict(scale=scale, reference_fs=250.0, fs=float(fs),
            h=[.125, .375, .375, .125], h_paper_indices=[-2, -1, 0, 1],
            g=[2., -2.], g_paper_indices=[-1, 0],
            reference_coefficients=reference_equivalent_filter(scale).tolist(),
            coefficients=q.tolist(), delay_samples=delay,
            alignment="exact_integer_delay" if delay.is_integer() else "linear_fractional_delay_adaptation",
            padding="reflect; border candidates flagged", support_radius_samples=support,
            gain_normalization="base h/8,g*2; adapted coefficients divided by fs/250",
            fs_adaptation="none" if fs == 250 else "Kaiser polyphase equivalent-filter resampling; project choice")
    return transformed, metadata


def modulus_extrema(w, start, end, include_edges=False):
    """Local maxima of |WT|, retaining polarity and deterministic plateau ties.

    Peak extraction alone is NOT T detection. Dominant/opposite slope tests,
    RMS detection and morphology rules are applied in delineate_single_wavelet.
    """
    w = np.asarray(w, float)
    if not 0 <= start < end < len(w):
        return np.array([], dtype=int)
    ix = find_peaks(abs(w[start:end+1]), plateau_size=1)[0] + start
    ix = ix[abs(w[ix]) > 0]
    if include_edges:
        edges = [p for p in (start, end) if abs(w[p]) > 0 and
                 abs(w[p]) >= abs(w[p+1 if p == start else p-1])]
        ix = np.unique(np.r_[ix, edges]).astype(int)
    return ix


def first_zero_crossing(w, start, end):
    """Reference zerocros rule: first sign crossing, nearer-to-zero sample."""
    if end <= start:
        return None
    a = np.asarray(w)[start:end+1]
    crossings = np.flatnonzero(a[:-1] * a[1:] <= 0)
    if not len(crossings):
        return None
    k = int(crossings[0])
    return start + k + int(abs(a[k+1]) < abs(a[k]))


def find_wave_boundary(w, extremum, lower, upper, fraction, boundary):
    """Nearest outward sample satisfying relative threshold OR local |WT| min.

    Unlike reference searchon/off's end-of-window fallback, absence of either
    criterion is unresolved. No artificial boundary is placed at a window edge.
    """
    if extremum is None:
        return None, "significant_extremum_missing"
    a = abs(np.asarray(w, float))
    threshold = float(fraction * a[extremum])
    indices = range(extremum - 1, lower - 1, -1) if boundary == "t_on" else range(extremum + 1, upper + 1)
    for k in indices:
        relative = bool(a[k] < threshold)
        local = (0 < k < len(a)-1 and a[k] <= a[k-1] and a[k] <= a[k+1]
                 and (a[k] < a[k-1] or a[k] < a[k+1]))
        if relative or local:
            return int(k), "relative_threshold" if relative else "local_modulus_minimum"
    return None, "no_threshold_or_local_minimum"


def delineate_single_wavelet(w, w_peak, window, fs, rms_interval,
                             cfg=AlmeidaConfig(), projected=False):
    """Martinez T slopes/morphology plus explicit reference-code details.

    The dominant slope and its nearest opposite-polarity slopes form positive,
    negative or two biphasic configurations. Lone slopes are recorded as
    only_upwards/downwards and unresolved rather than promoted to complete T.
    """
    w, w_peak = np.asarray(w, float), np.asarray(w_peak, float)
    lo, hi = map(int, window)
    result = dict(detection_status="unresolved", wave_detected=False, failure_reason="", morphology_type="unresolved",
        search_start=lo, search_end=hi, t_peak_candidate_sample=None, t_peak_candidates=[],
        t_on_candidate_sample=None, t_off_candidate_sample=None,
        first_significant_extremum=None, last_significant_extremum=None,
        selected_extrema=[], extrema=[], boundary_criteria={})
    if hi - lo < ms_to_samples(cfg.minimum_search_s * 1000, fs):
        result["failure_reason"] = "search_window_shorter_than_reference_100ms"
        return result
    ra, rb = rms_interval
    rms = float(np.sqrt(np.mean(w[ra:rb+1]**2)))
    detection_threshold = cfg.detection_rms_fraction * rms
    ix = modulus_extrema(w, lo, hi, include_edges=True)
    result.update(rms=rms, rms_interval=[int(ra), int(rb)], detection_threshold=detection_threshold)
    if not len(ix) or np.max(abs(w[ix])) <= cfg.numeric_floor:
        result["failure_reason"] = "no_nonzero_wavelet_extrema"
        return result
    dominant = int(ix[np.argmax(abs(w[ix]))])
    dominant_amp = float(abs(w[dominant]))
    significant_threshold = cfg.significant_fraction * float(np.max(abs(w[lo:hi+1])))
    result.update(dominant_extremum=dominant, significant_threshold=significant_threshold)
    # Paper II-D3 requires at least two extrema above epsilon. The later
    # reference code uses an OR on dominant amplitudes; this implementation
    # retains the paper's more conservative two-extremum presence requirement.
    present = ix[abs(w[ix]) > detection_threshold]
    if len(present) < 2:
        result["failure_reason"] = "fewer_than_two_extrema_above_detection_threshold"
    opposite = ix[np.sign(w[ix]) != np.sign(w[dominant])]
    before, after = opposite[opposite < dominant], opposite[opposite > dominant]
    left = int(before[-1]) if len(before) else None
    right = int(after[0]) if len(after) else None
    max_distance = (cfg.ml_opposite_max_distance_s if projected else cfg.sl_opposite_max_distance_s) * fs
    rejected = {}
    for name, p in (("left", left), ("right", right)):
        if p is not None and (abs(w[p]) < significant_threshold or abs(p-dominant) > max_distance):
            rejected[p] = "below_gamma" if abs(w[p]) < significant_threshold else "opposite_extremum_too_far_reference_rule"
            if name == "left": left = None
            else: right = None
    if left is not None and right is not None:
        weak = left if abs(w[left]) <= abs(w[right]) else right
        if abs(w[weak]) < cfg.significant_fraction * cfg.biphasic_weak_multiplier * dominant_amp:
            rejected[weak] = "weak_biphasic_lobe_reference_rule"
            if weak == left: left = None
            else: right = None
    selected = ([left] if left is not None else []) + [dominant] + ([right] if right is not None else [])
    signs = [int(np.sign(w[p])) for p in selected]
    shape = {(1, -1): "positive", (-1, 1): "negative",
             (1, -1, 1): "biphasic_positive_negative",
             (-1, 1, -1): "biphasic_negative_positive",
             (1,): "only_upwards", (-1,): "only_downwards"}.get(tuple(signs), "unresolved")
    result.update(selected_extrema=selected, morphology_type=shape)
    for p in ix:
        p = int(p)
        result["extrema"].append(dict(sample=p, amplitude=float(w[p]),
            exceeds_detection=bool(abs(w[p]) > detection_threshold),
            exceeds_significance=bool(abs(w[p]) >= significant_threshold),
            selected=p in selected,
            rejection_reason="" if p in selected else rejected.get(p, "not_adjacent_to_dominant_slope"),
            at_search_edge=p in (lo, hi)))
    if result["failure_reason"]:
        return result
    if len(selected) < 2:
        result["failure_reason"] = "only_one_significant_slope_incomplete_morphology"
        return result
    peaks = []
    for a, b in zip(selected[:-1], selected[1:]):
        peak = first_zero_crossing(w_peak, a, b)
        peak_source = "scale3"
        if peak is None:
            peak = first_zero_crossing(w, a, b)
            peak_source = "detection_scale_fallback"
        if peak is not None:
            peaks.append(int(peak))
        result.setdefault("peak_sources", []).append(peak_source if peak is not None else "unresolved")
    if len(peaks) != len(selected)-1:
        result["failure_reason"] = "t_peak_zero_crossing_unresolved"
        return result
    first, last = selected[0], selected[-1]
    radius = ms_to_samples(cfg.boundary_radius_s * 1000, fs)
    onset, on_criterion = find_wave_boundary(w, first, max(lo, first-radius), hi, cfg.onset_fraction, "t_on")
    end_fraction = cfg.projected_end_fraction if projected else cfg.single_lead_end_fraction
    offset, off_criterion = find_wave_boundary(w, last, lo, min(hi, last+radius), end_fraction, "t_off")
    result.update(first_significant_extremum=first, last_significant_extremum=last,
        wave_detected=True,
        t_peak_candidate_sample=peaks[0], t_peak_candidates=peaks,
        t_on_candidate_sample=onset, t_off_candidate_sample=offset,
        onset_threshold=float(cfg.onset_fraction*abs(w[first])),
        offset_threshold=float(end_fraction*abs(w[last])),
        boundary_criteria={"t_on": on_criterion, "t_off": off_criterion})
    if onset is None or offset is None:
        result["detection_status"] = "needs_review"
        result["failure_reason"] = "; ".join(k+":"+v for k,v in result["boundary_criteria"].items()
                                             if "no_threshold" in v)
    else:
        result["detection_status"] = "detected"
    return result


def single_lead_initialization(wavelets, window, fs, rms_interval, cfg=AlmeidaConfig()):
    records = []
    for axis_index, axis in enumerate("XYZ"):
        attempts, chosen = [], None
        for scale in (4, 5):
            result = delineate_single_wavelet(wavelets[scale][:, axis_index], wavelets[3][:, axis_index],
                                               window, fs, rms_interval, cfg)
            result.update(axis=axis, wavelet_scale=scale)
            attempts.append(result)
            # Scale fallback is about T-wave DETECTION, not whether both limits
            # happen to be resolved. A valid peak with one missing SL limit is
            # still a legitimate WT-loop initialization (Almeida II-C3).
            if result["wave_detected"]:
                chosen = result
                break
        chosen = dict(chosen if chosen is not None else attempts[-1])
        chosen["scale_selection_reason"] = ("scale4_detected" if chosen["wavelet_scale"] == 4
            else "scale4_unresolved_try_scale5")
        chosen["attempts"] = attempts
        records.append(chosen)
    return records


def select_multilead_scale(single_leads):
    """Almeida II-C3: m=5 iff at least TWO leads detected T at m=5."""
    count = sum(r.get("wave_detected", r["detection_status"] == "detected") and r["wavelet_scale"] == 5 for r in single_leads)
    return 5 if count >= 2 else 4


def tls_direction(points, previous=None, cfg=AlmeidaConfig()):
    """Affine orthogonal TLS line, exactly solved by centered SVD.

    The reference optimline estimates BOTH an intercept and direction; hence
    centering for the fit is justified. Projection still uses original WT @ u,
    without subtracting that intercept (Almeida eq.4).
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        return dict(status="unresolved", reason="tls_requires_three_finite_3d_points")
    center = p.mean(axis=0)
    _, singular, vt = np.linalg.svd(p-center, full_matrices=False)
    if singular[0] <= cfg.numeric_floor:
        return dict(status="unresolved", reason="tls_zero_variation", singular_values=singular.tolist())
    if (singular[0]-singular[1]) <= cfg.tls_equal_singular_tolerance*singular[0]:
        return dict(status="unresolved", reason="tls_direction_not_identifiable", singular_values=singular.tolist())
    u = vt[0] / np.linalg.norm(vt[0])
    if previous is not None:
        if np.dot(u, previous) < 0: u = -u
    elif u[int(np.argmax(abs(u)))] < 0:
        u = -u
    return dict(status="ok", direction=u.tolist(), center=center.tolist(),
        singular_values=singular.tolist(), fit_quality=float(singular[0]**2/np.sum(singular**2)),
        orthogonal_residual_rms=float(np.sqrt(np.sum(singular[1:]**2)/len(p))))


def project_wavelets(wavelets, direction):
    u = np.asarray(direction, float)
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) <= 0:
        raise ValueError("A finite nonzero 3-D direction is required")
    return np.asarray(wavelets) @ (u/np.linalg.norm(u))


def iterate_tls_boundary(wavelets, scale, initial_window, t_window, rms_interval,
                         fs, boundary, cfg=AlmeidaConfig()):
    if boundary not in ("t_on", "t_off"):
        raise ValueError("Boundary must be t_on or t_off")
    history, derived_history = [], []
    previous, previous_direction = None, None
    window = tuple(initial_window)
    margin = ms_to_samples(4*cfg.sigma_cse_ms, fs)
    stop_reason, technical_review = "max_iterations", []
    chosen_iteration = None
    for iteration in range(1, cfg.max_iterations+1):
        a, b = window
        fit = tls_direction(wavelets[scale][a:b+1], previous_direction, cfg)
        entry = dict(boundary=boundary, iteration=iteration, window_start=a, window_end=b, **fit,
                     accepted=False, candidate_sample=None, stop_reason="")
        history.append(entry)
        if fit["status"] != "ok":
            stop_reason = fit["reason"]
            if previous is not None: technical_review.append(stop_reason+"_previous_retained")
            break
        direction = fit["direction"]
        derived = project_wavelets(wavelets[scale], direction)
        derived_history.append(derived)
        peak_wavelet = project_wavelets(wavelets[3], direction)
        detected = delineate_single_wavelet(derived, peak_wavelet, t_window, fs, rms_interval, cfg, projected=True)
        ext_name = "first_significant_extremum" if boundary == "t_on" else "last_significant_extremum"
        ext = detected.get(ext_name)
        candidate = detected.get(boundary+"_candidate_sample")
        amplitude = float(abs(derived[ext])) if ext is not None else None
        entry.update(candidate_sample=candidate, extremum_sample=ext, extremum_amplitude=amplitude,
            morphology_type=detected["morphology_type"], wave=detected,
            derived_index=len(derived_history)-1,
            boundary_threshold=detected.get("onset_threshold" if boundary == "t_on" else "offset_threshold"))
        if ext is None or candidate is None:
            stop_reason = "no_significant_extremum_or_boundary"
            entry["failure_reason"] = detected["failure_reason"]
            # Almeida II-C1(d): keep previous mark, if one was established.
            break
        if previous is not None and amplitude <= previous["amplitude"]:
            stop_reason = "no_extremum_improvement_previous_retained"
            break
        old_candidate = previous["sample"] if previous is not None else None
        previous = dict(sample=int(candidate), amplitude=amplitude, extremum=int(ext),
                        direction=direction, peaks=detected["t_peak_candidates"], morphology=detected["morphology_type"])
        previous_direction = direction
        chosen_iteration = iteration
        entry["accepted"] = True
        if old_candidate is not None and abs(candidate-old_candidate) < cfg.stable_less_than_samples:
            stop_reason = "boundary_change_strictly_less_than_two_samples"
            break
        # Update W only; delineation is still applied on the full fixed T window I.
        if boundary == "t_on":
            window = (max(0, candidate-margin), int(ext))
        else:
            window = (int(ext), min(len(derived)-1, candidate+margin))
        if window[1]-window[0] < 2:
            stop_reason = "tls_updated_window_too_short"
            technical_review.append(stop_reason)
            break
    else:
        technical_review.append("max_iterations_reached_not_converged")
    if history: history[-1]["stop_reason"] = stop_reason
    if previous is not None:
        selected = history[chosen_iteration-1]
        if selected.get("fit_quality", 1) < cfg.tls_review_energy_fraction:
            technical_review.append("tls_low_directional_energy_review")
    return dict(sample=previous["sample"] if previous else None,
        extremum=previous["extremum"] if previous else None,
        peaks=previous["peaks"] if previous else [],
        direction=previous["direction"] if previous else None,
        morphology=previous["morphology"] if previous else "unresolved",
        chosen_iteration=chosen_iteration, history=history,
        derived_history=np.asarray(derived_history), stop_reason=stop_reason,
        converged=stop_reason == "boundary_change_strictly_less_than_two_samples",
        algorithm_stopped=stop_reason != "max_iterations",
        review_flags=technical_review)


def build_t_search_window(n, fs, anchor, qrs_off, rr_seconds, cfg=AlmeidaConfig()):
    """Reference-code RR window, adapted to the available frozen median.

    RR median replaces recursive beat-by-beat RR. QRS-off V1 replaces the
    single-lead S/off marks. No V1 T mark is accessed.
    """
    if not np.isfinite(rr_seconds) or rr_seconds <= 0:
        raise ValueError("Positive finite RR required")
    lo = max(int(qrs_off)+1, int(anchor)+ms_to_samples(cfg.search_start_s*1000, fs))
    end_s = min(cfg.search_max_s*max(1., rr_seconds), rr_seconds-cfg.next_qrs_guard_s)
    hi = min(n-1, int(anchor)+ms_to_samples(end_s*1000, fs))
    if hi <= lo:
        raise ValueError("No T search interval after QRS and before next-QRS guard")
    return lo, hi


def validate_t_bounds(onset, offset, qrs_off, n, peaks=()):
    if onset is None or offset is None:
        return False, "missing_t_boundary"
    if any(int(x) != x for x in (onset, offset)) or not 0 <= qrs_off < onset < offset < n:
        return False, "invalid_order_qrs_off_t_on_t_off_or_domain"
    if peaks and not all(onset < p < offset for p in peaks):
        return False, "selected_t_peaks_outside_joint_interval"
    return True, ""


def delineate_t(xyz, fs, anchor, qrs_on, qrs_off, rr_seconds, cfg=AlmeidaConfig()):
    """Complete numerical experiment, independent of all V1 T marks and labels."""
    a = np.asarray(xyz, float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("Finite Frank XYZ (N,3) required")
    if not (int(anchor) == anchor and 0 <= qrs_on < anchor < qrs_off < len(a)):
        raise ValueError("Median-local QRS-on < anchor < QRS-off required")
    window = build_t_search_window(len(a), fs, anchor, qrs_off, rr_seconds, cfg)
    wavelets, bank = wavelet_transform(a, fs, cfg)
    rms_interval = (int(anchor), min(len(a)-1, int(anchor)+ms_to_samples(rr_seconds*1000, fs)))
    sl = single_lead_initialization(wavelets, window, fs, rms_interval, cfg)
    valid_sl = [r for r in sl if r["wave_detected"]]
    output = dict(method=METHOD_VERSION, method_class=METHOD_CLASS, config_id=config_id(cfg),
        status="unresolved", failure_reason="", review_flags=[], t_on_sample=None,
        t_off_sample=None, t_on_ms=None, t_off_ms=None, t_duration_ms=None,
        qrs_on_sample=int(qrs_on), qrs_off_sample=int(qrs_off), anchor_sample=int(anchor),
        qrs_on_ms=sample_ms(qrs_on,anchor,fs), qrs_off_ms=sample_ms(qrs_off,anchor,fs),
        fs=float(fs), n_samples=len(a), search_window=list(window),
        rms_interval=list(rms_interval), rms_interval_complete_rr=rms_interval[1] < len(a)-1,
        rms_note="Available anchor-to-end median substitutes full QRS-to-next-QRS RMS; project adaptation",
        wavelets=wavelets, filter_bank=bank, single_leads=sl, branches={})
    if not valid_sl:
        output["failure_reason"] = "INITIALIZATION_FAILED_all_three_single_leads_unresolved"
        return output
    scale = select_multilead_scale(sl)
    initial_window = (min(r["search_start"] for r in valid_sl), max(r["search_end"] for r in valid_sl))
    output.update(selected_scale=scale, initial_window=list(initial_window), initialization_status="initialized",
        scale_selection_reason="At least two valid SL detections at scale5" if scale == 5 else "Fewer than two valid SL detections at scale5")
    for boundary in ("t_on", "t_off"):
        branch = iterate_tls_boundary(wavelets, scale, initial_window, window, rms_interval, fs, boundary, cfg)
        output["branches"][boundary] = branch
        output[boundary+"_sample"] = branch["sample"]
        output[boundary+"_ms"] = sample_ms(branch["sample"], anchor, fs)
        output["review_flags"].extend(branch["review_flags"])
    on, off = output["t_on_sample"], output["t_off_sample"]
    # The two independently optimized projections may select different lobes.
    # Require the first onset-projection peak and last offset-projection peak
    # inside the joint interval; report conflicts rather than stretching bounds.
    peaks = []
    for boundary, pick in (("t_on", 0), ("t_off", -1)):
        if output["branches"][boundary]["peaks"]:
            peaks.append(output["branches"][boundary]["peaks"][pick])
    valid, reason = validate_t_bounds(on, off, qrs_off, len(a), peaks)
    if not valid:
        output["failure_reason"] = reason
        # A resolved single boundary remains measurable even when its partner is
        # absent. Never export a loop from this partial result. Joint conflicts
        # (both present but invalid order/peaks) invalidate both final marks.
        output["candidate_t_on_sample"], output["candidate_t_off_sample"] = on, off
        if on is not None and off is not None:
            for key in ("t_on_sample", "t_off_sample", "t_on_ms", "t_off_ms"):
                output[key] = None
        return output
    duration = (off-on)*1000/fs
    output["t_duration_ms"] = float(duration)
    if not cfg.duration_review_ms[0] <= duration <= cfg.duration_review_ms[1]:
        output["review_flags"].append("t_duration_outside_review_range_not_clipped")
    for boundary in ("t_on", "t_off"):
        b = output["branches"][boundary]
        chosen = b["history"][b["chosen_iteration"]-1]
        if any(e["selected"] and e["at_search_edge"] for e in chosen["wave"]["extrema"]):
            output["review_flags"].append(boundary+"_extremum_at_search_edge")
        radius = bank[str(scale)]["support_radius_samples"]
        if b["sample"] < radius or b["sample"] >= len(a)-radius:
            output["review_flags"].append(boundary+"_wavelet_border_support_review")
    output["review_flags"] = sorted(set(output["review_flags"]))
    output["status"] = "needs_review" if output["review_flags"] else "detected"
    return output


def json_safe(value):
    """Strict JSON diagnostics; numeric missing values become null."""
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value
