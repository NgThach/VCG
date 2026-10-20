"""Read-only inventory of the *implemented* V1 QC rules for Notebook 07.

This module describes rules; it neither reruns nor changes V1 and never assigns
boundary correctness. Input/beat diagnostics and batch-only flags have distinct
scopes because they do not all participate in ``process_record`` status.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vcg_preprocessing import Config

PIPELINE = "vcg_pipeline.py"
PREPROCESSING = "vcg_preprocessing.py"
BATCH = "scripts/run_full_vcg_evaluation.py"
COLUMNS = ["qc_flag", "scope", "trigger_condition", "threshold_v1", "meaning",
           "false_alarm_risk", "threshold_basis", "source_file", "source_line",
           "contributes_to_v1_status"]


def _line(filename, snippet):
    """Resolve lines against the source actually audited, and fail on drift."""
    hits = [i for i, line in enumerate((ROOT / filename).read_text().splitlines(), 1)
            if snippet in line]
    if len(hits) != 1:
        raise ValueError(f"Source locator must be unique: {filename}: {snippet!r}: {hits}")
    return hits[0]


def qc_rules_inventory(cfg=None):
    """Return all categorical QC flags and beat-exclusion reasons in V1.

    ``cfg`` may be Config or a dict of Config fields. Default values describe
    unchanged V1. Source line numbers are looked up in current source, not copied
    from an old report. A row marked False can still indirectly affect beat
    availability; it is not directly merged into the successful V1 result flags.
    Dynamic prose warnings and status values are documented in findings, rather
    than represented as invented QC flags.
    """
    cfg = Config() if cfg is None else Config(**cfg) if isinstance(cfg, dict) else cfg
    rows = []
    heuristic = ("Heuristic nghiên cứu trong source V1; chưa có trích dẫn xác nhận "
                 "ngưỡng hoặc hiệu chuẩn độc lập trên cohort (vcg_preprocessing.py:1).")

    def add(flag, scope, condition, threshold, meaning, risk, filename=PIPELINE,
            snippet=None, contributes=True, basis=heuristic):
        rows.append(dict(qc_flag=flag, scope=scope, trigger_condition=condition,
                         threshold_v1=threshold, meaning=meaning, false_alarm_risk=risk,
                         threshold_basis=basis, source_file=filename,
                         source_line=_line(filename, snippet or f"append('{flag}')"),
                         contributes_to_v1_status=bool(contributes)))

    add("rr_extreme_or_missed_beat", "v1_detector", "any((RR < 0.30 s) | (RR > 2.0 s))",
        "RR < 300 ms hoặc > 2000 ms; bất đẳng thức nghiêm",
        "Khoảng cách giữa anchors cực đoan; chưa phân biệt loạn nhịp và nhận sai/bỏ sót nhịp.",
        "Nhịp nhanh/chậm hoặc loạn nhịp thật cũng có thể vượt ngưỡng.")
    add("rr_irregular", "v1_detector", "robust_sigma(RR) / max(median(RR), 1e-9) > 0.15",
        "1.4826 × MAD(RR) / median(RR) > 0.15",
        "RR giữa các anchors không đồng đều theo MAD chuẩn hóa.",
        "Không đồng đều sinh lý hoặc bệnh lý không chứng minh detector sai.")
    add("possible_missed_beat", "v1_detector", "any(RR > 1.7 * median(RR))",
        "RR > 1.7 × median RR", "Có RR dài tương đối; chỉ là nghi ngờ bỏ sót.",
        "Khoảng nghỉ thật/ngoại tâm thu cũng tạo RR dài.")

    # Input flags remain in preprocessing/input_qc_flags; process_record does not
    # merge this list directly. Their sample masks drive separate beat decisions.
    add("invalid_values", "v1_input_quality", "any(~isfinite(XYZ).all(axis=1)); return ngay",
        "Ít nhất một NaN/Inf", "Dữ liệu không hữu hạn; load_xyz/_validate_xyz thường dừng trước.",
        "Không xác nhận sai phân đoạn; pipeline có thể chưa tới delineation.", PREPROCESSING,
        contributes=False, basis="Ràng buộc dữ liệu hữu hạn; không phải tolerance accuracy.")
    add("all_axes_flat_segment_review", "v1_input_quality",
        "any run of all(abs(diff(XYZ)) <= tolerance, axis=1) lasting >= input_flat_min_s",
        f"tolerance={cfg.input_flat_tolerance_mV:g} mV; run >= {cfg.input_flat_min_s:g} s",
        "Cả ba trục cùng gần như hằng số trong một đoạn.",
        "Đoạn phẳng ngoài QRS không tự chứng minh biên sai.", PREPROCESSING, contributes=False)
    add("isolated_impulse_review", "v1_input_quality",
        "any axis j at sample i: d[i-1,j]*d[i,j]<0 and both abs(d)>max(input_spike_min_mV,input_spike_mad_factor*robust_sigma(d[:,j]))",
        f"Hai sai phân trái dấu, cùng > max({cfg.input_spike_min_mV:g} mV, {cfg.input_spike_mad_factor:g} × MAD chuẩn hóa)",
        "Xung một mẫu có sườn lên/xuống lớn; không đánh dấu mọi sườn QRS.",
        "QRS nhọn có thể thỏa điều kiện; mask gần anchor được giữ để review.", PREPROCESSING, contributes=False)
    add("near_nominal_adc_rail_review", "v1_input_quality",
        "any run(any(abs(XYZ)>=input_saturation_abs_mV,axis=1)) of >= input_saturation_min_ms",
        f"abs(axis) >= {cfg.input_saturation_abs_mV:g} mV trong >= {cfg.input_saturation_min_ms:g} ms",
        "Biên độ gần rail danh nghĩa; source không đo ADC rail riêng từng record.",
        "Rail danh nghĩa là heuristic; cần kiểm tra header/tín hiệu.", PREPROCESSING, contributes=False)
    add("low_frequency_transient_motion_or_ST_review", "v1_input_quality",
        "len(centers)>2 and any(norm(diff(median XYZ of each complete 1-s block)) > max(input_motion_center_step_mV,median(jumps)+input_motion_mad_factor*robust_sigma(jumps)))",
        f"jump > max({cfg.input_motion_center_step_mV:g} mV, median(jumps)+{cfg.input_motion_mad_factor:g} × MAD chuẩn hóa)",
        "Thay đổi tâm khối 1 giây; mask phủ hai khối cạnh jump.",
        "Có thể là thay đổi ST/T thật hoặc chuyển động; chưa phân biệt nguyên nhân.", PREPROCESSING, contributes=False)

    # These are actual values of beat_qc.reason, NOT record-level flags.
    reason_basis = "Quy tắc giữ/loại beat theo source V1; heuristic, không phải nhãn phân đoạn đúng/sai."
    def reject(flag, condition, threshold, meaning, risk, locator):
        add(flag, "v1_beat_exclusion", condition, threshold, meaning, risk,
            snippet=locator, contributes=False, basis=reason_basis)
    reject("edge", "i==0 or i==len(anchors)-1 or p-pre<0 or p+post+1>len(morph)",
           "pre=min(0.28,0.40×median RR) s; post=min(0.65,0.78×median RR) s (làm tròn mẫu)",
           "Loại nhịp đầu/cuối hoặc cửa sổ beat vượt tín hiệu.", "Nhịp ở mép có thể vẫn có QRS rõ.", "reason='edge'")
    reject("rr_outlier", "NOT(0.70*median_RR <= RR_prev <= 1.35*median_RR AND 0.70*median_RR <= RR_next <= 1.35*median_RR)",
           "Cả RR trước và sau phải nằm trong [0.70,1.35] × median RR",
           "Loại nhịp có RR lân cận ngoài khoảng; xét sau edge.", "Nhịp bệnh lý thật có thể bị loại.", "reason='rr_outlier'")
    reject("adjacent_qrs_in_window", "RR_prev < pre_s+0.10 or RR_next < post_s+0.08 (s)",
           "RR_prev < pre_s + 100 ms hoặc RR_next < post_s + 80 ms",
           "Loại cửa sổ có nguy cơ lẫn hoạt động nhịp kế cận.", "Không đồng nghĩa detector nhận sai nhịp.", "reason='adjacent_qrs_in_window'")
    reject("invalid_data", "quality['invalid_mask'][lo:hi].any()", "Một mẫu invalid trong beat",
           "Loại beat có dữ liệu không hữu hạn.", "Thường bị load chặn trước; không phải lỗi biên.", "reason='invalid_data'")
    reject("near_adc_rail", "quality['saturation_mask'][lo:hi].any()",
           f"Mask: abs(axis)>={cfg.input_saturation_abs_mV:g} mV liên tục >= {cfg.input_saturation_min_ms:g} ms",
           "Loại beat giao đoạn gần rail.", "Giới hạn rail cần đối chiếu dữ liệu gốc.", "reason='near_adc_rail'")
    reject("flat_data_over_qrs", "mean(dropout_mask[max(lo,p-ms_samples(80)):min(hi,p+ms_samples(80))]) > 0.5",
           f">50% cửa sổ ±80 ms nằm trong flat run >= {cfg.input_flat_min_s:g} s",
           "Loại beat có quá nhiều dữ liệu phẳng quanh anchor.", "Phụ thuộc anchor đã tìm; không định vị biên QRS.", "reason='flat_data_over_qrs'")
    reject("isolated_impulse_away_from_qrs", "any(abs(impulse_sample-p) > ms_samples(40,fs)) for impulses in [lo,hi)",
           "Xung trong beat cách anchor > 40 ms",
           "Loại beat có xung xa QRS; xung gần anchor chỉ cảnh báo.", "Thành phần sắc thật ngoài vùng ±40 ms có thể bị loại.", "reason='isolated_impulse_away_from_qrs'")

    add("pq_not_quiet_review", "v1_representative",
        "any included beat: pq_mean_speed > baseline_speed_fraction*max(QRS_speed_max,1e-12) OR pq_radius95 > baseline_radius_fraction*max(QRS_magnitude_max,1e-12)",
        f"PQ 25 ms: speed > {cfg.baseline_speed_fraction:g} × QRS speed hoặc radius95 > {cfg.baseline_radius_fraction:g} × QRS magnitude; QRS đo trong anchor[-120,+160] ms",
        "Cửa sổ PQ phẳng nhất vẫn có vận tốc hoặc độ phân tán tương đối cao; giữ beat.",
        "PQ có hoạt động sinh lý/baseline khác không tự chứng minh biên sai.")
    add("sharp_qrs_or_impulse_review", "v1_representative",
        "any included beat: spike_mask[max(lo,p-ms_samples(40)):min(hi,p+ms_samples(40))].any()",
        "Xung trong vùng ±40 ms quanh anchor (cận phải exclusive)",
        "Giữ beat có thành phần nhọn gần QRS và cảnh báo.", "QRS nhọn thật và xung nhiễu chưa được phân biệt.")
    add("motion_or_ST_review", "v1_representative",
        "any included beat: motion_suspect_mask[lo:hi].any()",
        f"Giao mask jump tâm 1 s > max({cfg.input_motion_center_step_mV:g} mV, median+{cfg.input_motion_mad_factor:g}×MAD chuẩn hóa)",
        "Giữ beat giao vùng biến thiên tâm; không mặc định là motion.", "ST/T bệnh lý thật có thể kích hoạt.")
    add("different_morphology_retained_review", "v1_representative",
        "any included beat: normalized dot correlation with first median template in [-120,+160] ms < min_template_correlation",
        f"correlation < {cfg.min_template_correlation:g}; flatten XYZ rồi trừ mean toàn vector",
        "Biến thể hình dạng được giữ, không loại chỉ vì tương quan thấp.",
        "Biến thể bệnh lý thật hoặc median gộp nhiều hình thái; chưa chứng minh biên sai.")

    qth = f"qthreshold=max(median(speed_q[PQ])+4×robust_sigma(speed_q[PQ]),{cfg.qrs_fraction:g}×speed_q[qpeak]); quiet={cfg.quiet_qrs_ms:g} ms"
    add("qrs_on_not_found", "v1_qrs_delineation", "q_on is None: quiet_boundary quét trái không đủ run mẫu <= qthreshold",
        qth, "Không tìm được đầu QRS theo đoạn yên trước cụm vận tốc.",
        "Có thể do baseline/search window; chưa có tham chiếu để biết biên thật.")
    add("qrs_off_not_found", "v1_qrs_delineation", "q_off is None: quiet_boundary quét phải không đủ run mẫu <= qthreshold",
        qth, "Không tìm được cuối QRS theo đoạn yên sau cụm vận tốc.",
        "QRS rộng, nhiễu hoặc ST hoạt động đều có thể ảnh hưởng.")
    add("qrs_duration_or_anchor_requires_review", "v1_qrs_delineation",
        "both QRS edges exist AND (NOT(q_on < anchor < q_off) OR NOT(40 <= (q_off-q_on)*1000/fs <= 240))",
        "Duration ngoài [40,240] ms hoặc anchor không nằm nghiêm bên trong",
        "Vi phạm khoảng kiểm tra hoặc thứ tự anchor; không tự sửa biên.", "QRS rộng bệnh lý thật có thể vượt 240 ms.")
    add("pq_overlaps_qrs", "v1_qrs_delineation", "both QRS edges exist AND q_on <= pq[1]",
        "q_on <= PQ stop (exclusive index)", "Mốc đầu QRS chạm/vượt đầu mút chỉ số cửa sổ PQ.",
        "PQ stop là cận exclusive; điều kiện <= bao gồm cả sát nhau một mẫu.")
    add("qrs_near_search_limit", "v1_qrs_delineation",
        "both QRS edges exist AND ((q_on-qlo)<ms_samples(8,fs) OR (qhi-q_off)<ms_samples(8,fs))",
        f"Cách mép <8 ms; search={tuple(cfg.qrs_search_ms)} ms quanh anchor, cắt theo mảng",
        "Biên QRS gần giới hạn tìm kiếm.", "QRS thật có thể nằm sát cửa sổ; không đủ để gắn nhãn sai.")

    add("t_search_too_short", "v1_t_delineation",
        "both QRS edges exist AND thi-tlo < ms_samples(100,fs); tlo=q_off+ms_samples(20); thi=min(n-1,anchor+int(min(0.60,0.70*median_RR)*fs))",
        "Vùng tìm T ngắn hơn 100 ms", "Dừng tìm T do thiếu cửa sổ đủ dài.", "RR ngắn/QRS dài thật có thể làm cửa sổ hẹp.")
    add("t_peak_not_resolved", "v1_t_delineation",
        "T search long enough AND no peak with ps>=q_off+ms_samples(50) AND ps<=thi-ms_samples(30)",
        "Peak prominence >= max(4×mnoise,0.015×QRS magnitude max); distance>=40 ms; margins 50/30 ms",
        "Không có đỉnh T thỏa prominence và vùng hợp lệ.", "T phẳng, T muộn hoặc hình thái khác có thể là tín hiệu thật.")
    add("t_low_amplitude", "v1_t_delineation", "resolved peak AND amp=mag_t[t_peak]-mbase <= 5*mnoise",
        "amp <= 5×max(robust_sigma(mag_t[PQ]),1e-6 mV)",
        "Biên độ T nhỏ tương đối với biến thiên magnitude PQ.", "T nhỏ thật không đồng nghĩa sai biên.")
    add("multiple_t_lobes_requires_review", "v1_t_delineation",
        "len(ps[(abs(ps-t_peak)<=ms_samples(100,fs)) & (prominences>=0.35*prominence_best)]) > 1; valid mask is NOT reapplied",
        "Ít nhất 2 peaks cách peak chính <=100 ms và prominence >=35% peak chính",
        "Magnitude có nhiều lobe được gộp; cần kiểm tra trên XYZ có dấu.",
        "T nhiều pha thật, hoặc peak gần ngoài valid mask; không tự là lỗi.")
    add("tp_not_quiet", "v1_t_delineation",
        "TP window exists AND mean(speed_q[tp[0]:tp[1]]) > 0.10*speed_q[qpeak]",
        "Vận tốc trung bình TP 25 ms >10% vận tốc đỉnh QRS",
        "Đoạn TP ít vận tốc nhất sau last_peak+100 ms vẫn hoạt động.", "Đuôi T hoặc nhịp nhanh làm TP không phẳng; chưa phân biệt drift.")
    add("tp_window_too_short", "v1_t_delineation",
        "resolved peak AND thi-(last_peak+ms_samples(100,fs)) < ms_samples(25,fs)",
        "Không đủ TP 25 ms sau last_peak+100 ms; fallback baseline PQ",
        "Thiếu cửa sổ TP để làm tham chiếu T-end.", "Thiếu vùng TP không chứng minh tiếp tuyến T-end sai.")
    add("t_on_tangent_unresolved", "v1_t_delineation",
        "tangent_on is None: NOT(dm[rising]>1e-9 AND tlo<=round(rising+(st-mag_t[rising])/dm[rising]*fs)<first_peak)",
        "Slope >1e-9 mV/s và giao điểm làm tròn trong [tlo,first_peak)",
        "Không có giao tiếp tuyến T-on hợp lệ; dùng threshold_on nếu có.", "Fallback có thể vẫn hợp lý; không phải nhãn biên sai.")
    add("t_off_tangent_unresolved", "v1_t_delineation",
        "tangent_off is None: NOT(dm[end_slope]<-1e-9 AND last_peak<round(end_slope+(t_end_baseline-mag_t[end_slope])/dm[end_slope]*fs)<thi)",
        "Slope <-1e-9 mV/s; giao điểm làm tròn trong (last_peak,thi)",
        "Không có giao tiếp tuyến T-end hợp lệ; dùng threshold_off nếu có.", "Fallback hoặc đuôi T phẳng chưa chắc sai.")
    add("st_level_requires_review", "v1_t_delineation",
        "st-mbase > max(5*mnoise,0.15*amp); st=median(mag_t[tlo:min(first_peak,tlo+ms_samples(25,fs))]); mbase=median(mag_t[PQ]); amp=mag_t[t_peak]-mbase",
        "Chênh magnitude ST−PQ > max(5×mnoise,15% amp); chỉ một phía, không abs",
        "Mức magnitude đoạn ST đầu cửa sổ cao hơn magnitude PQ theo ngưỡng tương đối.",
        "Magnitude không phải điện thế ST có dấu; có thể do ST/T thật, vị trí cửa sổ hoặc baseline.")
    add("t_off_no_baseline_return", "v1_t_delineation",
        "threshold_off is None: no quiet run from last_peak to thi with mag_t<=t_end_threshold",
        f"run={cfg.quiet_t_ms:g} ms; t_end_threshold=TPbaseline+max(3×end_noise,{cfg.t_fraction:g}×max(Tpeak−TPbaseline,0))",
        "Phương pháp ngưỡng không tìm thấy đoạn trở về gần baseline; tiếp tuyến vẫn có thể cho T-off.",
        "Đuôi T/drift/baseline/search window đều có thể gây ra; không đồng nghĩa mất T-off.")
    add("t_end_methods_disagree", "v1_t_delineation",
        "threshold_off is not None AND tangent_off is not None AND abs(threshold_off-tangent_off)*1000/fs > 100",
        "Chênh T-off tiếp tuyến và ngưỡng >100 ms; đúng 100 ms không gắn cờ",
        "Hai định nghĩa thao tác T-end bất đồng lớn.", "Hai phương pháp cùng dùng mag_t/baseline V1; bất đồng không xác định phương pháp đúng.")
    add("t_boundaries_invalid", "v1_t_delineation",
        "t_on and t_off exist AND NOT(q_off<t_on<first_peak<=t_peak<=last_peak<t_off<n); then t_on=t_off=None",
        "Thứ tự chỉ số nghiêm/không nghiêm đúng như condition",
        "Biên không tương thích thứ tự lobe; bỏ cả hai biên T.", "Đây là lỗi tính nhất quán nội bộ, chưa đo sai số so tham chiếu.",
        basis="Ràng buộc thứ tự nội bộ V1; không phải tolerance tham chiếu.")
    add("t_duration_requires_review", "v1_t_delineation",
        "both T edges exist AND ordering valid AND NOT(60 <= (t_off-t_on)*1000/fs <= 400)",
        "Duration ngoài [60,400] ms; vẫn giữ biên",
        "Độ dài T ngoài khoảng kiểm tra V1.", "Hình thái bệnh lý thật hoặc định nghĩa T-on có thể tạo thời lượng khác.")
    add("t_near_search_limit", "v1_t_delineation",
        "entered both-T-edges branch AND t_off is not None AND thi-t_off < ms_samples(15,fs)",
        "T-off cách thi <15 ms", "T-off gần giới hạn tìm T.", "T muộn thật cũng có thể sát giới hạn.")
    add("t_boundary_not_resolved", "v1_t_delineation",
        "resolved peak branch AND (t_on is None OR t_off is None) BEFORE final ordering test",
        "Thiếu ít nhất một biên sau tiếp tuyến/fallback",
        "Chưa có đủ biên T từ phương pháp V1.", "Không được thêm cờ này vào nhánh t_peak_not_resolved bằng suy diễn.")

    powerline_condition = (
        "powerline_status=='inconclusive'; aggregate: no channel noise_detected AND at least one inconclusive. "
        "A channel inconclusive when (globally_high AND persistence<persistence_threshold) OR "
        "(globally_high AND insufficient_quiet_evidence) OR "
        "(NOT globally_high AND finite(persistence) AND persistence>=inconsistent_persistence). "
        "All channels inconclusive also when PSD narrow bins<3 or background bins<4, or duration<minimum. "
        "globally_high = global_score>=score_threshold AND concentration>=peak_concentration. "
        "quiet sufficiency requires n_runs>=quiet_min_runs AND total>=quiet_min_total_s "
        "AND span>=quiet_min_span_s AND every harmonic-fit condition<=quiet_fit_condition_limit; "
        "quiet_positive requires sufficiency AND quiet_score>=quiet_line_score_db "
        "AND explained_fraction>=quiet_line_explained_fraction."
    )
    add("powerline_inconclusive_review", "v1_process", powerline_condition,
        f"PSD score>={cfg.powerline_score_threshold_db:g} dB; concentration>={cfg.powerline_peak_concentration:g}; persistence>={cfg.powerline_persistence_threshold:g}; inconsistent>={cfg.powerline_inconsistent_persistence:g}; minimum_duration={cfg.powerline_min_duration_s:g} s; "
        f"quiet: >= {cfg.quiet_min_runs} runs, mỗi run >= {cfg.quiet_min_run_s:g} s, total >= {cfg.quiet_min_total_s:g} s, span >= {cfg.quiet_min_span_s:g} s, condition <= {cfg.quiet_fit_condition_limit:g}; "
        f"quiet positive: score >= {cfg.quiet_line_score_db:g} dB, explained fraction >= {cfg.quiet_line_explained_fraction:g}; "
        f"quiet mask: magnitude/speed cùng <= quantile {cfg.quiet_quantile:g}, lowpass {cfg.quiet_lowpass_hz:g} Hz, smooth {cfg.quiet_activity_smooth_ms:g} ms, QRS exclusion {tuple(cfg.quiet_qrs_exclusion_ms)} ms, bỏ mép {cfg.psd_window_s/4:g} s",
        "Chưa đủ bằng chứng kết luận nhiễu điện lưới; auto notch giữ trục inconclusive.",
        "Có thể thiếu vùng yên vì tim/ST/T; không tự chứng minh phân đoạn sai.")
    add("few_representative_beats", "v1_process", "representative_beat succeeded AND rep['n_used'] < 5",
        "3 hoặc 4 beats được giữ (dưới 3 thì hàm đã raise)", "Median dựa trên ít nhịp.", "Ít nhịp chưa xác nhận biên sai.")
    add("many_excluded_beats", "v1_process", "rep['n_excluded']/len(detector['anchors']) > 0.4",
        ">40% anchors bị loại, bao gồm edge", "Tỷ lệ loại nhịp cao.", "Tính cả nhịp đầu/cuối; loạn nhịp thật cũng làm tăng tỷ lệ.")
    add("features_incomplete", "v1_process",
        "NOT isfinite(list(features.values())).all(); batch failure additionally appends this even when feature stage not reached",
        f"Thiếu ít nhất 1/12 features; loop phải >=max(5,ms_samples({cfg.derivative_window_ms:g},fs)) mẫu",
        "Không có đủ 12 giá trị hữu hạn; phải truy nguyên giai đoạn, không mặc định lỗi delineation.",
        "Đây là thiếu khả dụng; không đo accuracy hoặc nhãn bệnh.",
        basis="Ràng buộc hữu hạn/độ dài đủ cho đạo hàm V1; không phải ngưỡng correctness.")

    add("optional_highpass_may_change_ST_T", "v1_preprocessing_diagnostic",
        "cfg.morphology_highpass_hz is not None; only .5 Hz allowed else raises",
        f"morphology_highpass_hz={cfg.morphology_highpass_hz!r}; mặc định None (không bật)",
        "Cảnh báo nhánh thử nghiệm high-pass có thể làm đổi ST/T; không tự nhập result flags.",
        "Chỉ báo cấu hình, không phải lỗi biên.", PREPROCESSING, contributes=False)
    add("preliminary_detector: <error>", "v1_preprocessing_diagnostic",
        "preliminary_detector is not None AND it raises ValueError; exception text appended to warnings",
        "Không có ngưỡng riêng; giữ nội dung ValueError",
        "Lỗi detector sơ bộ dùng tìm vùng yên; detector chính chạy riêng sau đó.",
        "Không đồng nghĩa detector chính thất bại.", PREPROCESSING,
        snippet="warnings.append('preliminary_detector: '+preliminary_error)", contributes=False)
    add("processing_failed", "batch_failure", "describe_result called with result is None",
        "Ngoại lệ ở bất kỳ stage; qc_status='failed', 0/12 features trong row batch",
        "Flag do wrapper batch thêm; không phải flag do process_record trả về.",
        "Phải đọc failure_stage/error; có thể lỗi kỹ thuật hoặc không đủ beat.", BATCH,
        snippet="['processing_failed', 'features_incomplete']", contributes=False,
        basis="Trạng thái thực thi batch; không phải heuristic accuracy.")
    add("eval_t_on_methods_gap_gt100ms", "batch_evaluation",
        "boundaries and rep exist AND abs(t_tangent_on_ms-t_threshold_on_ms) > 100",
        ">100 ms; NaN không vượt ngưỡng", "Bất đồng T-on, chỉ ở evaluation_flags, không đổi QC V1.",
        "Hai phương pháp không độc lập; chưa biết bên nào gần biên thật.", BATCH,
        snippet="extra.append('eval_t_on_methods_gap_gt100ms')", contributes=False)
    add("eval_short_rr_lt0.6median_suspected_extra_or_irregular", "batch_evaluation",
        "boundaries and rep exist AND suspected_extra_rr_count=int((RR < .6*median(RR)).sum()) is nonzero",
        "Có RR <0.6×median RR; chỉ xuất extra flags trong nhánh có rep/boundaries",
        "Nghi ngờ nhịp thừa hoặc RR không đều; chỉ ở evaluation_flags.",
        "Ngoại tâm thu/nhịp không đều thật có thể làm RR ngắn.", BATCH,
        snippet="extra.append('eval_short_rr_lt0.6median_suspected_extra_or_irregular')", contributes=False)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.qc_flag.duplicated().any():
        raise AssertionError("QC rule names must be unique")
    return frame


def source_audit_findings():
    """Verified source behavior/caveats; no claim of cohort error or accuracy."""
    findings = []
    def finding(key, description, implication, filename, snippet):
        findings.append(dict(finding_id=key, description=description,
                             implication=implication, source_file=filename,
                             source_line=_line(filename, snippet),
                             evidence_scope="Mã nguồn xác nhận; không tự chứng minh lỗi trên cohort."))
    finding("status_is_any_merged_flag",
            "Successful result: flags = detector.flags + boundaries.flags + rep.quality_flags + process flags; deduplicate. Any flag => needs_review. eligible_for_features = not flags.",
            "Có đủ 12 features vẫn có thể eligible=False. Không có cờ không phải xác nhận biên đúng. Nhiều flags đồng thời do nối list.",
            PIPELINE, "eligible_for_features':not flags")
    finding("input_flags_are_separate",
            "Input-quality flags và toàn bộ preprocessing.warnings không được nhập trực tiếp vào process result flags; powerline inconclusive được thêm riêng. Các masks vẫn dùng để giữ/loại beat.",
            "input_qc_flags và qc_flags là hai phạm vi khác nhau; không cộng chúng rồi gọi là V1 needs_review.",
            PIPELINE, "flags=list(dict.fromkeys(detector['flags']+boundaries['flags']+rep['quality_flags']))")
    finding("st_is_unsigned_magnitude",
            "ST và PQ reference trong cờ ST là median magnitude XYZ đã làm mượt, không phải ST điện thế có dấu; trigger là st-mbase > threshold, không abs.",
            "Không diễn giải cờ thành ST elevation/depression lâm sàng hoặc MI; tần suất cao chưa chứng minh QC quá nhạy.",
            PIPELINE, "if st-mbase > max(5*mnoise, 0.15*amp):")
    finding("significant_lobes_do_not_reapply_valid_mask",
            "Peak chính chọn trong valid mask, nhưng significant lấy từ mọi ps trong ±100 ms với prominence>=35%; không áp dụng valid mask lần nữa.",
            "Có khả năng lobe ngoài valid region ảnh hưởng first/last_peak và tiếp tuyến; phải đo từng record trước khi khẳng định xảy ra lỗi.",
            PIPELINE, "significant = ps[(np.abs(ps-t_peak) <= ms_samples(100,fs)) &")
    finding("quiet_boundary_returns_inner_quiet_edge",
            "quiet_boundary trả i+run-1 khi quét trái và i-run+1 khi quét phải; đây là mẫu đầu đoạn quiet theo hướng từ peak ra ngoài, không phải điểm cuối run.",
            "Phải tái tạo đúng chỉ số và run; không dùng độ dài run như độ lệch biên cần hiệu chỉnh.",
            PIPELINE, "return i + run - 1 if direction < 0 else i - run + 1")
    finding("t_on_st_window_is_algorithm_dependent",
            "ST level đo từ tlo=QRSoff+20 ms đến min(first_peak,tlo+25 ms); rising là max gradient từ tlo đến first_peak inclusive.",
            "Baseline T-on và vị trí tiếp tuyến phụ thuộc QRS-off và lobe V1; cần đánh giá độc lập thay vì dùng lại như ground truth.",
            PIPELINE, "st = float(np.median(mag_t[tlo:min(first_peak,tlo+ms_samples(25,fs))]))")
    finding("t_end_methods_share_inputs",
            "Tiếp tuyến và threshold T-end cùng dùng mag_t và TP baseline (PQ nếu thiếu TP). T-off ưu tiên tiếp tuyến; no_baseline_return chỉ mô tả threshold.",
            "Không gọi đồng thuận hai phương pháp là accuracy; bất đồng cũng không chỉ ra biên nào sai.",
            PIPELINE, "t_off = tangent_off if tangent_off is not None else threshold_off")
    finding("invalid_boundaries_branch_is_distinct",
            "t_boundaries_invalid đặt t_on=t_off=None trong nhánh đã có cả hai biên; nhánh else t_boundary_not_resolved không chạy lại. Thiếu peak/search quá ngắn cũng không thêm cờ t_boundary_not_resolved.",
            "Không suy diễn cờ từ tên; dùng tất cả biên và flags thực tế khi truy nguyên 7 records thiếu features.",
            PIPELINE, "t_on = t_off = None")
    finding("failures_are_batch_status_not_v1_review",
            "Batch failure: qc_status='failed'; flags giữ detector nếu có rồi thêm processing_failed và features_incomplete. Không giữ toàn bộ rep/boundary flags đã tính trước lỗi.",
            "Phải tách failed khỏi needs_review và auto_candidate; không gọi mọi record không needs_review là không có cờ. Các artifact stage cần dùng cho chẩn đoán lỗi.",
            BATCH, "row['qc_flags'] = flags_string((detector['flags'] if detector else []) + ['processing_failed', 'features_incomplete'])")
    finding("time_reference_is_median_anchor",
            "qc_row và describe_result lấy time_ms từ rep.time_ms=(sample-rep.anchor)*1000/fs; detector anchors là mẫu tuyệt đối của raw recording.",
            "Không so annotation raw ms trực tiếp với CSV median-relative ms; cần anchor mapping/beat provenance phù hợp.",
            PIPELINE, "'time_ms':(np.arange(len(rep))-pre)*1000/fs")
    finding("feature_velocity_uses_full_beat_derivative",
            "process_record tính spatial_speed trên cả median beat trong delineate_loops, rồi cắt speed cùng biên để đưa extract_features. Khi speeds=None, extract_features tự đạo hàm trên loop đã cắt.",
            "Sensitivity phải giữ công thức V1: cắt lại speed từ cả beat, tránh đổi điều kiện biên Savitzky–Golay ngoài ý muốn.",
            PIPELINE, "speeds = {name: None if loop is None else boundaries['speed'][boundaries[a]:boundaries[z]+1]")
    finding("baseline_helper_uses_default_derivative_config",
            "flattest_pq gọi spatial_speed(xyz,fs) mà không truyền cfg; ở V1 mặc định trùng 11 ms nhưng cfg.derivative_window_ms tùy chỉnh không đi vào helper này.",
            "Giữ config V1; nếu nghiên cứu cfg khác sau audit cần ghi nhận điểm không nhất quán này.",
            PIPELINE, "speed = spatial_speed(xyz, fs)")
    finding("no_flag_threshold_citations_in_implementation",
            "Module preprocessing tuyên bố thresholds là research heuristics; không có trích dẫn tài liệu cho các ngưỡng QC cụ thể trong các module đã đọc.",
            "Không gán cơ sở lâm sàng cho các giá trị 7%, 8%, 15%, 100 ms hay khoảng duration chỉ vì tên flag.",
            PREPROCESSING, '"""Conditional Frank VCG preprocessing. Thresholds are research heuristics.')
    return findings


if __name__ == "__main__":
    print(qc_rules_inventory().to_string(index=False))
