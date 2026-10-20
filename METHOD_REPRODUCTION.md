# Notebook 09 — Truy vết phương pháp Wavelet Transform Loops + TLS

## Mức tái hiện được công bố

**`paper_equations_with_documented_project_adaptations`** — tái dựng các công thức có thể kiểm tra từ toàn văn, có tham khảo mã wavedet của tác giả và các điều chỉnh để chạy trên median V1. **Không gọi đây là faithful/bit-exact implementation của Almeida 2009 hoặc Martínez 2004.** Chưa có một fixture numerical hoặc bản chạy MATLAB của tác giả để chứng minh tương đương đầu ra.

Đọc trực tiếp phần Methods và Appendix, đồng thời render/quan sát các trang có công thức thay vì dựa vào PDF text mất ký hiệu. Hai PDF và bản text được lưu trong `outputs/vcg_t_almeida_v2_experiment/sources/`.

Nguồn chính:

1. R. Almeida, J. P. Martínez, A. P. Rocha, P. Laguna (2009), *Multilead ECG Delineation Using Spatially Projected Leads From Wavelet Transform Loops*, IEEE TBME 56(8), 1996–2005. [DOI](https://doi.org/10.1109/TBME.2009.2021658), [PDF tác giả](https://diec.unizar.es/~laguna/personal/publicaciones/MultileadDelineationECG.pdf).
2. J. P. Martínez, R. Almeida, S. Olmos, A. P. Rocha, P. Laguna (2004), *A Wavelet-Based ECG Delineator: Evaluation on Standard Databases*, IEEE TBME 51(4), 570–581. [DOI](https://doi.org/10.1109/TBME.2003.821031), [PDF tác giả](https://diec.unizar.es/~laguna/personal/publicaciones/wavedet_tbme04.pdf).
3. [ECG-kit source, commit `c8e3de47c54a9214138143676d2aa546b0540dd2`](https://github.com/marianux/ecg-kit/tree/c8e3de47c54a9214138143676d2aa546b0540dd2/common/wavedet). `qspfilt5.m`, `twavef.m`, `twave3D.m`, `searchon.m`, `searchoff.m`, `optimline.m`, `delineate3D.m` ghi tên J. P. Martínez/R. Almeida. Các bản này có cập nhật 2011–2012; chúng không phải bằng chứng mã xuất bản 2004/2009 giống từng chi tiết. Mã nguồn tham khảo được giữ cùng giấy phép GPL v2; các file Python mới của thí nghiệm được ghi `SPDX-License-Identifier: GPL-2.0-only`.

## Kiến trúc và dữ liệu V1 tái sử dụng

- Notebook 05 import `vcg_pipeline.py`, `vcg_preprocessing.py`, `vcg_preprocessing_evaluation.py`.
- Notebook 06 tạo các bảng `outputs/vcg_12_features_train_valid_test/{train,valid,test}_features.csv`, manifest và cấu hình V1.
- Notebook 07 dùng các module QC trong `scripts/`.
- Chỉ có **`08_QRS_T_Loop_Reference_Error_Analysis.ipynb`** trong repository khi bắt đầu nhiệm vụ; không có bản `(1)`. Notebook 08 dùng `scripts/vcg_reference_metrics.py`, `scripts/vcg_visual_review_data.py`, `scripts/vcg_visual_review_ui.py`, `scripts/vcg_visual_review_overview.py`, `scripts/vcg_boundary_reference_eval.py` và các module gallery/plots liên quan. Các tên không có prefix `scripts/` trong yêu cầu tương ứng với file ở thư mục này.
- Median thật nằm ở `outputs/vcg_full_evaluation/artifacts/{patient_record}.npz`, key `median_xyz`; `time_ms` và `speed` cũng được lưu. File JSON cùng stem có `fs`, `median_anchor_sample`, `rr_seconds`, `pq_indices`, các boundary sample-local và provenance tiền xử lý.
- `scripts/vcg_reference_metrics.load_median()` kiểm tra đúng ID, fs, shape, hữu hạn và lưới thời gian. Không tính lại recording/median.
- V1 dùng Frank `vx,vy,vz`, mV. Morphology là FIR 401 taps/cutoff thiết kế 175 Hz ở fs=1000 Hz; notch 50 Hz có điều kiện. Median hình thành sau chọn beat và trừ vector PQ ở từng beat, rồi trừ PQ trên median lần nữa. Không có high-pass morphology mặc định.
- Median là tín hiệu V1 đã xử lý, **không phải tín hiệu recording thô chưa lọc**. “XYZ gốc” trong Notebook 09 luôn có nghĩa chính mảng median đã dùng cắt loop V1, không đổi gốc hay lọc lại.
- V1 `spatial_speed()` tính đạo hàm Savitzky–Golay 11 ms, bậc 3 trên toàn median. Phần features mới dùng đúng mảng speed này rồi cắt. Không tính đạo hàm lại trên đoạn T riêng.
- Các số cohort/median/boundary được tính lại từ CSV và artifacts, không lấy số cố định trong yêu cầu làm kết quả.

Hệ thời gian và phép cắt:

\[
t_i=(i-\mathrm{anchor})1000/f_s,\qquad
T_{loop}=XYZ[T_{on}:T_{off}+1].
\]

Mọi chỉ số là **0-based, local trong median**, không phải sample của recording gốc. Duration bằng `(off-on)*1000/fs`; số mẫu loop bằng `off-on+1`. QRS-on/off, XYZ, split, labels, QRS loop và bảy QRS features được giữ nguyên.

## Bảng truy vết paper → hàm Python

Ký hiệu: **P** = công thức/quy tắc được ghi rõ trong paper; **R** = chi tiết từ mã tham khảo có tên tác giả; **A** = lựa chọn/điều chỉnh của dự án; **U** = chưa có bằng chứng tương đương toàn bộ bản gốc.

| Thành phần / nguồn | Hiện thực | Mức / kiểm chứng |
|---|---|---|
| Martínez II-A, Eq. (4): `Q_m(z)=G(z^(2^(m-1))) ∏ H(z^(2^l))` | `reference_equivalent_filter()` | P. Dilation bằng chèn zero; tích bằng convolution; test độc lập q1/q2/q3 và frequency response |
| Martínez II-B Eq. (6–7): h=[1,3,3,1]/8, g=2[1,-1] | `reference_equivalent_filter()` | P. Giữ đúng normalization; không dùng wavelet hai hệ số minh họa như toàn thuật toán |
| h support −2..1, g support −1..0; phase tuyến tính | `wavelet_transform()` + metadata bank | P/A. Dùng biểu diễn causal có cùng shape, bù delay tường minh; 1000 Hz có delay nguyên |
| À trous giữ temporal resolution | `wavelet_transform()` | P. Không decimate/resample XYZ; mọi scale trả N×3 |
| Martínez II-C: resample **equivalent filter**, giữ analog bands | `_adapted_filter()` | P về chiến lược; A về Kaiser polyphase beta5 và gain `/ (fs/250)`. Không tự nhận đúng MATLAB `interp` |
| Boundary handling của median ngắn | `wavelet_transform()` + flags | A. Reflect-padding; lưu support radius, gắn review nếu biên phụ thuộc border. Bản recording gốc dùng overlap-save, không được coi tương đương |
| Martínez II-D3: T search window phụ thuộc QRS/RR | `build_t_search_window()` | R/A. Chi tiết số lấy từ `twavef`/`twave3D`, QRS-off chung V1 và median RR là điều chỉnh |
| Begin≈QRS+100 ms, after QRS-off; end có giới hạn 600 ms/RR và next-QRS guard240 ms | `build_t_search_window()` | R/A. `end=min(0.6*max(1,RR), RR−0.240)` giây; chặn theo miền median. Không dùng T V1 |
| Martínez A.3: εT=0.25 RMS(W4) giữa hai QRS | `delineate_single_wavelet()` | P về hệ số; A về RMS trên phần post-anchor median có sẵn; fallback5/projected dùng RMS tại chính scale tương ứng, không claim exact cross-scale equivalence |
| Martínez II-D3: ít nhất hai extrema vượt ε để xem có T | `delineate_single_wavelet()` | P. Giữ điều kiện paper; mã `twavef` mới dùng OR trên hai dominant extrema, khác điều kiện này |
| Martínez A.7: γT=0.125 max modulus trong search window | `delineate_single_wavelet()` | P ở scale4; A khi dùng cùng quy tắc tại scale5. Threshold phát hiện, significance và boundary được lưu riêng |
| Cực trị giữ polarity | `modulus_extrema()` | P/R. `find_peaks` chỉ trích local extrema; toàn bộ logic T không phải một lời gọi chung `find_peaks` |
| Dominant slope và nearest opposite trước/sau | `delineate_single_wavelet()` | R. Đối chiếu `twavef`/`twave3D`; không lấy mọi lobe trong window làm một T |
| Opposite distance200 ms SL /150 ms projected; weak biphasic factor2.5 | `AlmeidaConfig`, `delineate_single_wavelet()` | R/U. Defaults của code 2011–2012; paper không công bố đủ chi tiết này |
| Positive, inverted, +/−, −/+ | `delineate_single_wavelet()` | P/R. Hai/ba extrema alternating signs; inversion-invariance và hai kiểu biphasic được test |
| Only upwards / downwards, thiếu cấu hình cực trị | `delineate_single_wavelet()` | P/A. Ghi hình thái, trả unresolved khi không đủ hai slopes/zero-crossing; không port đầy đủ recovery `picant` của author code |
| Martínez II-D3: peak ưu tiên zero crossing scale3, fallback detection scale | `first_zero_crossing()`, `delineate_single_wavelet()` | P/R. Chọn sample gần zero của first sign crossing; tie sớm. Quy tắc nhiều zero-crossings không bit-exact mọi morphology |
| Martínez II-D2/3, Almeida II-A: nearest threshold OR local minimum | `find_wave_boundary()` | P. Dò outward theo thứ tự thời gian; không tìm điểm thấp nhất toàn window |
| Martínez A.11: Ton=0.25×first slope | `delineate_single_wavelet(projected=False/True)` | P |
| Martínez A.12: SL Tend=0.4×last slope | `delineate_single_wavelet(projected=False)` | P |
| Almeida II-A footnote1: projected Tend=0.25×last slope | `delineate_single_wavelet(projected=True)` | P. Không nhầm 0.25 detection RMS với 0.25 boundary |
| Reference search radius120 ms | `find_wave_boundary()` calls | R; window-edge fallback trong author code bị bỏ có chủ ý, thay bằng unresolved (A) |
| Almeida II-C3: scale5 iff ≥2/3 SL dùng scale5 cho T detection | `select_multilead_scale()` | P. Phát hiện T với peak nhưng thiếu một SL boundary vẫn là wave detection; không đổi sang5 chỉ vì thiếu onset |
| Almeida II-C3: union các SL search windows, cùng initial W cho Ton/Toff | `delineate_t()` | P/A. Ba trục có QRS V1 chung nên search windows bằng nhau; không giả tạo ba cửa sổ khác nhau |
| Almeida Eq.(2): WT spatial loop | `wavelet_transform()` output `[Wx,Wy,Wz]` | P. Hướng tìm trên WT loop, không PCA trực tiếp XYZ |
| Almeida II-C1: TLS straight-line fit | `tls_direction()` | P/R. Affine fit có intercept như `optimline.m`; centered SVD giải toàn cục tiêu chí orthogonal TLS |
| Almeida Eq.(4): derived WT=W·u/‖u‖ | `project_wavelets()` | P. Không trừ intercept khỏi WT khi projection; chỉ centered trong bước fit |
| Eigenvector sign ambiguity | `tls_direction()` | A. Vòng đầu thành phần lớn nhất dương; vòng sau cùng chiều trước. Boundary dùng modulus, inversion test kiểm tra |
| Zero variance/equal leading singular values | `tls_direction()` | A. Unresolved cho hướng không xác định; không gọi “low SNR” là clinical noise đã xác nhận |
| Almeida II-C3: update Ton W=[oldTon−4σ, firstExt], Toff W=[lastExt,oldToff+4σ] | `iterate_tls_boundary()` | P. σ=15.3 ms; round widths nearest sample half-up. 61.2 ms là margin thời gian, không61.2 samples ở mọi fs |
| Almeida II-C1(d), II-C3: không cải thiện modulus/không có cực trị thì giữ vòng trước | `iterate_tls_boundary()` | P. Log cả candidate bị reject; không vô tình dùng direction của vòng bị reject |
| Almeida II-C3: hai boundary positions liên tiếp đổi **<2 samples** | `iterate_tls_boundary()` | P. Native fs; 2 samples không bị đổi thành2 ms hoặc2 vòng. Test riêng diff=1, diff=2 |
| Safeguard max_iterations20 | `iterate_tls_boundary()` | A. Flag riêng `max_iterations_reached_not_converged`; không nói đã hội tụ |
| Hai nhánh độc lập Ton/Toff | `delineate_t()` branches | P. Direction, stop reason, iterations và chosen iteration lưu độc lập |
| QRSoff<Ton<Toff và peak consistency | `validate_t_bounds()` | P/A. Joint consistency check là bảo vệ dự án; không dịch biên để sửa duration |
| T duration60–400 ms / TLS energy<0.6 | review flags | A. Chỉ yêu cầu xem lại; không dùng fit/duration để tối ưu hoặc ép kết quả, không phụ thuộc HC/MI |
| Loop cắt từ V1 median, inclusive | runner `feature_values()` | Theo yêu cầu dự án. Test array equality cả QRS và T |
| Feature velocity | runner `feature_values()` | Theo V1: full-median speed rồi slice, reuse `extract_features()`; assert parity của cả12 V1 features trước so sánh |
| References độc lập | runner `reference_evaluation()` | Tái dùng `validate_annotations()` của Notebook08; missing reference giữ NaN; V1 và V2 đều không được coi là ground truth |

## Cửa sổ, ngưỡng và trường hợp không giải được

Phân biệt ba nhóm ngưỡng: ε xác nhận có sóng, γ chọn slopes, ξ tìm boundary. Không dùng 15%/16% từ mô phỏng minh họa.

Mỗi trục bắt đầu ở scale4. Nếu không phát hiện được T (không đủ slopes/peaks), mới thử scale5. Nếu T được phát hiện nhưng SL-on/off chưa đủ, giữ scale detection và cho phép TLS thử tìm biên; điều này không dùng một mốc T V1 nào. Nếu cả ba trục đều không có T detection, `INITIALIZATION_FAILED`.

TLS window W dùng cho fit khác với fixed T search interval I dùng cho delineation. Các vòng sau cập nhật W theo paper; không thu hẹp I thành W rồi tự cắt mất lobe. Bản triển khai chỉ chiếu mảng median đang có, không dựng thêm sample thuộc RR kế tiếp. Update W có thể vượt initial I, nhưng luôn bị giới hạn bởi miền tín hiệu median; biên cuối vẫn phải thỏa next-QRS T search và QRS-off guard.

Các trạng thái V2:

- `detected`: có đủ biên và không có project review flag;
- `needs_review`: có biên nhưng có cảnh báo duration, border, fit hoặc safeguard;
- `unresolved`: thiếu biên, sai thứ tự/peak consistency, initialization/TLS failure hoặc missing V1 median.

Đây là trạng thái **của thuật toán**, không phải chuyên gia xác nhận đúng/sai. Candidate nội bộ có thể được giữ trong diagnostics khi final boundary là NaN. Không xuất `t_loop_v2` nếu final boundaries không hợp lệ.

## Những giới hạn chưa khắc phục

1. Paper không đủ để tái hiện mọi temporal guard, plateau/tie và morphology-recovery của single-lead. Reference code tìm được có cập nhật sau paper; một số default/nhánh khác mô tả paper. Các khác biệt quan trọng đã được chọn rõ, không trộn âm thầm.
2. Các MAT modern `dfilt.cascade` tải từ ECG-kit là MATLAB opaque objects trong SciPy. **Không sử dụng** chúng như thể đã giải mã đúng coefficients. Bank hiện dùng tích h/g có thể kiểm tra và kernel-resampling công khai.
3. Không có output numerical từ tác giả hoặc một MATLAB run tương ứng. Test bank/TLS/synthetic/integrity không chứng minh whole-algorithm equivalence.
4. Median V1 cắt ngắn hơn một RR đầy đủ. Do đó RMS giữa hai QRS không tái hiện chính xác được khi chỉ reuse median. Biến thể này dùng phần post-anchor có thật; lưu `rms_interval`, `rms_interval_complete_rr=False` và note trong mỗi diagnostic.
5. Frank axes được giả định orthogonal với noise không gian đồng nhất, như lập luận Almeida. Hướng TLS biến thiên lớn nhất không bảo đảm delineation tối ưu nếu noise/hình thái vi phạm giả định.
6. Border padding, RR-window adaptation và lựa chọn single-lead có thể làm thay đổi biên. Ca flat/monotonic/không đủ lobe được phép unresolved. Ca17 ms không được bắt buộc trở thành detected hoặc được kéo dài.
7. Reference trên beat ECG leadII hoặc dataset khác không tự chuyển thành ground truth bốn biên trên Frank median này. Khi chưa có annotation hợp lệ, MAE/RMSE không tính được và không kết luận V2 tốt hơn V1.

## Tính toàn vẹn và đánh giá

`protected_v1_hashes.json` chụp toàn bộ artifacts/bảng V1, các Notebook05–08 và module signal V1 trước thay đổi. Runner kiểm tra lại trước và sau chạy. `split_manifest.csv` của thí nghiệm là bản sao byte-identical của manifest nguồn. Mọi output mới nằm trong thư mục V2; checkpoint kiểm tra source row, config/engine và hash JSON/NPZ V1.

Hàm `delineate_t()` chỉ nhận XYZ, fs, anchor, QRS-on/off, RR và config; không nhận T-on/off V1, patient label hay split. Runner hoàn thành bước này trước khi đọc các mốc T V1 phục vụ so sánh. Cấu hình không chọn theo kết quả test; tham số khóa trước lần chạy cohort/test. Không huấn luyện ML.

Coverage, disagreement và accuracy được báo riêng. Bảng so sánh dùng record_id với one-to-one validation; giữ cả missing/failure. QRS features phải bằng V1; T features thay đổi không được mô tả là tốt hơn. Reference tùy chọn được validator hiện có kiểm tra identity, reviewer, provenance, độc lập, thời gian, signal và chất lượng trước khi dùng.

Các bước sau cần dữ liệu: chú giải độc lập, kiểm tra tác giả/MATLAB, sensitivity của RR/RMS/border trên train/valid, và đánh giá test sau khi khóa phiên bản. Không có bước nào trong Notebook09 thay thế V1 tự động.
