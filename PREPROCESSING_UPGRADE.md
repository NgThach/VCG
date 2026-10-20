# Nâng cấp tiền xử lý Frank VCG ngày 10 tháng 10 năm 2026

Notebook 05 và notebook 04 đã dùng chung tiền xử lý có điều kiện. XYZ gốc được giữ nguyên; nhánh morphology dùng FIR theo cấu hình nghiên cứu, còn Notch chỉ tự bật trên trục có đủ bằng chứng. Bộ detector, PQ, phân đoạn, PCA và công thức 12 feature được giữ nguyên. QC beat được mở rộng để giữ các hình thái khác thường có thể liên quan MI.

## File và phạm vi thay đổi

| File | Thay đổi |
|---|---|
| `05_QRS_T_Loops_Step_by_Step.ipynb` | Import engine chung; cập nhật Bước 1A/2/2A/2B, Bước 6–8 và QC; bổ sung A/B/C, hai loại so sánh, kiểm thử, nguồn; chạy lại output |
| `04_VCG_Only_QRS_T_Delineation.ipynb` | Cùng Config/functions với 05; Markdown tiền xử lý/QC đồng bộ; batch mặc định tắt để smoke test nhanh; chạy lại output |
| `vcg_preprocessing.py` | Config, FIR, Welch, QC đầu vào, phân tích 50 Hz, quyết định auto/on/off, nhánh morphology/detection |
| `vcg_pipeline.py` | Engine chung trích từ 04; detector/PQ/delineation/PCA/features không đổi; representative QC, process/qc diagnostics được mở rộng |
| `vcg_preprocessing_evaluation.py` | Đánh giá biên cố định và toàn pipeline; PSD/đáp ứng/XYZ/loops/features |
| `tests/test_vcg_preprocessing.py` | 11 tests A–F, mode/per-axis, lỗi dữ liệu, morphology sạch khác thường, API legacy, smoke HC/MI |

Bản gốc được sao lưu trước sửa tại `backups/2026-10-10-before-preprocessing-upgrade/`, có cả 04 và 05. Không xóa hoặc ghi đè các bản gốc đó.

Vị trí trong 05 (đếm ô từ 1): ô 3 import engine; ô 8 QC/PSD đầu vào; ô 10 `preprocess_xyz(..., return_diagnostics=True)`; ô 12 đáp ứng FIR/Notch; ô 14 quyết định/diagnostics; ô 22 `representative_beat(..., quality=...)`; ô 59 so sánh A/B/C; ô 61 chạy tests; ô 65 hàm thí nghiệm cửa sổ với cấu hình legacy. Các hàm sửa trực tiếp là `Config`, `preprocess_xyz`, `representative_beat`, `process_record`, `qc_row`; các hàm FIR/powerline/evaluation mới nằm trong hai module chuyên trách.

Notebook 02/03, `pipeline_core.py`, `test_run.py`, `create_visualizations.py`, `render_vcg_comparison.py` còn dùng pipeline lịch sử (có SG dài và detector khác). Chưa migration chúng để không viết lại dự án/ML. Cần đồng bộ riêng trước khi trộn feature hoặc dùng hình của chúng như kết quả pipeline mới. `vcg_12_features.csv` cũ không được ghi đè.

## Bộ lọc cũ và mới

| Thành phần | Cũ | Mới |
|---|---|---|
| Morphology low-pass | Butterworth bậc 4, 80 Hz | FIR 401 hệ số, Kaiser β=8, cutoff thiết kế 175 Hz, filtfilt, fs=1000 Hz |
| Mục tiêu băng thông | Không có kiểm chứng đáp ứng riêng | Kiểm tra gain đến 150 Hz và suy giảm 190–200 Hz; không coi 175 Hz là mép giữ nguyên |
| Notch | 50 Hz, Q30 cố định | Auto/on/off; Q50 ban đầu; auto lọc riêng trục có bằng chứng |
| Bằng chứng 50 Hz | Không có | Welch toàn record/cửa sổ 4 s, peak concentration, persistence, kiểm tra coherent ở vùng ít hoạt động |
| Baseline | PQ mỗi beat | Giữ PQ, thêm QC độ yên/radius và PQ–TP offset để review |
| High-pass morphology | Không có | Không mặc định; 0,5 Hz chỉ tùy chọn có cảnh báo ST/T |
| Detection | Butterworth bậc 3, 5–30 Hz + RMS 80 ms | Giữ công thức/dải/tham số detector |
| Hình thái khác template | Loại dưới correlation 0,85 | Giữ nếu không có lỗi dữ liệu rõ, gắn cờ hình thái khác |

FIR làm giảm thành phần cao tần và tránh hạ cutoff morphology xuống 80 Hz. Notch giảm quanh 50 Hz nhưng cũng có thể giảm thành phần tim thật. Nhánh 5–30 Hz làm rõ hoạt động QRS để tìm mốc, không dùng đo loop. PQ trừ một vector hằng để đặt gốc XYZ, không ép đoạn ST/T về phẳng.

## Ý nghĩa và giới hạn của auto Notch

`P50` là mean PSD density của các bin 49,5–50,5 Hz; nền là median PSD density của các bin 45–48 và 52–55 Hz. R_dB dùng tỷ lệ hai mật độ cùng đơn vị mV²/Hz. Welch ở dữ liệu đủ dài có grid 0,25 Hz; không zero-pad rồi coi là tăng độ phân giải thực.

Ngưỡng 8 dB, persistence 60%, peak concentration 55%, các ngưỡng quiet và PQ QC đều là heuristic nghiên cứu. Quiet candidates phải ít magnitude và ít slope trong cửa sổ hoạt động đã làm trung bình, tránh vùng quanh mốc QRS; không coi mọi khoảng giữa nhịp là yên. Fit sin/cos giữ timestamp rời rạc và fit thêm tần số lân cận để giảm spectral leakage do gaps.

- `noise_detected`: đủ bằng chứng theo các heuristic; không có nghĩa xác nhận lâm sàng chắc chắn.
- `noise_not_detected`: chưa tìm thấy bằng chứng đáng kể về nhiễu liên tục.
- `inconclusive`: thiếu vùng quiet/thời lượng hoặc bằng chứng không nhất quán; auto giữ chưa Notch, gắn cờ.

Forced `on` là thí nghiệm, không là quyết định dựa vào QC. Với auto, trục không chắc chắn không bị Notch dù trục khác được lọc. Q50 và mọi ngưỡng chưa được tối ưu theo annotation bệnh nhân. Mô hình harmonic có thể bị ảnh hưởng bởi pha/tần số thay đổi, gaps hoặc residual cardiac activity; không tự điều chỉnh tần số trung tâm ngoài 50 Hz.

## Kiểm chứng đã chạy

Hai notebook chạy toàn bộ, output cũ được xóa trước khi chạy lại. Kiểm tra AST xác nhận `detect_qrs_anchors`, `flattest_pq`, `delineate_loops`, `project_pca`, `extract_features` không đổi. Thí nghiệm cửa sổ thời gian được giữ bộ lọc legacy riêng để giữ các so sánh cũ và hỗ trợ fs250/500/1000; default FIR mới báo lỗi nếu fs khác 1000 Hz.

11 tests qua, gồm:

- A: sạch → không auto Notch.
- B: sin 50 Hz liên tục → phát hiện và giảm sai lệch so với tín hiệu sạch biết trước.
- C: packet gần 50 Hz tại QRS, kể cả PSD/persistence vượt ngưỡng → không auto Notch nếu thiếu bằng chứng ở vùng ít hoạt động.
- D: sin không ổn định, dữ liệu ngắn hoặc thiếu quiet → inconclusive.
- E: hệ số FIR đối xứng, đáp ứng hai chiều giữ 0–150 Hz và suy giảm stopband; notch center/width thực.
- F: shape, finite, fs không phù hợp, padlen, NaN, không có beat; không sinh giá trị đo thay thế.
- Bổ sung: mode on/off, chỉ lọc X khi X có line; phát hiện impulse/dropout/rail; beat sạch khác hình thái vẫn được giữ; smoke trên HC và MI thật.

Đo từ đáp ứng hệ số: sai lệch gain hai chiều lớn nhất 0–150 Hz khoảng **0,000478 dB**; gain cao nhất trong 190–200 Hz khoảng **−177,68 dB**; độ rộng Notch hai chiều ở mức ≤−3 dB khoảng **1,53125 Hz**. Đây là đáp ứng bộ lọc lý tưởng tuyến tính ở phần giữa tín hiệu; không chứng minh sai khác waveform/feature cũng nhỏ như vậy, và không bỏ qua transient mép.

Lỗi gặp và xử lý: kiểm tra quiet ban đầu bị ripple 50 Hz còn lại trong nhánh QC cắt vụn các đoạn, làm test sin liên tục trả inconclusive. Đã tách nhánh QC low-pass 15 Hz khỏi morphology và làm trung bình hoạt động ngắn để đánh giá khoảng yên thay vì từng mẫu; không hạ ngưỡng PSD/persistence để ép test qua. Các tests được chạy lại. Các figure diagnostics ban đầu bị `plt.show` hiển thị lặp mọi figure đang mở; hàm chuyển sang yield từng figure và notebook được chạy lại.

## Kết quả PTB thật và hai cách so sánh

Thử 30 giây đầu HC `patient104/s0306lre` và MI `patient001/s0010_re`. Cấu hình B không tự bật Notch trên cả hai; bảng chẩn đoán từng trục và lý do được lưu trong notebook. Điều đó không chứng minh hai recording hoàn toàn không có nhiễu điện lưới; chỉ phản ánh tiêu chí auto hiện tại.

| Record | A: QRS / T (ms) | B: QRS / T (ms) | C: QRS / T (ms) |
|---|---|---|---|
| HC patient104 | −31…48 / 179…330 | −31…48 / 180…329 | −31…48 / 180…330 |
| MI patient001 | −65…74 / 170…335 | −66…71 / 169…339 | −66…69 / 167…338 |

Đây là biên tự động theo từng mốc căn nhịp của mỗi pipeline, chưa là annotation thật. Case MI vẫn có cờ ST cần review.

`fixed` giữ anchors, tập beat, PQ của từng beat, PQ median và bốn biên B. Đã kiểm tra tái tạo B đúng bằng B gốc; không báo phép so sánh fixed nếu tái tạo lệch. `full` chạy lại từng cấu hình. A tái hiện bộ lọc cũ nhưng dùng QC mới như B/C để việc so sánh không bị lẫn chính sách loại hình thái.

Trong phép fixed, B so với C: MaxVel T đổi khoảng +1,06% ở HC và +2,22% ở MI; QRS_Arc đổi khoảng −0,11% và −0,48%. Bảng đầy đủ 12 feature, absolute/relative change và cả full comparison nằm trong notebook và CSV. Những tỷ lệ này chỉ cho hai record, không phải chứng minh bảo toàn tối ưu hay giảm nhiễu đúng trên cohort. Phần trăm được để NaN khi mẫu số gần 0/thiếu dữ liệu.

Kết quả mới nằm trong `vcg_preprocessing_results/`. Nghiên cứu cửa sổ 80 ms vẫn dùng `vcg_window_validation/` và vẫn ghi rõ giới hạn tổng hợp; không dùng kết quả đó xác nhận tiền xử lý FIR mới.

## Cơ sở khoa học và phần còn cần kiểm chứng

[Kligfield et al. 2007](https://www.jacc.org/doi/10.1016/j.jacc.2007.01.024) là nguồn tham khảo về ECG chẩn đoán/băng thông, không xác nhận cấu hình FIR VCG này. [Kijonka et al. 2022](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.941827/full) và [Representative QRS loop, volume 2023](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1260074/full) có xử lý baseline/Notch và phương pháp khác. Chưa tái hiện toàn bộ thuật toán wavelet/FIR của họ.

Cấu hình FIR/Notch trong yêu cầu và QC do project đề xuất được kiểm tra bằng đáp ứng cùng tín hiệu có đáp án, nhưng chưa tối ưu trên cohort. Cần annotation độc lập, dữ liệu nhiễu nhiều mức/tần số/pha, QRS rộng, ST bất thường và nhiều bệnh nhân để xác định false notch/missed noise, tác động feature và selection bias. “Gần tín hiệu chưa lọc” và “PSD giảm” riêng lẻ đều chưa chứng minh đúng; không lấy hai ca làm kết luận MI/HC.

Chạy nhanh: mở 05 từ thư mục project, chọn Python (ipcv), restart kernel và Run All. Bộ tests: `python -m unittest discover -s tests -v` trong môi trường có NumPy/SciPy/WFDB/sklearn. Nếu cần chạy batch mới, bật RUN_BATCH trong 04; xem warnings/QC trước khi dùng feature. Không cần cài thư viện mới trong workspace hiện tại.

Batch mới của 04 ghi vào `vcg_preprocessing_batch_results/` để tách khỏi `vcg_v4_results/` lịch sử. Pipeline tiền xử lý mới chưa chạy lại toàn cohort để sản xuất bảng feature/ML mới; các record PTB trong thí nghiệm 80 ms vẫn dùng bộ lọc legacy được ghi rõ ở mục đó.
