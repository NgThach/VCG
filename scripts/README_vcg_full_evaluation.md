# Chạy đánh giá PTB VCG baseline V1

Chạy từ thư mục project bằng Python của kernel `ipcv` (đã có numpy, scipy,
pandas, matplotlib, scikit-learn, wfdb và tqdm). Không huấn luyện classifier.
File notebook có hậu tố `(2)` không có trong workspace; baseline là notebook 05
hiện có và các module mà notebook đó import, được sao lưu kèm SHA-256.

```sh
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python scripts/run_full_vcg_evaluation.py --workers 4
```

Mặc định đọc PTB tại `content/ptbdb/physionet.org/files/ptbdb/1.0.0`, xử lý
30 giây đầu mỗi recording HC/MI, ghi vào `outputs/vcg_full_evaluation`.
Script từ chối ghi vào thư mục đã có kết quả khi không có `--resume`.
Mỗi record có checkpoint nguyên tử, JSON chẩn đoán và NPZ các đường tính toán.

```sh
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python scripts/run_full_vcg_evaluation.py --resume --workers 4
```

Resume kiểm tra hash baseline, mã batch/report/comparison, dữ liệu tín hiệu/header,
split, cấu hình và phiên bản thư viện; record đã có checkpoint được dùng lại.
Nếu thay mã/config/data, hãy dùng `--output-dir` mới cho thí nghiệm mới.
`--records patient104/s0306lre patient001/s0010_re --output-dir outputs/smoke_moi`
cho phép smoke test; split vẫn được tạo trên toàn cohort trước khi chọn subset.

Split cũ notebook 02 là 75/25 theo bệnh nhân, seed 42. Lần chạy này tái lập
đúng tập test cũ và chia 20% bệnh nhân train cũ sang validation, thành 60/15/25
(120/30/50 bệnh nhân; 263/70/115 recordings). Hai record lỗi trong thí nghiệm
cũ vẫn được đưa vào theo patient ID, không loại dựa trên QC hoặc bệnh lý.

## Đọc kết quả

- `features/all_features.csv`: toàn bộ 448 recording được thử, kể cả NaN và QC.
- `evaluation/qc_by_label.csv`: bao phủ/QC theo nhãn và split, mẫu số rõ ràng.
- `evaluation/beat_quality_results.csv`: lý do giữ/loại từng nhịp, kể cả khi không
  đủ ba nhịp để tạo median beat. Quan sát biến cục bộ của hàm V1 khi lỗi chỉ lưu
  chẩn đoán đã tính; không thay đổi quy tắc giữ/loại hoặc giá trị đầu ra V1.
- `artifacts/`: anchors thô, anchors được giữ, median anchor, hệ thời gian,
  biên/candidate tiếp tuyến/ngưỡng, input QC, và lỗi nếu có.
- `index.html`: mở trực tiếp để lọc gallery theo nhãn/split/QC/loại cảnh báo.
- `full_evaluation_report.md`: phân tích HC/MI và trả lời tám câu hỏi.

`processing_success` nghĩa là chạy tới cuối pipeline; `extraction_success`
nghĩa là đủ 12 giá trị hữu hạn. Cặp biên có sẵn, thứ tự biên, giới hạn duration
và QC là các kiểm tra khác nhau. `auto_candidate` không xác nhận biên đúng;
`needs_review` vẫn giữ features. `eligible_for_features` giữ nguyên ý nghĩa V1.
Các cờ bắt đầu `eval_` chỉ bổ sung hướng dẫn review, không đổi trạng thái QC V1.

## Chú giải độc lập

Sau batch và trước khi chuyên gia nhập dữ liệu, có thể tạo bộ review không vẽ
biên dự đoán và bổ sung phụ lục chẩn đoán chi tiết:

```sh
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python scripts/prepare_vcg_blind_review.py
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python scripts/compare_vcg_annotations.py --output-dir outputs/vcg_full_evaluation --annotations outputs/vcg_full_evaluation/annotations/manual_review_template.csv
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python scripts/finalize_vcg_evaluation.py
```

Lệnh comparison trên template pending tạo bảng rỗng và audit, không tạo ground
truth. Script blind review từ chối ghi đè CSV đã có để bảo vệ chú giải chuyên gia.
`manual_review_blinded.csv` bỏ dự đoán, QC và nhãn bệnh khỏi lượt đọc đầu.
Một số ca không có median được ghi `no_median_beat`; cần review tín hiệu gốc và
detector riêng, không chú giải biên trên median không tồn tại.

PTB cục bộ không có ground truth tương ứng đầy đủ cho bốn biên hoặc mốc nhịp.
Challenge 2006 dùng cùng record nhưng QT lead II khác endpoint và nhịp đại diện;
không dùng các QT đó làm nhãn chuẩn cho bốn biên Frank XYZ.

`annotations/manual_review_template.csv` có ít nhất 30 recording mỗi nhãn,
ưu tiên nhiều bệnh nhân, phân tầng QC và bổ sung các loại cờ hiếm. Các cột tham
chiếu trống; `pred_*` là dự đoán, không là nhãn. Khi chú giải, ẩn dự đoán/cờ trong
lượt đầu, dùng chuyên gia độc lập và ghi giao thức/nguồn. Chỉ điền `confirmed`,
reviewer và `independent_reference=true` sau khi xác nhận độc lập.

```sh
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python scripts/compare_vcg_annotations.py --output-dir outputs/vcg_full_evaluation --annotations reviewed_boundaries.csv
```

`median_relative_ms` là thời gian so với anchor trên đúng median beat đã lưu.
`raw_recording_ms` cần `reference_anchor_sample` là một anchor thuộc used_samples;
script trừ thời gian anchor để đối chiếu với median và ghi rõ phạm vi khác nhịp.
Biên ngoài cửa sổ median/beat bị từ chối. Sai số được lưu theo từng biên, kèm
MAE và tỷ lệ trong 10/20/40/60 ms; các mức này là phân tích kỹ thuật.
Thiếu dự đoán vẫn được tính trong mẫu số coverage, không tự xem là khớp.

QRS event reference cần xác nhận đầy đủ mọi nhịp trong khoảng thời gian khai báo,
để TP/FP/FN có nghĩa. Xem docstring `compare_vcg_annotations.py` cho schema.

```sh
/Users/nguyenbaothach/miniconda3/envs/ipcv/bin/python -m unittest discover -s tests -p 'test_vcg*.py' -v
```
