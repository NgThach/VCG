# VCG features đã trích xuất — đọc trực tiếp, không cần dataset gốc

Đây là các bảng số để phân tích/huấn luyện từ features có sẵn. Cả sáu CSV tổng cộng
308.976 byte (khoảng 309 KB), được lưu trực tiếp trong Git và có sẵn khi clone repo.
Không cần chạy `project_setup.py`, tải PTB, giải nén waveform hoặc kết nối Drive.

## Chọn bộ feature

- **`v1/`:** kết quả hiện tại của Notebook06, 7 feature QRS và 5 feature T theo V1.
  Có 448 dòng, 441 dòng đủ 12 feature, 7 dòng còn thiếu.
- **`v2/`:** giữ QRS V1, tính lại 5 feature T bằng biên Wavelet/TLS.
  Có 448 dòng, 312 dòng đủ 12 feature, 136 dòng còn thiếu.
  V2 là thí nghiệm; chưa có reference độc lập để kết luận chính xác hơn V1.

Mỗi thư mục có `train_features.csv` (263 dòng), `valid_features.csv` (70 dòng),
`test_features.csv` (115 dòng). Hai bộ giữ nguyên ID, patient, nhãn và split.
Nhãn: **HC=0, MI=1**. Không có bệnh nhân nằm trong nhiều tập.

## Đọc bằng Python

Từ root repo, chỉ cần pandas:

```python
import pandas as pd

version = "v1"  # đổi thành "v2" để lấy T features theo Wavelet/TLS
train = pd.read_csv(f"data/features/{version}/train_features.csv")
valid = pd.read_csv(f"data/features/{version}/valid_features.csv")
test = pd.read_csv(f"data/features/{version}/test_features.csv")
```

Nếu chưa clone repo, thay `data/features` bằng:

```text
https://raw.githubusercontent.com/NgThach/VCG/main/data/features
```

Notebook chỉ đọc features và tạo X/y có thể mở bằng
[Google Colab](https://colab.research.google.com/github/NgThach/VCG/blob/main/examples/Read_VCG_Features.ipynb)
hoặc từ [`examples/Read_VCG_Features.ipynb`](../../examples/Read_VCG_Features.ipynb).

## Các cột

- `patient_id`, `record_id`, `split`, `label_name`, `label`: nhận diện, chia tập và nhãn.
- `feature_version`: V1 hay V2.
- 12 cột số: `QRS_Arc`, `QRS_Area`, `QRS_MaxVec`, `QRS_MaxCentroidDistance`,
  `QRS_MaxVel`, `QRS_MeanVel`, `QRS_StdVel`, `T_MaxVec`, `T_Area`, `T_MaxVel`,
  `T_MeanVel`, `T_StdVel`.
- `feature_completeness`, `feature_complete`: số feature hữu hạn và có đủ 12 hay chưa.
- V1 giữ `qc_status`, `eligible_for_features`, `qc_flags`, `feature_missing_reasons`.
- V2 giữ `status`, `failure_reason`, `review_flags`, `config_id` và cờ QC V1 để đối chiếu.

Ô trống được pandas đọc thành NaN. Không xoá dòng thiếu, không điền số 0 và không
đổi cờ QC khi xuất. Đủ 12 feature không đồng nghĩa được QC duyệt hoặc biên đúng.

## Nguồn và phạm vi sử dụng

Snapshot ngày 21/10/2026, sao chép nguyên giá trị 12 feature từ:

- V1: `outputs/vcg_12_features_train_valid_test/{train,valid,test}_features.csv`.
- V2: `outputs/vcg_t_almeida_v2_experiment/v2_t_boundaries.csv`.

`metadata.json` ghi nguồn, checksum, tên cột và số dòng đủ/thiếu của từng tập.
File `vcg_12_features.csv` ở root là **bảng legacy 446 dòng**, không phải bộ V1 hiện tại.

Notebook01–09 phục vụ trích xuất/kiểm tra/vẽ tín hiệu. Để tách hoặc vẽ lại QRS/T loops
vẫn cần XYZ/median; không thể dựng lại hình dạng loop chỉ từ 12 giá trị feature.

Nguồn dataset: [PTB Diagnostic ECG Database 1.0.0](https://physionet.org/content/ptbdb/1.0.0/),
giấy phép Open Data Commons Attribution License v1.0; xem
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) để biết trích dẫn và giấy phép.
