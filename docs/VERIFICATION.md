# Kiểm tra chạy lại — 21/10/2026

## Môi trường và dữ liệu

- Clone riêng tại `/private/tmp/vcg-clean-check`, tách khỏi workspace nghiên cứu.
- Python 3.11.15, venv mới, không bật system-site-packages; macOS ARM64.
- Node.js 25.6.1 có sẵn để chạy các test JavaScript (thiếu Node thì các test đó được skip).
- Cài trực tiếp `requirements.txt` thành công; `pip check`: không có dependency lỗi.
- Ban đầu chỉ có source và gói PTB đã kiểm checksum; không chép median, features,
  checkpoint, hình hoặc gallery cũ sang clone.
- Kiểm tất cả 549 `.hea`, 549 `.dat`, 549 `.xyz` với SHA-256 từ PhysioNet: khớp.
- 1.651 file cần phân phối được khóa bằng manifest riêng; không đưa HTML duyệt file
  và `.DS_Store` vào dataset.

## Chạy hết các code cell

Mỗi notebook dùng một kernel mới, đúng Python của venv. Dừng ngay khi có exception;
không bật allow_errors và không tắt cell chạy toàn bộ dữ liệu của Notebook02.

- Notebook01: **20/20** cell, 8 output PNG.
- Notebook02: **21/21** cell, 29 output PNG.
- Notebook03: **9/9** cell, 4 output PNG.
- Notebook04: **12/12** cell, 9 output PNG.
- Notebook05: **39/39** cell, 37 output PNG.
- Notebook06: **11/11** cell, đủ 448 dòng train/valid/test.
- Notebook07: **19/19** cell, 10 output PNG; QC, sensitivity và mẫu review được tạo mới.
- Notebook08: **29/29** cell, 135 output PNG; kiểm tra các manifest/gallery đủ 448 ID,
  gồm 75 trang T loops V1/V2 cho toàn cohort.
- Notebook09: **19/19** cell, 11 output PNG; các phép kiểm T-wave trong notebook pass.

Tổng: **179/179 code cell, 0 lỗi thực thi**. Nhật ký máy đọc được:
[`notebook-execution.json`](notebook-execution.json). Lượt chạy này dùng code notebook
tại commit `f8b5b8a`; các cập nhật sau đó sửa hướng dẫn kernel và cách tải gói chia phần.

## Kiểm tra số liệu và thuật toán

- Bốn module `vcg_pipeline.py`, `vcg_preprocessing.py`,
  `vcg_preprocessing_evaluation.py`, `vcg_t_almeida.py` giống từng byte với máy nghiên cứu.
- **448 × 12 feature V1 khớp** với kết quả gốc: `rtol=1e-10`, `atol=1e-12`, NaN khớp NaN.
- Danh sách chia train/val/test theo từng ID khớp kết quả gốc, không đổi phân chia bệnh nhân.
- V1: 443 median, 443 QRS loop, 441 dòng đủ feature T. Giữ nguyên 7 dòng thiếu feature.
- V2: chạy đủ 448 ID, 0 exception; 314 T-on, 418 T-off, 312 T loops hoàn chỉnh.
  Không biến các ca chưa tìm được biên thành ca thành công.
- **178 tests pass**, gồm phép kiểm chuẩn bị dữ liệu, checksum notebook theo source,
  và các bộ kiểm thử xử lý/metric/giao diện sẵn có. Fixture notebook cũ được đổi từ
  chuỗi giả `original` sang JSON notebook hợp lệ để kiểm đúng chính sách hash mới.

## Cách tự lặp lại

Đã kiểm tra thêm một clone **trực tiếp từ GitHub**, commit `22a8165`, tại
`/private/tmp/vcg-github-check`: tải 41 phần qua HTTPS công khai bằng
`python project_setup.py --verify`, tự ghép gói, giải nén và xác nhận SHA-256
của đủ **1.651 file**, thiếu 0, hỏng 0. Lượt này không chép gói dataset từ máy gốc.
Sau khi tải, chạy riêng Notebook08 khi chưa có `outputs/`: tự tạo đầu vào của
06–07, tính V2 và chạy hết **29/29 cell**, 0 lỗi, 135 output PNG, 75 trang T loops
bao phủ 448 ID. Thời gian 376 giây trên máy kiểm tra. Notebook09 tiếp tục chạy hết 19/19 cell,
0 lỗi, 11 output PNG trên cùng dữ liệu tải từ GitHub.
Chi tiết: [`github-download-verification.json`](github-download-verification.json).

```bash
python project_setup.py --verify
python -m pip check
python -m unittest discover -s tests -p 'test_*.py'
python scripts/execute_notebook.py 08_QRS_T_Loop_Reference_Error_Analysis.ipynb
```

Các kết quả này xác nhận khả năng chạy và tính nhất quán dữ liệu; không phải đánh giá
chuyên gia về độ chính xác sinh lý của từng biên QRS/T. Chưa có reference độc lập
được chấp nhận để khẳng định V2 chính xác hơn V1.
