# Kiểm tra chạy lại — 21/10/2026

## Lượt kiểm tra mới nhất — source commit `3143d89`

Đã chạy lại toàn bộ **Notebook01–09: 179/179 code cell, 0 lỗi thực thi** trong
clone riêng `/private/tmp/vcg-notebook-check-v8a5yp74/repo`. Mỗi notebook dùng
kernel Python 3.11.15 mới, với thư viện theo `requirements.txt`. Không chép
artifacts, checkpoints hoặc kết quả cũ vào clone. Dữ liệu PTB đã có được liên kết
vào clone và kiểm SHA-256 đủ 1.651 file trước khi chạy; lượt này không tải lại PTB.

- Code cell 01–09 lần lượt: **20/21/9/12/39/11/19/29/19**, chạy đủ tất cả.
- Tổng thời gian runner: **452,15 giây**; Notebook08 gồm chuẩn bị V2 mất 272,39 giây.
- Notebook đọc features riêng: **3/3 cell cho V1 và 3/3 cho V2**, không lỗi.
  Sáu CSV công khai trên GitHub trả HTTP 200, dung lượng và SHA-256 khớp metadata.
- Các notebook có output và log của lượt này nằm trong thư mục tạm bên trên,
  không tạo thêm bộ kết quả lớn trong workspace nghiên cứu.

### Kiểm tra dữ liệu và hình sau thực thi

- V1 tính mới đủ 448 ID, có 443 median và 441 dòng đủ 12 feature. Chia tập
  263/70/115; thiếu feature 4/3/0. Các giá trị và NaN khớp chính xác bộ
  `data/features/v1/` đang phân phối; không trùng bệnh nhân giữa các tập.
- Notebook07 có 7.168 kịch bản dịch biên và 86.016 dòng sensitivity. Toàn bộ
  5.376 phép đối chiếu baseline đều khớp; gallery đủ 448 ID.
- Notebook08: 448 ID, 0 lỗi đọc/kiểm tra tín hiệu, 0 khác biệt so với V1 đã lưu,
  0 lỗi render. Có 38 trang tổng quan V1 và 75 trang T loops V1/V2; cả 113 PNG
  đều tồn tại và khớp checksum. Các ca thiếu median/T loop vẫn xuất hiện.
- V2: xử lý 448 ID trong bước chuẩn bị Notebook08, không có exception; tạo
  312 T loops, còn 136 dòng thiếu feature. Dữ liệu khớp chính xác
  `data/features/v2/`. Đủ 448 artifacts/checkpoints/diagnostics/payloads và 896 PNG.
  Notebook09 dùng lại kết quả này theo mặc định `REGENERATE_COHORT=False`.

### Phạm vi của kết luận

Notebook02 chạy pipeline legacy cho 448 ID: 446 thành công, 2 ca được ghi vào
failures (`patient007/s0029lre`, `patient010/s0042lre`, loop chỉ có 2 sample).
Đây là kết quả của pipeline cũ, khác 7 dòng thiếu feature V1 của Notebook06.
Notebook04 giữ mặc định `RUN_BATCH=False`, chạy đủ 3 demo. Notebook05 thực sự
kiểm tra 448 ID × 9 cửa sổ và 810 dòng tổng hợp, không lỗi record/cửa sổ.
Kết luận chạy thành công áp dụng cho cấu hình mặc định của source đã kiểm tra.

**Bản Notebook06 cũ ở workspace gốc có lỗi riêng:** khi chạy ba code cell đầu
bằng kernel sạch, cell số 7 báo `NameError: name 'importlib' is not defined`
tại `importlib.metadata.version(name)`. Source cũ thiếu `import importlib.metadata`;
source trên GitHub đã có import này và chạy đủ 11/11 cell. Chưa thay source cũ
hoặc kết quả đã đóng băng ở workspace. Vì vậy không gộp bản local cũ vào kết luận
179/179 cell của bản GitHub.

Nhật ký tóm tắt có số cell, source SHA-256 và phạm vi kiểm tra:
[`notebook-execution.json`](notebook-execution.json).

## Các lượt kiểm tra trước

Các số liệu bên dưới thuộc các lượt kiểm tra trước lượt `3143d89`; bộ tests riêng
178 ca chưa được chạy lại trong lượt mới nhất. Những phép kiểm có sẵn bên trong
Notebook05/09 vẫn chạy theo cấu hình mặc định của notebook.

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

Tổng của lượt trước: **179/179 code cell, 0 lỗi thực thi**, dùng code notebook
tại commit `f8b5b8a`; các cập nhật sau đó sửa hướng dẫn kernel và cách tải gói chia phần.
`notebook-execution.json` hiện ghi lượt kiểm tra mới nhất được nêu ở đầu trang.

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
