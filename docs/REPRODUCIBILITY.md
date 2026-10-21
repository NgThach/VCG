# Đầu vào để chạy lại toàn bộ notebook

## Nếu chỉ dùng VCG features có sẵn

Đọc trực tiếp `data/features/{v1,v2}/{train,valid,test}_features.csv`, hoặc mở
[`examples/Read_VCG_Features.ipynb`](../examples/Read_VCG_Features.ipynb).
Các CSV đã nằm trong Git, mỗi bộ có đủ 448 dòng với split 263/70/115.
V1 còn 7 dòng thiếu feature, V2 còn 136 dòng; giữ nguyên giá trị thiếu và cờ QC.
Không cần tải PTB hoặc gọi `project_setup.py` cho luồng này. Hướng dẫn bên dưới
dành cho việc chạy lại Notebook01–09 để trích xuất/kiểm tra/vẽ tín hiệu.

## Vì sao bản clone trước bị thiếu dữ liệu?

Git trước đây chỉ có source và các CSV legacy. Thư mục `content/` và `outputs/`
đều bị bỏ qua khi commit. Do đó không có waveform cho Notebook 01–05, median
cho 07–09, các bảng train/valid/test cho 07–09, hoặc kết quả Wavelet/TLS cho 08.

Đã bổ sung dataset tại release `data-v1`, manifest dung lượng/SHA-256 từng file,
và `project_setup.py`. Mỗi notebook gọi bước chuẩn bị ở đầu; các dữ liệu trung gian
được tạo bằng source hiện tại và đường dẫn của bản clone đang chạy.

## Những gì phải phân phối

1. **Source trong Git:** toàn bộ 9 notebook; các module `vcg_*.py`; các module/script
   được notebook import; thư mục `tests/`; `requirements.txt`; tài liệu phương pháp;
   `project_setup.py`, `project_integrity.py` và `scripts/execute_notebook.py`.
2. **Danh sách chia tập trong Git:** `data/split_manifest.csv` giữ 448 ID và phân chia
   theo bệnh nhân. Tên `val` trong manifest V1 được đổi thành `valid` ở bảng features.
3. **CSV lịch sử trong Git:** `vcg_12_features.csv`, `vcg_processing_qc.csv`,
   `vcg_processing_failures.csv`. Notebook02 và phép kiểm tra nguồn còn dùng chúng.
4. **Features hiện tại trong Git:** `data/features/` chứa sáu CSV V1/V2, metadata
   nguồn/checksum và hướng dẫn đọc; `examples/Read_VCG_Features.ipynb` chỉ cần pandas.
5. **Dataset ở GitHub Releases:** 1.651 file từ PhysioNet: 549 header, 549 ECG `.dat`, 549 Frank `.xyz`, `RECORDS`, `CONTROLS`,
   `README`, `SHA256SUMS.txt`. Gói nén 1.349.695.961 byte, chia thành 41 phần tối đa 32 MiB. `data/resources.json`
   khóa checksum của gói; `data/ptbdb-files.json` khóa checksum/dung lượng từng file.

Không cần phân phối `.DS_Store`, HTML liệt kê file tải từ website, môi trường Conda,
backup notebook, cache trình duyệt, hoặc các checkpoint có đường dẫn máy cũ.
Hình và gallery đều được tạo lại, gồm các ca không tạo được median/T loop.

## Chuỗi phụ thuộc thực tế

- **01:** raw PTB → bảng overview.
- **02:** raw PTB → thí nghiệm pipeline cũ, CSV legacy và mô hình phân loại.
- **03:** raw PTB → thí nghiệm QRS/T theo ca.
- **04:** raw PTB + module xử lý → ví dụ và batch tùy chọn.
- **05:** raw PTB + module xử lý → hướng dẫn từng bước và phân tích cửa sổ.
- **06:** raw PTB + manifest cố định → `outputs/vcg_full_evaluation/artifacts/`
  (448 JSON + 448 NPZ) và `outputs/vcg_12_features_train_valid_test/`
  (train/valid/test, manifest, cấu hình, checkpoint).
- **07:** các đầu ra 06 → `outputs/vcg_qc_audit_v1/`, gồm kiểm tra QC, độ nhạy,
  gallery và `manual_review_blinded.csv` cho 63 ca được chọn.
- **08:** các đầu ra 06–07 + raw header + kết quả V2 → gallery V1/V2 và hình toàn cohort.
  Bước chuẩn bị chạy 06, 07 và runner V2 nếu thiếu. Phần reference vẫn chấp nhận
  trạng thái chưa có nhãn độc lập; không tự tạo ground truth.
- **09:** các đầu ra 06 → tính V2, giải thích/vẽ các bước, kiểm thử và gallery.

Lệnh `python project_setup.py --notebook N` chỉ chuẩn bị **đầu vào** của notebook N.
Sau đó vẫn cần Run All hoặc `python scripts/execute_notebook.py <tên notebook>`.
Các lần chạy phụ thuộc lưu notebook có output vào `executed_*.ipynb`, source giữ nguyên.

## Kiểm tra tính toàn vẹn và chạy lại

```bash
python project_setup.py --verify
python -m pip check
python -m unittest discover -s tests -p 'test_*.py' -v
```

SHA-256 của NPZ/JSON/CSV/module vẫn tính trên từng byte. Với `.ipynb`, checksum
khóa loại cell và toàn bộ nội dung code/Markdown; bỏ output, execution count và
metadata giao diện. Vì vậy lưu hình vào notebook không làm thay đổi source digest,
còn sửa code hoặc Markdown vẫn được phát hiện.

Khi source/cấu hình/dataset/library thay đổi, các phép kiểm cũ vẫn chặn dùng checkpoint
không tương thích. Không sửa hash để ép chấp nhận kết quả cũ. Chọn thư mục kết quả mới.
Thứ tự chạy toàn bộ là 01 → 09; chạy lại Notebook02 sau 07–09 có thể sửa CSV legacy
đã được đóng băng, nên cần giữ các lần thí nghiệm riêng.

## Phạm vi kiểm tra

Kết quả chạy từ clone sạch được ghi trong `docs/VERIFICATION.md` sau đợt kiểm tra.
Chỉ báo notebook chạy thành công khi tất cả code cell chạy hết và không có exception.
Ca thiếu feature do thuật toán không tìm đủ beat/biên được giữ với trạng thái thiếu;
đó là kết quả nghiên cứu, không phải thiếu file dataset.

Việc chạy được không chứng minh biên QRS/T chính xác về lâm sàng. Các giới hạn V1/V2
và điều kiện dùng reference độc lập vẫn theo `METHOD_REPRODUCTION.md` và README.
